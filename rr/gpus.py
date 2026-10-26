"""Probe GPUs, decide which ones are free, and pick GPUs for a run within caps.

A GPU counts as free only when all of these hold:
  - no process belonging to another user (a process whose owner we cannot
    determine counts as another user's);
  - memory used < free_mem_mb and utilization < free_util_pct;
  - both true for `idle_probes` consecutive probes spanning at least
    (idle_probes - 1) probe intervals, so rapid manual `rr gpus` calls can't
    shortcut the hysteresis;
  - not reserved by one of our runs, not cooling down after a collision;
  - taking it keeps us within the total and per-node caps.

Probe results and idle streaks are stored in `state/gpus.json`.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional

from .config import Config, Policy
from .nodes import NodeError, NodeRunner
from .statefile import locked_json, read_json

# One round trip per node. Prefer gpustat (it attributes processes to users);
# fall back to nvidia-smi plus ps. `whoami` tells us which user is "us" on
# that node.
PROBE_COMMAND = r"""
echo @@WHOAMI
whoami
if command -v gpustat >/dev/null 2>&1 && __rr_gs=$(gpustat --json 2>/dev/null); then
  echo @@GPUSTAT
  printf '%s\n' "$__rr_gs"
else
  echo @@NVSMI
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || exit 3
  echo @@APPS
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
  echo @@USERS
  __rr_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | paste -sd, -)
  if [ -n "$__rr_pids" ]; then ps -o pid=,user= -p "$__rr_pids"; fi
fi
true
""".strip()

# A probe more than this many intervals after the previous one restarts the
# idle streak: we weren't watching in between.
STALE_PROBE_INTERVALS = 3
# Allow probes to arrive a little early when checking the idle duration.
INTERVAL_SLACK = 0.9


class ProbeError(Exception):
    pass


@dataclass
class GpuProc:
    pid: Optional[int]
    user: Optional[str]
    mem_mb: Optional[int]


@dataclass
class GpuSample:
    index: int
    uuid: str
    name: str
    mem_used_mb: Optional[int]
    mem_total_mb: Optional[int]
    util_pct: Optional[int]
    procs: list = field(default_factory=list)  # list[GpuProc]


@dataclass
class ProbeResult:
    user: str
    source: str  # "gpustat" or "nvidia-smi"
    gpus: list  # list[GpuSample]


# ---------------------------------------------------------------- parsing


def _int_or_none(text) -> Optional[int]:
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return int(text)
    s = str(text).strip()
    if not s or s.startswith("[") or s.upper() in ("N/A", "NONE"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def split_sections(stdout: str) -> dict:
    sections, current = {}, None
    for line in stdout.splitlines():
        if line.startswith("@@") and line[2:].isalpha():
            current = line[2:]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _csv_rows(text: str) -> list:
    return [row for row in csv.reader(io.StringIO(text), skipinitialspace=True) if row]


def parse_gpustat(text: str) -> list:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProbeError(f"gpustat output is not JSON: {e}") from None
    gpus = []
    for g in data.get("gpus", []):
        procs = [
            GpuProc(
                pid=_int_or_none(p.get("pid")),
                user=p.get("username") or None,
                mem_mb=_int_or_none(p.get("gpu_memory_usage")),
            )
            for p in (g.get("processes") or [])
        ]
        gpus.append(
            GpuSample(
                index=int(g["index"]),
                uuid=str(g.get("uuid") or f"index-{g['index']}"),
                name=str(g.get("name") or ""),
                mem_used_mb=_int_or_none(g.get("memory.used")),
                mem_total_mb=_int_or_none(g.get("memory.total")),
                util_pct=_int_or_none(g.get("utilization.gpu")),
                procs=procs,
            )
        )
    return gpus


def parse_nvidia_smi(gpu_csv: str, apps_csv: str, users_text: str) -> list:
    owners = {}
    for line in users_text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            owners[int(parts[0])] = parts[1]

    procs_by_uuid = {}
    for row in _csv_rows(apps_csv):
        if len(row) < 3 or "no running" in row[0].lower():
            continue
        pid = _int_or_none(row[1])
        procs_by_uuid.setdefault(row[0].strip(), []).append(
            GpuProc(pid=pid, user=owners.get(pid), mem_mb=_int_or_none(row[2]))
        )

    gpus = []
    for row in _csv_rows(gpu_csv):
        if len(row) < 6:
            raise ProbeError(f"unexpected nvidia-smi row: {row!r}")
        uuid = row[1].strip()
        gpus.append(
            GpuSample(
                index=int(row[0]),
                uuid=uuid,
                name=row[2].strip(),
                mem_used_mb=_int_or_none(row[3]),
                mem_total_mb=_int_or_none(row[4]),
                util_pct=_int_or_none(row[5]),
                procs=procs_by_uuid.get(uuid, []),
            )
        )
    return gpus


def parse_probe(stdout: str) -> ProbeResult:
    s = split_sections(stdout)
    user = s.get("WHOAMI", "").strip()
    if not user:
        raise ProbeError("probe did not report the current user")
    if "GPUSTAT" in s:
        return ProbeResult(user, "gpustat", parse_gpustat(s["GPUSTAT"]))
    if "NVSMI" in s:
        return ProbeResult(
            user, "nvidia-smi", parse_nvidia_smi(s["NVSMI"], s.get("APPS", ""), s.get("USERS", ""))
        )
    raise ProbeError("probe output has neither gpustat nor nvidia-smi section")


def probe_node(runner: NodeRunner, node: str) -> ProbeResult:
    res = runner.run(node, PROBE_COMMAND)
    if res.returncode != 0:
        raise ProbeError(f"{node}: probe exited {res.returncode}: {res.stderr.strip()[-300:]}")
    return parse_probe(res.stdout)


# ---------------------------------------------------------------- idleness


def foreign_users(sample: GpuSample, me: str) -> list:
    """Users other than `me` with processes on this GPU. Unknown owners count as foreign."""
    return sorted({p.user or "?" for p in sample.procs if p.user != me})


def is_idle_now(sample: GpuSample, me: str, policy: Policy) -> bool:
    """Instantaneous idleness. Unknown memory or utilization counts as busy."""
    if foreign_users(sample, me):
        return False
    if sample.mem_used_mb is None or sample.util_pct is None:
        return False
    return sample.mem_used_mb < policy.free_mem_mb and sample.util_pct < policy.free_util_pct


def update_node_state(prev: Optional[dict], probe: ProbeResult, policy: Policy, now: float) -> dict:
    """Fold a new probe into a node's stored state, carrying idle streaks forward."""
    prev = prev or {}
    stale = (
        not prev.get("ok")
        or prev.get("probed_at") is None
        or now - prev["probed_at"] > STALE_PROBE_INTERVALS * policy.probe_interval_s
    )
    prev_by_uuid = {g["uuid"]: g for g in prev.get("gpus", [])}
    gpus = []
    for sample in probe.gpus:
        old = {} if stale else prev_by_uuid.get(sample.uuid, {})
        idle = is_idle_now(sample, probe.user, policy)
        if idle:
            streak = old.get("idle_streak", 0) + 1
            idle_since = old.get("idle_since") or now
        else:
            streak, idle_since = 0, None
        g = asdict(sample)
        g.update(
            idle_now=idle,
            idle_streak=streak,
            idle_since=idle_since,
            foreign_users=foreign_users(sample, probe.user),
        )
        gpus.append(g)
    return {"ok": True, "error": None, "probed_at": now, "user": probe.user, "source": probe.source, "gpus": gpus}


def failed_node_state(prev: Optional[dict], error: str, now: float) -> dict:
    """A failed probe keeps the last GPU table for display but resets every streak."""
    prev = prev or {}
    gpus = [dict(g, idle_now=False, idle_streak=0, idle_since=None) for g in prev.get("gpus", [])]
    return {
        "ok": False,
        "error": error,
        "probed_at": now,
        "last_ok_at": prev.get("probed_at") if prev.get("ok") else prev.get("last_ok_at"),
        "user": prev.get("user"),
        "source": prev.get("source"),
        "gpus": gpus,
    }


def has_settled(gpu: dict, policy: Policy, now: float) -> bool:
    if gpu.get("idle_streak", 0) < policy.idle_probes or gpu.get("idle_since") is None:
        return False
    needed = (policy.idle_probes - 1) * policy.probe_interval_s * INTERVAL_SLACK
    return now - gpu["idle_since"] >= needed


# ---------------------------------------------------------------- state store


class GpuState:
    """`state/gpus.json`: last probe per node, idle streaks, and cooldowns."""

    def __init__(self, config: Config, clock: Callable[[], float] = time.time):
        self.config = config
        self.clock = clock
        self.path = config.state_dir / "gpus.json"

    def load(self) -> dict:
        data = read_json(self.path, None) or {}
        data.setdefault("nodes", {})
        data.setdefault("cooldowns", {})
        return data

    def refresh(self, runner: NodeRunner, nodes: Optional[Iterable[str]] = None) -> dict:
        """Probe the given nodes (default: all) and persist the updated state."""
        names = list(nodes) if nodes is not None else list(self.config.nodes)
        results = {}
        for name in names:
            try:
                results[name] = probe_node(runner, name)
            except (NodeError, ProbeError) as e:
                results[name] = e
        now = self.clock()
        with locked_json(self.path, {}) as data:
            data.setdefault("nodes", {})
            cooldowns = data.setdefault("cooldowns", {})
            for name, res in results.items():
                prev = data["nodes"].get(name)
                if isinstance(res, Exception):
                    data["nodes"][name] = failed_node_state(prev, str(res), now)
                else:
                    data["nodes"][name] = update_node_state(prev, res, self.config.policy, now)
            for key in [k for k, until in cooldowns.items() if until <= now]:
                del cooldowns[key]
            data["updated_at"] = now
            snapshot = json.loads(json.dumps(data))
        return snapshot

    def add_cooldown(self, node: str, index: int, minutes: Optional[int] = None) -> None:
        minutes = self.config.policy.cooldown_min if minutes is None else minutes
        with locked_json(self.path, {}) as data:
            data.setdefault("cooldowns", {})[gpu_key(node, index)] = self.clock() + minutes * 60


def gpu_key(node: str, index: int) -> str:
    return f"{node}:{index}"


# ---------------------------------------------------------------- allocation


@dataclass
class GpuStatus:
    node: str
    index: int
    status: str  # free | settling | busy | foreign | reserved | collision | cooldown | node-down
    gpu: dict


def classify(state: dict, config: Config, reserved: set, now: float) -> list:
    """Label every known GPU. `reserved` is a set of (node, index) held by our runs."""
    out = []
    cooldowns = state.get("cooldowns", {})
    for name in config.nodes:
        ns = state.get("nodes", {}).get(name)
        if ns is None:
            continue
        for g in ns.get("gpus", []):
            idx = g["index"]
            if (name, idx) in reserved:
                status = "collision" if ns.get("ok") and g.get("foreign_users") else "reserved"
            elif not ns.get("ok"):
                status = "node-down"
            elif cooldowns.get(gpu_key(name, idx), 0) > now:
                status = "cooldown"
            elif g.get("foreign_users"):
                status = "foreign"
            elif not g.get("idle_now"):
                status = "busy"
            elif has_settled(g, config.policy, now):
                status = "free"
            else:
                status = "settling"
            out.append(GpuStatus(name, idx, status, g))
    return out


@dataclass
class Allocation:
    node: str
    gpus: list  # list[int]
    gpu_type: str


def held_counts(reserved: set) -> tuple:
    per_node = {}
    for node, _ in reserved:
        per_node[node] = per_node.get(node, 0) + 1
    return len(reserved), per_node


def allocate(
    state: dict,
    config: Config,
    k: int,
    reserved: set,
    now: float,
    gpu_type: Optional[str] = None,
    wall_clock: Optional[datetime] = None,
) -> Optional[Allocation]:
    """Pick k free GPUs on one node within caps, or None if that isn't possible now.

    Prefers the node with the most free GPUs (least contended), then the
    lowest GPU indices. Runs never span nodes.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    policy = config.policy
    total_held, per_node = held_counts(reserved)
    wall_clock = wall_clock or datetime.fromtimestamp(now)
    if total_held + k > policy.max_total_at(wall_clock):
        return None

    free_by_node = {}
    for s in classify(state, config, reserved, now):
        if s.status == "free":
            free_by_node.setdefault(s.node, []).append(s.index)

    candidates = []
    for name, free in free_by_node.items():
        node = config.nodes[name]
        if gpu_type is not None and node.gpu_type != gpu_type:
            continue
        if per_node.get(name, 0) + k > policy.max_gpus_per_node:
            continue
        if len(free) < k:
            continue
        candidates.append((-len(free), name, sorted(free)))
    if not candidates:
        return None
    _, name, free = min(candidates)
    return Allocation(node=name, gpus=free[:k], gpu_type=config.nodes[name].gpu_type)


def still_idle(probe: ProbeResult, indices: Iterable[int], policy: Policy) -> bool:
    """Re-check chosen GPUs against a fresh probe right before launch."""
    by_index = {g.index: g for g in probe.gpus}
    return all(i in by_index and is_idle_now(by_index[i], probe.user, policy) for i in indices)
