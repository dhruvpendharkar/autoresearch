"""`rr`: the CLI agents and humans use instead of raw node commands."""

from __future__ import annotations

import json
import secrets
import shlex
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
import yaml

from . import approval as approvalmod
from . import gitutil
from . import gpus as gpumod
from . import queue as queuemod
from .config import Config, ConfigError, load_config
from .ledger import reserved_gpus
from .nodes import NodeError, NodeRunner
from .launcher import Launcher
from .project import Project, all_ledgers, find_project
from .spec import SpecError, load_spec
from .statefile import read_json

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

_state = {"config_path": None, "project_path": None}


@app.callback()
def main(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="RR_CONFIG", help="Path to nodes.yaml (default: search upward from cwd)."
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", envvar="RR_PROJECT", help="Project root holding studies/ (default: git toplevel of cwd)."
    ),
):
    _state["config_path"] = config
    _state["project_path"] = project


def _load() -> Config:
    try:
        return load_config(_state["config_path"])
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


def _project() -> Project:
    return find_project(_state["project_path"])


def _table(rows: list, headers: list) -> str:
    cells = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in cells)


def _ago(ts: Optional[float], now: float) -> str:
    if ts is None:
        return "never"
    s = int(now - ts)
    return f"{s}s ago" if s < 120 else f"{s // 60}m ago"


@app.command("gpus")
def gpus_cmd(
    node: Optional[List[str]] = typer.Option(None, "--node", "-n", help="Probe only these nodes (repeatable)."),
    cached: bool = typer.Option(False, "--cached", help="Show the last probe from state/gpus.json without probing."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
):
    """Probe GPUs on every node and show which ones are free to use."""
    cfg = _load()
    for n in node or []:
        try:
            cfg.node(n)
        except ConfigError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(2) from None
    store = gpumod.GpuState(cfg)
    state = store.load() if cached else store.refresh(NodeRunner(cfg), node or None)
    now = time.time()
    reserved = reserved_gpus(all_ledgers(cfg, _project()))
    statuses = gpumod.classify(state, cfg, reserved, now)
    total_held, _ = gpumod.held_counts(reserved)
    cap = cfg.policy.max_total_at(datetime.fromtimestamp(now))

    if as_json:
        out = {
            "probed_at": state.get("updated_at"),
            "held": total_held,
            "cap_total": cap,
            "cap_per_node": cfg.policy.max_gpus_per_node,
            "nodes": {
                name: {k: ns.get(k) for k in ("ok", "error", "probed_at", "user", "source")}
                for name, ns in state.get("nodes", {}).items()
                if name in cfg.nodes
            },
            "gpus": [
                {
                    "node": s.node,
                    "index": s.index,
                    "status": s.status,
                    "gpu_type": cfg.nodes[s.node].gpu_type,
                    "mem_used_mb": s.gpu.get("mem_used_mb"),
                    "mem_total_mb": s.gpu.get("mem_total_mb"),
                    "util_pct": s.gpu.get("util_pct"),
                    "foreign_users": s.gpu.get("foreign_users", []),
                    "idle_streak": s.gpu.get("idle_streak", 0),
                }
                for s in statuses
            ],
        }
        typer.echo(json.dumps(out, indent=2))
        return

    rows = []
    for s in statuses:
        g = s.gpu
        mem = f"{g.get('mem_used_mb', '?')}/{g.get('mem_total_mb', '?')}"
        util = "?" if g.get("util_pct") is None else f"{g['util_pct']}%"
        users = ",".join(g.get("foreign_users") or []) or "-"
        idle = f"{min(g.get('idle_streak', 0), cfg.policy.idle_probes)}/{cfg.policy.idle_probes}"
        rows.append([s.node, s.index, cfg.nodes[s.node].gpu_type, mem, util, users, idle, s.status])
    if rows:
        typer.echo(_table(rows, ["NODE", "GPU", "TYPE", "MEM MB", "UTIL", "OTHER USERS", "IDLE", "STATUS"]))
    for name in cfg.nodes:
        ns = state.get("nodes", {}).get(name)
        if ns is None:
            typer.secho(f"{name}: never probed", fg=typer.colors.YELLOW)
        elif not ns.get("ok"):
            err = str(ns.get("error"))
            err = err[len(name) + 2 :] if err.startswith(f"{name}: ") else err
            typer.secho(f"{name}: probe failed: {err}", fg=typer.colors.RED)
    free = sum(1 for s in statuses if s.status == "free")
    typer.echo(
        f"\nfree {free} · held {total_held}/{cap} total, {cfg.policy.max_gpus_per_node}/node"
        f" · probed {_ago(state.get('updated_at'), now)}"
    )


DOCTOR_COMMAND = """
echo @@HOST; hostname
echo @@WHOAMI; whoami
echo @@GPUSTAT; command -v gpustat || true
echo @@NVSMI; command -v nvidia-smi || true
echo @@TMUX; command -v tmux || true
echo @@NONCE; cat {nonce_path} 2>/dev/null || true
""".strip()

NET_TARGETS = {
    "Claude Code (api.anthropic.com)": "https://api.anthropic.com",
    "WandB (api.wandb.ai)": "https://api.wandb.ai",
    "ntfy (ntfy.sh)": "https://ntfy.sh",
}


def _reachable(url: str, timeout: float = 5) -> tuple:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
        return True, "ok"
    except urllib.error.HTTPError as e:
        return True, f"HTTP {e.code}"  # any HTTP answer means the network path works
    except Exception as e:  # noqa: BLE001 - report any network failure
        return False, str(getattr(e, "reason", e))


@app.command()
def doctor(
    node: Optional[List[str]] = typer.Option(None, "--node", "-n", help="Check only these nodes (repeatable)."),
    net: bool = typer.Option(True, "--net/--no-net", help="Check outbound internet from this machine."),
):
    """Check each node command and the cluster assumptions the design depends on."""
    cfg = _load()
    runner = NodeRunner(cfg)
    names = node or list(cfg.nodes)
    nonce = secrets.token_hex(8)
    nonce_path = cfg.state_dir / "doctor-nonce"
    nonce_path.parent.mkdir(parents=True, exist_ok=True)
    nonce_path.write_text(nonce + "\n")
    ok_all = True

    def line(ok: Optional[bool], label: str, detail: str = "") -> None:
        mark = {True: ("ok  ", typer.colors.GREEN), False: ("FAIL", typer.colors.RED), None: ("warn", typer.colors.YELLOW)}[ok]
        typer.secho(f"  [{mark[0]}] ", fg=mark[1], nl=False)
        typer.echo(f"{label}{': ' + detail if detail else ''}")

    for name in names:
        typer.echo(f"{name}  ({cfg.node(name).exec})")
        try:
            res = runner.run(name, DOCTOR_COMMAND.format(nonce_path=shlex.quote(str(nonce_path))))
        except NodeError as e:
            ok_all = False
            line(False, "runs a non-interactive command", str(e))
            continue
        s = gpumod.split_sections(res.stdout)
        line(True, "runs a non-interactive command", f"host {s.get('HOST')}, user {s.get('WHOAMI')}")
        shared = s.get("NONCE") == nonce
        ok_all &= shared
        line(shared, "shares this filesystem", str(cfg.state_dir) if shared else f"{nonce_path} not visible on node")
        has_tmux = bool(s.get("TMUX"))
        ok_all &= has_tmux
        line(has_tmux, "tmux installed", s.get("TMUX") or "not found")
        line(True if s.get("GPUSTAT") else None, "gpustat installed", s.get("GPUSTAT") or "not found; will use nvidia-smi")
        try:
            probe = gpumod.probe_node(runner, name)
        except (NodeError, gpumod.ProbeError) as e:
            ok_all = False
            line(False, "GPU probe", str(e))
            continue
        line(True, "GPU probe", f"{len(probe.gpus)} GPU(s) via {probe.source}")
        if len(probe.gpus) != cfg.node(name).gpus:
            line(None, "GPU count", f"nodes.yaml says {cfg.node(name).gpus}, probe found {len(probe.gpus)}")
        procs = [p for g in probe.gpus for p in g.procs]
        unknown = sum(1 for p in procs if p.user is None)
        if not procs:
            line(None, "process owners visible", "no GPU processes right now; can't tell")
        else:
            line(unknown == 0 or None, "process owners visible", f"{len(procs) - unknown}/{len(procs)} attributed")

    if net:
        typer.echo("this machine (controller)")
        for label, url in NET_TARGETS.items():
            reachable, detail = _reachable(url)
            line(reachable or None, f"outbound to {label}", detail)

    nonce_path.unlink(missing_ok=True)
    raise typer.Exit(0 if ok_all else 1)


# ---------------------------------------------------------------- studies and runs


def _launcher(cfg: Config) -> Launcher:
    return Launcher(cfg, _project())


def _spec(study: str):
    try:
        return load_spec(_project().study_dir(study) / "spec.yaml")
    except SpecError as e:
        typer.secho(f"spec error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


def _fail(msg: str, code: int = 1) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _print_outcomes(outcomes: list, show_waiting: bool = True) -> None:
    colors = {"launched": typer.colors.GREEN, "done": typer.colors.GREEN, "crashed": typer.colors.RED,
              "lost": typer.colors.RED, "invalid": typer.colors.RED, "waiting": None, "skipped": typer.colors.YELLOW}
    for o in outcomes:
        if o.action == "waiting" and not show_waiting:
            continue
        typer.secho(f"  {o}", fg=colors.get(o.action))


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


@app.command()
def approve(study: str = typer.Argument(..., help="Study id (directory under studies/).")):
    """Pin a study's spec, arm commits and eval data. Must be run by a human at a terminal."""
    cfg = _load()
    project = _project()
    spec = _spec(study)
    try:
        record = approvalmod.prepare(project, spec, cfg.state_dir / "hashcache.json")
    except (approvalmod.ApprovalError, gitutil.GitError) as e:
        _fail(f"cannot approve: {e}")
    current = approvalmod.check(project, spec, cfg.state_dir / "hashcache.json")
    typer.echo(f"study {spec.study}")
    typer.echo(f"  metric       {spec.metric.name} ({spec.metric.direction}) from {spec.metric.source}")
    typer.echo(f"  seeds        {', '.join(map(str, spec.seeds))}")
    typer.echo(f"  base_commit  {record['base_commit'][:12]} ({spec.base_commit})")
    for arm in spec.arms:
        src = f"branch {arm.branch}" if arm.branch else "base_commit"
        ov = f"  overrides {arm.overrides}" if arm.overrides else ""
        typer.echo(f"  arm {arm.id:<12} {record['commits'][arm.id][:12]} ({src}) config {arm.config}{ov}")
    for c in spec.comparisons:
        typer.echo(f"  compare      {c.treatment} vs {c.control} ({c.test})")
    typer.echo(f"  budget       {spec.gpu_hours:g} GPU-h, {spec.gpus_per_run} GPU(s)/run, {spec.run_timeout_h:g} h timeout")
    typer.echo(f"  protected    {', '.join(spec.protected_paths) or '-'}  (no arm changes them)")
    typer.echo(f"  eval data    {', '.join(spec.eval_data) or '-'}  {(record['eval_data_sha256'] or '')[:12]}")
    if current.valid and current.approval["approval_hash"] == record["approval_hash"]:
        typer.secho("already approved with these exact contents", fg=typer.colors.GREEN)
        return
    if not _stdin_is_tty():
        _fail("rr approve must be run by a human at a terminal; refusing non-interactive approval", 3)
    typed = typer.prompt(f"type the study id ({spec.study}) to approve")
    if typed.strip() != spec.study:
        _fail("not approved")
    approvalmod.write(project, spec, record)
    typer.secho(f"approved {spec.study} ({record['approval_hash'][:12]})", fg=typer.colors.GREEN)


def _parse_set(items: List[str]) -> dict:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            _fail(f"--set expects key=value, got {item!r}", 2)
        out[key] = yaml.safe_load(value) if value else ""
    return out


@app.command()
def submit(
    study: str = typer.Argument(..., help="Study id."),
    smoke: bool = typer.Option(False, "--smoke", help="Queue a short smoke run per arm (no approval needed)."),
    exploratory: bool = typer.Option(False, "--exploratory", help="Queue exploratory runs with --set overrides."),
    arm: Optional[List[str]] = typer.Option(None, "--arm", "-a", help="Only these arms (repeatable)."),
    seed: Optional[List[int]] = typer.Option(None, "--seed", "-s", help="Only these seeds (repeatable)."),
    set_: Optional[List[str]] = typer.Option(None, "--set", help="key=value override (exploratory only)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be queued."),
    no_launch: bool = typer.Option(False, "--no-launch", help="Queue only; don't try to launch now."),
):
    """Queue runs for a study, then try to launch them onto free GPUs."""
    if smoke and exploratory:
        _fail("choose one of --smoke and --exploratory", 2)
    kind = "smoke" if smoke else "exploratory" if exploratory else "full"
    cfg = _load()
    spec = _spec(study)
    lc = _launcher(cfg)
    ledger = lc.ledger(study)
    try:
        planned, skipped = queuemod.plan(
            lc.project, spec, ledger, kind, arm, seed, _parse_set(set_ or []), cache_path=lc.hash_cache
        )
    except (queuemod.SubmitError, SpecError, gitutil.GitError) as e:
        _fail(f"cannot submit: {e}")
    for s in skipped:
        typer.echo(f"  skip {s.arm} seed {s.seed}: {s.reason}")
    for p in planned:
        f = p.fields
        typer.echo(f"  {'would queue' if dry_run else 'queue'} {p.run_id}  {f['git_sha'][:7]}  {f['n_gpus']} GPU(s)  {f['command']}")
    if dry_run or not planned:
        if not planned:
            typer.echo("nothing to queue")
        return
    queuemod.enqueue(ledger, planned)
    if no_launch:
        return
    _print_outcomes(lc.tick([study]))


@app.command()
def launch(
    study: Optional[List[str]] = typer.Argument(None, help="Studies to consider (default: all)."),
    wait: float = typer.Option(0, "--wait", "-w", help="Keep trying for this many minutes until nothing is queued."),
):
    """One scheduling pass: collect finished runs, then launch queued runs that fit."""
    cfg = _load()
    lc = _launcher(cfg)
    deadline = time.time() + wait * 60
    while True:
        out = lc.tick(study or None)
        waiting = [o for o in out if o.action == "waiting"]
        last = not waiting or time.time() >= deadline
        _print_outcomes(out, show_waiting=last)
        if last:
            return
        time.sleep(cfg.policy.probe_interval_s)


def _heartbeat(run_dir: Path, now: float) -> str:
    hb = read_json(run_dir / "heartbeat.json", None)
    if not isinstance(hb, dict):
        return "-"
    return f"step {hb.get('step', '?')}, {_ago(hb.get('ts'), now)}"


def _duration(rec: dict, now: float) -> str:
    start = rec.get("started_at")
    if start is None:
        return f"queued {_ago(rec.get('created_at'), now)}" if rec["state"] == "queued" else "-"
    s = int((rec.get("ended_at") or now) - start)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


@app.command()
def status(
    study: Optional[List[str]] = typer.Argument(None, help="Studies to show (default: all)."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    no_reconcile: bool = typer.Option(False, "--no-reconcile", help="Show the ledger as is, without checking nodes."),
):
    """Show approval, budget and every run of each study."""
    cfg = _load()
    lc = _launcher(cfg)
    studies = study or lc.project.study_ids()
    if not no_reconcile:
        lc.reconcile(studies)
    now = time.time()
    report = []
    for sid in studies:
        spec = _spec(sid)
        st = approvalmod.check(lc.project, spec, lc.hash_cache)
        runs = list(lc.ledger(sid).runs().values())
        used = queuemod.committed_gpu_hours(runs, queuemod.MAIN_BUDGET_KINDS, now)
        report.append({"study": sid, "approval": {"valid": st.valid, "reason": st.reason},
                       "budget": {"committed_gpu_hours": round(used, 2), "gpu_hours": spec.gpu_hours}, "runs": runs})
        if as_json:
            continue
        typer.secho(f"{sid}", bold=True)
        typer.secho(f"  approval: {st.reason}", fg=typer.colors.GREEN if st.valid else typer.colors.YELLOW)
        typer.echo(f"  budget:   {used:.1f} / {spec.gpu_hours:g} GPU-h committed (live runs counted at their timeout)")
        if not runs:
            typer.echo("  no runs")
            continue
        rows = []
        for r in runs:
            where = f"{r['node']}:{','.join(map(str, r['gpus']))}" if r.get("node") else "-"
            metric = f"{r['metric']:.4g}" if isinstance(r.get("metric"), (int, float)) else "-"
            hb = _heartbeat(Path(r["run_dir"]), now) if r["state"] in ("launching", "running") else "-"
            rows.append([r["run_id"], r["kind"], r["state"], where, _duration(r, now), hb, metric, (r.get("notes") or "")[:60]])
        typer.echo(_table(rows, ["RUN", "KIND", "STATE", "WHERE", "TIME", "HEARTBEAT", "METRIC", "NOTES"]))
    if as_json:
        typer.echo(json.dumps(report, indent=2, default=str))


def _find_run(lc: Launcher, ref: str) -> tuple:
    """`study/run_id`, or a run_id that is unique across studies."""
    if "/" in ref:
        sid, rid = ref.split("/", 1)
        rec = lc.ledger(sid).get(rid)
        if rec is None:
            _fail(f"no run {rid!r} in study {sid!r}", 2)
        return sid, rec
    hits = [(sid, lc.ledger(sid).get(ref)) for sid in lc.project.study_ids()]
    hits = [(sid, rec) for sid, rec in hits if rec is not None]
    if not hits:
        _fail(f"no run {ref!r} in any study", 2)
    if len(hits) > 1:
        _fail(f"{ref!r} exists in several studies; use study/run_id: {', '.join(s for s, _ in hits)}", 2)
    return hits[0]


@app.command()
def logs(
    run: str = typer.Argument(..., help="run_id or study/run_id."),
    lines: int = typer.Option(200, "--lines", "-n", help="Number of lines from the end."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Keep printing new lines."),
):
    """Print the end of a run's log."""
    cfg = _load()
    _, rec = _find_run(_launcher(cfg), run)
    path = Path(rec["run_dir"]) / "log.txt"
    if not path.exists():
        _fail(f"no log yet at {path} (state: {rec['state']})")
    with open(path, errors="replace") as fh:
        for line in deque(fh, maxlen=lines):
            typer.echo(line, nl=False)
        while follow:
            line = fh.readline()
            if line:
                typer.echo(line, nl=False)
            else:
                time.sleep(1)


@app.command()
def cancel(
    run: Optional[List[str]] = typer.Argument(None, help="run_id or study/run_id (repeatable)."),
    study: Optional[str] = typer.Option(None, "--study", help="Cancel every queued or live run of this study."),
    reason: str = typer.Option("cancelled by user", "--reason", help="Recorded in the ledger."),
):
    """Cancel runs. Only kills rr-* tmux sessions that this system's ledger launched."""
    cfg = _load()
    lc = _launcher(cfg)
    targets = []
    if study:
        targets += [(study, r) for r in lc.ledger(study).runs().values() if r["state"] in ("queued", "launching", "running")]
    targets += [_find_run(lc, ref) for ref in run or []]
    if not targets:
        _fail("nothing to cancel", 2)
    failed = False
    for sid, rec in targets:
        try:
            typer.echo(f"  {lc.cancel(lc.ledger(sid), rec['run_id'], reason)}")
        except (NodeError, ValueError) as e:
            failed = True
            typer.secho(f"  {sid}/{rec['run_id']}: {e}", fg=typer.colors.RED)
    if failed:
        raise typer.Exit(1)


@app.command()
def drain(off: bool = typer.Option(False, "--off", help="Resume launching.")):
    """Stop launching new runs; running ones finish normally."""
    cfg = _load()
    flag = cfg.state_dir / "DRAIN"
    if off:
        flag.unlink(missing_ok=True)
        typer.echo("launching resumed")
    else:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        typer.echo(f"draining: nothing new will launch ({flag}); `rr drain --off` to resume")
