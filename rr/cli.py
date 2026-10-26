"""`rr`: the CLI agents and humans use instead of raw node commands."""

from __future__ import annotations

import json
import secrets
import shlex
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer

from . import gpus as gpumod
from .config import Config, ConfigError, load_config
from .ledger import project_ledgers, reserved_gpus
from .nodes import NodeError, NodeRunner

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

_state = {"config_path": None}


@app.callback()
def main(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="RR_CONFIG", help="Path to nodes.yaml (default: search upward from cwd)."
    ),
):
    _state["config_path"] = config


def _load() -> Config:
    try:
        return load_config(_state["config_path"])
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


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
    project: Path = typer.Option(Path("."), "--project", help="Project root whose studies' ledgers hold reservations."),
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
    reserved = reserved_gpus(project_ledgers(project))
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
