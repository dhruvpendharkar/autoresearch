# autoresearch
small autoresearch setup for student researchers

A file-driven research runner for a shared GPU cluster with no scheduler. See
[the design plan](Autoresearch%20System%20Design%20&%20Implementation%20Plan.md).

## Status

| Stage | Scope | State |
|---|---|---|
| 1. Foundations | `nodes.yaml`, node access, GPU probing and allocation, ledger, `rr gpus`, `rr doctor` | done |
| 2. Manual workflow | spec + `rr approve`, launcher (worktree, `run.sh`, tmux), frozen metric parser, `rr submit/status/logs/cancel/drain` | next |
| 3. Daemon | reconcile, deterministic recovery, health checks, PAUSE, keepalive, alerts | |
| 4. Claude integration | call budget, triage/report skills, report tables, hooks, permissions | |

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp nodes.example.yaml nodes.yaml   # fill in your per-node commands
.venv/bin/rr doctor                # checks the design's open questions on each node
.venv/bin/rr gpus                  # probe and show which GPUs are free
```

`rr` finds `nodes.yaml` via `--config`, then `$RR_CONFIG`, then by searching
upward from the current directory. State goes in `state/` next to it.

`rr doctor` checks, per node: that the node command runs one non-interactive
command, that the node sees this filesystem, and that tmux and gpustat are
installed. It also probes the GPUs and checks whether other users' process
owners are visible, then checks outbound internet from this machine.

## Tests

```bash
.venv/bin/pytest
```

The tests run against a fake cluster: each "node" is a local `bash -c` with
fake `nvidia-smi`, `gpustat`, `ps` and `whoami` on its PATH
([tests/fake_tools.py](tests/fake_tools.py)), so the real probe shell script
is exercised without GPUs.
