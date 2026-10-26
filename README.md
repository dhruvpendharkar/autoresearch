# autoresearch
small autoresearch setup for student researchers

A file-driven research runner for a shared GPU cluster with no scheduler. See
[the design plan](Autoresearch%20System%20Design%20&%20Implementation%20Plan.md).

## Status

| Stage | Scope | State |
|---|---|---|
| 1. Foundations | `nodes.yaml`, node access, GPU probing and allocation, ledger, `rr gpus`, `rr doctor` | done |
| 2. Manual workflow | spec + `rr approve`, launcher (worktree, `run.sh`, tmux), frozen metric parser, `rr submit/launch/status/logs/cancel/drain` | done |
| 3. Daemon | `rr daemon` loop, deterministic recovery, stall/NaN checks, keepalive, alerts, inbox | next |
| 4. Claude integration | call budget, triage/report skills, report tables, hooks, permissions | |

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp nodes.example.yaml nodes.yaml   # fill in your per-node commands
.venv/bin/rr doctor                # checks the design's open questions on each node
.venv/bin/rr gpus                  # probe and show which GPUs are free
```

`rr` finds `nodes.yaml` via `--config`, then `$RR_CONFIG`, then by searching
upward from the current directory. State goes in `state/` next to it. The
research project is `--project`, then `$RR_PROJECT`, then the git toplevel of
the current directory.

`rr doctor` checks, per node: that the node command runs one non-interactive
command, that the node sees this filesystem, and that tmux and gpustat are
installed. It also probes the GPUs and checks whether other users' process
owners are visible, then checks outbound internet from this machine.

## Running a study by hand

In your research project (add `runs/` to its `.gitignore`):

```
studies/<id>/brief.md      # prose: hypothesis, method, what would falsify it (you write)
studies/<id>/spec.yaml     # pre-registration (you write); see examples/studies/my-study/
studies/<id>/approval.json # written by rr approve
studies/<id>/ledger.jsonl  # append-only, one line per run state change (rr writes)
runs/<id>/<run_id>/        # code/ (git worktree at the exact commit), run.sh, log.txt,
                           # heartbeat.json, final_metrics.json, exit_code
```

```bash
rr submit my-study --smoke     # one short run per arm, at each arm's current commit
rr status                      # collects finished runs, shows heartbeats and metrics
rr approve my-study            # you, at a terminal: pins spec, arm commits, eval data
rr submit my-study             # queue every arm x seed; launches what fits now
rr launch --wait 600           # keep launching as GPUs free up (the daemon will do this)
rr logs method-s1-a01 -f       # or: tmux attach -t rr-my-study-method-s1-a01 on its node
rr cancel method-s1-a01        # or --study my-study
rr drain / rr drain --off      # stop / resume launching; `touch PAUSE` also stops launches
```

The rules `rr` enforces:

- **Approval pins everything.** Full and exploratory runs launch only with a
  valid approval, from the commits pinned at approval, even if a branch moves
  later. Editing the spec file in any way, or changing the eval data, makes the
  approval stale. `rr approve` refuses arms that change `protected_paths`
  relative to `base_commit`, and refuses to run without a terminal, so an agent
  can't approve its own work.
- **Smoke before full.** An arm's full runs wait for a successful smoke run of
  that arm at the same commit. A smoke run must exit 0 *and* write its metric.
- **Budget is worst-case.** Live runs count at their full timeout, so launching
  can never push a study past `budget.gpu_hours`.
- **Good neighbor.** GPU caps, idle hysteresis, and a re-probe of the chosen
  GPUs immediately before each launch.
- **Frozen judge.** Metrics are read only by rr's parser from the file named in
  the spec; a run collected after the eval data changed is marked `invalid`.

### What your training script needs to do

`run.sh` runs `run.command` from the spec inside the run's worktree, with
`CUDA_VISIBLE_DEVICES` set (and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so indices
match `nvidia-smi`). It exports:

| Variable | Use |
|---|---|
| `RR_HEARTBEAT` | write `{"step": ..., "ts": ..., "loss": ...}` here every N steps |
| `RR_FINAL_METRICS` | write the final metrics JSON here before exiting 0 |
| `RR_RUN_DIR`, `RR_RUN_ID`, `RR_STUDY`, `RR_ARM`, `RR_SEED`, `RR_KIND` | context |
| `WANDB_RUN_GROUP`, `WANDB_NAME`, `WANDB_JOB_TYPE`, `WANDB_TAGS` | picked up by `wandb.init()` |

[rr/runtime.py](rr/runtime.py) does this for you (standard library only):

```python
from rr import runtime
runtime.heartbeat(step, loss=loss.item())          # every N steps
runtime.wandb_run(wandb.run.path)                  # optional: links the ledger to WandB
runtime.final_metrics({"eval/pass@1": score})      # at the end
```

Timeouts and `rr cancel` stop the whole process group, dataloader workers
included. Exit code 124 means the run hit its timeout.

## Tests

```bash
.venv/bin/pytest
```

The tests run against a fake cluster: each "node" is a local `bash -c` with
fake `nvidia-smi`, `gpustat`, `ps`, `whoami` and `tmux` on its PATH
([tests/fake_tools.py](tests/fake_tools.py)), so the real probe script and the
real `run.sh` are exercised without GPUs, including cancel, timeout, lost
sessions and GPUs grabbed between allocation and launch.
