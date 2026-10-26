# Autoresearch System: Design & Implementation Plan

Oct 22, 2026 · @Dhruv

## Summary

Build a small, file-driven research runner that lives entirely on the cluster. A plain-Python scheduler in a persistent tmux session watches GPU availability across the shared nodes, launches pre-registered runs onto free GPUs, babysits them, and calls Claude Code on your Pro login only when a decision needs judgment. You write a research brief and an experiment spec, and wake up to a report built from run logs and WandB.

**You own:** the hypothesis, the method idea, which comparisons are valid, the eval protocol, and how many GPUs you're willing to take from the shared pool.

**Agents own:** turning the brief into code on a branch (interactively, during the day), smoke tests, finding free GPUs and launching, monitoring and recovery, result collection, and a first-pass analysis that reports every run, including failures.

Key decisions, in brief:

1. **Everything runs on the cluster, inside tmux.** This mirrors your manual routine (VPN, login command, tmux). The controller and every training run live in tmux sessions, so your laptop and VPN can drop overnight without stopping anything.
2. **GPU availability is the scheduler.** With no Slurm and other users on the same nodes, the scheduler polls `gpustat`/`nvidia-smi` on each node, counts a GPU as free only after it stays idle across several probes, and never touches a GPU that has another user's process on it.
3. **One narrow door.** Agents never run node commands directly; they call a small `rr` CLI (gpus, submit, status, logs, cancel) that wraps your hardcoded per-node commands and writes every action to a run ledger.
4. **Claude is the exception handler, not the loop.** Pro has a rolling five-hour limit and a weekly limit, shared with your claude.ai usage. So polling, retries and the report's numbers are plain Python, and headless Claude is called only for failure triage and the morning write-up, under a hard per-window call budget.
5. **Pre-registration and a frozen judge**, unchanged: comparisons, seeds and metrics are fixed in a spec before launch, and eval code is read-only to agents.

The main unknowns are in Open questions at the end: whether the nodes share a filesystem with the machine you log into, whether each node command can run one non-interactive command, and whether the cluster has outbound internet (Claude Code and WandB both need it).

## Landscape: what to borrow

No open-source system matches your exact shape (method development, pre-registered comparisons, a shared GPU cluster with no scheduler, WandB) out of the box, but each solves one piece well. The plan is a thin layer of your own that borrows these patterns rather than adopting any one framework.

| System | Core pattern | Borrow | Leave |
| --- | --- | --- | --- |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Agent edits only `train.py`; human edits only `program.md`; `prepare.py` (data + eval) is never modified; fixed 5-min budget; keep or discard | Frozen judge; human programs the instructions, not the code; fixed budgets make runs comparable | Single GPU, single metric, greedy hill-climbing |
| [SkyPilot: scaling autoresearch](https://skypilot.ai/blog/scaling-autoresearch) | Claude Code drove 16 GPUs for \~8 h, \~910 experiments, submitting detached jobs in waves of 10–13 | Wave-based parallel submission; screen cheaply, confirm on better hardware | Rankings shifted between H100 and H200, so hardware must be logged and held fixed per comparison |
| [CORAL](https://github.com/Human-Agent-Society/CORAL) | Agents in separate git worktrees; shared notes in `.coral/public/`; grader daemon scores commits; agents run as an unprivileged user that cannot read grader files; manager sends `reflect` / `consolidate` / `pivot` heartbeats | Grader isolation via OS permissions; shared notes file; heartbeat prompts | Open-ended evolution is overkill for pre-registered experiments |
| [research-loop](https://github.com/junjunjunbong/research-loop) | Approval bound to a plan hash; one worktree per attempt; append-only TSV ledger; six result states (promising, keep, discard, inconclusive, crash, invalid); smoke runs never count as results | Ledger schema and result states almost verbatim; plan-hash approval as your human gate | v0.3 explicitly excludes GPU, Slurm and SSH execution |
| [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) | Best-first tree search over experiment nodes, managed by an experiment-manager agent; nodes tagged buggy, non-buggy, replication, ablation | Node taxonomy, and treating debugging as tree expansion | Idea generation and paper writing: that is your job |
| [ARIS](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) | Markdown-only skills; executor model plus an independent reviewer model; watchdog alerts when an overnight loop stops updating its state | Cross-model review of claims before you see them; stall watchdog | Its paper-writing pipelines |
| [NanoResearch](https://github.com/OpenRaiser/NanoResearch), [ARK](https://github.com/kaust-ark/ARK) | End-to-end pipelines that run jobs on Slurm | Reference code for job launching and result parsing (their Slurm paths don't apply to your cluster) | End-to-end autonomy |

Two cautionary writeups are worth reading before you build: Nick Oak's [tennis XGBoost loop](https://nickoak.com/posts/tennis-xgboost-autoresearch/), which documents the agent gaming its setup, and Eliana's [GPUMode kernel retrospective](https://elianaive.com/posts/another-typical-autoresearch-post/), which catalogs reward hacks caught over 1,293 experiments. A curated index of the rest is at [awesome-autoresearch](https://github.com/webfuse-com/awesome-autoresearch).

## Design principles

Nine rules, each traceable to a failure mode the systems above ran into or to the constraints of a shared cluster and a Pro plan.

1. **You decide what counts; agents decide how to get it.** The research brief and experiment spec are human-authored and hashed at approval. Any change to baselines, metrics, seeds or eval data makes the approval stale, as in research-loop.
2. **The judge is out of reach.** Eval code, eval data and metric parsers live in a protected path. Enforce this with file permissions and Claude Code `permissions.deny` rules, and use hooks only as a second layer: the Claude Code docs say the hook `if` filter is best-effort and the permission system should do hard enforcement.
3. **Files and git are the memory; sessions are disposable.** The scheduler restarts from the ledger, and every Claude call is a short headless run that reads state from disk and exits. Nothing important lives in a context window or in the daemon's memory.
4. **One narrow door to the nodes.** Agents get an `rr` CLI with a handful of verbs, never raw node commands, `kill`, or `tmux kill-*`. This keeps permissions easy to state, puts every action in one log, and makes the system testable without GPUs.
5. **Be a good neighbor.** A GPU is free only after several consecutive idle probes and with no other user's process on it; the system caps how many GPUs it holds in total and per node, re-checks right before launch, and backs off when others start using a node.
6. **Plain code does the routine; Claude handles exceptions.** Polling, launching, retries with known fixes, and computing results are deterministic Python. Claude is called when something is new or ambiguous, which keeps Pro usage low and the behavior predictable.
7. **Every run is recorded, including failures.** The ledger is append-only and distinguishes crash, invalid, inconclusive and real regressions. The morning report shows all seeds and all runs, so cherry-picking is visible.
8. **Smoke before scale, confirm before claim.** New code runs a short smoke job on one GPU before any full launch, and smoke results never count. A "win" needs the pre-registered seed count on the same GPU type.
9. **Escalate, don't improvise.** When something needs judgment (a method bug that changes semantics, an ambiguous result, the Claude call budget exhausted), the agent writes a question to an inbox file and the scheduler moves on to other runs.

## Cluster model: nodes, tmux and GPU availability

The scheduler's core job is answering "which GPUs can I use right now without stepping on anyone?", then launching each run in its own tmux session on that node.

**Nodes are config, not discovery.** A `nodes.yaml` holds each node's hardcoded command and the sharing policy:

```yaml
nodes:
  gpu01: {exec: "<your command for node 1>", gpus: 8, gpu_type: RTX6000}
  gpu02: {exec: "<your command for node 2>", gpus: 8, gpu_type: RTX6000}
policy:
  max_gpus_total: 8          # never hold more than this across the cluster
  max_gpus_per_node: 4       # leave room for others on every node
  night: {hours: "23:00-07:00", max_gpus_total: 16}   # optional: take more when it's quiet
  free_mem_mb: 1000          # below this a GPU counts as idle
  free_util_pct: 5
  idle_probes: 3             # consecutive idle probes before a GPU counts as free
  probe_interval_s: 60
  cooldown_min: 60           # exclude a GPU after someone else grabbed it from under us
```

`rr` runs every remote command as `<exec> '<command>'`. The one requirement is that each node command accepts a single non-interactive command, the way `ssh gpu01 'nvidia-smi'` does. If your commands only open an interactive shell, the fallback is one long-lived tmux window per node that `rr` sends keystrokes into, which works but is more fragile.

**Probing GPUs.** Every probe interval the scheduler runs, on each node:

- `gpustat --json` if installed, which gives per-GPU memory, utilization and processes with usernames.
- Otherwise `nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`, plus `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader` and `ps -o user= -p <pid>` to attribute each process to a user.

Results go to `state/gpus.json` with timestamps; `rr gpus` prints the same table for you and for the agent.

**When a GPU counts as free.** All of these must hold:

- no process belonging to any other user;
- memory used below `free_mem_mb` and utilization below `free_util_pct`;
- both true for `idle_probes` consecutive probes, so a GPU between someone's runs or during their dataloader warmup isn't grabbed;
- not already reserved in our ledger, and taking it keeps us within the total and per-node caps.

**Launching a run.** For a run needing k GPUs, the scheduler prefers one node with k free GPUs (multi-GPU runs don't span nodes), re-probes those GPUs immediately before launch, writes a reservation to the ledger, then starts:

```bash
<exec> "tmux new-session -d -s rr-<run_id> \
  'cd <run_dir> && CUDA_VISIBLE_DEVICES=<ids> bash run.sh > log.txt 2>&1; echo \$? > exit_code'"
```

Each run gets its own tmux session, so you can attach to `rr-<run_id>` on that node and watch it exactly as you do now. It also gets its own directory holding a git worktree of the exact commit, the log, and an `exit_code` file written when it ends. The training script writes a small `heartbeat.json` (step, loss, timestamp) every N steps, which detects stalls without depending on WandB.

**Status and collisions.** Each probe also checks every running run: tmux session alive, `exit_code` present, heartbeat fresh. If another user's process appears on one of our GPUs, the run is flagged; if it then runs out of memory, it is requeued as a collision rather than counted as a failure, and that GPU is excluded for `cooldown_min`.

**Day and night caps.** Optional time-based caps let the system hold more GPUs overnight. It never kills its own runs to shrink back in the morning; it just stops launching until it is under the daytime cap.

## Running on Claude Code Pro

Pro is enough for this design because Claude's overnight job is small: a few triage calls and one report. Most of your quota stays available for daytime implementation sessions.

**Auth on the cluster.** Install Claude Code on the machine your login command lands on and sign in with your Pro account. Either run `claude` inside tmux and finish the login URL in your laptop's browser, or generate a long-lived token with `claude setup-token` and export it as `CLAUDE_CODE_OAUTH_TOKEN` in the controller's environment. Make sure `ANTHROPIC_API_KEY` is not set anywhere on the cluster: if it is, Claude Code bills that key instead of your plan ([support article](https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan)).

**Headless calls use your subscription, with one caveat.** `claude -p` runs on your login, but `--bare` mode skips OAuth and needs an API key, and the [headless docs](https://code.claude.com/docs/en/headless) say `--bare` will become the default for `-p` in a future release. Pin the Claude Code version on the cluster and re-test headless auth whenever you upgrade.

**Usage limits shape the design.** Pro has a five-hour session limit and a weekly limit, and the same pool covers claude.ai and IDE use. So the scheduler treats Claude as a scarce resource:

- It keeps a call budget per rolling five-hour window (start with 6 overnight calls) and per day, logged in `state/claude_usage.jsonl`.
- Each call is wrapped in `timeout 15m` and gets a narrow, pre-digested input: the failing run's last 200 log lines, its config diff and the GPU state, not the whole repo.
- Overnight triage uses Sonnet (`--model sonnet`); heavier models are for your interactive sessions.
- If a call fails on a usage limit, the scheduler stops calling Claude until the window resets, keeps applying its deterministic rules, and parks anything unresolved in `inbox.md`. Training never waits on Claude.

| Task | Done by | Claude calls |
| --- | --- | --- |
| Probe GPUs, launch, track, resubmit on OOM, collision or lost node | Python | 0 |
| Parse metrics, per-seed tables, bootstrap CIs | Python | 0 |
| Triage an unfamiliar crash and write a diagnosis or proposed patch | Claude, headless | 0–3 per night |
| Morning narrative over the computed tables | Claude, headless | 1 per night |
| Implement a method from a brief, write tests, smoke run | Claude, interactive with you | daytime sessions |

## Architecture

Everything runs on the cluster. The controller lives in a `rr-control` tmux session on the machine your login command lands on, and each run lives in its own tmux session on a GPU node, so nothing depends on your laptop or the VPN staying up.

&#91;embedded content: system architecture · login machine, shared GPU nodes, WandB\]

The daemon probes every node through `rr`, launches runs onto free GPUs, and reads progress from heartbeat files and WandB. Only an unfamiliar failure or the morning report wakes Claude Code, whose sole route to the nodes is `rr`. You connect the way you do today, attach to `rr-control`, and read the study files.

## Experiment spec and ledger

Two files carry the whole system: a spec you write per study, and a ledger agents append to per run.

**The brief** (`studies/<id>/brief.md`) is prose: hypothesis, the method change in a paragraph, what result would falsify it, and pointers to relevant code. Agents read it; only you edit it.

**The spec** (`studies/<id>/spec.yaml`) is the pre-registration. A sketch:

```yaml
study: async-rollout-offpolicy-v2
base_commit: 3f2a9c1          # code the baseline runs on
metric: {name: eval/pass@1, direction: max, source: final_metrics.json}
seeds: [0, 1, 2]
resources: {gpus_per_run: 2, gpu_type: RTX6000, run_timeout_h: 10}
budget: {gpu_hours: 96, max_concurrent_runs: 4, max_retries: 2}
arms:
  - id: baseline
    config: configs/grpo_sync.yaml
  - id: method
    config: configs/grpo_async.yaml
    branch: study/async-rollout-offpolicy-v2   # agent-implemented
  - id: ablate_no_is_correction
    config: configs/grpo_async.yaml
    overrides: {is_correction: false}
comparisons:
  - {treatment: method, control: baseline, test: mean_diff_bootstrap_ci}
  - {treatment: method, control: ablate_no_is_correction}
exploratory: {allowed: true, gpu_hours: 16}   # agent may add HP probes, labeled exploratory
protected_paths: [eval/, data/eval/, src/metrics/]
```

At approval, `rr approve` hashes the spec plus `base_commit` and writes `approval.json`. Every launch checks the hash; any edit to the spec invalidates it until you re-approve.

**The ledger** (`studies/<id>/ledger.jsonl`) is append-only, one line per state change, with this schema:

| Field | Example | Purpose |
| --- | --- | --- |
| `run_id` | `method-s1-a03` | Arm, seed, attempt |
| `arm`, `seed`, `kind` | `method`, `1`, `full` | `kind` is smoke, full, confirm or exploratory |
| `git_sha` | `9b1e0d4` | Exact code, committed before launch |
| `spec_hash` | `e3b0c4…` | Ties the run to an approved spec |
| `tmux_session`, `node`, `gpu_type` | `rr-method-s1-a03`, `gpu03`, `RTX6000, GPUs 4–5` | Hardware is a confound; log it |
| `wandb_run` | `entity/proj/abc123` | Group = study, tags = arm and seed |
| `state` | `running` | queued, running, crash, invalid, done |
| `result` | `promising` | keep, promising, discard, inconclusive (from research-loop) |
| `metric` | `0.412` | Parsed only by the frozen parser |
| `notes` | `OOM at step 1.2k; resubmitted with bs/2` | Agent's one-line explanation |

WandB stays the place you look at curves. The ledger is the place agents reason over, because it is small, local, and cannot be edited after the fact. Use WandB `group` for the study and `tags` for arm, seed and kind so every comparison is one filter away.

## Agent roles and the overnight loop

The overnight loop is a Python daemon, not an agent. It runs in a window of a `rr-control` tmux session on the controller and calls Claude only through two fixed entry points.

| Role | Runs as | Trigger | Writes | Can do |
| --- | --- | --- | --- | --- |
| Scheduler (`rr daemon`) | Python in tmux | Every 60 s | ledger, `state/`, `notes.md` | Probe GPUs; launch, requeue and cancel its own runs within caps |
| Implementer | Claude Code, interactive with you | Daytime | Study branch in a git worktree: code, unit tests, smoke config, `IMPLEMENTATION.md` | Edit non-protected code, run tests, `rr submit --smoke` |
| Triage | `claude -p "/triage ..."` | Daemon, on a failure no rule covers | `triage/<run_id>.md`, inbox | Read logs; change spec-tunable fields; resubmit once |
| Analyst | `claude -p "/report ..."` | Daemon, at 07:00 or when all arms finish | `report.md` | Read-only |

On Pro, a separate reviewer pass is optional: run one interactively when a result looks like a win, rather than spending overnight quota on it.

**The daemon loop**, every 60 seconds:

1. Probe GPUs on every node and update `state/gpus.json`.
2. Reconcile runs: finished (exit code 0, metric parsed by the frozen parser into the ledger), crashed, stalled (no heartbeat for 20 minutes), or collided with another user.
3. Apply deterministic recovery rules: out of memory means retry with half the micro-batch and double the gradient accumulation (same effective batch); a collision or lost node means requeue unchanged; NaN means one same-seed retry. Anything else goes to Triage if the call budget allows, otherwise to the inbox.
4. Launch: fill free GPUs, within caps, with the next pending arm and seed, smoke before full.
5. When all arms finish or at 07:00: compute the report tables in Python, then make the Analyst call.

A triage call looks like this:

```bash
timeout 15m claude -p "/triage studies/async-rollout-offpolicy-v2 method-s1-a03" \
  --model sonnet --permission-mode dontAsk --permission-prompts none \
  --output-format json >> logs/claude_calls.jsonl
```

Overnight, Claude never edits method code. Triage may change only fields the spec marks tunable (micro-batch, gradient accumulation, dataloader workers), and it writes a diagnosis either way.

**A typical night:**

1. Evening: you write the brief and spec, review the implementer's branch and smoke run, run `rr approve`, then detach from tmux and drop the VPN.
2. The daemon launches baseline and method seeds as GPUs free up, staying inside the caps.
3. Overnight: known failures are retried by rule, unknown ones get one triage call, and anything unresolved lands in `inbox.md` with a phone notification.
4. \~07:00: the report lists every run with per-seed numbers and bootstrap CIs for each pre-registered comparison, followed by Claude's short narrative.
5. Morning: VPN, login command, `tmux attach -t rr-control`, read the report and inbox, decide the next study.

**Ratchet mode is a daytime tool on Pro.** An autoresearch-style keep-or-discard loop needs a Claude call per iteration, so run it interactively for single-metric sub-problems rather than overnight.

## Safety, budgets and failure handling

Three layers, strongest first: limits enforced in `rr` and the filesystem, Claude Code permission rules, then hooks for logging and soft checks. Your existing Claude Code safety harness slots in as the hook and transcript layer.

**Layer 1: `rr` and the filesystem.** GPU caps, the call budget and the PAUSE file are enforced inside `rr` and the daemon, not by the agent's goodwill. `rr cancel` only kills tmux sessions named `rr-*` that the ledger says this system launched, so it can never touch your manual sessions or other users' jobs. Protected paths (`eval/`, `data/eval/`, `src/metrics/`, `nodes.yaml`) are made read-only; since agents run as your user this is a speed bump rather than a wall, which is why Layer 2 matters.

**Layer 2: Claude Code permissions** in the project's `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": ["Bash(rr *)", "Bash(uv run pytest *)", "Bash(git add *)", "Bash(git commit *)",
              "Edit(src/**)", "Edit(configs/**)", "Edit(studies/*/notes.md)", "Edit(studies/*/inbox.md)"],
    "deny":  ["Bash(ssh *)", "Bash(scp *)", "Bash(rsync *)", "Bash(git push *)", "Bash(rm -rf *)",
              "Bash(kill *)", "Bash(pkill *)", "Bash(killall *)", "Bash(tmux kill-session *)", "Bash(tmux kill-server *)",
              "Edit(eval/**)", "Edit(src/metrics/**)", "Edit(nodes.yaml)",
              "Edit(studies/*/spec.yaml)", "Edit(studies/*/ledger.jsonl)"]
  }
}
```

Denying edits to `nodes.yaml` keeps agents from raising their own GPU caps. Headless calls run with `--permission-mode dontAsk` so anything outside the allow list is denied instead of hanging on a prompt.

**Layer 3: hooks.** A `PreToolUse` hook re-checks `rr submit` calls against the caps and the PAUSE file, with `onFailure: "block"` so a broken hook fails closed. A `PostToolUse` hook appends every tool call to an audit log. A `Stop` hook verifies that a triage call wrote its diagnosis file.

**Kill switches and keep-alive.** `touch PAUSE` stops all launches; `rr drain` lets running jobs finish but starts nothing new; `rr cancel --study <id>` stops only that study's runs. If `crontab` is available on the controller, a cron entry every 10 minutes checks the daemon's heartbeat file, restarts it in tmux if it's dead, and sends a phone notification (ntfy over `curl`, if the cluster can reach the internet). Runs live in their own tmux sessions on the GPU nodes, so a daemon crash doesn't stop training; on restart the daemon reconciles from the ledger.

| Failure | Detected by | Response |
| --- | --- | --- |
| Out of memory | Exit code + log pattern | Retry with half micro-batch, doubled grad accumulation; once |
| Another user grabs our GPU | Probe sees a foreign process on a reserved GPU | Flag the run; if it then dies, requeue as a collision and cool that GPU down |
| Node command fails or times out | `rr` error | Back off; mark the node down after 3 failures; alert |
| Node rebooted | Run's tmux session gone, no `exit_code` | Requeue from last checkpoint |
| NaN or divergence | Heartbeat or WandB check | One same-seed retry, then inbox |
| Unfamiliar crash | Traceback in log | One triage call if budget allows, else inbox; no overnight method edits |
| No free GPUs for hours | Queue age | Keep waiting; report shows queue time |
| Claude usage limit hit | Error from `claude -p` | Stop calling Claude until the window resets; rules keep running |
| Claude auth expired | Auth error | Alert; deterministic mode continues |
| Controller tmux lost | Missing daemon heartbeat | Cron restarts it; running jobs unaffected |
| Agent edits eval, spec or caps | Permission deny + audit log | Blocked; logged for your review |
| Metric gaming (e.g. eval on train split) | Frozen parser, eval-data hash check in `rr` | Run marked invalid |

## Implementation plan

Four phases of about a week each, each ending in a gate you can check, so the system is useful from week one and never runs unattended before it has survived injected failures.

&#91;embedded content: implementation roadmap · 4 phases, 4 gates\]

Phases 0 and 1 contain no LLM at all: they build the deterministic plumbing the agents will stand on, and they pay off on their own as a cleaner manual workflow. If a gate fails, stay in that phase rather than moving on.

## Repo layout and tooling

Keep the runner in its own repo (`labos/`) on the cluster and install it into each research project as a Claude Code plugin plus a small Python package, so every project gets the same skills, hooks and `rr` CLI.

```
labos/
  rr/                      # Python CLI (typer): gpus, approve, submit, status, logs, cancel, drain, daemon
    nodes.py               # runs <exec> '<cmd>' per node from nodes.yaml; timeouts, retries, node-down marking
    gpus.py                # gpustat/nvidia-smi probes, idle hysteresis, caps, reservations, cooldowns
    launcher.py            # per-run git worktree, run.sh, tmux session, exit_code + heartbeat checks
    ledger.py              # append-only JSONL, state machine, result states
    health.py              # heartbeat and WandB checks: NaN, stall, throughput
    metrics.py             # frozen parser; hashes eval data
    claude.py              # wraps `timeout ... claude -p`, call budget, usage-limit backoff
  templates/run.sh.j2      # sets CUDA_VISIBLE_DEVICES, WANDB group/tags, writes heartbeat + exit_code
  plugin/
    skills/implement/  skills/triage/  skills/report/
    hooks/budget_gate.py  hooks/audit_log.py  hooks/triage_done.py
  nodes.yaml               # hardcoded node commands + sharing policy (protected)
  cron/keepalive.sh        # restarts the daemon in tmux if its heartbeat is stale
your-project/
  .claude/settings.json    # permissions from the safety section
  eval/  src/metrics/      # protected
  studies/<id>/brief.md spec.yaml approval.json ledger.jsonl notes.md inbox.md report.md
  runs/<run_id>/           # worktree, log.txt, heartbeat.json, exit_code, final_metrics.json
```

| Concern | Choice | Why |
| --- | --- | --- |
| Where it runs | Controller in a `rr-control` tmux session on the machine your login command lands on | Matches your routine; survives laptop sleep and VPN drops |
| Node access | Your hardcoded per-node commands in `nodes.yaml` | No scheduler to integrate with; one place to change if access changes |
| GPU discovery | `gpustat --json`, falling back to `nvidia-smi` queries | Gives memory, utilization and process owners per GPU |
| Run execution | One detached tmux session per run on its node | You can attach to any run; runs outlive the controller |
| Code shipping | Per-run git worktree at the exact commit, on the shared filesystem | Every run is reproducible from its ledger row |
| Agent runtime | Claude Code headless (`claude -p`) on your Pro login, plus interactive sessions | No API key; usage stays inside the plan |
| Monitoring | Heartbeat file first, WandB Python API second; [W&B MCP server](https://docs.wandb.ai/platform/mcp-server) for your interactive analysis | Deterministic checks are free; the MCP server is for ad-hoc questions |
| Alerts | ntfy via `curl`, or email if the cluster has `mail` | Phone notification for inbox items and stalls |

## Open questions

- [ ] Can each node command run a single non-interactive command (like `ssh gpu01 'nvidia-smi'`), or does it only open an interactive shell?
- [ ] Do the GPU nodes share a filesystem (home or project directory) with the machine you log into? If not, `rr` needs an rsync step per run.
- [ ] Does the cluster have outbound internet for Claude Code, WandB and ntfy?
- [ ] Is `gpustat` installed or installable in your user environment, and can you see other users' process owners in `nvidia-smi`?
- [ ] Which RTX 6000 generation (24 GB or 48 GB)? It sets default micro-batch sizes and the OOM retry ladder.
- [ ] What are the lab's norms for sharing: a GPU cap per person, quiet hours, or a channel to announce long jobs? These become the `policy` values.
- [ ] Is `crontab` allowed on the login machine, and is it OK to keep a tmux session there long-term?

## Sources

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [SkyPilot: Scaling Karpathy's Autoresearch](https://skypilot.ai/blog/scaling-autoresearch)
- [CORAL](https://github.com/Human-Agent-Society/CORAL)
- [research-loop](https://github.com/junjunjunbong/research-loop)
- [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- [ARIS (Auto-claude-code-research-in-sleep)](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)
- [awesome-autoresearch](https://github.com/webfuse-com/awesome-autoresearch)
- [Claude Code hooks reference](https://code.claude.com/docs/en/hooks)
- [Claude Code headless mode](https://code.claude.com/docs/en/headless)
- [Use Claude Code with your Pro or Max plan](https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan)
- [W&B MCP server docs](https://docs.wandb.ai/platform/mcp-server)
