"""Turn a spec into queued ledger rows, and account for GPU-hours.

A queued row carries everything needed to launch and judge the run (commit,
rendered command, metric definition, expected eval-data hash), so launching
and collecting never depend on a spec file that may have been edited since.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Iterable, Optional

from . import approval as approvalmod
from .ledger import HOLDS_GPUS, Ledger, make_run_id
from .project import Project
from .spec import Arm, Spec, format_value

# A run that already exists in one of these states is not submitted again.
LIVE_OR_DONE = frozenset({"queued", "launching", "running", "done"})
MAIN_BUDGET_KINDS = frozenset({"smoke", "full", "confirm"})


class SubmitError(Exception):
    pass


@dataclass
class Planned:
    run_id: str
    fields: dict


@dataclass
class Skipped:
    arm: str
    seed: int
    reason: str


def session_name(study: str, run_id: str) -> str:
    return f"rr-{study}-{run_id}"


def render_command(spec: Spec, arm: Arm, seed: int, overrides: dict, run_dir: str, run_id: str) -> str:
    ov = " ".join(
        shlex.quote(spec.override_format.format(key=k, value=format_value(v))) for k, v in overrides.items()
    )
    return spec.command.format(
        config=shlex.quote(arm.config),
        seed=seed,
        overrides=ov,
        run_dir=shlex.quote(run_dir),
        run_id=shlex.quote(run_id),
        arm=shlex.quote(arm.id),
        study=shlex.quote(spec.study),
    ).strip()


def _attempts(runs: Iterable[dict], arm: str, seed: int, kind: str) -> list:
    return [r for r in runs if r.get("arm") == arm and r.get("seed") == seed and r.get("kind") == kind]


def plan(
    project: Project,
    spec: Spec,
    ledger: Ledger,
    kind: str,
    arms: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
    extra_overrides: Optional[dict] = None,
    cache_path=None,
) -> tuple:
    """Work out which runs `rr submit` would queue: (list[Planned], list[Skipped])."""
    arm_list = [spec.arm(a) for a in arms] if arms else list(spec.arms)
    if seeds:
        unknown = set(seeds) - set(spec.seeds)
        if unknown and kind != "exploratory":
            raise SubmitError(f"seed(s) {sorted(unknown)} are not pre-registered in the spec")
        seed_list = list(seeds)
    else:
        seed_list = [spec.seeds[0]] if kind in ("smoke", "exploratory") else list(spec.seeds)
    extra_overrides = dict(extra_overrides or {})
    if extra_overrides and kind != "exploratory":
        raise SubmitError("--set overrides are only allowed for exploratory runs")

    if kind == "smoke":
        base, commits = approvalmod.arm_commits(project, spec)
        commits = {a.id: commits[a.id] for a in arm_list}
        bad = approvalmod.protected_violations(project, spec, base, commits)
        if bad:
            detail = "; ".join(f"{a}: {', '.join(f)}" for a, f in bad.items())
            raise SubmitError(f"arms change protected paths relative to base_commit: {detail}")
        spec_hash, eval_sha = None, None
    else:
        if kind == "exploratory" and not spec.exploratory_allowed:
            raise SubmitError("the spec does not allow exploratory runs")
        st = approvalmod.check(project, spec, cache_path)
        if not st.valid:
            raise SubmitError(f"{kind} runs need a valid approval: {st.reason}")
        commits = st.approval["commits"]
        spec_hash, eval_sha = st.approval["approval_hash"], st.approval["eval_data_sha256"]

    runs = list(ledger.runs().values())
    planned, skipped = [], []
    # Seed-major order, so partial results stay balanced across arms.
    for seed in seed_list:
        for arm in arm_list:
            prior = _attempts(runs, arm.id, seed, kind)
            sha = commits[arm.id]
            if kind == "smoke":
                live = [r for r in prior if r["state"] in LIVE_OR_DONE and r.get("git_sha") == sha]
                if live:
                    skipped.append(Skipped(arm.id, seed, f"smoke at {sha[:7]} already {live[-1]['state']}"))
                    continue
            elif kind == "full":
                live = [r for r in prior if r["state"] in LIVE_OR_DONE]
                if live:
                    skipped.append(Skipped(arm.id, seed, f"already {live[-1]['state']} ({live[-1]['run_id']})"))
                    continue
                if len(prior) > spec.max_retries:
                    skipped.append(Skipped(arm.id, seed, f"retries exhausted ({len(prior)} attempts)"))
                    continue
            attempt = max([r.get("attempt", 0) for r in prior] + [0]) + 1
            run_id = make_run_id(arm.id, seed, attempt, kind)
            overrides = dict(arm.overrides)
            if kind == "smoke":
                overrides.update(spec.smoke_overrides)
            overrides.update(extra_overrides)
            run_dir = str(project.run_dir(spec.study, run_id))
            planned.append(
                Planned(
                    run_id,
                    dict(
                        arm=arm.id,
                        seed=seed,
                        kind=kind,
                        attempt=attempt,
                        study=spec.study,
                        git_sha=sha,
                        spec_hash=spec_hash,
                        config=arm.config,
                        overrides=overrides,
                        command=render_command(spec, arm, seed, overrides, run_dir, run_id),
                        run_dir=run_dir,
                        n_gpus=spec.smoke_gpus if kind == "smoke" else spec.gpus_per_run,
                        gpu_type=spec.gpu_type,
                        timeout_s=int(spec.smoke_timeout_min * 60 if kind == "smoke" else spec.run_timeout_h * 3600),
                        metric={"name": spec.metric.name, "direction": spec.metric.direction, "source": spec.metric.source},
                        eval_data=list(spec.eval_data) if kind != "smoke" else [],
                        eval_data_sha256=eval_sha,
                    ),
                )
            )
    return planned, skipped


def enqueue(ledger: Ledger, planned: Iterable[Planned]) -> list:
    out = []
    for p in planned:
        f = dict(p.fields)
        out.append(ledger.create(p.run_id, f.pop("arm"), f.pop("seed"), f.pop("kind"), f.pop("attempt"), **f))
    return out


# ---------------------------------------------------------------- budget


def committed_gpu_hours(runs: Iterable[dict], kinds: Iterable[str], now: float) -> float:
    """GPU-hours used by finished runs plus the worst case (full timeout) for live ones.

    Counting live runs at their timeout means launching never risks exceeding
    the budget, whatever the runs end up doing.
    """
    kinds = set(kinds)
    total = 0.0
    for r in runs:
        if r.get("kind") not in kinds:
            continue
        n = r.get("n_gpus") or len(r.get("gpus") or []) or 1
        if r.get("state") in HOLDS_GPUS:
            total += n * r.get("timeout_s", 0) / 3600
        elif r.get("started_at") is not None and r.get("ended_at") is not None:
            total += n * max(0.0, r["ended_at"] - r["started_at"]) / 3600
    return total


def budget_allows(spec: Spec, runs: list, rec: dict, now: float) -> Optional[str]:
    """None if launching `rec` stays within budget, else the reason it doesn't."""
    need = rec.get("n_gpus", 1) * rec.get("timeout_s", 0) / 3600
    if rec.get("kind") == "exploratory":
        used = committed_gpu_hours(runs, {"exploratory"}, now)
        cap = spec.exploratory_gpu_hours
        label = "exploratory budget"
    else:
        used = committed_gpu_hours(runs, MAIN_BUDGET_KINDS, now)
        cap = spec.gpu_hours
        label = "budget"
    if used + need > cap + 1e-9:
        return f"{label}: {used:.1f} + {need:.1f} GPU-h would exceed {cap:g}"
    return None
