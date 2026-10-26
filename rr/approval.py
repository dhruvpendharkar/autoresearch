"""The human gate: `rr approve` pins a study's spec, code and eval data.

Approval records the sha256 of the spec file's exact bytes, the resolved
commit of every arm (base_commit, or the arm's branch tip at approval time),
and a hash of the eval data. Full runs launch only from those pinned commits,
so code can't change between your review and the overnight launch. Any edit
to the spec file, or any change to the eval data, makes the approval stale.

Approval also refuses if any arm's commit differs from base_commit under
`protected_paths`: the judge must be the same for every arm.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import gitutil
from .metrics import MetricError, hash_paths
from .project import Project
from .spec import Spec
from .statefile import read_json, write_json_atomic


class ApprovalError(Exception):
    pass


@dataclass
class ApprovalStatus:
    valid: bool
    reason: str
    approval: Optional[dict]


def spec_sha256(spec: Spec) -> str:
    return hashlib.sha256(spec.raw).hexdigest()


def compute_hash(spec_sha: str, base_commit: str, commits: dict, eval_sha: Optional[str]) -> str:
    payload = json.dumps(
        {"spec": spec_sha, "base_commit": base_commit, "commits": commits, "eval_data": eval_sha}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def arm_commits(project: Project, spec: Spec) -> tuple:
    """(base sha, {arm: sha}) resolved now from base_commit and arm branches."""
    base = gitutil.resolve_commit(project.root, spec.base_commit)
    commits = {a.id: gitutil.resolve_commit(project.root, a.branch) if a.branch else base for a in spec.arms}
    return base, commits


def protected_violations(project: Project, spec: Spec, base: str, commits: dict) -> dict:
    """{arm: [changed protected files]} for arms whose code touches the judge."""
    out = {}
    for arm, sha in commits.items():
        changed = gitutil.changed_files(project.root, base, sha, spec.protected_paths)
        if changed:
            out[arm] = changed
    return out


def eval_data_sha(project: Project, spec: Spec, cache_path: Optional[Path] = None) -> Optional[str]:
    if not spec.eval_data:
        return None
    return hash_paths(project.root, spec.eval_data, cache_path)


def approval_path(project: Project, spec: Spec) -> Path:
    return project.study_dir(spec.study) / "approval.json"


def prepare(project: Project, spec: Spec, cache_path: Optional[Path] = None) -> dict:
    """Everything `rr approve` would record, after the protected-path check."""
    base, commits = arm_commits(project, spec)
    bad = protected_violations(project, spec, base, commits)
    if bad:
        detail = "; ".join(f"{arm}: {', '.join(files)}" for arm, files in bad.items())
        raise ApprovalError(f"arms change protected paths relative to base_commit: {detail}")
    try:
        eval_sha = eval_data_sha(project, spec, cache_path)
    except MetricError as e:
        raise ApprovalError(str(e)) from None
    s = spec_sha256(spec)
    return {
        "study": spec.study,
        "spec_sha256": s,
        "base_commit": base,
        "commits": commits,
        "eval_data_sha256": eval_sha,
        "approval_hash": compute_hash(s, base, commits, eval_sha),
    }


def write(project: Project, spec: Spec, record: dict, clock: Callable[[], float] = time.time) -> dict:
    record = dict(record, approved_at=clock(), approved_by=getpass.getuser())
    write_json_atomic(approval_path(project, spec), record)
    return record


def check(project: Project, spec: Spec, cache_path: Optional[Path] = None, check_eval: bool = True) -> ApprovalStatus:
    """Is the study's approval still valid for the spec and eval data on disk?"""
    a = read_json(approval_path(project, spec), None)
    if not a:
        return ApprovalStatus(False, "not approved (run `rr approve`)", None)
    expected = compute_hash(a.get("spec_sha256"), a.get("base_commit"), a.get("commits"), a.get("eval_data_sha256"))
    if a.get("approval_hash") != expected:
        return ApprovalStatus(False, "approval.json was modified after approval", a)
    if a["spec_sha256"] != spec_sha256(spec):
        return ApprovalStatus(False, "spec.yaml changed since approval", a)
    if set(a["commits"]) != {arm.id for arm in spec.arms}:
        return ApprovalStatus(False, "arms differ from the approved set", a)
    if check_eval and spec.eval_data:
        try:
            current = eval_data_sha(project, spec, cache_path)
        except MetricError as e:
            return ApprovalStatus(False, str(e), a)
        if current != a["eval_data_sha256"]:
            return ApprovalStatus(False, "eval data changed since approval", a)
    return ApprovalStatus(True, "approved", a)
