import json

import pytest

from conftest import git
from rr import approval
from rr.spec import load_spec


def approve(fp, **spec_over):
    spec = load_spec(fp.write_spec(**spec_over))
    rec = approval.write(fp.project, spec, approval.prepare(fp.project, spec))
    return spec, rec


def test_approve_pins_commits_and_is_valid(fproject):
    spec, rec = approve(fproject)
    main = git(fproject.root, "rev-parse", "main")
    method = git(fproject.root, "rev-parse", "study/method")
    assert rec["commits"] == {"baseline": main, "method": method}
    assert rec["base_commit"] == main
    assert rec["eval_data_sha256"]
    assert approval.check(fproject.project, spec).valid


def test_missing_approval(fproject):
    spec = load_spec(fproject.write_spec())
    st = approval.check(fproject.project, spec)
    assert not st.valid and "not approved" in st.reason


def test_any_spec_edit_makes_approval_stale(fproject):
    spec, _ = approve(fproject)
    path = spec.path
    path.write_bytes(spec.raw + b"# just a comment\n")
    st = approval.check(fproject.project, load_spec(path))
    assert not st.valid and "spec.yaml changed" in st.reason


def test_eval_data_change_makes_approval_stale(fproject):
    spec, _ = approve(fproject)
    (fproject.root / "data" / "eval" / "test.jsonl").write_text('{"q": 2}\n')
    st = approval.check(fproject.project, spec)
    assert not st.valid and "eval data changed" in st.reason


def test_tampered_approval_detected(fproject):
    spec, rec = approve(fproject)
    p = approval.approval_path(fproject.project, spec)
    rec["commits"]["method"] = rec["commits"]["baseline"]
    p.write_text(json.dumps(rec))
    st = approval.check(fproject.project, spec)
    assert not st.valid and "modified" in st.reason


def test_branch_moving_after_approval_does_not_change_pinned_commit(fproject):
    spec, rec = approve(fproject)
    git(fproject.root, "checkout", "-q", "study/method")
    (fproject.root / "train.py").write_text("print('changed after review')\n")
    fproject.commit("sneaky")
    git(fproject.root, "checkout", "-q", "main")
    st = approval.check(fproject.project, spec)
    assert st.valid and st.approval["commits"]["method"] == rec["commits"]["method"]


def test_refuses_arm_that_touches_protected_paths(fproject):
    git(fproject.root, "checkout", "-q", "study/method")
    (fproject.root / "eval" / "judge.py").write_text("THRESHOLD = 0.0\n")
    fproject.commit("loosen the judge")
    git(fproject.root, "checkout", "-q", "main")
    spec = load_spec(fproject.write_spec())
    with pytest.raises(approval.ApprovalError, match="method: eval/judge.py"):
        approval.prepare(fproject.project, spec)


def test_unknown_branch(fproject):
    from rr.gitutil import GitError

    spec = load_spec(fproject.write_spec(arms=[{"id": "m", "config": "c", "branch": "nope"}], comparisons=[]))
    with pytest.raises(GitError, match="not a commit"):
        approval.prepare(fproject.project, spec)


def test_missing_eval_data(fproject):
    spec = load_spec(fproject.write_spec(eval_data=["data/missing"]))
    with pytest.raises(approval.ApprovalError, match="not found"):
        approval.prepare(fproject.project, spec)
