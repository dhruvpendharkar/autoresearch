import os
import signal
import time
from pathlib import Path

import pytest

from conftest import git, gpu
from rr import approval, gitutil, launcher as launchmod
from rr.gpus import PROBE_COMMAND
from rr.launcher import Launcher
from rr.nodes import NodeError, NodeRunner
from rr.queue import SubmitError, enqueue, plan
from rr.spec import load_spec


def setup(cluster, fproject, n_gpus=2, policy=None, runner_cls=NodeRunner, **spec_over):
    cluster.set_node("gpu01", [gpu(i) for i in range(n_gpus)])
    cfg = cluster.write_config(policy={"idle_probes": 1, **(policy or {})})
    spec = load_spec(fproject.write_spec(**spec_over))
    lc = Launcher(cfg, fproject.project, runner=runner_cls(cfg))
    return cfg, spec, lc


def submit(lc, spec, kind, **kw):
    planned, skipped = plan(lc.project, spec, lc.ledger(spec.study), kind, cache_path=lc.hash_cache, **kw)
    return enqueue(lc.ledger(spec.study), planned), skipped


def do_approve(fproject, spec):
    return approval.write(fproject.project, spec, approval.prepare(fproject.project, spec))


def wait_exit(lc, spec, run_ids, timeout=20):
    deadline = time.time() + timeout
    ledger = lc.ledger(spec.study)
    while time.time() < deadline:
        if all((Path(ledger.get(r)["run_dir"]) / "exit_code").exists() for r in run_ids):
            return
        time.sleep(0.05)
    raise AssertionError(f"runs did not finish: {run_ids}")


def actions(outcomes):
    return {o.run_id: o.action for o in outcomes}


def state(lc, spec, run_id):
    return lc.ledger(spec.study).get(run_id)


def pid_gone(pid, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------- the happy path


def test_smoke_runs_end_to_end(cluster, fproject):
    _, spec, lc = setup(cluster, fproject)
    created, _ = submit(lc, spec, "smoke")
    ids = [r["run_id"] for r in created]
    assert ids == ["baseline-s0-smoke-a01", "method-s0-smoke-a01"]

    out = lc.tick()
    assert actions(out) == {i: "launched" for i in ids}
    wait_exit(lc, spec, ids)
    out = lc.tick()
    assert actions(out) == {i: "done" for i in ids}

    base, method = (state(lc, spec, i) for i in ids)
    assert base["metric"] == 0.5 and method["metric"] == 0.6  # method branch's code ran
    assert {tuple(base["gpus"]), tuple(method["gpus"])} == {(0,), (1,)}
    assert base["exit_code"] == 0 and base["ended_at"] >= base["started_at"]
    run_dir = Path(method["run_dir"])
    assert gitutil.head_of(run_dir / "code") == git(fproject.root, "rev-parse", "study/method")
    log = (run_dir / "log.txt").read_text()
    assert f"cuda {method['gpus'][0]}" in log and '"steps": "1"' in log  # smoke overrides applied
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in (run_dir / "run.sh").read_text()
    # GPUs are released once runs finish.
    out = lc.tick()
    assert out == []


def test_full_runs_need_approval_then_smoke(cluster, fproject):
    _, spec, lc = setup(cluster, fproject)
    with pytest.raises(SubmitError, match="not approved"):
        submit(lc, spec, "full")
    rec = do_approve(fproject, spec)
    created, _ = submit(lc, spec, "full")
    assert [r["run_id"] for r in created] == ["baseline-s0-a01", "method-s0-a01", "baseline-s1-a01", "method-s1-a01"]
    assert all(r["spec_hash"] == rec["approval_hash"] for r in created)

    out = lc.tick()
    assert set(actions(out).values()) == {"waiting"}
    assert "no successful smoke" in out[0].detail

    smokes, _ = submit(lc, spec, "smoke")
    out = lc.tick()  # smoke runs jump the queue
    assert actions(out)["baseline-s0-smoke-a01"] == "launched"
    wait_exit(lc, spec, [r["run_id"] for r in smokes])
    out = lc.tick()
    launched = [o.run_id for o in out if o.action == "launched"]
    assert launched == ["baseline-s0-a01", "method-s0-a01"]  # 2 GPUs, seed-major order
    wait_exit(lc, spec, launched)
    out = lc.tick()
    assert [o.run_id for o in out if o.action == "done"] == launched
    assert [o.run_id for o in out if o.action == "launched"] == ["baseline-s1-a01", "method-s1-a01"]


def test_resubmit_skips_live_and_done(cluster, fproject):
    _, spec, lc = setup(cluster, fproject)
    created, _ = submit(lc, spec, "smoke")
    again, skipped = submit(lc, spec, "smoke")
    assert again == [] and len(skipped) == 2 and "already queued" in skipped[0].reason


def test_full_run_uses_approved_commit_even_if_branch_moves(cluster, fproject):
    _, spec, lc = setup(cluster, fproject, require_smoke=False)
    rec = do_approve(fproject, spec)
    git(fproject.root, "checkout", "-q", "study/method")
    (fproject.root / "train.py").write_text("raise SystemExit('unreviewed code')\n")
    fproject.commit("after review")
    git(fproject.root, "checkout", "-q", "main")
    submit(lc, spec, "full", arms=["method"], seeds=[0])
    lc.tick()
    wait_exit(lc, spec, ["method-s0-a01"])
    lc.tick()
    r = state(lc, spec, "method-s0-a01")
    assert r["state"] == "done" and r["git_sha"] == rec["commits"]["method"]


# ---------------------------------------------------------------- failures


def test_crash_records_exit_code_and_log_tail(cluster, fproject):
    arms = [{"id": "boom", "config": "c.yaml", "overrides": {"mode": "crash"}}]
    _, spec, lc = setup(cluster, fproject, arms=arms, comparisons=[])
    submit(lc, spec, "smoke")
    lc.tick()
    wait_exit(lc, spec, ["boom-s0-smoke-a01"])
    out = lc.tick()
    r = state(lc, spec, "boom-s0-smoke-a01")
    assert r["state"] == "crash" and r["exit_code"] == 1
    assert "illegal memory access" in r["notes"]
    assert actions(out) == {"boom-s0-smoke-a01": "crashed"}


def test_missing_metrics_is_a_crash(cluster, fproject):
    arms = [{"id": "quiet", "config": "c.yaml", "overrides": {"mode": "nometrics"}}]
    _, spec, lc = setup(cluster, fproject, arms=arms, comparisons=[])
    submit(lc, spec, "smoke")
    lc.tick()
    wait_exit(lc, spec, ["quiet-s0-smoke-a01"])
    lc.tick()
    r = state(lc, spec, "quiet-s0-smoke-a01")
    assert r["state"] == "crash" and "not written" in r["notes"]


def test_timeout_kills_run(cluster, fproject):
    arms = [{"id": "slow", "config": "c.yaml", "overrides": {"mode": "hang"}}]
    _, spec, lc = setup(cluster, fproject, arms=arms, comparisons=[], smoke={"timeout_min": 0.02})
    submit(lc, spec, "smoke")
    lc.tick()
    rid = "slow-s0-smoke-a01"
    run_dir = Path(state(lc, spec, rid)["run_dir"])
    wait_exit(lc, spec, [rid])
    lc.tick()
    r = state(lc, spec, rid)
    assert r["state"] == "crash" and r["exit_code"] == 124 and "timed out" in r["notes"]
    assert pid_gone(int((run_dir / "pid").read_text()))


def test_cancel_kills_the_whole_process_group(cluster, fproject):
    arms = [{"id": "slow", "config": "c.yaml", "overrides": {"mode": "spawn"}}]
    _, spec, lc = setup(cluster, fproject, arms=arms, comparisons=[])
    submit(lc, spec, "smoke")
    lc.tick()
    rid = "slow-s0-smoke-a01"
    run_dir = Path(state(lc, spec, rid)["run_dir"])
    deadline = time.time() + 10
    while not (run_dir / "worker_pid").exists() and time.time() < deadline:
        time.sleep(0.05)
    child = int((run_dir / "pid").read_text())
    worker = int((run_dir / "worker_pid").read_text())
    o = lc.cancel(lc.ledger(spec.study), rid)
    assert o.action == "cancelled"
    assert state(lc, spec, rid)["state"] == "cancelled"
    assert pid_gone(child), "training process survived cancel"
    assert pid_gone(worker), "training's worker process survived cancel"
    wait_exit(lc, spec, [rid])  # run.sh still records how it ended
    assert lc.tick() == []  # cancelled runs aren't collected


def test_lost_session_is_reported(cluster, fproject):
    arms = [{"id": "slow", "config": "c.yaml", "overrides": {"mode": "hang"}}]
    _, spec, lc = setup(cluster, fproject, arms=arms, comparisons=[])
    submit(lc, spec, "smoke")
    lc.tick()
    rid = "slow-s0-smoke-a01"
    rec = state(lc, spec, rid)
    run_dir = Path(rec["run_dir"])
    deadline = time.time() + 10
    while not (run_dir / "pid").exists() and time.time() < deadline:
        time.sleep(0.05)
    # Simulate a node reboot: everything dies at once, nothing gets to write exit_code.
    session_pid = int((cluster.data_dir / "tmux-gpu01" / rec["tmux_session"]).read_text())
    os.killpg(int((run_dir / "pid").read_text()), signal.SIGKILL)
    os.killpg(session_pid, signal.SIGKILL)
    assert pid_gone(session_pid)
    out = lc.tick()
    r = state(lc, spec, rid)
    assert actions(out)[rid] == "lost" and r["state"] == "crash" and r["lost"] is True


def test_reprobe_catches_gpu_taken_after_allocation(cluster, fproject):
    class GrabAfterFirstProbe(NodeRunner):
        probes = 0

        def run(self, node, command, timeout=None):
            res = super().run(node, command, timeout)
            if command == PROBE_COMMAND:
                type(self).probes += 1
                if type(self).probes == 1:  # someone starts a job right after our probe
                    cluster.set_node(
                        "gpu01", [gpu(0, mem_used=9000, util=90)], procs=[{"gpu_uuid": "GPU-0000", "pid": 9, "mem": 9000, "user": "bob"}]
                    )
            return res

    _, spec, lc = setup(cluster, fproject, n_gpus=1, runner_cls=GrabAfterFirstProbe)
    submit(lc, spec, "smoke", arms=["baseline"])
    out = lc.tick()
    assert actions(out) == {"baseline-s0-smoke-a01": "waiting"}
    assert "no longer idle" in out[0].detail
    assert state(lc, spec, "baseline-s0-smoke-a01")["state"] == "queued"


def test_unanswered_launch_is_resolved_by_reconcile(cluster, fproject, monkeypatch):
    class DropTmuxReply(NodeRunner):
        drop = True

        def run(self, node, command, timeout=None):
            res = super().run(node, command, timeout)
            if self.drop and command.startswith("tmux new-session"):
                raise NodeError("connection reset")  # the session did start
            return res

    _, spec, lc = setup(cluster, fproject, runner_cls=DropTmuxReply)
    arms = [{"id": "slow", "config": "c.yaml", "overrides": {"mode": "hang"}}]
    spec = load_spec(fproject.write_spec(arms=arms, comparisons=[]))
    submit(lc, spec, "smoke")
    out = lc.tick()
    rid = "slow-s0-smoke-a01"
    assert actions(out)[rid] == "launching"
    lc.runner.drop = False
    out = lc.tick()
    assert actions(out)[rid] == "running"  # found the session instead of launching twice
    lc.cancel(lc.ledger(spec.study), rid)


def test_launch_that_never_started_is_requeued(cluster, fproject, monkeypatch):
    class FailTmux(NodeRunner):
        fail = True

        def run(self, node, command, timeout=None):
            if self.fail and command.startswith("tmux new-session"):
                raise NodeError("connection reset")
            return super().run(node, command, timeout)

    _, spec, lc = setup(cluster, fproject, runner_cls=FailTmux)
    submit(lc, spec, "smoke", arms=["baseline"])
    rid = "baseline-s0-smoke-a01"
    assert actions(lc.tick())[rid] == "launching"
    assert lc.tick() == []  # within the grace period: wait
    monkeypatch.setattr(launchmod, "LAUNCH_GRACE_S", 0)
    lc.runner.fail = False
    out = lc.tick()
    assert [o.action for o in out if o.run_id == rid] == ["requeued", "launched"]


# ---------------------------------------------------------------- limits


def test_caps_and_concurrency(cluster, fproject):
    _, spec, lc = setup(cluster, fproject, n_gpus=4, policy={"max_gpus_total": 1})
    submit(lc, spec, "smoke")
    out = lc.tick()
    assert sorted(actions(out).values()) == ["launched", "waiting"]


def test_max_concurrent_runs(cluster, fproject):
    _, spec, lc = setup(
        cluster, fproject, n_gpus=4, budget={"gpu_hours": 100, "max_concurrent_runs": 1}, require_smoke=False
    )
    do_approve(fproject, spec)
    submit(lc, spec, "full")
    out = lc.tick()
    assert list(actions(out).values()).count("launched") == 1
    assert any("max_concurrent_runs" in o.detail for o in out)


def test_budget_counts_live_runs_at_their_timeout(cluster, fproject):
    _, spec, lc = setup(
        cluster, fproject, n_gpus=4, budget={"gpu_hours": 1.5, "max_concurrent_runs": 4}, require_smoke=False
    )
    do_approve(fproject, spec)
    submit(lc, spec, "full")
    out = lc.tick()
    assert list(actions(out).values()).count("launched") == 1  # 1 GPU x 1 h timeout each
    assert any("would exceed 1.5" in o.detail for o in out)


def test_pause_and_drain_block_launches(cluster, fproject):
    cfg, spec, lc = setup(cluster, fproject)
    submit(lc, spec, "smoke")
    (fproject.root / "PAUSE").touch()
    out = lc.tick()
    assert out[0].action == "skipped" and "paused" in out[0].detail
    (fproject.root / "PAUSE").unlink()
    (cfg.state_dir / "DRAIN").touch()
    assert "draining" in lc.tick()[0].detail
    (cfg.state_dir / "DRAIN").unlink()
    assert "launched" in actions(lc.tick()).values()


def test_spec_edit_after_queueing_blocks_full_runs(cluster, fproject):
    _, spec, lc = setup(cluster, fproject, require_smoke=False)
    do_approve(fproject, spec)
    submit(lc, spec, "full")
    spec.path.write_bytes(spec.raw + b"# edited\n")
    out = lc.tick()
    assert set(actions(out).values()) == {"waiting"} and "spec.yaml changed" in out[0].detail


def test_eval_data_change_marks_run_invalid(cluster, fproject):
    _, spec, lc = setup(cluster, fproject, require_smoke=False)
    do_approve(fproject, spec)
    submit(lc, spec, "full", arms=["baseline"], seeds=[0])
    lc.tick()
    wait_exit(lc, spec, ["baseline-s0-a01"])
    (fproject.root / "data" / "eval" / "test.jsonl").write_text('{"q": "leaked"}\n')
    out = lc.tick()
    r = state(lc, spec, "baseline-s0-a01")
    assert r["state"] == "invalid" and r["metric"] == 0.5
    assert actions(out)["baseline-s0-a01"] == "invalid"


def test_exploratory_runs(cluster, fproject):
    _, spec, lc = setup(cluster, fproject, exploratory={"allowed": True, "gpu_hours": 4})
    with pytest.raises(SubmitError, match="only allowed for exploratory"):
        submit(lc, spec, "full", extra_overrides={"lr": 1})
    do_approve(fproject, spec)
    created, _ = submit(lc, spec, "exploratory", arms=["method"], extra_overrides={"score": 0.9})
    assert created[0]["run_id"] == "method-s0-exploratory-a01"
    lc.tick()
    wait_exit(lc, spec, ["method-s0-exploratory-a01"])
    lc.tick()
    assert state(lc, spec, "method-s0-exploratory-a01")["metric"] == 0.9


def test_smoke_refuses_branch_touching_protected_paths(cluster, fproject):
    git(fproject.root, "checkout", "-q", "study/method")
    (fproject.root / "eval" / "judge.py").write_text("THRESHOLD = 0\n")
    fproject.commit("judge edit")
    git(fproject.root, "checkout", "-q", "main")
    _, spec, lc = setup(cluster, fproject)
    with pytest.raises(SubmitError, match="protected paths"):
        submit(lc, spec, "smoke")
