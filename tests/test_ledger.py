import json
import threading

import pytest

from rr.ledger import Ledger, LedgerError, make_run_id, project_ledgers, reserved_gpus


@pytest.fixture
def ledger(tmp_path, clock):
    return Ledger(tmp_path / "studies" / "s1" / "ledger.jsonl", clock=clock)


def new_run(ledger, rid="method-s1-a01", kind="full"):
    return ledger.create(rid, arm="method", seed=1, kind=kind, attempt=1, spec_hash="abc")


def test_run_ids():
    assert make_run_id("method", 1, 3) == "method-s1-a03"
    assert make_run_id("baseline", 0, 1, "smoke") == "baseline-s0-smoke-a01"
    with pytest.raises(LedgerError):
        make_run_id("m", 0, 1, "weird")


def test_lifecycle_and_fold(ledger, clock):
    new_run(ledger)
    clock.advance(5)
    ledger.transition("method-s1-a01", "launching", node="gpu03", gpus=[4, 5], gpu_type="RTX6000")
    ledger.transition("method-s1-a01", "running", tmux_session="rr-method-s1-a01")
    ledger.append("method-s1-a01", wandb_run="entity/proj/abc123")
    clock.advance(100)
    rec = ledger.transition("method-s1-a01", "done", metric=0.412, result="promising")
    assert rec["state"] == "done"
    folded = ledger.get("method-s1-a01")
    assert folded["node"] == "gpu03" and folded["wandb_run"] == "entity/proj/abc123" and folded["metric"] == 0.412
    assert folded["updated_at"] - folded["created_at"] == 105
    assert len(ledger.events()) == 5  # append-only: one line per change


@pytest.mark.parametrize(
    "path,bad",
    [
        (["launching", "running"], "queued"),
        (["launching", "running", "done"], "running"),
        (["cancelled"], "queued"),
        (["launching", "running", "crash"], "running"),
    ],
)
def test_illegal_transitions(ledger, path, bad):
    new_run(ledger)
    for st in path:
        extra = {"node": "gpu01", "gpus": [0]} if st == "launching" else {}
        ledger.transition("method-s1-a01", st, **extra)
    with pytest.raises(LedgerError, match="illegal transition"):
        ledger.transition("method-s1-a01", bad)


def test_done_can_be_invalidated(ledger):
    new_run(ledger)
    ledger.transition("method-s1-a01", "launching", node="gpu01", gpus=[0])
    ledger.transition("method-s1-a01", "running")
    ledger.transition("method-s1-a01", "done", metric=0.9)
    assert ledger.transition("method-s1-a01", "invalid", notes="eval data hash mismatch")["state"] == "invalid"


def test_validation(ledger):
    with pytest.raises(LedgerError, match="unknown run"):
        ledger.append("nope", notes="x")
    with pytest.raises(LedgerError, match="needs"):
        ledger.transition("x-s0-a01", "queued", arm="x")
    new_run(ledger)
    with pytest.raises(LedgerError, match="cannot change"):
        ledger.append("method-s1-a01", seed=2)
    with pytest.raises(LedgerError, match="unknown result"):
        ledger.append("method-s1-a01", result="great")
    with pytest.raises(LedgerError, match="requires node and gpus"):
        ledger.transition("method-s1-a01", "launching")
    with pytest.raises(LedgerError, match="set by the ledger"):
        ledger.append("method-s1-a01", ts=0)


def test_smoke_runs_never_count(ledger):
    new_run(ledger, "method-s1-smoke-a01", kind="smoke")
    with pytest.raises(LedgerError, match="smoke"):
        ledger.append("method-s1-smoke-a01", result="keep")


def test_reservations_across_studies(tmp_path, clock):
    a = Ledger(tmp_path / "studies" / "a" / "ledger.jsonl", clock=clock)
    b = Ledger(tmp_path / "studies" / "b" / "ledger.jsonl", clock=clock)
    new_run(a, "r1")
    a.transition("r1", "launching", node="gpu01", gpus=[0, 1])
    new_run(a, "r2")
    a.transition("r2", "launching", node="gpu01", gpus=[2])
    a.transition("r2", "queued")  # re-probe found the GPU taken: released
    new_run(b, "r3")
    b.transition("r3", "launching", node="gpu02", gpus=[5])
    b.transition("r3", "running")
    new_run(b, "r4")
    b.transition("r4", "launching", node="gpu02", gpus=[6])
    b.transition("r4", "running")
    b.transition("r4", "crash")
    assert reserved_gpus(project_ledgers(tmp_path)) == {("gpu01", 0), ("gpu01", 1), ("gpu02", 5)}


def test_torn_line_is_skipped_and_not_glued(ledger):
    new_run(ledger)
    with open(ledger.path, "a") as fh:
        fh.write('{"run_id": "method-s1-a01", "sta')  # crash mid-write
    ledger.append("method-s1-a01", notes="after crash")
    assert ledger.get("method-s1-a01")["notes"] == "after crash"
    assert ledger.bad_lines() == [2]


def test_concurrent_appends_are_serialized(ledger):
    for i in range(20):
        ledger.create(f"r{i}", arm="a", seed=i, kind="full", attempt=1)

    def worker(i):
        for j in range(10):
            ledger.append(f"r{i}", notes=f"n{j}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = ledger.path.read_text().splitlines()
    assert len(lines) == 20 + 200
    assert all(json.loads(line) for line in lines)
    assert ledger.bad_lines() == []
