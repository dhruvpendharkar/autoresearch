from rr.ledger import Ledger, reserved_gpus
from rr.project import Project, all_ledgers, find_project, register_ledger


def test_find_project_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("RR_PROJECT", raising=False)
    assert find_project(tmp_path / "x").root == (tmp_path / "x").resolve()
    monkeypatch.setenv("RR_PROJECT", str(tmp_path))
    assert find_project().root == tmp_path.resolve()


def test_reservations_span_registered_projects(cluster, tmp_path):
    from conftest import gpu

    cluster.set_node("gpu01", [gpu(0)])
    cfg = cluster.write_config()
    other = Project(tmp_path / "other-project")
    here = Project(tmp_path / "this-project")
    lg = other.ledger("s1")
    lg.create("r1", arm="a", seed=0, kind="full", attempt=1)
    lg.transition("r1", "launching", node="gpu01", gpus=[3])
    register_ledger(cfg, lg)
    register_ledger(cfg, lg)  # idempotent
    assert reserved_gpus(all_ledgers(cfg, here)) == {("gpu01", 3)}
    assert len(all_ledgers(cfg, here)) == 1


def test_queued_run_can_crash_when_unpreparable(tmp_path):
    lg = Ledger(tmp_path / "l.jsonl")
    lg.create("r1", arm="a", seed=0, kind="full", attempt=1)
    assert lg.transition("r1", "crash", notes="commit not found")["state"] == "crash"
