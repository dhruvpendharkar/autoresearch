import json

from typer.testing import CliRunner

from conftest import gpu
from rr.cli import app
from rr.ledger import Ledger

runner = CliRunner()


def invoke(cfg, *args):
    return runner.invoke(app, ["--config", str(cfg.path), *args])


def test_gpus_table_and_json(cluster, tmp_path, monkeypatch):
    cluster.set_node(
        "gpu01",
        [gpu(0), gpu(1, mem_used=9000, util=90)],
        procs=[{"gpu_uuid": "GPU-0001", "pid": 4242, "mem": 9000, "user": "bob"}],
    )
    cfg = cluster.write_config(policy={"idle_probes": 1})
    monkeypatch.chdir(tmp_path)

    res = invoke(cfg, "gpus")
    assert res.exit_code == 0, res.output
    assert "free" in res.output and "foreign" in res.output and "bob" in res.output
    assert "held 0/8 total" in res.output

    ledger = Ledger(tmp_path / "studies" / "s" / "ledger.jsonl")
    ledger.create("r1", arm="a", seed=0, kind="full", attempt=1)
    ledger.transition("r1", "launching", node="gpu01", gpus=[0])

    res = invoke(cfg, "gpus", "--cached", "--json")
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["held"] == 1
    assert {(x["index"], x["status"]) for x in out["gpus"]} == {(0, "reserved"), (1, "foreign")}


def test_gpus_reports_failed_node(cluster, tmp_path, monkeypatch):
    cluster.set_node("gpu01", [gpu(0)])
    cluster.set_node("gpu02", [gpu(0)])
    cluster.execs["gpu02"] = "false"
    cfg = cluster.write_config()
    monkeypatch.chdir(tmp_path)
    res = invoke(cfg, "gpus")
    assert res.exit_code == 0
    assert "gpu02: probe failed" in res.output


def test_gpus_unknown_node(cluster):
    cluster.set_node("gpu01", [gpu(0)])
    res = invoke(cluster.write_config(), "gpus", "--node", "gpu99")
    assert res.exit_code == 2 and "unknown node" in res.output


def test_config_error_exit_code(tmp_path):
    (tmp_path / "nodes.yaml").write_text("nodes: {}\n")
    res = runner.invoke(app, ["--config", str(tmp_path / "nodes.yaml"), "gpus"])
    assert res.exit_code == 2 and "config error" in res.output


def test_doctor_on_healthy_fake_cluster(cluster):
    cluster.set_node(
        "gpu01", [gpu(0), gpu(1)], procs=[{"gpu_uuid": "GPU-0001", "pid": 7, "mem": 9000, "user": "bob"}]
    )
    res = invoke(cluster.write_config(), "doctor", "--no-net")
    assert res.exit_code == 0, res.output
    for expected in ("runs a non-interactive command", "shares this filesystem", "tmux installed", "2 GPU(s) via nvidia-smi", "1/1 attributed"):
        assert expected in res.output


def test_doctor_flags_unreachable_node(cluster):
    cluster.set_node("gpu01", [gpu(0)])
    cluster.execs["gpu01"] = "false"
    res = invoke(cluster.write_config(), "doctor", "--no-net")
    assert res.exit_code == 1
    assert "FAIL" in res.output
