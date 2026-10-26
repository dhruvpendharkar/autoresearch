import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import gpu
from rr import cli
from rr.cli import app

runner = CliRunner()


@pytest.fixture
def env(cluster, fproject):
    cluster.set_node("gpu01", [gpu(0), gpu(1)])
    cfg = cluster.write_config(policy={"idle_probes": 1, "probe_interval_s": 1})
    fproject.write_spec()

    def rr(*args, input=None):
        return runner.invoke(app, ["--config", str(cfg.path), "--project", str(fproject.root), *args], input=input)

    return rr, cfg, fproject


def wait_for_exit(fproject, run_ids, timeout=20):
    deadline = time.time() + timeout
    paths = [fproject.root / "runs" / "demo" / r / "exit_code" for r in run_ids]
    while not all(p.exists() for p in paths):
        assert time.time() < deadline, "runs did not finish"
        time.sleep(0.05)


def test_approve_refuses_without_a_terminal(env):
    rr, _, fproject = env
    res = rr("approve", "demo")
    assert res.exit_code == 3 and "human at a terminal" in res.output
    assert not (fproject.root / "studies" / "demo" / "approval.json").exists()


def test_approve_requires_typing_the_study_id(env, monkeypatch):
    rr, _, fproject = env
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: True)
    res = rr("approve", "demo", input="nope\n")
    assert res.exit_code == 1 and "not approved" in res.output
    res = rr("approve", "demo", input="demo\n")
    assert res.exit_code == 0, res.output
    assert "arm method" in res.output and "approved demo" in res.output
    res = rr("approve", "demo")
    assert "already approved" in res.output


def test_manual_workflow(env, monkeypatch):
    rr, cfg, fproject = env

    res = rr("submit", "demo")
    assert res.exit_code == 1 and "need a valid approval" in res.output

    res = rr("submit", "demo", "--smoke", "--dry-run")
    assert res.exit_code == 0 and "would queue baseline-s0-smoke-a01" in res.output
    assert not (fproject.root / "studies" / "demo" / "ledger.jsonl").exists()

    res = rr("submit", "demo", "--smoke")
    assert res.exit_code == 0, res.output
    assert res.output.count("launched") == 2
    wait_for_exit(fproject, ["baseline-s0-smoke-a01", "method-s0-smoke-a01"])

    res = rr("status")
    assert res.exit_code == 0, res.output
    assert "not approved" in res.output and res.output.count(" done ") == 2

    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: True)
    assert rr("approve", "demo", input="demo\n").exit_code == 0
    res = rr("submit", "demo")
    assert res.exit_code == 0, res.output
    assert "baseline-s0-a01" in res.output and res.output.count("launched") == 2

    wait_for_exit(fproject, ["baseline-s0-a01", "method-s0-a01"])
    res = rr("launch")
    assert "done" in res.output and res.output.count("launched") == 2

    res = rr("status", "demo", "--json", "--no-reconcile")
    report = json.loads(res.output)
    assert report[0]["approval"]["valid"] is True
    states = {r["run_id"]: r["state"] for r in report[0]["runs"]}
    assert states["baseline-s0-a01"] == "done" and states["method-s1-a01"] in ("running", "done")

    res = rr("logs", "baseline-s0-a01", "-n", "5")
    assert res.exit_code == 0 and "[rr]" in res.output and "exit 0" in res.output

    res = rr("logs", "nope")
    assert res.exit_code == 2


def test_cancel_and_drain(env):
    rr, cfg, fproject = env
    spec_path = fproject.root / "studies" / "demo" / "spec.yaml"
    text = spec_path.read_text().replace("steps: 1", "steps: 1\n    mode: hang")
    spec_path.write_text(text)

    assert rr("drain").exit_code == 0
    res = rr("submit", "demo", "--smoke")
    assert "draining" in res.output
    assert rr("drain", "--off").exit_code == 0
    res = rr("launch")
    assert res.output.count("launched") == 2

    res = rr("cancel", "baseline-s0-smoke-a01")
    assert res.exit_code == 0 and "killed rr-demo-baseline-s0-smoke-a01" in res.output
    res = rr("cancel", "--study", "demo", "--reason", "testing")
    assert res.exit_code == 0 and "method-s0-smoke-a01: cancelled" in res.output
    res = rr("status", "--json", "--no-reconcile")
    assert {r["state"] for r in json.loads(res.output)[0]["runs"]} == {"cancelled"}
    assert rr("cancel", "--study", "demo").exit_code == 2  # nothing left


def test_exploratory_set_parsing(env, monkeypatch):
    rr, _, fproject = env
    spec_path = fproject.root / "studies" / "demo" / "spec.yaml"
    spec_path.write_text(spec_path.read_text() + "exploratory: {allowed: true, gpu_hours: 2}\n")
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: True)
    rr("approve", "demo", input="demo\n")
    res = rr("submit", "demo", "--exploratory", "--arm", "method", "--set", "score=0.8", "--set", "flag=true", "--no-launch")
    assert res.exit_code == 0, res.output
    assert "score=0.8 flag=true" in res.output
    res = rr("submit", "demo", "--set", "x=1")
    assert res.exit_code == 1 and "only allowed for exploratory" in res.output
