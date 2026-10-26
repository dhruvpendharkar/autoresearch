"""A fake cluster: each node is a local `bash -c` with fake GPU tools on PATH."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest
import yaml

from rr.config import load_config

FAKE_TOOLS = Path(__file__).with_name("fake_tools.py")


def gpu(index, mem_used=3, util=0, mem_total=24576, uuid=None, name="Quadro RTX 6000"):
    return {
        "index": index,
        "uuid": uuid or f"GPU-{index:04d}",
        "name": name,
        "mem_used": mem_used,
        "mem_total": mem_total,
        "util": util,
    }


class FakeClock:
    def __init__(self, t: float = 1_700_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeCluster:
    def __init__(self, root: Path):
        self.root = root
        self.data_dir = root / "fake-nodes"
        self.bin_dir = root / "fake-bin"
        self.data_dir.mkdir()
        self.bin_dir.mkdir()
        for tool in ("nvidia-smi", "gpustat", "ps", "whoami", "tmux"):
            p = self.bin_dir / tool
            p.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_TOOLS))} {tool} "$@"\n')
            p.chmod(0o755)
        self.execs = {}

    def set_node(self, name, gpus, procs=(), me="me", **flags):
        spec = {"me": me, "gpus": list(gpus), "procs": list(procs), **flags}
        (self.data_dir / f"{name}.json").write_text(json.dumps(spec))
        self.execs.setdefault(name, self.exec_for(name))

    def exec_for(self, name):
        path = f"{self.bin_dir}:/usr/bin:/bin"
        return (
            f"env RR_FAKE_NODE={name} FAKE_CLUSTER={shlex.quote(str(self.data_dir))} "
            f"PATH={shlex.quote(path)} bash -c"
        )

    def write_config(self, policy=None, gpu_type="RTX6000", gpus_per_node=None):
        nodes = {}
        for name, exec_str in self.execs.items():
            n_gpus = gpus_per_node or len(json.loads((self.data_dir / f"{name}.json").read_text())["gpus"]) or 1
            nodes[name] = {"exec": exec_str, "gpus": n_gpus, "gpu_type": gpu_type}
        data = {"nodes": nodes, "policy": policy or {}}
        path = self.root / "nodes.yaml"
        path.write_text(yaml.safe_dump(data))
        return load_config(path)


@pytest.fixture
def cluster(tmp_path):
    return FakeCluster(tmp_path)


@pytest.fixture
def clock():
    return FakeClock()


# ---------------------------------------------------------------- a fake research project

FAKE_TRAIN = r'''
"""Fake training script. Behaviour is driven by key=value overrides."""
import json, math, os, sys, time

args = sys.argv[1:]
opts = {"mode": "ok", "score": "0.5", "sleep": "0", "steps": "3"}
seed = None
i = 0
while i < len(args):
    if args[i] == "--seed":
        seed = int(args[i + 1]); i += 2; continue
    if args[i] == "--config":
        opts["config"] = args[i + 1]; i += 2; continue
    k, _, v = args[i].partition("=")
    opts[k] = v
    i += 1

def write(path, data):
    if path:
        with open(path + ".tmp", "w") as fh:
            json.dump(data, fh)
        os.replace(path + ".tmp", path)

print("args", json.dumps(opts), "seed", seed, "cuda", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
for step in range(int(opts["steps"])):
    write(os.environ.get("RR_HEARTBEAT"), {"step": step, "ts": time.time(), "loss": 1.0 / (step + 1)})
    time.sleep(float(opts["sleep"]))
mode = opts["mode"]
if mode == "crash":
    raise RuntimeError("CUDA error: an illegal memory access was encountered")
if mode == "spawn":  # like a dataloader worker: a grandchild of run.sh
    import subprocess
    worker = subprocess.Popen(["sleep", "3600"])
    with open(os.path.join(os.environ["RR_RUN_DIR"], "worker_pid"), "w") as fh:
        fh.write(str(worker.pid))
if mode in ("hang", "spawn"):
    time.sleep(3600)
if mode != "nometrics":
    write(os.environ.get("RR_FINAL_METRICS"), {"eval/pass@1": float(opts["score"]) + (seed or 0) * 0.01})
print("done", flush=True)
'''


def git(root, *args):
    import subprocess

    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def spec_yaml(study="demo", **over):
    spec = {
        "study": study,
        "base_commit": "main",
        "metric": {"name": "eval/pass@1", "direction": "max", "source": "final_metrics.json"},
        "seeds": [0, 1],
        "resources": {"gpus_per_run": 1, "gpu_type": "RTX6000", "run_timeout_h": 1},
        "budget": {"gpu_hours": 100, "max_concurrent_runs": 4, "max_retries": 2},
        "run": {"command": f"{shlex.quote(sys.executable)} train.py --config {{config}} --seed {{seed}} {{overrides}}"},
        "smoke": {"overrides": {"steps": 1}, "timeout_min": 5},
        "arms": [
            {"id": "baseline", "config": "configs/base.yaml"},
            {"id": "method", "config": "configs/base.yaml", "branch": "study/method"},
        ],
        "comparisons": [{"treatment": "method", "control": "baseline"}],
        "protected_paths": ["eval/"],
        "eval_data": ["data/eval"],
    }
    spec.update(over)
    return yaml.safe_dump(spec, sort_keys=False)


class FakeProject:
    def __init__(self, root: Path):
        from rr.project import Project

        self.root = root
        root.mkdir(parents=True)
        git(root, "init", "-q")
        (root / "train.py").write_text(FAKE_TRAIN)
        (root / "configs").mkdir()
        (root / "configs" / "base.yaml").write_text("lr: 0.1\n")
        (root / "eval").mkdir()
        (root / "eval" / "judge.py").write_text("THRESHOLD = 0.5\n")
        (root / ".gitignore").write_text("runs/\nstudies/\ndata/\n")
        self.commit("base")
        git(root, "branch", "-M", "main")
        git(root, "checkout", "-q", "-b", "study/method")
        (root / "train.py").write_text(FAKE_TRAIN.replace('"score": "0.5"', '"score": "0.6"'))
        self.commit("method")
        git(root, "checkout", "-q", "main")
        # Eval data is untracked, as large datasets usually are.
        (root / "data" / "eval").mkdir(parents=True)
        (root / "data" / "eval" / "test.jsonl").write_text('{"q": 1}\n')
        self.project = Project(root.resolve())

    def commit(self, msg):
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", msg)
        return git(self.root, "rev-parse", "HEAD")

    def write_spec(self, study="demo", **over):
        d = self.root / "studies" / study
        d.mkdir(parents=True, exist_ok=True)
        (d / "spec.yaml").write_text(spec_yaml(study, **over))
        return d / "spec.yaml"


@pytest.fixture
def fproject(tmp_path):
    return FakeProject(tmp_path / "proj")
