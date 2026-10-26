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
        for tool in ("nvidia-smi", "gpustat", "ps", "whoami"):
            p = self.bin_dir / tool
            p.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_TOOLS))} {tool} "$@"\n')
            p.chmod(0o755)
        (self.bin_dir / "tmux").write_text("#!/bin/sh\nexit 0\n")
        (self.bin_dir / "tmux").chmod(0o755)
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
