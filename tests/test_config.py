from datetime import datetime
from pathlib import Path

import pytest
import yaml

from rr.config import ConfigError, find_config, load_config, parse_config

BASE = {
    "nodes": {
        "gpu01": {"exec": "ssh gpu01", "gpus": 8, "gpu_type": "RTX6000"},
        "gpu02": {"exec": "ssh gpu02", "gpus": 8, "gpu_type": "RTX6000"},
    },
    "policy": {
        "max_gpus_total": 8,
        "max_gpus_per_node": 4,
        "night": {"hours": "23:00-07:00", "max_gpus_total": 16},
        "free_mem_mb": 1000,
        "free_util_pct": 5,
        "idle_probes": 3,
        "probe_interval_s": 60,
        "cooldown_min": 60,
    },
}


def parse(data):
    return parse_config(data, Path("/lab/nodes.yaml"))


def test_parses_design_doc_example():
    cfg = parse(BASE)
    assert set(cfg.nodes) == {"gpu01", "gpu02"}
    assert cfg.nodes["gpu01"].exec == "ssh gpu01"
    assert cfg.policy.max_gpus_per_node == 4
    assert cfg.state_dir == Path("/lab/state")


def test_policy_defaults_when_omitted():
    cfg = parse({"nodes": BASE["nodes"]})
    assert cfg.policy.max_gpus_total == 8
    assert cfg.policy.night is None


@pytest.mark.parametrize(
    "when,expected",
    [("2026-10-25 22:59", 8), ("2026-10-25 23:00", 16), ("2026-10-26 03:00", 16), ("2026-10-26 07:00", 8), ("2026-10-26 12:00", 8)],
)
def test_night_cap_wraps_midnight(when, expected):
    cfg = parse(BASE)
    assert cfg.policy.max_total_at(datetime.strptime(when, "%Y-%m-%d %H:%M")) == expected


def test_daytime_night_window():
    data = {**BASE, "policy": {"night": {"hours": "12:00-13:00", "max_gpus_total": 2}}}
    cfg = parse(data)
    assert cfg.policy.max_total_at(datetime(2026, 1, 1, 12, 30)) == 2
    assert cfg.policy.max_total_at(datetime(2026, 1, 1, 13, 0)) == 8


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: d["policy"].update(max_gpu_total=99), "unknown key"),
        (lambda d: d["policy"].update(max_gpus_total="8"), "expected an integer"),
        (lambda d: d["policy"].update(max_gpus_total=True), "expected an integer"),
        (lambda d: d["policy"].update(idle_probes=0), "must be >= 1"),
        (lambda d: d["policy"]["night"].update(hours="23:00"), "HH:MM-HH:MM"),
        (lambda d: d["policy"]["night"].update(hours="25:00-07:00"), "expected HH:MM"),
        (lambda d: d["nodes"]["gpu01"].pop("exec"), "missing required"),
        (lambda d: d["nodes"]["gpu01"].update(exec="  "), "non-empty"),
        (lambda d: d["nodes"]["gpu01"].update(gpu=8), "unknown key"),
        (lambda d: d.update(nodes={}), "non-empty mapping"),
        (lambda d: d.update(extra=1), "unknown key"),
    ],
)
def test_rejects_bad_config(mutate, message):
    data = yaml.safe_load(yaml.safe_dump(BASE))  # deep copy
    mutate(data)
    with pytest.raises(ConfigError, match=message):
        parse(data)


def test_find_config_searches_upward(tmp_path, monkeypatch):
    monkeypatch.delenv("RR_CONFIG", raising=False)
    (tmp_path / "nodes.yaml").write_text(yaml.safe_dump(BASE))
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_config(start=deep) == tmp_path.resolve() / "nodes.yaml"


def test_env_var_wins(tmp_path, monkeypatch):
    p = tmp_path / "custom.yaml"
    p.write_text(yaml.safe_dump(BASE))
    monkeypatch.setenv("RR_CONFIG", str(p))
    assert load_config().path == p.resolve()


def test_invalid_yaml(tmp_path):
    p = tmp_path / "nodes.yaml"
    p.write_text("nodes: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p)
