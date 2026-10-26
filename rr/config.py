"""Load and validate nodes.yaml: node commands plus the GPU sharing policy.

Validation is strict. Unknown keys are errors, so a typo such as
`max_gpu_total` fails loudly instead of silently falling back to a default
that may take more GPUs than intended.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_NAME = "nodes.yaml"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Node:
    name: str
    exec: str
    gpus: int
    gpu_type: str


@dataclass(frozen=True)
class NightPolicy:
    start: time
    end: time
    max_gpus_total: int

    def active(self, now: datetime) -> bool:
        t = now.time()
        if self.start <= self.end:
            return self.start <= t < self.end
        # Window wraps midnight, e.g. 23:00-07:00.
        return t >= self.start or t < self.end


@dataclass(frozen=True)
class Policy:
    max_gpus_total: int = 8
    max_gpus_per_node: int = 4
    night: Optional[NightPolicy] = None
    free_mem_mb: int = 1000
    free_util_pct: int = 5
    idle_probes: int = 3
    probe_interval_s: int = 60
    cooldown_min: int = 60
    # Node access: per-command timeout, failures before a node is marked
    # down, and how long a down node is skipped before trying it again.
    node_timeout_s: int = 30
    node_max_failures: int = 3
    node_down_backoff_s: int = 600

    def max_total_at(self, now: datetime) -> int:
        if self.night is not None and self.night.active(now):
            return self.night.max_gpus_total
        return self.max_gpus_total


@dataclass(frozen=True)
class Config:
    path: Path
    nodes: dict = field(default_factory=dict)  # name -> Node
    policy: Policy = field(default_factory=Policy)
    state_dir: Path = Path("state")

    def node(self, name: str) -> Node:
        try:
            return self.nodes[name]
        except KeyError:
            raise ConfigError(f"unknown node {name!r}; known: {', '.join(self.nodes)}") from None


def _require_int(where: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{where}: must be >= {minimum}, got {value}")
    return value


def _check_keys(where: str, data: dict, allowed: set, required: set = frozenset()) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = set(required) - set(data)
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {sorted(missing)}")


def _parse_hhmm(where: str, text: str) -> time:
    try:
        return datetime.strptime(text.strip(), "%H:%M").time()
    except ValueError:
        raise ConfigError(f"{where}: expected HH:MM, got {text!r}") from None


def _parse_night(data: Any) -> NightPolicy:
    if not isinstance(data, dict):
        raise ConfigError("policy.night: expected a mapping")
    _check_keys("policy.night", data, {"hours", "max_gpus_total"}, {"hours", "max_gpus_total"})
    hours = data["hours"]
    if not isinstance(hours, str) or hours.count("-") != 1:
        raise ConfigError(f"policy.night.hours: expected 'HH:MM-HH:MM', got {hours!r}")
    start_s, end_s = hours.split("-")
    start = _parse_hhmm("policy.night.hours", start_s)
    end = _parse_hhmm("policy.night.hours", end_s)
    if start == end:
        raise ConfigError("policy.night.hours: start and end must differ")
    return NightPolicy(start, end, _require_int("policy.night.max_gpus_total", data["max_gpus_total"]))


def _parse_policy(data: Any) -> Policy:
    if data is None:
        return Policy()
    if not isinstance(data, dict):
        raise ConfigError("policy: expected a mapping")
    int_fields = {
        "max_gpus_total": 0,
        "max_gpus_per_node": 0,
        "free_mem_mb": 0,
        "free_util_pct": 0,
        "idle_probes": 1,
        "probe_interval_s": 1,
        "cooldown_min": 0,
        "node_timeout_s": 1,
        "node_max_failures": 1,
        "node_down_backoff_s": 0,
    }
    _check_keys("policy", data, set(int_fields) | {"night"})
    kwargs = {k: _require_int(f"policy.{k}", data[k], lo) for k, lo in int_fields.items() if k in data}
    if data.get("night") is not None:
        kwargs["night"] = _parse_night(data["night"])
    return Policy(**kwargs)


def _parse_nodes(data: Any) -> dict:
    if not isinstance(data, dict) or not data:
        raise ConfigError("nodes: expected a non-empty mapping of node name -> settings")
    nodes = {}
    for name, spec in data.items():
        where = f"nodes.{name}"
        if not isinstance(spec, dict):
            raise ConfigError(f"{where}: expected a mapping")
        _check_keys(where, spec, {"exec", "gpus", "gpu_type"}, {"exec", "gpus", "gpu_type"})
        if not isinstance(spec["exec"], str) or not spec["exec"].strip():
            raise ConfigError(f"{where}.exec: expected a non-empty command string")
        if not isinstance(spec["gpu_type"], str) or not spec["gpu_type"]:
            raise ConfigError(f"{where}.gpu_type: expected a string")
        nodes[str(name)] = Node(
            name=str(name),
            exec=spec["exec"],
            gpus=_require_int(f"{where}.gpus", spec["gpus"], 1),
            gpu_type=spec["gpu_type"],
        )
    return nodes


def parse_config(data: Any, path: Path) -> Config:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    _check_keys(str(path), data, {"nodes", "policy", "state_dir"}, {"nodes"})
    state_dir = Path(data.get("state_dir") or "state")
    if not state_dir.is_absolute():
        state_dir = path.parent / state_dir
    return Config(
        path=path,
        nodes=_parse_nodes(data["nodes"]),
        policy=_parse_policy(data.get("policy")),
        state_dir=state_dir,
    )


def find_config(explicit: Optional[Path] = None, start: Optional[Path] = None) -> Path:
    """Resolve nodes.yaml: explicit path, then $RR_CONFIG, then search upward from cwd."""
    if explicit is not None:
        return explicit
    env = os.environ.get("RR_CONFIG")
    if env:
        return Path(env)
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        candidate = d / CONFIG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigError(f"no {CONFIG_NAME} found in {here} or its parents; set RR_CONFIG or pass --config")


def load_config(explicit: Optional[Path] = None) -> Config:
    path = find_config(explicit).resolve()
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from None
    return parse_config(data, path)
