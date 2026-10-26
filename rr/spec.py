"""Parse and validate a study's pre-registration: `studies/<id>/spec.yaml`.

The spec is human-authored. Validation is strict (unknown keys are errors)
because a silently ignored typo in seeds, metrics or comparisons would change
what the study claims without anyone noticing.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ID_RE = re.compile(r"^[A-Za-z0-9_]+(-[A-Za-z0-9_]+)*$")
COMMAND_FIELDS = {"config", "seed", "overrides", "run_dir", "run_id", "arm", "study"}
TESTS = ("mean_diff_bootstrap_ci",)


class SpecError(Exception):
    pass


@dataclass(frozen=True)
class Metric:
    name: str
    direction: str  # max | min
    source: str  # file in the run dir written by the training script


@dataclass(frozen=True)
class Arm:
    id: str
    config: str
    branch: Optional[str] = None
    overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Comparison:
    treatment: str
    control: str
    test: str = "mean_diff_bootstrap_ci"


@dataclass(frozen=True)
class Spec:
    path: Path
    raw: bytes
    study: str
    base_commit: str
    metric: Metric
    seeds: tuple
    gpus_per_run: int
    gpu_type: Optional[str]
    run_timeout_h: float
    gpu_hours: float
    max_concurrent_runs: int
    max_retries: int
    command: str
    override_format: str
    smoke_overrides: dict
    smoke_timeout_min: float
    smoke_gpus: int
    require_smoke: bool
    arms: tuple  # tuple[Arm]
    comparisons: tuple  # tuple[Comparison]
    exploratory_allowed: bool
    exploratory_gpu_hours: float
    protected_paths: tuple
    eval_data: tuple
    tunable: tuple

    def arm(self, arm_id: str) -> Arm:
        for a in self.arms:
            if a.id == arm_id:
                return a
        raise SpecError(f"unknown arm {arm_id!r}; arms: {', '.join(a.id for a in self.arms)}")


# ---------------------------------------------------------------- field helpers


def _keys(where: str, data: Any, allowed: set, required: set = frozenset()) -> dict:
    if not isinstance(data, dict):
        raise SpecError(f"{where}: expected a mapping")
    unknown = set(data) - allowed
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = set(required) - set(data)
    if missing:
        raise SpecError(f"{where}: missing required key(s) {sorted(missing)}")
    return data


def _str(where: str, v: Any) -> str:
    if not isinstance(v, str) or not v.strip():
        raise SpecError(f"{where}: expected a non-empty string, got {v!r}")
    return v


def _int(where: str, v: Any, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise SpecError(f"{where}: expected an integer >= {lo}, got {v!r}")
    return v


def _pos(where: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise SpecError(f"{where}: expected a positive number, got {v!r}")
    return float(v)


def _bool(where: str, v: Any) -> bool:
    if not isinstance(v, bool):
        raise SpecError(f"{where}: expected true or false, got {v!r}")
    return v


def _id(where: str, v: Any) -> str:
    v = _str(where, v)
    if not ID_RE.match(v):
        raise SpecError(f"{where}: {v!r} may contain only letters, digits, '_' and single '-'")
    return v


def _overrides(where: str, v: Any) -> dict:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise SpecError(f"{where}: expected a mapping of key -> scalar")
    for k, val in v.items():
        if not isinstance(k, str) or not k:
            raise SpecError(f"{where}: keys must be strings")
        if isinstance(val, (dict, list)):
            raise SpecError(f"{where}.{k}: values must be scalars")
    return dict(v)


def _str_list(where: str, v: Any) -> tuple:
    if v is None:
        return ()
    if not isinstance(v, list):
        raise SpecError(f"{where}: expected a list of strings")
    return tuple(_str(f"{where}[{i}]", x) for i, x in enumerate(v))


def _command(where: str, v: Any) -> str:
    v = _str(where, v)
    try:
        names = {f for _, f, _, _ in string.Formatter().parse(v) if f is not None}
    except ValueError as e:
        raise SpecError(f"{where}: bad template: {e}") from None
    unknown = names - COMMAND_FIELDS
    if unknown:
        raise SpecError(f"{where}: unknown placeholder(s) {sorted(unknown)}; allowed: {sorted(COMMAND_FIELDS)}")
    return v


# ---------------------------------------------------------------- parsing


def parse_spec(data: Any, path: Path, raw: bytes = b"") -> Spec:
    top = {
        "study", "base_commit", "metric", "seeds", "resources", "budget", "run", "smoke",
        "require_smoke", "arms", "comparisons", "exploratory", "protected_paths", "eval_data", "tunable",
    }
    d = _keys("spec", data, top, {"study", "base_commit", "metric", "seeds", "resources", "budget", "run", "arms"})

    study = _id("study", d["study"])
    if path.parent.name and path.parent.name != study:
        raise SpecError(f"study: {study!r} must match its directory name {path.parent.name!r}")

    m = _keys("metric", d["metric"], {"name", "direction", "source"}, {"name", "direction"})
    if m["direction"] not in ("max", "min"):
        raise SpecError(f"metric.direction: expected max or min, got {m['direction']!r}")
    metric = Metric(_str("metric.name", m["name"]), m["direction"], _str("metric.source", m.get("source", "final_metrics.json")))

    if not isinstance(d["seeds"], list) or not d["seeds"]:
        raise SpecError("seeds: expected a non-empty list of integers")
    seeds = tuple(_int(f"seeds[{i}]", s, 0) for i, s in enumerate(d["seeds"]))
    if len(set(seeds)) != len(seeds):
        raise SpecError("seeds: duplicates are not allowed")

    r = _keys("resources", d["resources"], {"gpus_per_run", "gpu_type", "run_timeout_h"}, {"gpus_per_run", "run_timeout_h"})
    b = _keys("budget", d["budget"], {"gpu_hours", "max_concurrent_runs", "max_retries"}, {"gpu_hours"})
    run = _keys("run", d["run"], {"command", "override_format"}, {"command"})
    override_format = _str("run.override_format", run.get("override_format", "{key}={value}"))
    if "{key}" not in override_format or "{value}" not in override_format:
        raise SpecError("run.override_format: must contain {key} and {value}")
    smoke = _keys("smoke", d.get("smoke") or {}, {"overrides", "timeout_min", "gpus"})

    if not isinstance(d["arms"], list) or not d["arms"]:
        raise SpecError("arms: expected a non-empty list")
    arms = []
    for i, a in enumerate(d["arms"]):
        a = _keys(f"arms[{i}]", a, {"id", "config", "branch", "overrides"}, {"id", "config"})
        arms.append(
            Arm(
                id=_id(f"arms[{i}].id", a["id"]),
                config=_str(f"arms[{i}].config", a["config"]),
                branch=_str(f"arms[{i}].branch", a["branch"]) if a.get("branch") is not None else None,
                overrides=_overrides(f"arms[{i}].overrides", a.get("overrides")),
            )
        )
    ids = [a.id for a in arms]
    if len(set(ids)) != len(ids):
        raise SpecError("arms: ids must be unique")

    comparisons = []
    for i, c in enumerate(d.get("comparisons") or []):
        c = _keys(f"comparisons[{i}]", c, {"treatment", "control", "test"}, {"treatment", "control"})
        for k in ("treatment", "control"):
            if c[k] not in ids:
                raise SpecError(f"comparisons[{i}].{k}: unknown arm {c[k]!r}")
        if c["treatment"] == c["control"]:
            raise SpecError(f"comparisons[{i}]: treatment and control must differ")
        test = c.get("test", TESTS[0])
        if test not in TESTS:
            raise SpecError(f"comparisons[{i}].test: expected one of {TESTS}, got {test!r}")
        comparisons.append(Comparison(c["treatment"], c["control"], test))

    x = _keys("exploratory", d.get("exploratory") or {}, {"allowed", "gpu_hours"})
    exploratory_allowed = _bool("exploratory.allowed", x.get("allowed", False))

    return Spec(
        path=path,
        raw=raw,
        study=study,
        base_commit=_str("base_commit", str(d["base_commit"]) if isinstance(d["base_commit"], int) else d["base_commit"]),
        metric=metric,
        seeds=seeds,
        gpus_per_run=_int("resources.gpus_per_run", r["gpus_per_run"], 1),
        gpu_type=_str("resources.gpu_type", r["gpu_type"]) if r.get("gpu_type") is not None else None,
        run_timeout_h=_pos("resources.run_timeout_h", r["run_timeout_h"]),
        gpu_hours=_pos("budget.gpu_hours", b["gpu_hours"]),
        max_concurrent_runs=_int("budget.max_concurrent_runs", b.get("max_concurrent_runs", 4), 1),
        max_retries=_int("budget.max_retries", b.get("max_retries", 2), 0),
        command=_command("run.command", run["command"]),
        override_format=override_format,
        smoke_overrides=_overrides("smoke.overrides", smoke.get("overrides")),
        smoke_timeout_min=_pos("smoke.timeout_min", smoke.get("timeout_min", 30)),
        smoke_gpus=_int("smoke.gpus", smoke.get("gpus", 1), 1),
        require_smoke=_bool("require_smoke", d.get("require_smoke", True)),
        arms=tuple(arms),
        comparisons=tuple(comparisons),
        exploratory_allowed=exploratory_allowed,
        exploratory_gpu_hours=_pos("exploratory.gpu_hours", x["gpu_hours"]) if "gpu_hours" in x else 0.0,
        protected_paths=_str_list("protected_paths", d.get("protected_paths")),
        eval_data=_str_list("eval_data", d.get("eval_data")),
        tunable=_str_list("tunable", d.get("tunable")),
    )


def load_spec(path: Path) -> Spec:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        raise SpecError(f"spec not found: {path}") from None
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise SpecError(f"{path}: invalid YAML: {e}") from None
    return parse_spec(data, Path(path), raw)


# ---------------------------------------------------------------- command rendering


def format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)
