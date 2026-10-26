from pathlib import Path

import pytest
import yaml

from conftest import spec_yaml
from rr.spec import SpecError, format_value, load_spec, parse_spec

PATH = Path("/p/studies/demo/spec.yaml")


def base():
    return yaml.safe_load(spec_yaml())


def test_parses_example(fproject):
    spec = load_spec(fproject.write_spec())
    assert spec.study == "demo"
    assert spec.seeds == (0, 1)
    assert spec.arm("method").branch == "study/method"
    assert spec.comparisons[0].test == "mean_diff_bootstrap_ci"
    assert spec.smoke_gpus == 1 and spec.smoke_overrides == {"steps": 1}
    assert spec.require_smoke is True
    assert spec.raw.startswith(b"study: demo")


def test_design_doc_example_parses():
    doc = """
study: async-rollout-offpolicy-v2
base_commit: 3f2a9c1
metric: {name: eval/pass@1, direction: max, source: final_metrics.json}
seeds: [0, 1, 2]
resources: {gpus_per_run: 2, gpu_type: RTX6000, run_timeout_h: 10}
budget: {gpu_hours: 96, max_concurrent_runs: 4, max_retries: 2}
run: {command: "python train.py --config {config} --seed {seed} {overrides}"}
arms:
  - id: baseline
    config: configs/grpo_sync.yaml
  - id: method
    config: configs/grpo_async.yaml
    branch: study/async-rollout-offpolicy-v2
  - id: ablate_no_is_correction
    config: configs/grpo_async.yaml
    overrides: {is_correction: false}
comparisons:
  - {treatment: method, control: baseline, test: mean_diff_bootstrap_ci}
  - {treatment: method, control: ablate_no_is_correction}
exploratory: {allowed: true, gpu_hours: 16}
protected_paths: [eval/, data/eval/, src/metrics/]
"""
    spec = parse_spec(yaml.safe_load(doc), Path("/p/studies/async-rollout-offpolicy-v2/spec.yaml"))
    assert spec.arm("ablate_no_is_correction").overrides == {"is_correction": False}
    assert spec.exploratory_allowed and spec.exploratory_gpu_hours == 16


def test_all_digit_commit_is_accepted():
    d = base()
    d["base_commit"] = 1234567
    assert parse_spec(d, PATH).base_commit == "1234567"


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: d.update(seed=[0]), "unknown key"),
        (lambda d: d.update(study="other"), "must match its directory"),
        (lambda d: d["metric"].update(direction="up"), "max or min"),
        (lambda d: d.update(seeds=[]), "non-empty"),
        (lambda d: d.update(seeds=[0, 0]), "duplicates"),
        (lambda d: d.update(seeds=["0"]), "integer"),
        (lambda d: d["resources"].update(gpus_per_run=0), ">= 1"),
        (lambda d: d["resources"].update(run_timeout_h=-1), "positive"),
        (lambda d: d["budget"].pop("gpu_hours"), "missing required"),
        (lambda d: d["run"].update(command="python train.py {cfg}"), "unknown placeholder"),
        (lambda d: d["run"].update(command="python train.py {"), "bad template"),
        (lambda d: d["run"].update(override_format="--{key}"), "{key} and {value}"),
        (lambda d: d["arms"].append({"id": "baseline", "config": "c"}), "unique"),
        (lambda d: d["arms"].append({"id": "a b", "config": "c"}), "may contain only"),
        (lambda d: d["arms"].append({"id": "x", "config": "c", "overrides": {"k": [1]}}), "scalars"),
        (lambda d: d["comparisons"].append({"treatment": "method", "control": "nope"}), "unknown arm"),
        (lambda d: d["comparisons"].append({"treatment": "method", "control": "method"}), "must differ"),
        (lambda d: d["comparisons"].append({"treatment": "method", "control": "baseline", "test": "t"}), "expected one of"),
        (lambda d: d.update(exploratory={"allowed": "yes"}), "true or false"),
        (lambda d: d.update(smoke={"timeout": 5}), "unknown key"),
    ],
)
def test_rejects_bad_spec(mutate, message):
    d = base()
    mutate(d)
    with pytest.raises(SpecError, match=message):
        parse_spec(d, PATH)


def test_format_value():
    assert [format_value(v) for v in (True, False, None, 3, 0.5, "x")] == ["true", "false", "null", "3", "0.5", "x"]
