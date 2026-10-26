import json
import os

import pytest

from rr import runtime
from rr.metrics import MetricError, hash_paths, read_metric
from rr.spec import Metric

M = Metric("eval/pass@1", "max", "final_metrics.json")


def write(d, data):
    (d / "final_metrics.json").write_text(json.dumps(data) if not isinstance(data, str) else data)


def test_flat_and_nested_keys(tmp_path):
    write(tmp_path, {"eval/pass@1": 0.41})
    assert read_metric(tmp_path, M) == 0.41
    write(tmp_path, {"eval": {"pass@1": 0.5}})
    assert read_metric(tmp_path, M) == 0.5


@pytest.mark.parametrize(
    "content,message",
    [
        (None, "not written"),
        ("{nope", "not valid JSON"),
        ([1, 2], "JSON object"),
        ({"other": 1}, "has no"),
        ({"eval/pass@1": "0.4"}, "not a number"),
        ({"eval/pass@1": True}, "not a number"),
        ('{"eval/pass@1": NaN}', "not finite"),
    ],
)
def test_bad_metrics(tmp_path, content, message):
    if content is not None:
        write(tmp_path, content)
    with pytest.raises(MetricError, match=message):
        read_metric(tmp_path, M)


def test_source_cannot_escape_run_dir(tmp_path):
    (tmp_path / "outside.json").write_text('{"eval/pass@1": 1.0}')
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(MetricError, match="escapes"):
        read_metric(run, Metric("eval/pass@1", "max", "../outside.json"))


def test_hash_paths_detects_content_and_name_changes(tmp_path):
    d = tmp_path / "data"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("a")
    (d / "sub" / "b.txt").write_text("b")
    cache = tmp_path / "cache.json"
    h1 = hash_paths(tmp_path, ["data"], cache)
    assert hash_paths(tmp_path, ["data"], cache) == h1 == hash_paths(tmp_path, ["data"])
    (d / "sub" / "b.txt").write_text("B")
    h2 = hash_paths(tmp_path, ["data"], cache)
    assert h2 != h1
    (d / "sub" / "b.txt").rename(d / "sub" / "c.txt")
    assert hash_paths(tmp_path, ["data"], cache) != h2


def test_hash_paths_cache_keyed_on_size_and_mtime(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"1234")
    cache = tmp_path / "cache.json"
    h1 = hash_paths(tmp_path, ["x.bin"], cache)
    st = f.stat()
    f.write_bytes(b"5678")  # same size
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert hash_paths(tmp_path, ["x.bin"], cache) != h1


def test_runtime_helpers(tmp_path, monkeypatch):
    runtime.heartbeat(1, loss=0.5)  # no env: no-op
    monkeypatch.setenv("RR_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RR_HEARTBEAT", str(tmp_path / "heartbeat.json"))
    monkeypatch.setenv("RR_FINAL_METRICS", str(tmp_path / "final_metrics.json"))
    runtime.heartbeat(7, loss=float("nan"))
    hb = json.loads((tmp_path / "heartbeat.json").read_text())
    assert hb["step"] == 7 and hb["loss"] == "nan"
    runtime.final_metrics({"eval/pass@1": 0.3})
    assert read_metric(tmp_path, M) == 0.3
    runtime.wandb_run("ent/proj/abc")
    assert (tmp_path / "wandb_run").read_text().strip() == "ent/proj/abc"
