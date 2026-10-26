"""The frozen judge: the only code that turns a run's output into a metric.

It lives in rr, outside the research project, so code under test cannot
change how its own results are read. Eval data is hashed at approval and
re-checked when a run is collected; a mismatch marks the run invalid.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Optional

from .spec import Metric
from .statefile import locked_json


class MetricError(Exception):
    pass


def read_metric(run_dir: Path, metric: Metric) -> float:
    """Read `metric.name` from `run_dir/metric.source`. Nested keys may use '/'."""
    run_dir = Path(run_dir).resolve()
    path = (run_dir / metric.source).resolve()
    if run_dir not in path.parents:
        raise MetricError(f"metric source {metric.source!r} escapes the run directory")
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise MetricError(f"{metric.source} not written") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise MetricError(f"{metric.source} is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise MetricError(f"{metric.source}: expected a JSON object")
    if metric.name in data:
        value = data[metric.name]
    else:
        value = data
        for part in metric.name.split("/"):
            if not isinstance(value, dict) or part not in value:
                raise MetricError(f"{metric.source} has no {metric.name!r}")
            value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricError(f"{metric.name} is not a number: {value!r}")
    if not math.isfinite(value):
        raise MetricError(f"{metric.name} is not finite: {value!r}")
    return float(value)


def _files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            yield Path(root) / f


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(base: Path, paths: Iterable[str], cache_path: Optional[Path] = None) -> str:
    """One sha256 over every file under `paths` (relative to `base` unless absolute).

    Per-file digests are cached by (size, mtime) in `cache_path`, so re-checking
    a large, unchanged eval set is cheap.
    """
    base = Path(base)
    entries = []
    for p in paths:
        root = Path(p) if Path(p).is_absolute() else base / p
        if not root.exists():
            raise MetricError(f"eval data path not found: {root}")
        for f in _files(root):
            entries.append((p, f, f.relative_to(root) if f != root else Path(".")))

    def digest_all(cache: dict) -> str:
        h = hashlib.sha256()
        for label, f, rel in entries:
            st = f.stat()
            key = str(f.resolve())
            hit = cache.get(key)
            if hit and hit["size"] == st.st_size and hit["mtime_ns"] == st.st_mtime_ns:
                digest = hit["sha256"]
            else:
                digest = _file_sha256(f)
                cache[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
            h.update(f"{label}\0{rel.as_posix()}\0{digest}\n".encode())
        return h.hexdigest()

    if cache_path is None:
        return digest_all({})
    with locked_json(cache_path, {}) as cache:
        return digest_all(cache)
