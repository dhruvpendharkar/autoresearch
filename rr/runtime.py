"""Helpers for training scripts launched by rr. Standard library only.

run.sh exports RR_RUN_DIR, RR_HEARTBEAT and RR_FINAL_METRICS. Outside rr
(e.g. a local debug run) these helpers do nothing.

    from rr import runtime
    for step in range(steps):
        ...
        if step % 50 == 0:
            runtime.heartbeat(step, loss=loss.item())
    runtime.final_metrics({"eval/pass@1": score})

If you'd rather not depend on rr, write the same files yourself: both are
plain JSON objects, written atomically (write a temp file, then rename).
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Optional


def _write_atomic(path: str, data: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def _clean(v: Any) -> Any:
    # NaN/inf are not valid JSON; record them as strings so health checks can see them.
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


def heartbeat(step: int, **values: Any) -> None:
    """Record progress. rr treats a heartbeat older than the stall limit as a stalled run."""
    path = os.environ.get("RR_HEARTBEAT")
    if path:
        _write_atomic(path, {"step": int(step), "ts": time.time(), **{k: _clean(v) for k, v in values.items()}})


def final_metrics(metrics: dict) -> None:
    """Write the run's final metrics, read by rr's frozen parser when the run exits 0."""
    path = os.environ.get("RR_FINAL_METRICS")
    if path:
        _write_atomic(path, {k: _clean(v) for k, v in metrics.items()})


def wandb_run(path: Optional[str]) -> None:
    """Record the WandB run path (entity/project/id) so the ledger can link to it."""
    run_dir = os.environ.get("RR_RUN_DIR")
    if run_dir and path:
        with open(os.path.join(run_dir, "wandb_run"), "w") as fh:
            fh.write(path + "\n")
