"""Append-only run ledger: `studies/<id>/ledger.jsonl`, one line per change.

Each line is a partial run record. A line carrying `state` is a transition and
is checked against the state machine; a line without it only adds fields
(wandb_run, metric, notes, ...). A run's current record is all its lines
merged in order. Lines are never rewritten.
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

STATES = ("queued", "launching", "running", "done", "crash", "invalid", "cancelled")
TERMINAL = frozenset({"done", "crash", "invalid", "cancelled"})
# States that hold GPUs: `launching` is the reservation written before the
# tmux session starts, so a crash mid-launch still shows what we held.
HOLDS_GPUS = frozenset({"launching", "running"})

TRANSITIONS = {
    None: {"queued"},
    # queued -> crash: the run could not be prepared (e.g. its commit is gone).
    "queued": {"launching", "crash", "cancelled"},
    # launching -> queued: the pre-launch re-probe found the GPUs taken.
    "launching": {"running", "queued", "crash", "cancelled"},
    "running": {"done", "crash", "invalid", "cancelled"},
    # A finished run can still be invalidated, e.g. by the eval-data hash check.
    "done": {"invalid"},
    "crash": set(),
    "invalid": set(),
    "cancelled": set(),
}

KINDS = ("smoke", "full", "confirm", "exploratory")
RESULTS = ("keep", "promising", "discard", "inconclusive")

IMMUTABLE_FIELDS = ("arm", "seed", "kind", "attempt")


class LedgerError(Exception):
    pass


def make_run_id(arm: str, seed: int, attempt: int, kind: str = "full") -> str:
    """`method-s1-a03` for full runs; other kinds are named, e.g. `method-s1-smoke-a01`."""
    if kind not in KINDS:
        raise LedgerError(f"unknown kind {kind!r}")
    middle = "" if kind == "full" else f"-{kind}"
    return f"{arm}-s{seed}{middle}-a{attempt:02d}"


class Ledger:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.clock = clock

    # ------------------------------------------------------------ reading

    def _text(self) -> str:
        try:
            return self.path.read_text()
        except FileNotFoundError:
            return ""

    def events(self) -> list:
        return _parse_lines(self._text())[0]

    def bad_lines(self) -> list:
        """Line numbers that aren't valid JSON, e.g. a write torn by a crash."""
        return _parse_lines(self._text())[1]

    def runs(self) -> dict:
        """run_id -> current merged record, in order of first appearance."""
        return fold(self.events())

    def get(self, run_id: str) -> Optional[dict]:
        return self.runs().get(run_id)

    # ------------------------------------------------------------ writing

    def append(self, run_id: str, **fields) -> dict:
        """Append one change for `run_id` and return the run's new merged record."""
        if "run_id" in fields or "ts" in fields:
            raise LedgerError("run_id and ts are set by the ledger")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                text = fh.read()
                current = fold(_parse_lines(text)[0]).get(run_id)
                _validate(run_id, current, fields)
                event = {"ts": self.clock(), "run_id": run_id, **fields}
                fh.seek(0, 2)
                # Don't glue our line onto a torn one left by a crash mid-write.
                prefix = "\n" if text and not text.endswith("\n") else ""
                fh.write(prefix + json.dumps(event, sort_keys=True) + "\n")
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        merged = dict(current or {})
        merged.update(event)
        merged.setdefault("created_at", event["ts"])
        merged["updated_at"] = event["ts"]
        return merged

    def create(self, run_id: str, arm: str, seed: int, kind: str, attempt: int, **fields) -> dict:
        return self.append(run_id, state="queued", arm=arm, seed=seed, kind=kind, attempt=attempt, **fields)

    def transition(self, run_id: str, state: str, **fields) -> dict:
        return self.append(run_id, state=state, **fields)


def _parse_lines(text: str) -> tuple:
    """Return (events, bad_line_numbers). Unparseable lines are skipped, not fatal."""
    events, bad = [], []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(n)
    return events, bad


def fold(events: Iterable[dict]) -> dict:
    runs = {}
    for e in events:
        rid = e.get("run_id")
        if rid is None:
            continue
        rec = runs.setdefault(rid, {"created_at": e.get("ts")})
        rec.update(e)
        rec["updated_at"] = e.get("ts")
    return runs


def _validate(run_id: str, current: Optional[dict], fields: dict) -> None:
    cur_state = current.get("state") if current else None
    if "state" in fields:
        new = fields["state"]
        if new not in STATES:
            raise LedgerError(f"{run_id}: unknown state {new!r}")
        if new not in TRANSITIONS[cur_state]:
            raise LedgerError(f"{run_id}: illegal transition {cur_state} -> {new}")
    elif current is None:
        raise LedgerError(f"{run_id}: unknown run; create it first")

    if current is None:
        missing = [f for f in IMMUTABLE_FIELDS if f not in fields]
        if missing:
            raise LedgerError(f"{run_id}: new run needs {missing}")
    else:
        for f in IMMUTABLE_FIELDS:
            if f in fields and fields[f] != current.get(f):
                raise LedgerError(f"{run_id}: {f} cannot change ({current.get(f)!r} -> {fields[f]!r})")

    if "kind" in fields and fields["kind"] not in KINDS:
        raise LedgerError(f"{run_id}: unknown kind {fields['kind']!r}")
    if fields.get("result") is not None and fields["result"] not in RESULTS:
        raise LedgerError(f"{run_id}: unknown result {fields['result']!r}")
    if fields.get("result") is not None and (fields.get("kind") or (current or {}).get("kind")) == "smoke":
        raise LedgerError(f"{run_id}: smoke runs never count as results")

    if fields.get("state") in HOLDS_GPUS:
        node = fields.get("node", (current or {}).get("node"))
        gpus = fields.get("gpus", (current or {}).get("gpus"))
        if not node or not gpus:
            raise LedgerError(f"{run_id}: state {fields['state']} requires node and gpus")


def project_ledgers(project_root: Path) -> list:
    return [Ledger(p) for p in sorted(Path(project_root).glob("studies/*/ledger.jsonl"))]


def reserved_gpus(ledgers: Iterable[Ledger]) -> set:
    """(node, gpu_index) pairs held by launching or running runs across ledgers."""
    held = set()
    for ledger in ledgers:
        for rec in ledger.runs().values():
            if rec.get("state") in HOLDS_GPUS:
                for idx in rec.get("gpus") or []:
                    held.add((rec["node"], int(idx)))
    return held
