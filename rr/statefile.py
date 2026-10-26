"""Small helpers for state files shared by the daemon and the CLI.

The daemon and a human running `rr gpus` can touch the same files at the same
time, so every read-modify-write takes an exclusive lock on a sidecar file and
every write is atomic (write to a temp file, then rename).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on `<path>.lock` for the duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def locked_json(path: Path, default: Any = None) -> Iterator[Any]:
    """Read a JSON file under lock, yield it for mutation, then write it back."""
    with file_lock(path):
        data = read_json(path, default)
        yield data
        write_json_atomic(path, data)
