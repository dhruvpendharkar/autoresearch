"""Where a research project keeps its studies and runs.

    <project>/studies/<study>/  brief.md spec.yaml approval.json ledger.jsonl ...
    <project>/runs/<study>/<run_id>/  code/ (git worktree) run.sh log.txt exit_code ...

GPU reservations must be global across projects that share the cluster, so
every ledger rr writes to is registered in `state/ledgers.json`, and
reservations are computed over all registered ledgers.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config
from .ledger import Ledger
from .statefile import locked_json, read_json


class ProjectError(Exception):
    pass


@dataclass(frozen=True)
class Project:
    root: Path

    @property
    def studies_dir(self) -> Path:
        return self.root / "studies"

    def study_dir(self, study: str) -> Path:
        return self.studies_dir / study

    def study_ids(self) -> list:
        if not self.studies_dir.is_dir():
            return []
        return sorted(p.name for p in self.studies_dir.iterdir() if (p / "spec.yaml").is_file())

    def ledger(self, study: str) -> Ledger:
        return Ledger(self.study_dir(study) / "ledger.jsonl")

    def run_dir(self, study: str, run_id: str) -> Path:
        return self.root / "runs" / study / run_id


def find_project(explicit: Optional[Path] = None) -> Project:
    """Explicit path, then $RR_PROJECT, then the git toplevel of cwd, then cwd."""
    if explicit is not None:
        return Project(Path(explicit).resolve())
    env = os.environ.get("RR_PROJECT")
    if env:
        return Project(Path(env).resolve())
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if top:
            return Project(Path(top).resolve())
    except (OSError, subprocess.CalledProcessError):
        pass
    return Project(Path.cwd().resolve())


def _registry_path(config: Config) -> Path:
    return config.state_dir / "ledgers.json"


def register_ledger(config: Config, ledger: Ledger) -> None:
    path = str(ledger.path.resolve())
    with locked_json(_registry_path(config), []) as paths:
        if path not in paths:
            paths.append(path)


def all_ledgers(config: Config, project: Optional[Project] = None) -> list:
    """Every registered ledger that still exists, plus the given project's ledgers."""
    paths = {Path(p) for p in read_json(_registry_path(config), []) or []}
    if project is not None:
        paths |= {p.resolve() for p in project.studies_dir.glob("*/ledger.jsonl")}
    return [Ledger(p) for p in sorted(paths) if p.is_file()]
