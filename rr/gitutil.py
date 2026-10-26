"""Thin git wrappers. Kept to commands that work on old git (2.15 era)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


class GitError(Exception):
    pass


def git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    except OSError as e:
        raise GitError(f"could not run git: {e}") from None
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def resolve_commit(root: Path, rev: str) -> str:
    try:
        return git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").strip()
    except GitError:
        raise GitError(f"{rev!r} is not a commit in {root}") from None


def changed_files(root: Path, a: str, b: str, paths: Iterable[str]) -> list:
    paths = list(paths)
    if not paths:
        return []
    out = git(root, "diff", "--name-only", a, b, "--", *paths)
    return [line for line in out.splitlines() if line]


def worktree_add(root: Path, path: Path, sha: str) -> None:
    git(root, "worktree", "add", "--detach", str(path), sha)


def head_of(path: Path) -> str:
    return git(path, "rev-parse", "HEAD").strip()
