"""Run one non-interactive command on a node as `<exec> '<command>'`.

Every command is wrapped in begin/end markers so we can tell a command that
ran and failed (marker present, non-zero exit code) from a node we could not
reach (marker missing, timeout, or the exec command itself failing). Only the
latter counts toward marking a node down. The markers also strip any login
banner the node command prints before running the command.

Node health lives in `state/nodes.json` so the daemon and the CLI agree on
which nodes are down.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .statefile import locked_json, read_json

BEGIN = "__RR_BEGIN__"
END = "__RR_END__"


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


class NodeError(Exception):
    """The node could not be reached or did not answer in time."""


class NodeDown(NodeError):
    """The node is marked down and is being skipped until its backoff expires."""


# An executor takes argv and a timeout in seconds and returns the raw result.
# It raises subprocess.TimeoutExpired on timeout. Tests swap in fakes.
Executor = Callable[[list, float], ExecResult]


def subprocess_executor(argv: list, timeout: float) -> ExecResult:
    proc = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return ExecResult(proc.returncode, proc.stdout, proc.stderr)


def wrap_command(command: str) -> str:
    # The subshell keeps an `exit` inside the command from skipping the end marker.
    return f"printf '%s\\n' {BEGIN}\n(\n{command}\n)\n__rr_rc=$?\nprintf '\\n{END} %s\\n' \"$__rr_rc\""


def build_argv(exec_str: str, command: str) -> list:
    """`ssh gpu01` + cmd -> ['ssh', 'gpu01', cmd]: the command is one argument."""
    return shlex.split(exec_str) + [command]


def unwrap_output(raw: str) -> Optional[tuple]:
    """Return (returncode, stdout) from wrapped output, or None if markers are missing."""
    begin = raw.find(BEGIN + "\n")
    end = raw.rfind("\n" + END + " ")
    if begin < 0 or end < 0 or end < begin:
        return None
    body = raw[begin + len(BEGIN) + 1 : end]
    rc_line = raw[end + len(END) + 2 :].split("\n", 1)[0].strip()
    try:
        rc = int(rc_line)
    except ValueError:
        return None
    return rc, body


class NodeRunner:
    def __init__(
        self,
        config: Config,
        executor: Executor = subprocess_executor,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self.executor = executor
        self.clock = clock
        self.health_path: Path = config.state_dir / "nodes.json"

    def health(self) -> dict:
        return read_json(self.health_path, {}) or {}

    def is_down(self, node: str) -> bool:
        h = self.health().get(node, {})
        return h.get("down_until") is not None and self.clock() < h["down_until"]

    def run(self, node: str, command: str, timeout: Optional[float] = None) -> ExecResult:
        """Run `command` on `node`. Returns the command's own exit code and output.

        Raises NodeDown if the node is in backoff, NodeError if it can't be reached.
        """
        n = self.config.node(node)
        if self.is_down(node):
            raise NodeDown(f"{node} is marked down; skipping until backoff expires")
        timeout = timeout if timeout is not None else self.config.policy.node_timeout_s
        argv = build_argv(n.exec, wrap_command(command))
        try:
            raw = self.executor(argv, timeout)
        except subprocess.TimeoutExpired:
            self._record_failure(node, f"timed out after {timeout}s")
            raise NodeError(f"{node}: timed out after {timeout}s") from None
        except OSError as e:
            self._record_failure(node, f"exec failed: {e}")
            raise NodeError(f"{node}: could not run node command: {e}") from None
        parsed = unwrap_output(raw.stdout)
        if parsed is None:
            detail = (raw.stderr.strip() or raw.stdout.strip())[-500:]
            msg = f"node command exited {raw.returncode} without running the command"
            self._record_failure(node, f"{msg}: {detail}")
            raise NodeError(f"{node}: {msg}: {detail}")
        rc, stdout = parsed
        self._record_success(node)
        return ExecResult(rc, stdout, raw.stderr)

    def _record_failure(self, node: str, error: str) -> None:
        policy = self.config.policy
        now = self.clock()
        with locked_json(self.health_path, {}) as health:
            h = health.setdefault(node, {"failures": 0})
            h["failures"] = h.get("failures", 0) + 1
            h["last_error"] = error
            h["last_error_at"] = now
            if h["failures"] >= policy.node_max_failures:
                h["down_until"] = now + policy.node_down_backoff_s

    def _record_success(self, node: str) -> None:
        now = self.clock()
        with locked_json(self.health_path, {}) as health:
            h = health.setdefault(node, {})
            h.update(failures=0, down_until=None, last_ok_at=now)
