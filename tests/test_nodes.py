import subprocess

import pytest

from conftest import gpu
from rr.nodes import ExecResult, NodeDown, NodeError, NodeRunner, build_argv, unwrap_output, wrap_command


def test_build_argv_passes_command_as_one_argument():
    assert build_argv("ssh -p 2222 gpu01", "nvidia-smi; echo hi") == ["ssh", "-p", "2222", "gpu01", "nvidia-smi; echo hi"]


@pytest.mark.parametrize("output", ["", "abc", "abc\n", "line1\nline2\n\n"])
def test_wrap_roundtrip_preserves_output_exactly(output):
    raw =subprocess.run(["bash", "-c", wrap_command(f"printf '%s' '{output}'")], capture_output=True, text=True)
    assert unwrap_output(raw.stdout) == (0, output)


def test_unwrap_strips_login_banner():
    raw = "Welcome to gpu01!\nLast login: yesterday\n__RR_BEGIN__\nhello\n\n__RR_END__ 3\n"
    assert unwrap_output(raw) == (3, "hello\n")


def test_unwrap_missing_markers():
    assert unwrap_output("ssh: connect to host gpu01 port 22: Connection refused\n") is None
    assert unwrap_output("__RR_BEGIN__\npartial output") is None


def test_command_failure_is_not_a_node_failure(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    runner = NodeRunner(cluster.write_config(), clock=clock)
    res = runner.run("gpu01", "echo out; echo err >&2; exit 7")
    assert (res.returncode, res.stdout) == (7, "out\n")
    assert "err" in res.stderr
    assert runner.health()["gpu01"]["failures"] == 0


def test_node_marked_down_after_repeated_failures_then_recovers(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    cluster.execs["gpu01"] = "false"  # the node command itself fails
    cfg = cluster.write_config(policy={"node_max_failures": 3, "node_down_backoff_s": 600})
    runner = NodeRunner(cfg, clock=clock)
    for _ in range(3):
        with pytest.raises(NodeError) as e:
            runner.run("gpu01", "true")
        assert not isinstance(e.value, NodeDown)
    assert runner.is_down("gpu01")
    with pytest.raises(NodeDown):
        runner.run("gpu01", "true")

    # After the backoff the node is tried again; one success clears it.
    clock.advance(601)
    calls = []

    def ok_executor(argv, timeout):
        calls.append(argv)
        return ExecResult(0, "__RR_BEGIN__\nfine\n__RR_END__ 0\n", "")

    runner.executor = ok_executor
    assert runner.run("gpu01", "true").stdout == "fine"
    assert len(calls) == 1
    assert not runner.is_down("gpu01")
    assert runner.health()["gpu01"]["failures"] == 0


def test_timeout_counts_as_failure(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    runner = NodeRunner(cluster.write_config(), clock=clock)

    def hang(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    runner.executor = hang
    with pytest.raises(NodeError, match="timed out"):
        runner.run("gpu01", "true", timeout=1)
    assert runner.health()["gpu01"]["failures"] == 1


def test_real_timeout(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    runner = NodeRunner(cluster.write_config(), clock=clock)
    with pytest.raises(NodeError, match="timed out"):
        runner.run("gpu01", "sleep 5", timeout=0.5)


def test_exit_inside_command_still_reports_exit_code(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    runner = NodeRunner(cluster.write_config(), clock=clock)
    assert runner.run("gpu01", "echo before; exit 3; echo after").returncode == 3
    assert runner.health()["gpu01"]["failures"] == 0
