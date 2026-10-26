from datetime import datetime

import pytest

from conftest import gpu
from rr import gpus as g
from rr.nodes import NodeRunner

POLICY = {"idle_probes": 3, "probe_interval_s": 60, "max_gpus_total": 8, "max_gpus_per_node": 4, "cooldown_min": 60}
NOON = datetime(2026, 10, 25, 12, 0)


def probe_n(store, runner, clock, n, interval=60):
    state = None
    for _ in range(n):
        state = store.refresh(runner)
        clock.advance(interval)
    clock.advance(-interval)  # leave the clock at the last probe
    return state


def statuses(state, cfg, clock, reserved=frozenset()):
    return {(s.node, s.index): s.status for s in g.classify(state, cfg, set(reserved), clock())}


# ---------------------------------------------------------------- parsing


def test_parse_nvidia_smi_with_owners():
    gpu_csv = "0, GPU-a, Quadro RTX 6000, 3, 24576, 0\n1, GPU-b, Quadro RTX 6000, 9000, 24576, 87\n2, GPU-c, Quadro RTX 6000, [N/A], 24576, [N/A]"
    apps = "GPU-b, 4242, 8990\nGPU-b, 4243, 10"
    users = "   4242 bob\n   4243 me"
    out = g.parse_nvidia_smi(gpu_csv, apps, users)
    assert [x.index for x in out] == [0, 1, 2]
    assert out[1].util_pct == 87
    assert [(p.pid, p.user) for p in out[1].procs] == [(4242, "bob"), (4243, "me")]
    assert out[2].mem_used_mb is None and out[2].util_pct is None


def test_parse_probe_requires_user():
    with pytest.raises(g.ProbeError, match="current user"):
        g.parse_probe("@@NVSMI\n0, GPU-a, X, 0, 1, 0\n")


def test_foreign_and_unknown_owners():
    s = g.GpuSample(0, "u", "n", 0, 1, 0, [g.GpuProc(1, "me", 5), g.GpuProc(2, None, 5), g.GpuProc(3, "bob", 5)])
    assert g.foreign_users(s, "me") == ["?", "bob"]


@pytest.mark.parametrize(
    "mem,util,procs,idle",
    [
        (3, 0, [], True),
        (999, 4, [], True),
        (1000, 0, [], False),  # threshold is strict
        (3, 5, [], False),
        (None, 0, [], False),  # unknown counts as busy
        (3, None, [], False),
        (3, 0, [g.GpuProc(1, "bob", 1)], False),  # tiny foreign process still blocks
        (3, 0, [g.GpuProc(1, None, 1)], False),
    ],
)
def test_is_idle_now(mem, util, procs, idle):
    from rr.config import Policy

    assert g.is_idle_now(g.GpuSample(0, "u", "n", mem, 24576, util, procs), "me", Policy()) is idle


# ---------------------------------------------------------------- end-to-end probe via fake cluster


@pytest.mark.parametrize("use_gpustat", [False, True])
def test_probe_both_backends(cluster, clock, use_gpustat):
    cluster.set_node(
        "gpu01",
        [gpu(0), gpu(1, mem_used=9000, util=90)],
        procs=[{"gpu_uuid": "GPU-0001", "pid": 4242, "mem": 9000, "user": "bob"}],
        gpustat=use_gpustat,
    )
    runner = NodeRunner(cluster.write_config(), clock=clock)
    res = g.probe_node(runner, "gpu01")
    assert res.source == ("gpustat" if use_gpustat else "nvidia-smi")
    assert res.user == "me"
    assert [x.index for x in res.gpus] == [0, 1]
    assert [(p.pid, p.user) for p in res.gpus[1].procs] == [(4242, "bob")]


def test_hidden_owners_count_as_foreign(cluster, clock):
    cluster.set_node(
        "gpu01", [gpu(0)], procs=[{"gpu_uuid": "GPU-0000", "pid": 77, "mem": 50, "user": "bob"}], hide_owners=True
    )
    runner = NodeRunner(cluster.write_config(), clock=clock)
    res = g.probe_node(runner, "gpu01")
    assert g.foreign_users(res.gpus[0], res.user) == ["?"]


def test_nvidia_smi_failure_is_probe_error_not_node_failure(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)], nvsmi_fail=True)
    runner = NodeRunner(cluster.write_config(), clock=clock)
    with pytest.raises(g.ProbeError, match="exited 3"):
        g.probe_node(runner, "gpu01")
    assert runner.health()["gpu01"]["failures"] == 0


# ---------------------------------------------------------------- hysteresis


def test_gpu_free_only_after_idle_probes_spanning_intervals(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    cfg = cluster.write_config(policy=POLICY)
    runner, store = NodeRunner(cfg, clock=clock), g.GpuState(cfg, clock=clock)

    state = probe_n(store, runner, clock, 2)
    assert statuses(state, cfg, clock)[("gpu01", 0)] == "settling"
    clock.advance(60)
    state = store.refresh(runner)  # third idle probe, 120 s after the first
    assert statuses(state, cfg, clock)[("gpu01", 0)] == "free"


def test_rapid_probes_cannot_shortcut_hysteresis(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    cfg = cluster.write_config(policy=POLICY)
    runner, store = NodeRunner(cfg, clock=clock), g.GpuState(cfg, clock=clock)
    state = probe_n(store, runner, clock, 5, interval=1)
    assert state["nodes"]["gpu01"]["gpus"][0]["idle_streak"] == 5
    assert statuses(state, cfg, clock)[("gpu01", 0)] == "settling"


def test_busy_probe_resets_streak(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    cfg = cluster.write_config(policy=POLICY)
    runner, store = NodeRunner(cfg, clock=clock), g.GpuState(cfg, clock=clock)
    probe_n(store, runner, clock, 3)
    cluster.set_node("gpu01", [gpu(0, mem_used=5000, util=60)])  # someone's dataloader warms up
    clock.advance(60)
    state = store.refresh(runner)
    assert statuses(state, cfg, clock)[("gpu01", 0)] == "busy"
    cluster.set_node("gpu01", [gpu(0)])
    clock.advance(60)
    state = store.refresh(runner)
    assert state["nodes"]["gpu01"]["gpus"][0]["idle_streak"] == 1
    assert statuses(state, cfg, clock)[("gpu01", 0)] == "settling"


def test_stale_gap_restarts_streak(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    cfg = cluster.write_config(policy=POLICY)
    runner, store = NodeRunner(cfg, clock=clock), g.GpuState(cfg, clock=clock)
    probe_n(store, runner, clock, 3)
    clock.advance(3600)  # daemon was down for an hour
    state = store.refresh(runner)
    assert state["nodes"]["gpu01"]["gpus"][0]["idle_streak"] == 1


def test_failed_probe_resets_streaks_and_marks_node(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    cfg = cluster.write_config(policy=POLICY)
    runner, store = NodeRunner(cfg, clock=clock), g.GpuState(cfg, clock=clock)
    probe_n(store, runner, clock, 3)
    cluster.set_node("gpu01", [gpu(0)], nvsmi_fail=True)
    clock.advance(60)
    state = store.refresh(runner)
    ns = state["nodes"]["gpu01"]
    assert ns["ok"] is False and "exited 3" in ns["error"]
    assert ns["gpus"][0]["idle_streak"] == 0
    assert statuses(state, cfg, clock)[("gpu01", 0)] == "node-down"


def test_one_unreachable_node_does_not_block_others(cluster, clock):
    cluster.set_node("gpu01", [gpu(0)])
    cluster.set_node("gpu02", [gpu(0)])
    cluster.execs["gpu02"] = "false"
    cfg = cluster.write_config(policy=POLICY)
    runner, store = NodeRunner(cfg, clock=clock), g.GpuState(cfg, clock=clock)
    state = probe_n(store, runner, clock, 3)
    assert state["nodes"]["gpu01"]["ok"] and not state["nodes"]["gpu02"]["ok"]
    assert statuses(state, cfg, clock)[("gpu01", 0)] == "free"


# ---------------------------------------------------------------- statuses, cooldowns, allocation


def settled_cluster(cluster, clock, nodes, policy=POLICY):
    for name, gs in nodes.items():
        cluster.set_node(name, gs[0], procs=gs[1] if len(gs) > 1 else ())
    cfg = cluster.write_config(policy=policy)
    runner, store = NodeRunner(cfg, clock=clock), g.GpuState(cfg, clock=clock)
    state = probe_n(store, runner, clock, 3)
    return cfg, store, state


def test_classify_labels(cluster, clock):
    cfg, store, state = settled_cluster(
        cluster,
        clock,
        {
            "gpu01": (
                [gpu(0), gpu(1), gpu(2, mem_used=8000, util=50), gpu(3, mem_used=8000, util=50), gpu(4)],
                [
                    {"gpu_uuid": "GPU-0002", "pid": 10, "mem": 8000, "user": "bob"},
                    {"gpu_uuid": "GPU-0003", "pid": 11, "mem": 8000, "user": "me"},
                ],
            )
        },
    )
    store.add_cooldown("gpu01", 4)
    state = store.load()
    labels = statuses(state, cfg, clock, reserved={("gpu01", 1)})
    assert labels == {
        ("gpu01", 0): "free",
        ("gpu01", 1): "reserved",
        ("gpu01", 2): "foreign",
        ("gpu01", 3): "busy",
        ("gpu01", 4): "cooldown",
    }


def test_collision_label(cluster, clock):
    cfg, _, state = settled_cluster(
        cluster, clock, {"gpu01": ([gpu(0, mem_used=9000, util=99)], [{"gpu_uuid": "GPU-0000", "pid": 1, "mem": 9, "user": "bob"}])}
    )
    assert statuses(state, cfg, clock, reserved={("gpu01", 0)})[("gpu01", 0)] == "collision"


def test_cooldown_expires_on_refresh(cluster, clock):
    cfg, store, _ = settled_cluster(cluster, clock, {"gpu01": ([gpu(0)],)})
    store.add_cooldown("gpu01", 0, minutes=1)
    clock.advance(61)
    state = store.refresh(NodeRunner(cfg, clock=clock))
    assert state["cooldowns"] == {}
    assert statuses(state, cfg, clock)[("gpu01", 0)] == "free"


def test_allocate_prefers_least_contended_node_and_low_indices(cluster, clock):
    cfg, _, state = settled_cluster(
        cluster,
        clock,
        {
            "gpu01": ([gpu(0), gpu(1), gpu(2, mem_used=9000, util=99)],),
            "gpu02": ([gpu(0), gpu(1), gpu(2), gpu(3)],),
        },
    )
    a = g.allocate(state, cfg, 2, set(), clock(), wall_clock=NOON)
    assert (a.node, a.gpus) == ("gpu02", [0, 1])


def test_allocate_never_spans_nodes(cluster, clock):
    cfg, _, state = settled_cluster(cluster, clock, {"gpu01": ([gpu(0), gpu(1)],), "gpu02": ([gpu(0), gpu(1)],)})
    assert g.allocate(state, cfg, 3, set(), clock(), wall_clock=NOON) is None


def test_allocate_respects_per_node_cap(cluster, clock):
    cfg, _, state = settled_cluster(cluster, clock, {"gpu01": ([gpu(i) for i in range(8)],)})
    reserved = {("gpu01", 6), ("gpu01", 7)}
    assert g.allocate(state, cfg, 2, reserved, clock(), wall_clock=NOON).gpus == [0, 1]
    assert g.allocate(state, cfg, 3, reserved, clock(), wall_clock=NOON) is None


def test_allocate_respects_total_cap_and_night_cap(cluster, clock):
    policy = dict(POLICY, max_gpus_total=4, night={"hours": "23:00-07:00", "max_gpus_total": 6})
    cfg, _, state = settled_cluster(cluster, clock, {"gpu01": ([gpu(i) for i in range(4)],), "gpu02": ([gpu(i) for i in range(4)],)}, policy)
    reserved = {("gpu01", 0), ("gpu01", 1), ("gpu01", 2)}
    assert g.allocate(state, cfg, 2, reserved, clock(), wall_clock=NOON) is None
    assert g.allocate(state, cfg, 1, reserved, clock(), wall_clock=NOON).node == "gpu02"
    night = g.allocate(state, cfg, 2, reserved, clock(), wall_clock=datetime(2026, 10, 25, 23, 30))
    assert night.node == "gpu02" and night.gpus == [0, 1]


def test_allocate_filters_gpu_type(cluster, clock):
    cfg, _, state = settled_cluster(cluster, clock, {"gpu01": ([gpu(0)],)})
    assert g.allocate(state, cfg, 1, set(), clock(), gpu_type="H100", wall_clock=NOON) is None
    assert g.allocate(state, cfg, 1, set(), clock(), gpu_type="RTX6000", wall_clock=NOON).node == "gpu01"


def test_still_idle_recheck(cluster, clock):
    cfg, _, _ = settled_cluster(cluster, clock, {"gpu01": ([gpu(0), gpu(1)],)})
    runner = NodeRunner(cfg, clock=clock)
    assert g.still_idle(g.probe_node(runner, "gpu01"), [0, 1], cfg.policy)
    cluster.set_node("gpu01", [gpu(0), gpu(1)], procs=[{"gpu_uuid": "GPU-0001", "pid": 5, "mem": 300, "user": "bob"}])
    assert not g.still_idle(g.probe_node(runner, "gpu01"), [0, 1], cfg.policy)
    assert not g.still_idle(g.probe_node(runner, "gpu01"), [7], cfg.policy)
