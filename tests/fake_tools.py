"""Stand-ins for nvidia-smi, gpustat, ps and whoami on a fake node.

Invoked as `python fake_tools.py <tool> <args...>` by tiny wrappers that
FakeCluster puts on PATH. Each fake node is described by
$FAKE_CLUSTER/$RR_FAKE_NODE.json:

    {"me": "alice", "gpustat": false, "hide_owners": false, "nvsmi_fail": false,
     "gpus":  [{"index": 0, "uuid": "GPU-a0", "name": "Quadro RTX 6000",
                "mem_used": 3, "mem_total": 24576, "util": 0}],
     "procs": [{"gpu_uuid": "GPU-a0", "pid": 4242, "mem": 9000, "user": "bob"}]}
"""

import json
import os
import sys


def load():
    path = os.path.join(os.environ["FAKE_CLUSTER"], os.environ["RR_FAKE_NODE"] + ".json")
    with open(path) as fh:
        return json.load(fh)


def nvidia_smi(node, args):
    if node.get("nvsmi_fail"):
        print("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", file=sys.stderr)
        return 9
    query = next((a for a in args if a.startswith("--query-")), "")
    if query.startswith("--query-gpu="):
        for g in node["gpus"]:
            print(f"{g['index']}, {g['uuid']}, {g['name']}, {g['mem_used']}, {g['mem_total']}, {g['util']}")
    elif query == "--query-compute-apps=pid":
        for p in node.get("procs", []):
            print(p["pid"])
    elif query.startswith("--query-compute-apps="):
        for p in node.get("procs", []):
            print(f"{p['gpu_uuid']}, {p['pid']}, {p['mem']}")
    else:
        print(f"fake nvidia-smi: unsupported args {args}", file=sys.stderr)
        return 2
    return 0


def gpustat(node, args):
    if not node.get("gpustat"):
        return 1
    gpus = []
    for g in node["gpus"]:
        gpus.append(
            {
                "index": g["index"],
                "uuid": g["uuid"],
                "name": g["name"],
                "memory.used": g["mem_used"],
                "memory.total": g["mem_total"],
                "utilization.gpu": g["util"],
                "processes": [
                    {"pid": p["pid"], "username": p["user"], "gpu_memory_usage": p["mem"], "command": "python"}
                    for p in node.get("procs", [])
                    if p["gpu_uuid"] == g["uuid"]
                ],
            }
        )
    print(json.dumps({"hostname": os.environ["RR_FAKE_NODE"], "gpus": gpus}))
    return 0


def ps(node, args):
    if node.get("hide_owners"):
        return 1
    pids = set(args[args.index("-p") + 1].split(","))
    for p in node.get("procs", []):
        if str(p["pid"]) in pids:
            print(f"{p['pid']:>7} {p['user']}")
    return 0


def whoami(node, args):
    print(node.get("me", "me"))
    return 0


def _sessions_dir():
    d = os.path.join(os.environ["FAKE_CLUSTER"], "tmux-" + os.environ["RR_FAKE_NODE"])
    os.makedirs(d, exist_ok=True)
    return d


def _session_pid(name):
    path = os.path.join(_sessions_dir(), name)
    try:
        pid = int(open(path).read())
    except (FileNotFoundError, ValueError):
        return None, path
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        os.unlink(path)
        return None, path
    return pid, path


def tmux(node, args):
    """Detached sessions as process groups: new-session -d -s, has-session -t, kill-session -t."""
    import signal
    import subprocess

    verb, rest = args[0], args[1:]
    target = rest[rest.index("-t") + 1].lstrip("=") if "-t" in rest else None
    if verb == "new-session":
        name = rest[rest.index("-s") + 1]
        cmd = rest[rest.index("-s") + 2]
        if _session_pid(name)[0] is not None:
            print(f"duplicate session: {name}", file=sys.stderr)
            return 1
        proc = subprocess.Popen(
            ["bash", "-c", cmd], start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with open(os.path.join(_sessions_dir(), name), "w") as fh:
            fh.write(str(proc.pid))
        return 0
    pid, path = _session_pid(target)
    if pid is None:
        print(f"can't find session: {target}", file=sys.stderr)
        return 1
    if verb == "has-session":
        return 0
    if verb == "kill-session":
        os.killpg(pid, signal.SIGHUP)  # what closing the pane's pty does
        os.unlink(path)
        return 0
    print(f"fake tmux: unsupported {args}", file=sys.stderr)
    return 2


TOOLS = {"nvidia-smi": nvidia_smi, "gpustat": gpustat, "ps": ps, "whoami": whoami, "tmux": tmux}

if __name__ == "__main__":
    tool, rest = sys.argv[1], sys.argv[2:]
    sys.exit(TOOLS[tool](load(), rest))
