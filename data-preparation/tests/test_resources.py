from pathlib import Path

import pytest

from b_replay.resources import GIB, memory_snapshot, require_memory


def proc_fixture(tmp_path, *, version=2, namespaced=False):
    proc, mount = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    mount.mkdir()
    (proc / "meminfo").write_text(f"MemTotal: {251 * GIB // 1024} kB\nMemAvailable: {200 * GIB // 1024} kB\n")
    if version == 2:
        root, member = ("/docker/container", "/") if namespaced else ("/tenant", "/tenant/job")
        fs, membership = "cgroup2 cgroup rw", f"0::{member}"
    else:
        root, member = "/", "/job"
        fs, membership = "cgroup cgroup rw,memory", f"5:memory:{member}"
    (proc / "self/cgroup").write_text(membership + "\n")
    (proc / "self/mountinfo").write_text(f"1 0 0:1 {root} {mount} rw - {fs}\n")
    leaf = mount if namespaced else mount / "job"
    leaf.mkdir(exist_ok=True)
    return proc, mount, leaf


def v2(node, maximum="max", current=0, inactive=0, high="max"):
    (node / "memory.max").write_text(str(maximum))
    (node / "memory.high").write_text(str(high))
    (node / "memory.current").write_text(str(current))
    (node / "memory.stat").write_text(f"inactive_file {inactive}\nactive_file {10 * GIB}\n")
    (node / "memory.events").write_text("oom 0\noom_kill 0\n")


def test_v2_parent_limit_overrides_251gib_host_and_leaf_unlimited(tmp_path):
    proc, mount, leaf = proc_fixture(tmp_path)
    v2(leaf, current=2 * GIB)
    v2(mount, maximum=30_000_000_000, current=9 * GIB, inactive=2 * GIB)
    observed = memory_snapshot(proc)
    assert observed["host_available_bytes"] == 200 * GIB
    assert observed["effective_limit_bytes"] == 30_000_000_000
    assert observed["available_bytes"] == 30_000_000_000 - 7 * GIB
    assert len(observed["cgroups"]) == 2
    with pytest.raises(RuntimeError, match="container limits"):
        require_memory(observed, 25 * GIB, phase="Startup")


def test_namespaced_root_and_memory_high_are_respected(tmp_path):
    proc, mount, leaf = proc_fixture(tmp_path, namespaced=True)
    v2(leaf, maximum=30 * GIB, high=26 * GIB, current=8 * GIB, inactive=GIB)
    observed = memory_snapshot(proc)
    assert observed["available_bytes"] == 19 * GIB
    assert observed["effective_limit_bytes"] == 26 * GIB
    assert observed["cgroups"][0]["raw_headroom_bytes"] == 18 * GIB


def test_v1_hierarchical_memory_limit_and_inactive_file(tmp_path):
    proc, mount, leaf = proc_fixture(tmp_path, version=1)
    for node, maximum, used in ((leaf, 2 ** 63 - 4096, 2 * GIB), (mount, 30 * GIB, 12 * GIB)):
        (node / "memory.limit_in_bytes").write_text(str(maximum))
        (node / "memory.usage_in_bytes").write_text(str(used))
        (node / "memory.stat").write_text(f"inactive_file 0\ntotal_inactive_file {GIB}\n")
        (node / "memory.failcnt").write_text("0")
    observed = memory_snapshot(proc)
    assert observed["effective_limit_bytes"] == 30 * GIB
    assert observed["available_bytes"] == 19 * GIB


@pytest.mark.parametrize("missing", ["mount", "counter"])
def test_unreadable_cgroup_fails_closed_instead_of_trusting_host(tmp_path, missing):
    proc, mount, leaf = proc_fixture(tmp_path)
    v2(leaf, maximum=30 * GIB)
    if missing == "mount":
        (proc / "self/mountinfo").write_text("")
    else:
        (leaf / "memory.current").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        memory_snapshot(proc)


def test_host_available_still_caps_reclaimable_cgroup_bytes(tmp_path):
    proc, mount, leaf = proc_fixture(tmp_path)
    v2(leaf, maximum=30 * GIB, current=28 * GIB, inactive=8 * GIB)
    (proc / "meminfo").write_text(f"MemTotal: {251 * GIB // 1024} kB\nMemAvailable: {4 * GIB // 1024} kB\n")
    observed = memory_snapshot(proc)
    assert observed["available_bytes"] == 4 * GIB
    assert observed["cgroups"][0]["available_bytes"] == 10 * GIB
