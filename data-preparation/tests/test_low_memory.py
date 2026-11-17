import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from b_replay import low_memory, server
from b_replay.common import read_json
from b_replay.resources import GIB
from tests.test_server import args_for


@pytest.mark.parametrize("method", ["kairos", "continuum"])
def test_ram30_plan_consistent_budgets_and_unchanged_data(tmp_path, method):
    args = args_for(tmp_path)
    args.server_profile, args.method = "ram30", method
    plan = server.make_plan(args)
    engine, helper, env = plan["engine_command"], plan["helper_command"], plan["environment"]
    assert engine[engine.index("--max-num-seqs") + 1] == "8"
    assert engine[engine.index("--safetensors-load-strategy") + 1] == "lazy"
    assert "--enforce-eager" not in engine and "--compilation-config" in engine
    assert engine[engine.index("--max-model-len") + 1] == "16384"
    assert helper[helper.index("--l1-size-gb") + 1] == "8.0"
    assert helper[helper.index("--l1-init-size-gb") + 1] == "1"
    assert env["LMCACHE_MAX_LOCAL_CPU_SIZE"] == env["CONTINUUM_LMCACHE_CPU_GB"] == "8.0"
    assert env["MALLOC_ARENA_MAX"] == "2"
    assert plan["server_profile"] == "ram30" and plan["calls"] == 10
    assert not plan["resource_profile"]["performance_comparable_to_archive"]
    assert not args.output.exists()
    # A later frozen plan is unaffected by the resource adaptation.
    args.server_profile = "frozen"
    frozen = server.make_plan(args)
    assert frozen["environment"]["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "8.0"
    assert frozen["resource_profile"]["max_num_seqs"] == 8


@pytest.mark.parametrize("available,foreign", [(GIB // 2, set()), (25 * GIB, {999})])
def test_ram30_guard_rejects_pressure_and_foreign_gpu_jobs(monkeypatch, available, foreign):
    harness = SimpleNamespace(smi=lambda: [dict(index="0", used="100", total="24564")],
        compute_apps=lambda uuid: foreign,
        gpu_process_scope=lambda apps, roots: (set(), apps))
    monkeypatch.setattr(low_memory, "memory_snapshot", lambda: dict(available_bytes=available))
    with pytest.raises(RuntimeError, match="ram30 resource guard"):
        low_memory.guard(SimpleNamespace(harness=harness), SimpleNamespace(gpu=0), [], "test")


def test_watchdog_stops_only_owned_groups_on_memory_failure(monkeypatch):
    stopped, monitoring = [], []
    processes = [SimpleNamespace(pid=101), SimpleNamespace(pid=202)]
    upstream = SimpleNamespace(harness=SimpleNamespace(stop=lambda process: stopped.append(process.pid)))
    def fail(*args):
        raise RuntimeError("container RAM exhausted")
    monkeypatch.setattr(low_memory, "guard", fail)
    watch = low_memory.ResourceWatch(upstream, None, processes, "test", monitoring)
    watch._run()
    assert stopped == [202, 101] and "container RAM" in monitoring[0]["guard_error"]
    with pytest.raises(RuntimeError, match="container RAM"):
        watch.check()


@pytest.mark.parametrize("available", [25 * GIB, GIB // 2])
def test_record_only_pid_policy_keeps_memory_guard(monkeypatch, available):
    harness = SimpleNamespace(smi=lambda: [dict(index="0", used="350", total="24564")],
        compute_apps=lambda uuid: {845943},
        gpu_process_scope=lambda apps, roots: ({604}, apps))
    monkeypatch.setattr(low_memory, "memory_snapshot", lambda: dict(available_bytes=available))
    args = SimpleNamespace(gpu=0, gpu_pid_policy="record-only")
    if available < GIB:
        with pytest.raises(RuntimeError, match="ram30 resource guard"):
            low_memory.guard(SimpleNamespace(harness=harness), args, [], "test")
    else:
        row = low_memory.guard(SimpleNamespace(harness=harness), args, [], "test")
        assert row["unattributed_gpu_pids"] == [845943]
        assert row["gpu_ownership_verified"] is False


def test_new_container_oom_counter_invalidates_run_and_stops_owned_groups(monkeypatch):
    stopped, monitoring = [], []
    upstream = SimpleNamespace(harness=SimpleNamespace(stop=lambda process: stopped.append(process.pid)))
    baseline = {"cgroups": [{"path": "/cg", "events": {"oom": 3, "oom_kill": 0}}]}
    current = {"cgroups": [{"path": "/cg", "events": {"oom": 4, "oom_kill": 1}}]}
    monkeypatch.setattr(low_memory, "guard", lambda *args: {"memory": current})
    watch = low_memory.ResourceWatch(upstream, None, [SimpleNamespace(pid=101)], "test", monitoring, baseline)
    watch._run()
    assert stopped == [101]
    with pytest.raises(RuntimeError, match="OOM/allocation failure"):
        watch.check()


def test_v1_reclaim_pressure_is_not_misreported_as_oom(monkeypatch):
    stopped, monitoring = [], []
    upstream = SimpleNamespace(harness=SimpleNamespace(stop=lambda process: stopped.append(process.pid)))
    watch = low_memory.ResourceWatch(upstream, None, [], "test", monitoring,
        {"cgroups": [{"path": "/cg", "events": {"failcnt": 0, "oom_kill": 0}}]})
    def reclaimed(*args):
        watch.stopped.set()
        return {"memory": {"cgroups": [{"path": "/cg", "events": {"failcnt": 17, "oom_kill": 0}}]}}
    monkeypatch.setattr(low_memory, "guard", reclaimed)
    watch._run()
    assert watch.error is None and not stopped and len(monitoring) == 1


def test_preflight_blocks_before_model_import_and_process_creation(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    native, _ = server.native_args(args, actual=False)
    upstream = server.load_upstream()
    monkeypatch.setattr(low_memory, "memory_snapshot", lambda: dict(available_bytes=2 * GIB))
    def forbidden(*args, **kwargs):
        pytest.fail("Low RAM must stop before imports or Popen")
    monkeypatch.setattr(upstream, "check_model", forbidden)
    monkeypatch.setattr(low_memory.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="Before model import"):
        low_memory.run(upstream, native)
    assert not args.output.exists()


@pytest.mark.parametrize("failure", [None, "warmup", "fixtures", "replay", "incomplete"])
def test_low_memory_lifecycle_keeps_fixtures_and_cleans_up(tmp_path, monkeypatch, failure):
    args = args_for(tmp_path)
    args.server_profile = "ram30"
    events, stopped = [], []
    upstream = server.load_upstream()
    original = upstream.teacher_replay
    monkeypatch.setattr(server.sys, "platform", "linux")
    monkeypatch.setattr(low_memory.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(server, "require_idle_before_model_import", lambda *a: dict(index="0", uuid="CPU_STUB"))
    monkeypatch.setattr(low_memory, "memory_snapshot", lambda: dict(available_bytes=25 * GIB))
    monkeypatch.setattr(upstream, "check_model", lambda model: dict(weights_verified=True))
    monkeypatch.setattr(low_memory.importlib.metadata, "version", lambda name: "CPU_TEST_STUB")
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def connect_ex(self, address): return 1
    monkeypatch.setattr(low_memory, "socket", SimpleNamespace(socket=Socket))
    def stage(name):
        events.append(name)
        if name == failure:
            raise RuntimeError("CPU test failure at " + name)
    def popen(command, **kwargs):
        stage("helper" if len(events) == 0 else "engine")
        return SimpleNamespace(pid=101 if "engine" not in events else 202, poll=lambda: None)
    monkeypatch.setattr(low_memory.subprocess, "Popen", popen)
    class Watch:
        error = None
        def __init__(self, *args): self.thread = SimpleNamespace(start=lambda: events.append("watch"))
        def check(self): pass
        def close(self): events.append("watch_closed")
    monkeypatch.setattr(low_memory, "ResourceWatch", Watch)
    monkeypatch.setattr(low_memory, "_wait_ready", lambda *args: None)
    monkeypatch.setattr(upstream.harness, "warmup", lambda ns: stage("warmup"))
    def fixtures(*args):
        assert os.environ["CONTINUUM_PREFIX_ONLY_CORRECTNESS"] == "1"
        stage("fixtures")
        return [{}] * 5
    monkeypatch.setattr(importlib.import_module("kairos._runtime.layout.correctness"), "validate", fixtures)
    monkeypatch.setattr(upstream.harness, "reset_owned_lmcache", lambda *a: stage("reset"))
    monkeypatch.setattr(upstream.harness, "snapshot_metrics", lambda ns, out, phase, *a: stage(phase))
    monkeypatch.setattr(upstream.harness, "stop", lambda process: stopped.append(process.pid))
    async def replay(selected):
        stage("replay")
        assert selected.bundle == args.bundle.resolve()
        return dict(strict_complete=failure != "incomplete", expected_calls=10)
    monkeypatch.setattr(server, "replay", replay)
    if failure:
        with pytest.raises(RuntimeError, match="every recorded call" if failure == "incomplete" else "CPU test failure"):
            server.run(args)
    else:
        server.run(args)
        assert events[:8] == ["helper", "watch", "engine", "warmup", "fixtures", "reset", "before", "replay"]
    assert stopped == [202, 101] and events[-1] == "watch_closed"
    assert upstream.teacher_replay is original
    state = read_json(args.output / "state.json")
    assert state["server_profile"] == "ram30"
    assert state["status"] == ("failed" if failure else "complete")
    assert state["gpu_input_smoke_validated"] is (failure is None)
    if failure in ("warmup", "fixtures"):
        assert "replay" not in events


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Requires real Linux /proc and process groups")
def test_orphan_worker_is_cleaned_without_touching_other_group(tmp_path):
    upstream=server.load_upstream()
    env=os.environ.copy()
    env["CONTINUUM_LOG_PATH"]=str(tmp_path/"scheduler.jsonl")
    code="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True)"
    leader=subprocess.Popen([sys.executable,"-c",code],env=env,start_new_session=True,
                            stdout=subprocess.PIPE,text=True)
    foreign=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],
                             start_new_session=True,stdout=subprocess.DEVNULL)
    try:
        worker=int(leader.stdout.readline())
        leader.wait(timeout=5)
        low_memory.stop_owned(upstream,leader,tmp_path)
        try:
            state=Path(f"/proc/{worker}/stat").read_text().rsplit(")",1)[1].split()[0]
        except FileNotFoundError:
            state="gone"
        assert state in ("Z","gone")
        assert foreign.poll() is None
    finally:
        low_memory.stop_owned(upstream,leader,tmp_path)
        upstream.harness.stop(foreign)
        leader.stdout.close()
