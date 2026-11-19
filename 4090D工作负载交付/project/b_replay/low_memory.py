"""30GB input-smoke launcher using the frozen runtime's serving components.

The upstream source stays byte-identical. Full concurrency and cache capacity
are retained; weight loading and initial allocation reduce transient RAM use.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

from .resources import GIB, memory_snapshot, require_memory

PROFILE = dict(name="h800_27c230g", advertised_ram_bytes=230*1024**3,
               cpu_kv_gib=64, cpu_kv_initial_gib=1, max_num_seqs=8,
               safetensors_load_strategy="lazy", cuda_graphs=True,
               malloc_arena_max=2, startup_available_gib=8, runtime_available_gib=4,
               performance_comparable_to_archive=False)


def _option(command, name, value):
    if command.count(name) != 1:
        raise ValueError("Frozen launch option missing or duplicated: " + name)
    index = command.index(name)
    if value is None:
        del command[index:index + 2]
    else:
        command[index + 1] = str(value)


def prepare(upstream, args, out, control_socket):
    config, engine, helper, env = upstream.prepare(args, out, control_socket)
    _option(engine, "--safetensors-load-strategy", PROFILE["safetensors_load_strategy"])
    _option(helper, "--l1-init-size-gb", "1")
    # Bound allocator fragmentation without lowering request or cache capacity.
    config["environment"]["MALLOC_ARENA_MAX"] = env["MALLOC_ARENA_MAX"] = "2"
    for name in ("CUDA_HOME", "CUDACXX", "TRITON_LIBCUDA_PATH", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "HF_HOME",
                 "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                 "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "CUPY_CACHE_DIR",
                 "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "HF_DATASETS_CACHE", "RAY_TMPDIR"):
        if name in env:
            config["environment"][name] = env[name]
    config.update(server_profile="ram30")
    return config, engine, helper, env


def guard(upstream, args, processes, gpu_uuid):
    gpu = next(g for g in upstream.harness.smi() if int(g["index"]) == args.gpu)
    pid_policy = getattr(args, "gpu_pid_policy", "strict")
    process_query_error = None
    try:
        apps=upstream.harness.compute_apps(gpu_uuid)
    except subprocess.TimeoutExpired as error:
        if pid_policy=='strict':raise
        apps=[]
        process_query_error=repr(error)
    own, foreign = upstream.harness.gpu_process_scope(
        apps, [p.pid for p in processes])
    rss = 0
    for pid in own:
        try:
            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
            rss += sum(int(line.split()[1]) * 1024 for line in lines if line.startswith("VmRSS:"))
        except FileNotFoundError:
            pass
    memory = memory_snapshot()
    row = dict(time=time.time(), gpu=gpu, foreign_gpu_pids=sorted(foreign),
               gpu_pid_policy=pid_policy, gpu_ownership_verified=pid_policy=="strict",
               owned_rss_bytes=rss, available_ram_bytes=memory["available_bytes"], memory=memory)
    if pid_policy == "record-only":
        row["unattributed_gpu_pids"] = row.pop("foreign_gpu_pids")
        if process_query_error:
            row['unattributed_gpu_pids']=None
            row['process_query_available']=False
            row['process_query_error']=process_query_error
    if ((foreign and pid_policy=="strict") or memory["available_bytes"] < PROFILE["runtime_available_gib"] * GIB
            or int(gpu["used"]) > int(gpu["total"])):
        raise RuntimeError("ram30 resource guard violated: " + json.dumps(row))
    if any(p.poll() is not None for p in processes):
        raise RuntimeError("Owned serving process exited: " + json.dumps(row))
    return row


def _tagged_group(pgid, out):
    """Recognize surviving private workers even after their leader has exited."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    expected = b"CONTINUUM_LOG_PATH=" + os.fsencode(Path(out) / "scheduler.jsonl")
    members = []
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            if os.getpgid(int(path.name)) == pgid and expected in (path / "environ").read_bytes().split(b"\0"):
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return members


def stop_owned(upstream, process, out):
    # The normal stop handles a live Popen leader. A dead leader can leave
    # worker children holding CUDA; validate their exact run tag before signals.
    if process.poll() is None:
        try:
            upstream.harness.stop(process)
        except ProcessLookupError:
            pass
    for sig, timeout in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        if not _tagged_group(process.pid, out):
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _tagged_group(process.pid, out):
                return
            time.sleep(.1)
    raise RuntimeError("Owned workers survived cleanup for process group " + str(process.pid))


class ResourceWatch:
    """Keep RAM/ownership protection active during loading and fixtures too."""

    def __init__(self, upstream, args, processes, uuid, monitoring, baseline_memory=None):
        self.upstream, self.args, self.processes = upstream, args, processes
        self.uuid, self.monitoring = uuid, monitoring
        self.error = None
        self.baseline_oom = self._oom_counts(baseline_memory or {})
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _oom_counts(memory):
        return {(row["path"], event): value
                for row in memory.get("cgroups", [])
                for event, value in row.get("events", {}).items()
                if event in ("oom", "oom_kill", "oom_group_kill")}

    def _run(self):
        while not self.stopped.is_set():
            try:
                row = guard(self.upstream, self.args, list(self.processes), self.uuid)
                self.monitoring.append(row)
                if any(count > self.baseline_oom.get(key, 0)
                       for key, count in self._oom_counts(row["memory"]).items()):
                    raise RuntimeError("Container OOM/allocation failure counter increased; see resources.json")
            except Exception as error:
                self.error = error
                self.monitoring.append(dict(time=time.time(), guard_error=repr(error)))
                # PIDs come solely from this launcher's private Popen groups.
                for process in reversed(list(self.processes)):
                    try:
                        if self.args is None:
                            self.upstream.harness.stop(process)
                        else:
                            stop_owned(self.upstream, process, self.args.output)
                    except Exception as stop_error:
                        self.monitoring.append(dict(time=time.time(), cleanup_error=repr(stop_error)))
                return
            self.stopped.wait(1)

    def check(self):
        if self.error is not None:
            raise self.error

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=90)
        if self.thread.is_alive():
            raise RuntimeError("Resource watchdog did not stop")


def _wait_ready(url, timeout, processes, watch):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        watch.check()
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("Owned serving process exited before readiness")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.5)
    raise RuntimeError("Serving readiness timeout")


def run(upstream, args):
    """Retain model/checkout/fixture checks; use measured 30GB RAM guards."""
    from kairos.verify import verify_checkout, verify_workload
    verify_checkout()
    verify_workload()
    require_memory(memory_snapshot(), PROFILE["startup_available_gib"] * GIB, phase="Before model import")
    verified_model = upstream.check_model(args.model)
    if upstream.affinity(args.engine_cpus) & upstream.affinity(args.helper_cpus):
        raise ValueError("Engine/helper CPU sets must not overlap")
    if not upstream.affinity(args.engine_cpus + "," + args.helper_cpus) <= os.sched_getaffinity(0):
        raise ValueError("Selected CPU affinity is outside the allowed cpuset")
    for port in (18765, 18766, 18767, 18768):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already occupied")
    from .server import require_idle_before_model_import
    gpu = require_idle_before_model_import(upstream, args.gpu)
    memory = memory_snapshot()
    require_memory(memory, PROFILE["startup_available_gib"] * GIB, phase="Before serving startup")
    packages = {p: importlib.metadata.version(p) for p in ("torch", "vllm", "lmcache", "transformers", "httpx")}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    processes, monitoring, files = [], [], []
    watch = None
    state = dict(status="starting", method=args.method, started_at=time.time(), gpu=gpu,
        gpu_pid_policy=getattr(args,"gpu_pid_policy","strict"),
        gpu_ownership_verified=getattr(args,"gpu_pid_policy","strict")=="strict",
        server_profile="ram30", resource_profile=PROFILE, memory_before_start=memory,
        launcher="B ram30; frozen serving components, adapted resource settings and lifecycle guards",
        packages=packages, controller_pid=os.getpid(), owned_process_groups=[],
        profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        reload_sha256=hashlib.sha256(args.reload_profile.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256((upstream.ROOT / "data/workload/trace.json").read_bytes()).hexdigest(),
        expected_model_revision=upstream.REVISION, verified_model=verified_model,
        performance_comparable_to_archive=False, packaging_gpu_revalidation=True)
    upstream.harness.dump(out / "state.json", state)
    try:
        # Keep temporary files in the selected output tree, including on remote nodes.
        with tempfile.TemporaryDirectory(prefix="control-", dir=out) as temporary:
            config, engine, helper, env = prepare(upstream, args, out, Path(temporary) / "control.sock")
            upstream.harness.dump(out / "launch.json", dict(engine=engine, helper=helper,
                environment=upstream.recorded_environment(config, env), resource_profile=PROFILE))
            for name, path in (("profile.json", args.profile), ("reload-calibration.json", args.reload_profile)):
                (out / name).write_bytes(path.read_bytes())
            for command, name, url in ((helper, "helper", "http://127.0.0.1:18768/metrics"),
                                       (engine, "server", "http://127.0.0.1:18765/health")):
                if watch:
                    watch.check()
                log = (out / (name + ".log")).open("x")
                files.append(log)
                processes.append(subprocess.Popen(command, env=env, cwd=upstream.ROOT,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
                state["owned_process_groups"].append(dict(role=name, pid=processes[-1].pid, pgid=processes[-1].pid))
                upstream.harness.dump(out / "state.json", state)
                if watch is None:
                    watch = ResourceWatch(upstream, args, processes, gpu["uuid"], monitoring, memory)
                    watch.thread.start()
                _wait_ready(url, args.start_timeout, processes, watch)
            namespace = SimpleNamespace(model=args.model, port=18765, max_num_seqs=PROFILE["max_num_seqs"],
                request_timeout=120, run_timeout=900, reuse_engine=True, gapped_reuse=True,
                current_method=upstream.RUNTIME_ACTIVATION[args.method] + "~heterogeneous", current_root=str(out))
            upstream.harness.warmup(namespace)
            watch.check()
            from kairos._runtime.layout.correctness import validate
            # The frozen serving configurations use native prefix execution;
            # the fixture's legacy segmented method is not accepted by them.
            previous_fixture_mode = os.environ.get("CONTINUUM_PREFIX_ONLY_CORRECTNESS")
            os.environ["CONTINUUM_PREFIX_ONLY_CORRECTNESS"] = "1"
            try:
                fixtures = validate(namespace, processes[0], out)
            finally:
                if previous_fixture_mode is None:
                    os.environ.pop("CONTINUUM_PREFIX_ONLY_CORRECTNESS", None)
                else:
                    os.environ["CONTINUUM_PREFIX_ONLY_CORRECTNESS"] = previous_fixture_mode
            watch.check()
            state["correctness_results"] = len(fixtures)
            state["correctness_mode"] = "prefix-only; five layouts"
            upstream.harness.reset_owned_lmcache(processes[0], out)
            upstream.harness.snapshot_metrics(namespace, out, "before", True)
            state["status"] = "replaying"
            upstream.harness.dump(out / "state.json", state)
            # The B hook reads its own bundle; no archived program is submitted.
            result = asyncio.run(upstream.teacher_replay(namespace, [], out))
            watch.check()
            result.update(method=args.method, server_profile="ram30")
            upstream.harness.dump(out / "summary.json", result)
            if result.get("strict_complete") is not True:
                raise RuntimeError("B replay did not complete every recorded call; see summary.json")
            upstream.harness.snapshot_metrics(namespace, out, "after", True)
            watch.check()
            state.update(status="complete", summary=result)
            print(json.dumps(result, indent=2))
    except BaseException as error:
        state.update(status="failed", error=repr(watch.error if watch and watch.error else error))
        raise
    finally:
        cleanup_errors = []
        if watch:
            try:
                watch.close()
            except Exception as error:
                cleanup_errors.append(repr(error))
            if watch.error:
                state.update(status="failed", error=repr(watch.error))
        for process in reversed(processes):
            try:
                stop_owned(upstream, process, out)
            except Exception as error:
                cleanup_errors.append(repr(error))
        for file in files:
            file.close()
        if cleanup_errors:
            state.update(status="failed", cleanup_errors=cleanup_errors)
        state["ended_at"] = time.time()
        upstream.harness.dump(out / "state.json", state)
        upstream.harness.dump(out / "resources.json", monitoring)
        if cleanup_errors:
            raise RuntimeError("Owned service cleanup failed: " + repr(cleanup_errors))
        if watch and watch.error:
            raise watch.error
