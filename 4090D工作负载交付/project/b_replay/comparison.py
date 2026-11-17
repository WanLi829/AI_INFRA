"""Five-method comparison. Plans are CPU-only; runs require the private Linux GPU environment."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

from .client import load_bundle, validate_arrivals, replay
from .common import ROOT, read_json, write_json, sha
from . import low_memory, server
from .resources import memory_snapshot, require_memory, GIB

METHODS = ("native", "lmcache", "static_ttl", "continuum", "kairos")
COMMON_PROTOCOL = "comparison-v1-cold-process"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--arrivals", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ttl-seconds", type=float,
                        help="Required for static_ttl; frozen engineering constant, not a fitted paper threshold")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-pid-policy", choices=("strict", "record-only"), default="strict",
                        help="Record unresolvable host PIDs on namespaced pilot runs; excludes formal speedup claims")
    parser.add_argument("--start-timeout", type=float, default=1200)
    parser.add_argument("--run-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--engine-cpus", default="auto")
    parser.add_argument("--helper-cpus", default="auto")
    args = parser.parse_args(argv)
    if args.method == "static_ttl":
        if args.ttl_seconds is None or not math.isfinite(args.ttl_seconds) or args.ttl_seconds <= 0:
            parser.error("static_ttl requires a positive, finite --ttl-seconds")
    elif args.ttl_seconds is not None:
        parser.error("--ttl-seconds is exclusive to static_ttl")
    for value in (args.start_timeout, args.run_timeout, args.request_timeout):
        if not math.isfinite(value) or value <= 0:
            parser.error("Timeouts must be positive and finite")
    args.server_profile = "ram30"
    args.output = args.output.resolve()
    args.bundle = args.bundle.resolve()
    args.arrivals = args.arrivals.resolve()
    return args


def build(args):
    source = server.verify_runtime()
    manifest = server.validate_inputs(args)
    upstream = server.load_upstream()
    native, deferred = server.native_args(args, actual=args.action == "run")
    native.method = "kairos" if args.method == "kairos" else "continuum"
    config, engine, helper, env = low_memory.prepare(
        upstream, native, args.output, args.output / "control.sock")
    if args.method in ("native", "lmcache"):
        for flag in ("--worker-cls", "--scheduler-cls", "--kv-transfer-config"):
            low_memory._option(engine, flag, None)
        low_memory._option(engine, "--scheduling-policy", "fcfs")
        env = {k: v for k, v in env.items()
               if not k.startswith(("CONTINUUM_", "CONTINUITY_", "KAIROS_", "LMCACHE_"))}
        env["PYTHONPATH"] = str(ROOT)
        helper = None
        if args.method == "lmcache":
            engine += ["--kv-transfer-config", json.dumps({
                "kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"})]
            env.update(LMCACHE_LOCAL_CPU="True", LMCACHE_MAX_LOCAL_CPU_SIZE="8.0",
                       LMCACHE_CHUNK_SIZE="256", LMCACHE_SAVE_DECODE_CACHE="False",
                       LMCACHE_ENABLE_ASYNC_LOADING="False")
        # Used only to identify this launcher's surviving process group.
        env["CONTINUUM_LOG_PATH"] = str(args.output / "scheduler.jsonl")
    else:
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(server.RUNTIME_ROOT)))
        if args.method == "static_ttl":
            scheduler = "b_replay.static_scheduler.StaticTTLScheduler"
            low_memory._option(engine, "--scheduler-cls", scheduler)
            env["CONTINUUM_EXPECTED_SCHEDULER"] = scheduler
            env["CONTINUUM_TTL_SECONDS"] = str(args.ttl_seconds)
    config["environment"] = {k: v for k, v in env.items()
                             if k in config["environment"] or k.startswith("LMCACHE_")}
    code = {p.name: sha(p.read_bytes()) for p in sorted((ROOT / "b_replay").glob("*.py"))}
    plan = dict(schema="comparison/1", status="plan_only", gpu_started=False,
                method=args.method, ttl_seconds=args.ttl_seconds, protocol=COMMON_PROTOCOL,
                model=str(Path(args.model).resolve()), runtime=source, adapter_sha256=code,
                programs=manifest["programs"], calls=manifest["calls"],
                bundle_manifest_sha256=sha((args.bundle / "manifest.json").read_bytes()),
                arrivals_sha256=sha(args.arrivals.read_bytes()),
                trace_sha256=manifest["files"]["trace.json"],
                arrivals=read_json(args.arrivals), engine_command=engine, helper_command=helper,
                environment=upstream.recorded_environment(config, env),
                cpu_affinity_deferred_to_linux=deferred, gpu_pid_policy=args.gpu_pid_policy,
                warmup="Fresh process; engine initialization/graph capture only; no client warmup or correctness requests",
                caveats=["GPU package/connector compatibility and policy activation await live smoke test",
                         "Static TTL is a named engineering ablation",
                         "Legacy warm-start observations are not comparable to this cold-process protocol"])
    return plan, upstream, native, env


def snapshot(out, label):
    with urllib.request.urlopen("http://127.0.0.1:18765/metrics", timeout=10) as response:
        (out / f"metrics-{label}.txt").write_bytes(response.read())


def run(args, plan, upstream, native, env):
    if sys.platform != "linux":
        raise RuntimeError("GPU run requires Linux")
    if len(os.fsencode(args.output / "control.sock")) >= 108:
        raise ValueError("Output path is too long for a Unix socket")
    gpu = server.require_idle_before_model_import(upstream, args.gpu)
    memory = memory_snapshot()
    require_memory(memory, low_memory.PROFILE["startup_available_gib"] * GIB, phase="Before model validation")
    verified = upstream.check_model(args.model)
    engine_cpus, helper_cpus = upstream.affinity(native.engine_cpus), upstream.affinity(native.helper_cpus)
    if engine_cpus & helper_cpus or not (engine_cpus | helper_cpus) <= os.sched_getaffinity(0):
        raise ValueError("Invalid or overlapping CPU affinity")
    packages = {p: importlib.metadata.version(p) for p in ("vllm", "torch", "lmcache", "httpx", "transformers")}
    if packages["vllm"] != "0.28.0":
        raise RuntimeError("This comparison was prepared for vLLM 0.28.0; verify a changed version before running")
    ports = (18765, 18766, 18767, 18768) if plan["helper_command"] else (18765,)
    for port in ports:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already occupied")
    # Model verification may have taken time. Recheck ownership before starting.
    gpu = server.require_idle_before_model_import(upstream, args.gpu)
    args.output.mkdir(parents=True, exist_ok=False)
    native.output = args.output
    state = dict(schema="comparison/1", method=args.method, status="starting", started_at=time.time(),
                 gpu=gpu, gpu_ownership_verified=args.gpu_pid_policy == "strict",
                 gpu_pid_policy=args.gpu_pid_policy,
                 memory_before_start=memory,
                 verified_model=verified, packages=packages, plan=plan, protocol=COMMON_PROTOCOL)
    write_json(args.output / "launch.json", plan)
    write_json(args.output / "state.json", state)
    processes, logs, monitoring = [], [], []
    watch = None
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt("Comparison interrupted")

    signal.signal(signal.SIGTERM, terminate)
    try:
        commands = []
        if plan["helper_command"]:
            commands.append(("helper", plan["helper_command"], "http://127.0.0.1:18768/metrics"))
        commands.append(("server", plan["engine_command"], "http://127.0.0.1:18765/health"))
        deadline = time.monotonic() + args.start_timeout
        for name, command, url in commands:
            if watch:
                watch.check()
            log = (args.output / f"{name}.log").open("x", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            state["owned_process_groups"] = [p.pid for p in processes]
            write_json(args.output / "state.json", state)
            if watch is None:
                watch = low_memory.ResourceWatch(upstream, native, processes, gpu["uuid"], monitoring, memory)
                watch.thread.start()
            low_memory._wait_ready(url, max(0, deadline-time.monotonic()), processes, watch)
        snapshot(args.output, "before")
        state["status"] = "replaying"
        write_json(args.output / "state.json", state)
        selected = SimpleNamespace(**vars(args))
        selected.output = args.output / "b_replay"
        selected.endpoint = "http://127.0.0.1:18765"
        selected.model = plan["model"]
        selected.mock = False
        result = asyncio.run(replay(selected))
        watch.check()
        snapshot(args.output, "after")
        if args.method == "static_ttl":
            events = [json.loads(line) for line in (args.output / "scheduler.jsonl").read_text().splitlines()]
            if not any(e.get("event") == "fixed_ttl_active" and e.get("seconds") == args.ttl_seconds for e in events):
                raise RuntimeError("Fixed TTL activation was not recorded")
            pins = [e for e in events if e.get("event") == "pin"]
            if not pins or any(abs(e["ttl"]-args.ttl_seconds) > 1e-8 for e in pins):
                raise RuntimeError("No fixed-TTL pins observed or a different TTL was applied")
        state.update(status="complete", summary=result)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        cleanup_errors = []
        if watch:
            try:
                watch.close()
            except Exception as error:
                cleanup_errors.append(repr(error))
            if watch.error:
                cleanup_errors.append(repr(watch.error))
        for process in reversed(processes):
            try:
                low_memory.stop_owned(upstream, process, args.output)
            except Exception as error:
                cleanup_errors.append(repr(error))
        for log in logs:
            log.close()
        signal.signal(signal.SIGTERM, previous)
        if cleanup_errors:
            state.update(status="failed", cleanup_errors=cleanup_errors)
        state["ended_at"] = time.time()
        write_json(args.output / "state.json", state)
        write_json(args.output / "resources.json", monitoring)
        if cleanup_errors:
            raise RuntimeError("Cleanup/watchdog failed: " + repr(cleanup_errors))


def main():
    args = arguments()
    plan, upstream, native, env = build(args)
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        run(args, plan, upstream, native, env)


if __name__ == "__main__":
    main()
