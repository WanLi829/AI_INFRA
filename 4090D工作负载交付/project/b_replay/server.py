"""Run B inputs with frozen Kairos components and an explicit resource profile."""
from __future__ import annotations
import asyncio
import importlib
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from .common import ROOT, MODELS, sha, read_json, write_json
from .client import load_bundle, validate_arrivals, replay

RUNTIME_ROOT = ROOT / "runtime/Kairos"
RUNTIME_COMMIT = "7971fa40323f16e8e4ce8f12dfd5db9755a98cf1"

def verify_runtime():
    manifest=read_json(RUNTIME_ROOT/"provenance/release_manifest.json")
    for row in manifest["files"]:
        p=RUNTIME_ROOT/row["path"]
        if not p.resolve().is_relative_to(RUNTIME_ROOT.resolve()) or not p.is_file() or sha(p.read_bytes())!=row["sha256"]:
            raise ValueError("Frozen runtime integrity mismatch: "+row["path"])
    return dict(commit=RUNTIME_COMMIT,files_verified=len(manifest["files"]),core_modified=False)

def validate_inputs(args):
    if getattr(args,"gpu_pid_policy","strict")!="strict" and server_profile(args)!="ram30":
        raise ValueError("GPU PID record-only policy requires the ram30 launcher")
    manifest,trace,_=load_bundle(args.bundle)
    if manifest["model"]!="Qwen3-8B" or manifest["model_revision"]!=MODELS["Qwen3-8B"]:
        raise ValueError("Frozen runtime supports only the pinned Qwen3-8B snapshot; 4B data is conversion-only")
    validate_arrivals(manifest,read_json(args.arrivals))
    if args.method=="kairos":
        trained=set(read_json(RUNTIME_ROOT/"assets/forest.json")["training_ids"])
        overlap=sorted(trained.intersection(manifest["program_ids"]))
        if overlap:raise ValueError(f"Frozen Kairos forest rejects training identities: {len(overlap)} overlaps; use pilot_* or heldout_candidates")
    if args.request_timeout<=0 or args.run_timeout<=0 or args.start_timeout<=0:
        raise ValueError("Timeouts must be positive")
    if Path(args.output).exists():raise FileExistsError("Output directory must not exist: "+str(args.output))
    return manifest

def load_upstream():
    loaded=sys.modules.get("kairos")
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(RUNTIME_ROOT.resolve()):
        raise RuntimeError("A different Kairos package is already imported")
    location=str(RUNTIME_ROOT.resolve())
    if location not in sys.path:sys.path.insert(0,location)
    module=importlib.import_module("kairos.run")
    if not Path(module.__file__).resolve().is_relative_to(RUNTIME_ROOT.resolve()):
        raise RuntimeError("Kairos module shadowed the bundled runtime")
    return module

def cpu_sets(engine, helper, *, actual):
    if (engine=="auto")!=(helper=="auto"):
        raise ValueError("Specify both CPU sets or use auto for both")
    if engine!="auto":return engine,helper,False
    if not hasattr(os,"sched_getaffinity"):
        if actual:raise RuntimeError("GPU startup requires Linux CPU affinity support")
        return "AUTO_ENGINE_ON_LINUX","AUTO_HELPER_ON_LINUX",True
    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<4:raise RuntimeError("Require at least four allowed CPU IDs for disjoint engine/helper sets")
    return ",".join(map(str,allowed[:2])),",".join(map(str,allowed[2:4])),False

def native_args(args, *, actual):
    engine,helper,deferred=cpu_sets(args.engine_cpus,args.helper_cpus,actual=actual)
    return SimpleNamespace(method=args.method,model=str(Path(args.model).expanduser().resolve()),
        output=Path(args.output).resolve(),gpu=args.gpu,engine_cpus=engine,helper_cpus=helper,
        profile=RUNTIME_ROOT/"assets/deployment_profile.json",reload_profile=RUNTIME_ROOT/"assets/reload_profile.json",
        start_timeout=args.start_timeout,gpu_pid_policy=getattr(args,"gpu_pid_policy","strict"),dry_run=False),deferred

def server_profile(args):
    profile=getattr(args,"server_profile","frozen")
    if profile not in ("frozen","ram30"):
        raise ValueError("Unknown server profile: "+profile)
    return profile

def implementation_hashes():
    result={name:sha(Path(__file__).with_name(name).read_bytes()) for name in
            ("server.py","client.py","low_memory.py","resources.py","common.py","cli.py")}
    result["scripts/gpu_python.py"]=sha((ROOT/"scripts/gpu_python.py").read_bytes())
    return result

def make_plan(args):
    source=verify_runtime()
    manifest=validate_inputs(args)
    upstream=load_upstream()
    native,deferred=native_args(args,actual=False)
    profile=server_profile(args)
    if profile=="ram30":
        from .low_memory import prepare,PROFILE
        config,engine,helper,env=prepare(upstream,native,native.output,"OUTPUT/control-EXAMPLE/control.sock")
        resource_profile=PROFILE
    else:
        config,engine,helper,env=upstream.prepare(native,native.output,"/tmp/b-replay-example/control.sock")
        resource_profile=dict(name="frozen",cpu_kv_gib=8,max_num_seqs=8,startup_available_gib=40)
    return dict(status="plan_only",gpu_started=False,packages_or_weights_validated=False,
        gpu_pid_policy=getattr(args,"gpu_pid_policy","strict"),
        runtime=source,method=args.method,server_profile=profile,resource_profile=resource_profile,
        b_runtime_sha256=implementation_hashes(),
        programs=manifest["programs"],calls=manifest["calls"],
        bundle=str(Path(args.bundle).resolve()),arrivals=str(Path(args.arrivals).resolve()),
        trace_sha256=manifest["files"]["trace.json"],
        model=native.model,engine_command=engine,helper_command=helper,
        cpu_affinity_deferred_to_linux=deferred,
        environment=upstream.recorded_environment(config,env),
        lifecycle=["Verify frozen checkout, original workload integrity, pinned model files/weights",
                   "Check Linux CPU affinity, free ports, idle GPU and profile RAM limits",
                   "ram30: preserve 8 sequences/8 GiB CPU KV, lazy weight loading, cgroup RAM/ownership watchdog" if profile=="ram30" else "Frozen guard: >=40 GiB available host RAM",
                   "Start owned helper/engine; readiness with resource watchdog",
                   "Original warmup, five correctness fixtures, private-cache drain/reset",
                   "Run B selected bundle via client hook; original 16-program replay is not called",
                   "Save B input provenance, per-call records, metrics/resources; stop owned groups only"],
        limits=["Input smoke test; no performance conclusion", "GPU execution remains unverified until an actual run",
                "Archived cost profiles are retained for interface validation",
                "ram30 adapts resource settings and RAM watchdog; no archived performance comparison" if profile=="ram30" else "Original safety/ownership checks remain enabled"])

def make_adapter(args,manifest):
    async def adapter(namespace, archived_programs, run_dir):
        # archived_programs is intentionally unused; keep upstream source/data intact.
        selected=SimpleNamespace(bundle=Path(args.bundle).resolve(),arrivals=Path(args.arrivals).resolve(),
            endpoint=f"http://127.0.0.1:{namespace.port}",model=namespace.model,method=args.method,
            output=Path(run_dir)/"b_replay",request_timeout=args.request_timeout,run_timeout=args.run_timeout,mock=False)
        result=await replay(selected)
        result.update(input_source="B bundle",runtime_commit=RUNTIME_COMMIT,
                      performance_claim_supported=False,archived_replay_executed=False)
        return result
    return adapter

def correct_provenance(out,manifest,args,runtime_hashes):
    path=out/"state.json"
    if not path.exists():return
    state=read_json(path)
    state["upstream_archived_trace_sha256"]=state.get("trace_sha256")
    state["trace_sha256"]=manifest["files"]["trace.json"]
    state["b_bundle_manifest_sha256"]=sha((Path(args.bundle)/"manifest.json").read_bytes())
    state["b_arrival_manifest_sha256"]=sha(Path(args.arrivals).read_bytes())
    state["runtime_commit"]=RUNTIME_COMMIT
    state["server_profile"]=server_profile(args)
    state["b_runtime_sha256"]=runtime_hashes
    state["gpu_pid_policy"]=getattr(args,"gpu_pid_policy","strict")
    state["gpu_ownership_verified"]=state["gpu_pid_policy"]=="strict"
    state["adapter"]=("B ram30 launcher; frozen serving source and fixtures, adapted memory/settings/guards"
                       if state["server_profile"]=="ram30" else
                       "B client hook; source, cache ownership, guards, warmup and correctness fixtures unchanged")
    verified=state.get("verified_model",{}).get("weights_verified") is True
    successful=state.get("status")=="complete" and state.get("summary",{}).get("strict_complete") is True
    state["gpu_input_smoke_validated"]=bool(successful and verified and state.get("gpu"))
    state["performance_claim_supported"]=False
    write_json(path,state)
    write_json(out/"b_input_provenance.json",dict(runtime_commit=RUNTIME_COMMIT,
        manifest=manifest,arrivals=read_json(args.arrivals),gpu_input_smoke_validated=state["gpu_input_smoke_validated"],
        server_profile=state["server_profile"],b_runtime_sha256=runtime_hashes,performance_claim_supported=False))

def require_idle_before_model_import(upstream,gpu_index):
    """Check before model validation can import torch/vLLM; native guards still run later."""
    matches=[g for g in upstream.harness.smi() if int(g["index"])==gpu_index]
    if len(matches)!=1:raise RuntimeError("Requested GPU index is unavailable")
    gpu=matches[0]
    if int(gpu["used"])>128 or int(gpu["util"])>5 or upstream.harness.compute_apps(gpu["uuid"]):
        raise RuntimeError("GPU is busy before model import; no service started: "+json.dumps(gpu))
    return gpu

def run(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("server-run requires the Linux GPU server; use server-plan locally")
    verify_runtime()
    manifest=validate_inputs(args)
    upstream=load_upstream()
    require_idle_before_model_import(upstream,args.gpu)
    profile=server_profile(args)
    native,_=native_args(args,actual=True)
    runtime_hashes=implementation_hashes()
    previous=upstream.teacher_replay
    upstream.teacher_replay=make_adapter(args,manifest)
    previous_handler=signal.getsignal(signal.SIGTERM)
    def terminate(signum,frame):raise KeyboardInterrupt("Received SIGTERM")
    signal.signal(signal.SIGTERM,terminate)
    try:
        if profile=="ram30":
            from .low_memory import run as run_low_memory
            run_low_memory(upstream,native)
        else:
            upstream.run(native)
    finally:
        upstream.teacher_replay=previous
        signal.signal(signal.SIGTERM,previous_handler)
        correct_provenance(native.output,manifest,args,runtime_hashes)
