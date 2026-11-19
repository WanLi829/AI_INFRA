"""Validate raw comparison outputs and aggregate only matching completed runs."""
import argparse
import json
import math
from pathlib import Path
import statistics

from .client import load_bundle
from .common import read_json, sha, stats, write_json
from .comparison import COMMON_PROTOCOL, METHODS


def inspect(run, bundle, *, legacy=False):
    manifest, trace, _ = load_bundle(bundle)
    state = read_json(run / "state.json")
    plan = state["plan"]
    expected_schema='comparison/1' if legacy else 'comparison/2'
    expected_protocol='comparison-v1-cold-process' if legacy else COMMON_PROTOCOL
    if state.get("schema") != expected_schema or state.get("protocol") != expected_protocol:
        raise ValueError("Legacy or mock result is outside this comparison protocol")
    if state["method"] not in METHODS or state["method"] != plan["method"]:
        raise ValueError("Method mismatch")
    if plan["bundle_manifest_sha256"] != sha((bundle / "manifest.json").read_bytes()):
        raise ValueError("Bundle identity mismatch")
    from .workload import validate_workload, concurrency
    mode = validate_workload(manifest, plan['arrivals'])
    row = dict(run=str(run), method=state["method"], jps=plan["arrivals"].get('jps'),
               workload_mode=mode, active_programs=plan['arrivals'].get('active_programs'),
               seed=plan['arrivals']['seed'], dataset_id=manifest.get('dataset_id','unregistered'),
               status=state["status"], ttl_seconds=plan["ttl_seconds"], error=state.get("error"),
               completed_calls=None, completed_programs=None, speedup_eligible=False)
    summary_path = run / "b_replay/summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        row.update(completed_calls=summary["completed_calls"], completed_programs=summary["completed_programs"])
    if state["status"] != "complete":
        return row, None
    policy = state.get("gpu_pid_policy")
    verified = policy == "strict"
    if (not state["verified_model"]["weights_verified"] or policy not in ("strict", "record-only")
            or state.get("gpu_ownership_verified") is not verified or plan.get("gpu_pid_policy") != policy):
        raise ValueError("Missing model/ownership evidence")
    client = read_json(run / "b_replay/state.json")
    if (client["method"] != state["method"] or client["status"] != "complete"
            or (not legacy and client.get('generation_protocol')!='fixed-work-private-stop-v1')
            or "mock" in client["evidence"].lower()
            or client["bundle_manifest_sha256"] != plan["bundle_manifest_sha256"]
            or client["trace_sha256"] != manifest["files"]["trace.json"]
            or client["arrivals_sha256"] != plan["arrivals_sha256"]):
        raise ValueError("Invalid client provenance")
    records = [json.loads(line) for line in (run / "b_replay/requests.jsonl").read_text().splitlines()]
    expected = {(p["id"], i): t for p in trace["programs"] for i, t in enumerate(p["turns"])}
    actual = {(r["program_id"], r["turn_id"]): r for r in records}
    if len(actual) != len(records) or actual.keys() != expected.keys():
        raise ValueError("Missing or duplicate requests")
    for key, request in actual.items():
        turn = expected[key]
        from .client import replay_metadata
        if (not request["success"] or request["error"] is not None
                or (not legacy and request.get('public_max_tokens') != trace['model']['max_sequence_tokens']-turn['prompt_tokens'])
                or (not legacy and request.get('replay_key') != replay_metadata(state['method'],key[0],key[1],[])['cont_replay_key'])
                or request["prompt_sha256"] != turn["prompt_sha256"]
                or request["prompt_tokens"] != turn["prompt_tokens"]
                or len(request["output_ids"]) != request["output_tokens"]
                or request["planned_output_tokens"] != turn["output_tokens"]
                or request["output_tokens"] != turn["output_tokens"]):
            raise ValueError(f"Invalid request {key}")
        if key[1]:
            previous = actual[(key[0], key[1]-1)]
            if request["start_s"] + .002 < previous["end_s"] + expected[(key[0], key[1]-1)]["gap_after_s"]:
                raise ValueError("Continuation sent too early")
    jobs = read_json(run / "b_replay/jobs.json")
    if (len(jobs) != manifest["programs"] or {j["program_id"] for j in jobs} != set(manifest["program_ids"])
            or not all(j["success"] for j in jobs) or summary["strict_complete"] is not True):
        raise ValueError("Incomplete programs")
    arrival_times = {a["program_id"]: a["arrival_s"] for a in plan["arrivals"].get('arrivals',[])}
    for job in jobs:
        if mode == 'open' and abs(job["planned_arrival_s"] - arrival_times[job["program_id"]]) > .000001:
            raise ValueError("Job used a different arrival schedule")
        first = actual[(job["program_id"], 0)]
        turns = [r for (pid, _), r in actual.items() if pid == job["program_id"]]
        if (not math.isfinite(job["actual_arrival_s"])
                or (mode == 'open' and job["actual_arrival_s"] + .002 < job["planned_arrival_s"])
                or first["start_s"] + .002 < job["actual_arrival_s"]
                or job["end_s"] + .002 < max(r["end_s"] for r in turns)):
            raise ValueError("Invalid job timing")
    jct = stats([j["end_s"]-j["actual_arrival_s"] for j in jobs])
    concurrent = concurrency(jobs, plan['arrivals'].get('active_programs'))
    if mode == 'closed':
        if concurrent['peak'] > plan['arrivals']['active_programs']:
            raise ValueError('Closed loop exceeded active program target')
        observed = [j['program_id'] for j in sorted(jobs,key=lambda j:j['actual_arrival_s'])]
        if observed != plan['arrivals']['program_order']:
            raise ValueError('Closed loop activation order differs')
    if not math.isfinite(jct["mean"]) or jct["mean"] <= 0 or abs(jct["mean"] - summary["jct_s"]["mean"]) > 1e-6:
        raise ValueError("Summary does not match program durations")
    output_tokens = sum(r["output_tokens"] for r in records)
    resources = read_json(run / "resources.json")
    samples = [s for s in resources if "memory" in s]
    if (not samples or any(s.get("foreign_gpu_pids") or s.get("gpu_pid_policy") != policy
                           or s.get("gpu_ownership_verified", verified) is not verified for s in samples)
            or any("guard_error" in s or "cleanup_error" in s for s in resources)):
        raise ValueError("No clean resource samples")
    window = summary["observation_from_first_actual_arrival_s"]
    minimum_window = max(j["end_s"] for j in jobs) - min(j["actual_arrival_s"] for j in jobs)
    if not math.isfinite(window) or window <= 0 or window + .002 < minimum_window:
        raise ValueError("Invalid throughput window")
    ttft = stats([r['ttft_s'] for r in records])
    eligible = verified and not legacy and not plan.get('readiness_blockers') and not plan.get('exploratory',False)
    row.update(speedup_eligible=eligible, gpu_pid_policy=policy, gpu_uuid=state['gpu'].get('uuid'), jct_mean_s=jct["mean"], jct_p95_s=jct["p95"],
               output_tokens=output_tokens,
               output_tokens_per_s=output_tokens/window,
               ttft_mean_s=ttft['mean'], ttft_p95_s=ttft['p95'],
               programs_per_s=len(jobs)/window, observation_s=window,
               active_mean=concurrent['mean'],active_peak=concurrent['peak'],
               sampled_gpu_peak_mib=max(int(s["gpu"]["used"]) for s in samples),
               sampled_cgroup_peak_bytes=max((c["current_bytes"] for s in samples
                                              for c in s["memory"].get("cgroups", [])), default=None))
    for prefix,values in (('jct',jct),('ttft',ttft),
            ('request_e2e',stats([r['e2e_s'] for r in records])),
            ('tpot',stats([r.get('tpot_s') for r in records])),
            ('itl_single_token_chunk',stats([v for r in records for v in r.get('single_token_chunk_itl_s',[])]))):
        for metric in ('mean','p50','p90','p95','p99'):
            row[prefix+'_'+metric+'_s']=values[metric]
    target=concurrent.get('target_window') or {}
    row.update(active_target_mean=target.get('mean'),active_target_fraction=target.get('target_fraction'),
        active_below_fraction=target.get('below_fraction'),active_above_fraction=target.get('above_fraction'),
        target_window_s=target.get('duration_s'),initial_fill_s=concurrent.get('initial_fill_s'),drain_s=concurrent.get('drain_s'),
        refill_delay_mean_s=concurrent['refill_delay_s']['mean'],refill_delay_p95_s=concurrent['refill_delay_s']['p95'],
        jct_max_s=jct['max'],launch_replay_cleanup_s=state.get('timing',{}).get('total_launch_replay_cleanup_s'),
        resource_group=plan.get('resource_group','pilot'))
    command = plan["engine_command"]
    limits = {flag: command[command.index(flag)+1] for flag in
              ("--max-model-len", "--max-num-seqs", "--max-num-batched-tokens", "--gpu-memory-utilization")}
    identity = dict(protocol=state["protocol"], model=plan["model"], packages=state["packages"],
                    gpu=state["gpu"], input=plan["bundle_manifest_sha256"],
                    arrivals=plan["arrivals_sha256"], code=plan["adapter_sha256"],
                    memory_limit=state["memory_before_start"]["effective_limit_bytes"],
                    engine_cpus=command[2], limits=limits,
                    runtime=plan['runtime'], cpu_kv_gib=plan.get('cpu_kv_gib'),
                    joint_release=plan.get('joint_release_sha256'),
                    gpu_kv_bytes=plan.get('gpu_kv_bytes'), profile=plan.get('profile_sha256'),
                    reload_profile=plan.get('reload_profile_sha256'))
    identity['resource_group']=plan.get('resource_group')
    identity['helper_cpus']=plan['helper_command'][2] if plan.get('helper_command') else None
    # Helper presence is a method implementation detail; the reserved helper CPU
    # allocation itself is identical in the common resource declaration.
    identity.pop('helper_cpus')
    # Utilization/free memory fluctuate between runs; GPU identity is stable.
    identity["gpu"] = {k: state["gpu"].get(k) for k in ("uuid", "total", "index")}
    return row, json.dumps(identity, sort_keys=True)


def aggregate(rows):
    groups = {}
    for row, identity in rows:
        if row["speedup_eligible"]:
            method = row["method"] + (f"({row['ttl_seconds']}s)" if row["method"] == "static_ttl" else "")
            groups.setdefault(identity, {}).setdefault(method, []).append(row)
    output = []
    for methods in groups.values():
        baseline = methods.get("native", [])
        native_jct = statistics.mean(r["jct_mean_s"] for r in baseline) if baseline else None
        native_tps = statistics.mean(r["output_tokens_per_s"] for r in baseline) if baseline else None
        for method, values in methods.items():
            mean = statistics.mean(r["jct_mean_s"] for r in values)
            tps = statistics.mean(r["output_tokens_per_s"] for r in values)
            output.append(dict(method=method, jps=values[0]["jps"], n=len(values), native_n=len(baseline),
                               mean_jct_s=mean, jct_mean_range_s=[min(r["jct_mean_s"] for r in values),
                                                               max(r["jct_mean_s"] for r in values)],
                               jct_speedup=native_jct/mean if native_jct else None,
                               throughput_ratio=tps/native_tps if native_tps else None,
                               repeated_comparison=len(values)>=3 and len(baseline)>=3,
                               runs=[r["run"] for r in values]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [r.resolve() for r in args.run]
    if len(set(runs)) != len(runs):
        parser.error("A run was listed more than once")
    if args.output.exists():
        parser.error("Choose a new report path")
    rows = [inspect(run, args.bundle) for run in runs]
    write_json(args.output, dict(runs=[r for r, _ in rows], comparisons=aggregate(rows),
                                note="Ratios require matched native runs; n<3 remains exploratory."))


if __name__ == "__main__":
    main()
