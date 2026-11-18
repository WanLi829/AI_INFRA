"""Replay to an already-running Kairos-compatible engine; never starts a GPU service."""
from __future__ import annotations
import asyncio
import json
import math
from pathlib import Path
import random
import time
from .common import read_json, write_json, sha, stats, json_bytes, RUNTIME_ACTIVATION
from .workload import validate_workload, concurrency

def replay_metadata(method, pid, turn_id, messages):
    experiment = RUNTIME_ACTIVATION.get(method, RUNTIME_ACTIVATION['continuum']) + '~heterogeneous'
    key = sha(json.dumps([experiment, pid, turn_id], separators=(',', ':')).encode())
    return dict(cont_program_id=pid, cont_tool='unknown', cont_experiment=experiment,
                cont_replay_key=key, cont_visible_messages=json.dumps(messages, separators=(',', ':')))

def load_bundle(bundle):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha((bundle / name).read_bytes()) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    trace = read_json(bundle / "trace.json")
    visible = read_json(bundle / "visible_messages.json")
    ids = [p["id"] for p in trace["programs"]]
    if not ids or len(ids) != len(set(ids)) or ids != manifest["program_ids"] or set(ids) != set(visible):
        raise ValueError("Empty, duplicate, or inconsistent program identities")
    calls = 0
    for program in trace["programs"]:
        prompt = []
        turns = program["turns"]
        if not turns or len(turns) != len(visible[program["id"]]):
            raise ValueError("Invalid visible turn counts")
        for i, turn in enumerate(turns):
            keep = turn["keep_prefix_tokens"]
            if not isinstance(keep, int) or not 0 <= keep <= len(prompt):
                raise ValueError("Invalid prefix reference")
            prompt = prompt[:keep] + turn["added_token_ids"]
            if any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in prompt):
                raise ValueError("Invalid token identity")
            if len(prompt) != turn["prompt_tokens"] or sha(json.dumps(prompt).encode()) != turn["prompt_sha256"]:
                raise ValueError("Prompt reconstruction/hash mismatch")
            if sha(json_bytes(visible[program["id"]][i], pretty=False)) != turn["visible_messages_sha256"]:
                raise ValueError("Visible message hash mismatch")
            output = turn["output_tokens"]
            if not isinstance(output, int) or isinstance(output, bool) or output < 1:
                raise ValueError("Invalid output budget")
            if len(prompt)+output > trace["model"]["max_sequence_tokens"]:
                raise ValueError("Sequence exceeds bundle limit")
            gap = turn["gap_after_s"]
            if not isinstance(gap, (int,float)) or not math.isfinite(gap) or gap < 0:
                raise ValueError("Invalid tool interval")
            calls += 1
    if calls != manifest["calls"] or len(ids) != manifest["programs"]:
        raise ValueError("Bundle count mismatch")
    if manifest.get('schema_version') == 'b-replay-bundle/2.0':
        if not {'completed_actions.json','data_audit.json'} <= manifest['files'].keys():
            raise ValueError('Prepared bundle lacks hashed completion/audit evidence')
        payload=read_json(bundle/'completed_actions.json')
        if payload.get('schema')!='recorded-current-actions-v1' or set(payload.get('actions',{}))!=set(ids):
            raise ValueError('Invalid completed-action identities')
        for p in trace['programs']:
            actions=payload['actions'][p['id']]
            if len(actions)!=len(p['turns']): raise ValueError('Completion count mismatch')
            for i,a in enumerate(actions):
                if set(a)!={'tool','terminal','evidence','assistant_sha256'} or type(a['terminal']) is not bool:
                    raise ValueError('Invalid action fields')
                if a['terminal'] and i+1<len(actions): raise ValueError('Program continues after explicit terminal action')
    return manifest, trace, visible

def make_arrivals(bundle, jps, seed, output):
    if not math.isfinite(jps) or jps <= 0:
        raise ValueError("JPS must be positive and finite")
    manifest, _, _ = load_bundle(bundle)
    rng = random.Random(seed)
    arrivals = []
    clock = 0.0
    for pid in manifest["program_ids"]:
        clock += rng.expovariate(jps)
        arrivals.append(dict(program_id=pid, arrival_s=clock))
    write_json(output, dict(schema_version="b-replay-arrivals/1.0", seed=seed, jps=jps,
        first_arrival_rule="Includes the first exponential inter-arrival from clock zero",
        program_order="Bundle program_ids order", trace_sha256=manifest["files"]["trace.json"],
        arrivals=arrivals))

def validate_arrivals(manifest, arrival):
    rows = arrival["arrivals"]
    if arrival["trace_sha256"] != manifest["files"]["trace.json"]:
        raise ValueError("Arrival manifest targets another trace")
    if [r["program_id"] for r in rows] != manifest["program_ids"]:
        raise ValueError("Arrival identities/order mismatch")
    values = [r["arrival_s"] for r in rows]
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Invalid arrival timestamp")
    if values != sorted(values): raise ValueError("Arrival times are not ordered")
    return {r["program_id"]: r["arrival_s"] for r in rows}

async def replay(args, transport=None):
    import httpx
    manifest, trace, visible = load_bundle(args.bundle)
    if args.method not in (*RUNTIME_ACTIVATION, "native", "lmcache", "static_ttl", "zero_ttl"):
        raise ValueError("Unsupported method")
    arrival = read_json(args.arrivals)
    mode = validate_workload(manifest, arrival)
    planned = validate_arrivals(manifest, arrival) if mode == 'open' else {p: None for p in manifest['program_ids']}
    if args.request_timeout <= 0 or args.run_timeout <= 0:
        raise ValueError("Timeouts must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    evidence = "CPU mock protocol test only" if args.mock else "Serving replay; hardware/engine identity requires C's run record"
    state = dict(status="starting", evidence=evidence, endpoint=args.endpoint, served_model=args.model,
        method=args.method, workload_mode=mode, generation_protocol='fixed-work-private-stop-v1',
        bundle_manifest_sha256=sha((Path(args.bundle)/"manifest.json").read_bytes()),
        trace_sha256=manifest["files"]["trace.json"], arrivals_sha256=sha(Path(args.arrivals).read_bytes()),
        expected_programs=manifest["programs"], expected_calls=manifest["calls"],
        tokenizer=trace["model"], tools_executed=False,
        online_tool_label='unknown',
        future_tool_annotation_sent=False)
    write_json(out/"state.json", state)
    records, jobs = [], []
    start = time.monotonic()
    state['clock_origin_monotonic_s']=start
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout, connect=min(10,args.request_timeout)),
            limits=httpx.Limits(max_connections=max(64, manifest["programs"])), transport=transport,
            trust_env=False) as client:
        async def program(row, refill_at=None):
            pid = row["id"]
            job_start = None
            completed = 0
            try:
                if mode == 'open':
                    await asyncio.sleep(max(0, planned[pid]-(time.monotonic()-start)))
                job_start = time.monotonic()
                prompt = []
                for turn_id, part in enumerate(row["turns"]):
                    prompt = prompt[:part["keep_prefix_tokens"]] + part["added_token_ids"]
                    # The current teacher output's tool class is not visible
                    # before generation. Keep it offline and use the protocol's
                    # pooled fallback for online decisions.
                    meta = replay_metadata(args.method, pid, turn_id, visible[pid][turn_id])
                    payload = dict(model=args.model, prompt=prompt, max_tokens=trace['model']['max_sequence_tokens']-len(prompt),
                        ignore_eos=True, temperature=0, seed=1, stream=True, return_token_ids=True,
                        stream_options=dict(include_usage=True), cache_salt=row.get("cache_salt", pid),
                        vllm_xargs=meta)
                    # All methods share a private stop adapter. Public max_tokens
                    # is the context ceiling, never the recorded output label.
                    begin = time.monotonic()
                    output, chunks, usage, error, request_id = [], [], {}, None, None
                    try:
                        async with client.stream("POST", args.endpoint.rstrip("/")+"/v1/completions", json=payload) as response:
                            if response.status_code != 200:
                                raise RuntimeError(f"HTTP {response.status_code}: {(await response.aread()).decode()[:1000]}")
                            async for line in response.aiter_lines():
                                if not line.startswith("data:"): continue
                                data = line[5:].strip()
                                if data == "[DONE]": break
                                item = json.loads(data)
                                if item.get("error"): raise RuntimeError(str(item["error"]))
                                request_id = item.get("id", request_id)
                                for choice in item.get("choices", []):
                                    new_ids = choice.get("token_ids") or []
                                    if any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in new_ids):
                                        raise RuntimeError("Invalid generated token IDs")
                                    if new_ids:
                                        chunks.append((time.monotonic()-start, len(new_ids)))
                                        output.extend(new_ids)
                                    if len(output) > part["output_tokens"]:
                                        raise RuntimeError("Output exceeded fixed budget")
                                if item.get("usage"): usage = item["usage"]
                        if len(output) != part["output_tokens"]:
                            raise RuntimeError(f"Incomplete output: expected {part['output_tokens']}, got {len(output)}")
                        if usage.get("completion_tokens") not in (None, len(output)):
                            raise RuntimeError("Usage/token-ID count mismatch")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    end = time.monotonic()
                    record = dict(program_id=pid, turn_id=turn_id, request_id=request_id,
                        planned_arrival_s=planned[pid] if turn_id == 0 else None,
                        start_s=begin-start, end_s=end-start, e2e_s=end-begin,
                        ttft_s=chunks[0][0]-(begin-start) if chunks else None,
                        single_token_chunk_itl_s=[b[0]-a[0] for a,b in zip(chunks,chunks[1:]) if a[1]==b[1]==1],
                        stream_chunk_sizes=[n for _,n in chunks], prompt_tokens=len(prompt),
                        prompt_sha256=part["prompt_sha256"], output_tokens=len(output),
                        planned_output_tokens=part["output_tokens"], output_ids=output, usage=usage,
                        public_max_tokens=payload['max_tokens'], replay_key=meta['cont_replay_key'],
                        gap_after_s=part["gap_after_s"],
                        cont_tool_sent=meta["cont_tool"] if "vllm_xargs" in payload else None,
                        error=error, success=error is None)
                    records.append(record)
                    with (out/"requests.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(record, ensure_ascii=False)+"\n")
                    if error: raise RuntimeError(error)
                    completed += 1
                    if turn_id+1 < len(row["turns"]):
                        await asyncio.sleep(part["gap_after_s"])
                job = dict(program_id=pid, success=True, completed_calls=completed)
            except BaseException as exc:
                job = dict(program_id=pid, success=False, completed_calls=completed,
                           error=f"{type(exc).__name__}: {exc}")
                if isinstance(exc, (KeyboardInterrupt, SystemExit)): raise
            finish = time.monotonic()
            job.update(planned_arrival_s=planned[pid], actual_arrival_s=job_start-start if job_start else None,
                arrival_lag_s=job_start-start-planned[pid] if job_start and mode == 'open' else None,
                refill_delay_s=job_start-refill_at if job_start and refill_at is not None else None,
                end_s=finish-start, jct_s=finish-job_start if job_start else None,
                tool_dependency_s=sum(t["gap_after_s"] for t in row["turns"][:-1]))
            jobs.append(job)
            # Persist on every completion, including before a global timeout.
            write_json(out/'jobs.json', jobs)
        if mode == 'closed':
            by_id = {p['id']: p for p in trace['programs']}
            pending = iter(arrival['program_order'])
            async def worker():
                refill_at = time.monotonic()
                while True:
                    pid = next(pending, None)
                    if pid is None:
                        return
                    await program(by_id[pid], refill_at)
                    refill_at = time.monotonic()
                    if asyncio.current_task().cancelling():
                        return
            tasks = [asyncio.create_task(worker()) for _ in range(arrival['active_programs'])]
        else:
            tasks = [asyncio.create_task(program(row)) for row in trace["programs"]]
        state["status"] = "replaying"
        write_json(out/"state.json", state)
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.run_timeout)
        except TimeoutError:
            timed_out = True
            await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.monotonic()-start
    successful = [r for r in records if r["success"]]
    good_jobs = [j for j in jobs if j["success"]]
    strict = len(successful)==manifest["calls"] and len(good_jobs)==manifest["programs"] and not timed_out
    first_actual = min((j["actual_arrival_s"] for j in jobs if j["actual_arrival_s"] is not None), default=None)
    observation = wall-first_actual if first_actual is not None else None
    summary = dict(status="complete" if strict else "failed", strict_complete=strict, evidence=evidence,
        gpu_validated=False, expected_programs=manifest["programs"], expected_calls=manifest["calls"],
        completed_programs=len(good_jobs), failed_programs=sum(not j['success'] for j in jobs),
        unattempted_programs=manifest['programs']-len(jobs), workload_mode=mode,
        completed_calls=len(successful), failed_calls=sum(not r["success"] for r in records),
        unattempted_calls=manifest["calls"]-len(records), timed_out=timed_out,
        wall_from_clock_zero_s=wall, observation_from_first_actual_arrival_s=observation,
        jct_s=stats([j["jct_s"] for j in good_jobs]), arrival_lag_s=stats([j["arrival_lag_s"] for j in jobs]),
        request_ttft_s=stats([r["ttft_s"] for r in successful]), request_e2e_s=stats([r["e2e_s"] for r in successful]),
        programs_per_s=len(good_jobs)/observation if observation else None,
        output_tokens_per_s=sum(r["output_tokens"] for r in successful)/observation if observation else None,
        metric_notes=["Program JCT includes tool wait and queueing; starts at actual arrival",
                      "Throughput window includes tool idle and drain, excludes initial pre-arrival idle",
                      "ITL samples use adjacent single-token chunks only; chunk sizes retained",
                      "No cache-hit/transfer metrics are inferred by the client; unavailable values remain absent",
                      "GPU validation requires independent model, engine and hardware evidence"])
    write_json(out/"jobs.json", sorted(jobs, key=lambda j:j["program_id"]))
    write_json(out/'concurrency.json', concurrency(jobs, arrival.get('active_programs')))
    write_json(out/"summary.json", summary)
    state.update(status=summary["status"], summary=summary)
    write_json(out/"state.json", state)
    if not strict: raise RuntimeError(f"Replay incomplete; see {out/'summary.json'}")
    return summary
