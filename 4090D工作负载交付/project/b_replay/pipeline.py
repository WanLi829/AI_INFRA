from __future__ import annotations
from collections import Counter
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
from .common import ROOT, DATASET, REVISION, KAIROS, MODELS, sha, json_bytes, read_json, write_json, write_csv, stats, lcp

MESSAGE_FIELDS = ("role", "content", "tool_calls", "tool_call_id", "name")
ALLOWED_ROLES = {"system", "user", "assistant", "tool", "exit"}

def model_message(message):
    return {key: message[key] for key in MESSAGE_FIELDS if key in message}

def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def tool_label(assistant):
    # Same symbolic command labels as Kairos; strings are never executed.
    commands = [a.get("command", "") for a in assistant.get("extra", {}).get("actions", [])]
    if not commands:
        for call in assistant.get("tool_calls") or []:
            try:
                arguments = call.get("function", {}).get("arguments", "{}")
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                commands.append(arguments.get("command", ""))
            except (ValueError, TypeError, AttributeError):
                continue
    if not commands:
        commands = re.findall(r"```bash\s*\n(.*?)\n```", assistant.get("content") or "", re.S)
    return "+".join(sorted({(c.strip().split() or ["unknown"])[0] for c in commands if isinstance(c,str)})) or "no-action"

def split_rounds(document):
    messages = document.get("messages")
    if not isinstance(messages, list) or not messages:
        return [], ["missing_messages"]
    issues = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ALLOWED_ROLES:
            issues.append(f"invalid_role:{index}")
    if issues:
        return [], issues
    indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not indices:
        return [], ["no_assistant_rounds"]
    result = []
    for turn, index in enumerate(indices):
        assistant = messages[index]
        end = indices[turn + 1] if turn + 1 < len(indices) else len(messages)
        feedback = [m for m in messages[index + 1:end] if m["role"] != "exit"]
        tools = [m for m in feedback if m["role"] == "tool"]
        timestamp = assistant.get("extra", {}).get("timestamp")
        if not finite_number(timestamp):
            issues.append(f"invalid_assistant_timestamp:{turn}")
        call_ids = [call.get("id") for call in (assistant.get("tool_calls") or [])]
        if len(call_ids) != len(set(call_ids)) or any(value is None for value in call_ids):
            issues.append(f"invalid_tool_call_ids:{turn}")
        observed_ids = [m.get("tool_call_id") for m in tools]
        if tools and (not call_ids or Counter(call_ids) != Counter(observed_ids)):
            issues.append(f"unmatched_tool_feedback:{turn}")
        gap = None
        if turn + 1 < len(indices):
            stamps = [m.get("extra", {}).get("timestamp") for m in tools]
            if not stamps or not all(finite_number(v) for v in stamps) or not finite_number(timestamp):
                issues.append(f"missing_tool_interval:{turn}")
            elif stamps != sorted(stamps) or stamps[0] < timestamp:
                issues.append(f"invalid_tool_interval:{turn}")
            else:
                gap = stamps[-1] - timestamp
                next_stamp = messages[indices[turn + 1]].get("extra", {}).get("timestamp")
                if finite_number(next_stamp) and next_stamp < stamps[-1]:
                    issues.append(f"nonmonotonic_next_assistant:{turn}")
        response = assistant.get("extra", {}).get("response") or {}
        usage = response.get("usage") if isinstance(response, dict) else None
        result.append(dict(turn_id=turn, message_index=index, assistant_timestamp=timestamp,
                           feedback_end_index=end, feedback_count=len(feedback), tool_count=len(tools),
                           gap_after_s=gap, gap_origin="observed_feedback" if gap is not None else
                           ("terminal_no_wait" if turn + 1 == len(indices) else "missing"),
                           tool=tool_label(assistant), original_usage=usage,
                           feedback_returncodes=[m.get("extra", {}).get("returncode") for m in tools]))
    return result, sorted(set(issues))

def sources():
    return sorted((ROOT / "sources/teacher/trajectories").glob("*.traj.json"))

def split_map():
    data = read_json(ROOT / "sources/kairos/data/training/causal_states.json.gz")
    assert len(data["programs"]) == 499
    return {p["id"]: p for p in data["programs"]}

def audit():
    splits = split_map()
    files = sources()
    if len(files) != 499:
        raise ValueError(f"Expected 499 raw trajectories; got {len(files)}")
    programs, turns, quarantined = [], [], []
    seen = set()
    for path in files:
        document = read_json(path)
        pid = document.get("instance_id")
        rows, issues = split_rounds(document)
        if pid != path.name.removesuffix(".traj.json") or pid in seen or pid not in splits:
            issues.append("identity_or_split_mismatch")
        seen.add(pid)
        metadata = splits.get(pid, {})
        if metadata.get("source_sha256") != sha(path.read_bytes()):
            issues.append("kairos_raw_source_hash_mismatch")
        status = document.get("info", {}).get("exit_status")
        if not isinstance(status, str):
            issues.append("missing_exit_status")
        declared_calls = document.get("info", {}).get("model_stats", {}).get("api_calls")
        format_errors = sum(m.get("extra", {}).get("interrupt_type")=="FormatError"
                            for m in document.get("messages", []))
        # API attempts may include rejected/format-error calls absent from assistant messages.
        row = dict(program_id=pid, source_path=str(path.relative_to(ROOT)), source_sha256=sha(path.read_bytes()),
                   split=metadata.get("split"), exit_status=status, stored_calls=len(rows),
                   declared_api_calls=declared_calls, message_count=len(document.get("messages", [])),
                   api_calls_minus_stored=declared_calls-len(rows) if isinstance(declared_calls, int) else None,
                   format_error_feedbacks=format_errors,
                   structure_valid=not issues, issues=";".join(sorted(set(issues))))
        programs.append(row)
        if issues:
            quarantined.append(dict(program_id=pid, reasons=sorted(set(issues))))
        # Boundaries refer to the original list, so retain that list's indices here.
        write_json(ROOT / f"intermediate/{pid}.json.gz", dict(
            program_id=pid, teacher_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            source_revision=REVISION, exit_status=status, split=metadata.get("split"),
            source_path=row["source_path"], source_sha256=row["source_sha256"],
            api_attempts=declared_calls, unrecorded_api_attempts=row["api_calls_minus_stored"],
            format_error_feedbacks=format_errors, coverage="recorded_assistant_calls_only",
            messages=[model_message(m) if m["role"] != "exit" else {"role": "exit"}
                      for m in document.get("messages", [])],
            turns=rows, issues=sorted(set(issues))), pretty=False)
        for item in rows:
            turns.append(dict(program_id=pid, **item))
    gaps = [t["gap_after_s"] for t in turns if t["gap_after_s"] is not None]
    summary = dict(source_programs=len(programs), stored_calls=len(turns),
                   exit_status_counts=dict(Counter(p["exit_status"] for p in programs)),
                   split_counts=dict(Counter(p["split"] for p in programs)),
                   structurally_valid=sum(p["structure_valid"] for p in programs),
                   quarantined=len(quarantined), rounds=stats([p["stored_calls"] for p in programs]),
                   raw_feedback_gaps_s=stats(gaps),
                   missing_usage_rounds=sum(t["original_usage"] is None for t in turns),
                   observed_api_attempt_mismatches=sum(p["api_calls_minus_stored"] not in (None,0) for p in programs),
                   missing_api_attempt_counts=sum(p["api_calls_minus_stored"] is None for p in programs),
                   api_attempts=sum(p["declared_api_calls"] or 0 for p in programs),
                   unrecorded_api_attempts=sum(p["api_calls_minus_stored"] or 0 for p in programs),
                   format_error_feedbacks=sum(p["format_error_feedbacks"] for p in programs),
                   attempt_delta_matches_format_errors=all(p["api_calls_minus_stored"]==p["format_error_feedbacks"] for p in programs),
                   grain="One program is one source trajectory; one stored call is one assistant message",
                   notes=["Submitted is termination status, not correctness",
                          "Feedback intervals include sandbox/client overhead",
                          "All 499 source hashes are checked against the pinned Kairos metadata",
                          "Existing blind split has already been used for offline evaluation"])
    write_csv(ROOT / "audit/raw_programs.csv", programs)
    write_json(ROOT / "audit/raw_turns.json.gz", turns, pretty=False)
    write_json(ROOT / "audit/quarantine.json", quarantined)
    write_json(ROOT / "audit/raw_summary.json", summary)
    write_json(ROOT / "audit/splits.json", {p["program_id"]: p["split"] for p in programs})
    audit_a()
    audit_bfcl()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

def audit_a():
    trace = read_json(ROOT / "sources/a_handoff/sources/Kairos/data/workload/trace.json")
    comparison, statuses, raw_gaps, published_gaps = [], Counter(), [], []
    for program in trace["programs"]:
        pid = program["id"]
        document = read_json(ROOT / f"sources/teacher/trajectories/{pid}.traj.json")
        raw_turns, _ = split_rounds(document)
        statuses[document["info"]["exit_status"]] += 1
        assert len(raw_turns) == len(program["turns"])
        prompt = []
        for row, turn in zip(raw_turns, program["turns"]):
            prompt = prompt[:turn["keep_prefix_tokens"]] + turn["added_token_ids"]
            assert len(prompt) == turn["prompt_tokens"]
            if row["gap_after_s"] is None:
                continue
            raw_gaps.append(row["gap_after_s"])
            published_gaps.append(turn["gap_after_s"])
            comparison.append(dict(program_id=pid, turn_id=row["turn_id"], raw_gap_s=row["gap_after_s"],
                                   published_gap_s=turn["gap_after_s"],
                                   changed=abs(row["gap_after_s"] - turn["gap_after_s"]) > 1e-5))
    write_csv(ROOT / "audit/a_gap_comparison.csv", comparison)
    write_json(ROOT / "audit/a_recomputation.json", dict(programs=len(trace["programs"]),
        calls=sum(len(p["turns"]) for p in trace["programs"]), exit_status_counts=dict(statuses),
        raw_gaps_s=stats(raw_gaps), published_gaps_s=stats(published_gaps),
        changed_intervals=sum(r["changed"] for r in comparison),
        changed_only_first_turn=all(r["turn_id"] == 0 for r in comparison if r["changed"])))

def audit_bfcl():
    base = ROOT / "sources/a_handoff/sources"
    path = base / "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_web_search.json"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    candidates = []
    for source in ("gorilla", "vllm-continuum"):
        tree = read_json(base / f"{source}_tree.json")
        entries = tree.get("tree", []) if isinstance(tree, dict) else tree
        for item in entries:
            name = item.get("path", "")
            if any(term in name.lower() for term in ("bfcl", "web_search", "trace")):
                candidates.append(dict(repository=source, path=name, type=item.get("type")))
    write_json(ROOT / "audit/bfcl_availability.json", dict(
        status="not_replay_ready", questions=len(rows), keys=sorted(rows[0]),
        questions_sha256=sha(path.read_bytes()), contains_recorded_timestamps=False,
        missing=["per-call visible messages", "recorded assistant outputs", "tool feedback timestamps"],
        inspected_repository_candidates=candidates,
        limitation="Bounded search and pinned repository inspection; does not establish universal absence",
        public_log_search=read_json(ROOT / "sources/public_log_search.json") if
            (ROOT / "sources/public_log_search.json").exists() else None,
        next_step="Obtain provenance-preserving execution logs or explicitly authorize new collection; do not synthesize real timings"))

def load_tokenizer(model):
    os.environ.setdefault("USE_TORCH", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT / f"sources/tokenizers/{model}", local_files_only=True,
                                         trust_remote_code=False)

def ids(tokenizer, messages, generation):
    value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation,
                                          enable_thinking=False)
    return list(value["input_ids"] if hasattr(value, "keys") else value)

def convert(model, max_length=16384):
    tokenizer = load_tokenizer(model)
    model_root = ROOT / f"encoded/{model}"
    programs, all_turns, failures = [], [], []
    intermediates = sorted((ROOT / "intermediate").glob("*.json.gz"))
    if len(intermediates) != 499:
        raise ValueError("Run full audit first")
    template = tokenizer.chat_template
    template_hash = sha((template if isinstance(template, str) else json.dumps(template, sort_keys=True)).encode())
    for count, path in enumerate(intermediates, 1):
        doc = read_json(path)
        pid = doc["program_id"]
        previous, encoded, rows = [], [], []
        errors = list(doc["issues"])
        for turn in doc["turns"]:
            index = turn["message_index"]
            visible = [m for m in doc["messages"][:index] if m["role"] != "exit"]
            assistant = doc["messages"][index]
            try:
                prompt = ids(tokenizer, visible, True)
                completed = ids(tokenizer, visible + [assistant], False)
                output_lcp = lcp(prompt, completed)
                # Mirrors the pinned Kairos template-difference budget; includes special-token suffix.
                output = len(completed) - output_lcp
                if output <= 0 or not prompt:
                    raise ValueError("Empty prompt/output template difference")
                keep = lcp(previous, prompt)
                added = prompt[keep:]
                rebuilt = previous[:keep] + added
                assert rebuilt == prompt
                prompt_hash = sha(json.dumps(prompt).encode())  # pinned Kairos request signature encoding
                usage = turn["original_usage"] or {}
                usage_details = usage.get("prompt_tokens_details") or {}
                source_cached = usage_details.get("cached_tokens")
                returncodes = turn["feedback_returncodes"]
                known_codes = all(isinstance(c, int) for c in returncodes)
                encoded.append(dict(keep_prefix_tokens=keep, added_token_ids=added,
                    output_tokens=output, prompt_tokens=len(prompt), gap_after_s=turn["gap_after_s"] or 0.0,
                    source_gap_s=turn["gap_after_s"], source_cached_tokens=source_cached,
                    source_tool_failures=sum(c != 0 for c in returncodes) if known_codes else None,
                    tool=turn["tool"], original_round=turn["turn_id"],
                    original_assistant_timestamp=turn["assistant_timestamp"],
                    feedback_messages=turn["feedback_count"], prompt_sha256=prompt_hash,
                    visible_messages_sha256=sha(json_bytes(visible, pretty=False)),
                    original_usage=turn["original_usage"],
                    gap_origin=turn["gap_origin"]))
                rows.append(dict(program_id=pid, turn_id=turn["turn_id"], prompt_tokens=len(prompt),
                    output_budget_tokens=output, prompt_plus_output=len(prompt)+output,
                    previous_prompt_lcp=keep, added_tokens=len(added),
                    context_growth=len(prompt)-len(previous), template_output_lcp=output_lcp,
                    prompt_hash=prompt_hash, raw_prompt_usage=usage.get("prompt_tokens"),
                    raw_completion_usage=usage.get("completion_tokens"),
                    gap_after_s=turn["gap_after_s"], tool_count=turn["tool_count"]))
                previous = prompt
            except Exception as exc:
                errors.append(f"tokenization_error:{turn['turn_id']}:{type(exc).__name__}:{exc}")
                break
        if doc["exit_status"] != "Submitted":
            errors.append("non_submitted:" + str(doc["exit_status"]))
        maximum = max((row["prompt_plus_output"] for row in rows), default=0)
        if maximum > max_length:
            errors.append(f"overlength:{maximum}>{max_length}")
        if len(rows) != len(doc["turns"]):
            errors.append("incomplete_conversion")
        accepted = not errors
        disposition = "accepted" if accepted else ("quarantined" if doc["issues"] or
            any(e.startswith("tokenization_error") for e in errors) else "excluded")
        program = dict(program_id=pid, split=doc["split"], exit_status=doc["exit_status"],
            stored_calls=len(doc["turns"]), converted_calls=len(rows), accepted=accepted,
            disposition=disposition,
            reasons=";".join(sorted(set(errors))), max_prompt_plus_output=maximum,
            final_prompt_tokens=rows[-1]["prompt_tokens"] if rows else None,
            total_prompt_tokens=sum(r["prompt_tokens"] for r in rows),
            total_output_budget_tokens=sum(r["output_budget_tokens"] for r in rows),
            declared_prefix_tokens=sum(r["previous_prompt_lcp"] for r in rows),
            source_sha256=doc["source_sha256"])
        programs.append(program)
        all_turns.extend(rows)
        if accepted:
            write_json(model_root / f"programs/{pid}.json.gz", dict(program_id=pid, turns=encoded), pretty=False)
        elif errors:
            failures.append(dict(program_id=pid, reasons=sorted(set(errors))))
        if count % 25 == 0:
            print(f"{model}: converted {count}/499; accepted {sum(p['accepted'] for p in programs)}", flush=True)
    packages = {name: importlib.metadata.version(name) for name in
                ("transformers", "tokenizers", "jinja2", "httpx")}
    metadata = dict(model="Qwen/"+model, revision=MODELS[model], chat_template_sha256=template_hash,
        enable_thinking=False, add_generation_prompt=True, max_sequence_tokens=max_length,
        output_budget_rule="len(completed_template_ids) - LCP(prompt_ids, completed_template_ids); includes template special tokens",
        python=platform.python_version(), packages=packages,
        tokenizer_files={p.name: sha(p.read_bytes()) for p in sorted((ROOT / f"sources/tokenizers/{model}").iterdir())
                         if p.is_file() and p.name != "upstream_metadata.json"})
    accepted = [p for p in programs if p["accepted"]]
    summary = dict(source_programs=len(programs), converted_calls=len(all_turns),
        accepted_programs=len(accepted), accepted_calls=sum(p["stored_calls"] for p in accepted),
        rejected_total=len(failures),
        quarantined_programs=sum(p["disposition"]=="quarantined" for p in programs),
        excluded_programs=sum(p["disposition"]=="excluded" for p in programs),
        accepted_by_split=dict(Counter(p["split"] for p in accepted)),
        exclusion_reasons=dict(Counter(reason.split(":")[0] for f in failures for reason in f["reasons"])),
        all_prompt_tokens=stats([r["prompt_tokens"] for r in all_turns]),
        all_output_budget_tokens=stats([r["output_budget_tokens"] for r in all_turns]),
        accepted_prompt_tokens=stats([r["prompt_tokens"] for r in all_turns if
            next(p["accepted"] for p in programs if p["program_id"] == r["program_id"])]),
        accepted_rounds=stats([p["stored_calls"] for p in accepted]),
        accepted_final_context_tokens=stats([p["final_prompt_tokens"] for p in accepted]),
        declared_prefix_fraction=sum(p["declared_prefix_tokens"] for p in accepted) /
            max(1, sum(p["total_prompt_tokens"] for p in accepted)),
        prefix_metric_note="Identical input history between teacher prompts; not measured GPU cache hits")
    write_json(model_root / "tokenizer_manifest.json", metadata)
    write_json(model_root / "programs.json", programs)
    write_csv(model_root / "programs.csv", programs)
    write_json(model_root / "turns.json.gz", all_turns, pretty=False)
    write_csv(model_root / "turns.csv", all_turns)
    write_json(model_root / "exclusions.json", failures)
    write_json(model_root / "summary.json", summary)
    refresh_summaries(model)
    verify_a_tokens(model, tokenizer)
    print(json.dumps(summary, indent=2), flush=True)

def refresh_summaries(model):
    base=ROOT/f"encoded/{model}"
    summary=read_json(base/"summary.json")
    programs=read_json(base/"programs.json")
    turns=read_json(base/"turns.json.gz")
    accepted={p["program_id"] for p in programs if p["accepted"]}
    selected=[r for r in turns if r["program_id"] in accepted]
    summary.update(accepted_output_budget_tokens=stats([r["output_budget_tokens"] for r in selected]),
        accepted_context_growth_tokens=stats([r["context_growth"] for r in selected if r["turn_id"]>0]),
        accepted_feedback_gaps_s=stats([r["gap_after_s"] for r in selected]),
        accepted_by_split_details={split:dict(
            programs=sum(p["program_id"] in accepted and p["split"]==split for p in programs),
            calls=sum(p["stored_calls"] for p in programs if p["program_id"] in accepted and p["split"]==split))
            for split in ("train","development","blind")},
        coverage="recorded_assistant_calls_only")
    assert summary["accepted_programs"]+summary["excluded_programs"]+summary["quarantined_programs"]==499
    write_json(base/"summary.json",summary)

def verify_a_tokens(model, tokenizer):
    trace = read_json(ROOT / "sources/a_handoff/sources/Kairos/data/workload/trace.json")
    mismatches, checked = [], 0
    for program in trace["programs"]:
        doc = read_json(ROOT / f"intermediate/{program['id']}.json.gz")
        previous = []
        for row, historical in zip(doc["turns"], program["turns"]):
            previous = previous[:historical["keep_prefix_tokens"]] + historical["added_token_ids"]
            visible = [m for m in doc["messages"][:row["message_index"]] if m["role"] != "exit"]
            prompt = ids(tokenizer, visible, True)
            completed = ids(tokenizer, visible+[doc["messages"][row["message_index"]]], False)
            budget = len(completed)-lcp(prompt, completed)
            checked += 1
            if prompt != previous or budget != historical["output_tokens"]:
                mismatches.append(dict(program_id=program["id"], turn_id=row["turn_id"],
                    prompt_equal=prompt == previous, output_budget_equal=budget == historical["output_tokens"]))
    write_json(ROOT / f"encoded/{model}/a_token_verification.json", dict(
        checked=checked, mismatches=mismatches, identical=not mismatches,
        note="Independent re-tokenization versus archived 4B-encoded prompts; no 4B/8B identity assumed"))

def export(model):
    base = ROOT / f"encoded/{model}"
    programs = read_json(base / "programs.json")
    accepted = {p["program_id"]: p for p in programs if p["accepted"]}
    pilot = sorted(read_json(ROOT / "sources/a_handoff/audit/submitted_candidate_ids.json"))
    if not set(pilot) <= set(accepted):
        raise ValueError("Pilot candidates failed conversion; inspect exclusions instead of silently shrinking")
    subsets = {"pilot_1": pilot[:1], "pilot_4": pilot[:4], "pilot_14": pilot,
               "all_eligible": sorted(accepted),
               "heldout_candidates": sorted(pid for pid,p in accepted.items() if p["split"] == "blind")}
    catalog = []
    for name, selected in subsets.items():
        path = ROOT / f"bundles/{model}/{name}"
        rows, visible, signatures = [], {}, []
        for pid in selected:
            doc = read_json(ROOT / f"intermediate/{pid}.json.gz")
            encoded = read_json(base / f"programs/{pid}.json.gz")
            turns = encoded["turns"]
            visible[pid] = [[m for m in doc["messages"][:turn["message_index"]] if m["role"] != "exit"]
                            for turn in doc["turns"]]
            rows.append(dict(id=pid, source_instance_id=pid, append_generated_outputs=False,
                session_id="b-replay:"+pid, synthetic_user="b-replay:"+pid,
                source_turn_id=pid, source_turn_index=0, cohort=name, cache_salt="b-replay:"+pid,
                human_gap_original_s=0.0, human_gap_replay_s=0.0, predecessor=None,
                split=doc["split"], exit_status=doc["exit_status"], turns=turns,
                source_api_attempts=doc.get("api_attempts"),
                unrecorded_api_attempts=doc.get("unrecorded_api_attempts"),
                coverage="recorded_assistant_calls_only"))
            signatures.extend(dict(program_id=pid, turn_id=i, prompt_sha256=t["prompt_sha256"],
                                   prompt_tokens=t["prompt_tokens"], output_tokens=t["output_tokens"])
                              for i,t in enumerate(turns))
        trace = dict(schema_version="b-replay/1.0", source_dataset=DATASET, source_revision=REVISION,
            kairos_interface_commit=KAIROS, model=read_json(base / "tokenizer_manifest.json"),
            arrival_policy="External manifest required; program-only Poisson, local completion plus feedback gap",
            output_policy="Fixed template-difference teacher budget; ignore_eos; generated text not appended",
            coverage="Recorded assistant calls only; omitted FormatError API attempts are not reconstructed",
            tools_executed=False, programs=rows)
        write_json(path / "trace.json", trace, pretty=False)
        write_json(path / "visible_messages.json", visible, pretty=False)
        write_json(path / "visible_protocol.json", dict(request_signatures=signatures,
            current_assistant_included=False, future_outcomes_included=False,
            trace_sha256=sha((path / "trace.json").read_bytes()),
            visible_messages_sha256=sha((path / "visible_messages.json").read_bytes())))
        manifest = dict(schema_version="b-replay-bundle/1.0", subset=name, model=model,
            model_revision=MODELS[model], program_ids=selected, programs=len(rows),
            calls=sum(len(p["turns"]) for p in rows),
            total_output_tokens=sum(t["output_tokens"] for p in rows for t in p["turns"]),
            files={p.name: sha(p.read_bytes()) for p in sorted(path.iterdir()) if p.name != "manifest.json"},
            intended_use="Pilot/development" if name.startswith("pilot") else
                ("Disjoint from existing train/development; previously used in offline evaluation" if
                 name == "heldout_candidates" else "Data inventory and stress preparation; contains training programs"),
            gpu_validated=False)
        manifest["coverage"] = "recorded_assistant_calls_only"
        manifest["unrecorded_api_attempts"] = sum(p.get("unrecorded_api_attempts") or 0 for p in rows)
        write_json(path / "manifest.json", manifest)
        catalog.append({k:v for k,v in manifest.items() if k != "files"})
    write_json(ROOT / f"bundles/{model}/catalog.json", catalog)
    print(json.dumps([{k:c[k] for k in ("model","subset","programs","calls","total_output_tokens")}
                      for c in catalog], ensure_ascii=False, indent=2), flush=True)

def validate_bundle(bundle, tokenizer=None):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha((bundle / name).read_bytes()) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    trace, visible = read_json(bundle / "trace.json"), read_json(bundle / "visible_messages.json")
    calls = 0
    if sorted(visible) != sorted(manifest["program_ids"]):
        raise ValueError("Visible sidecar identity mismatch")
    for program in trace["programs"]:
        pid, prompt = program["id"], []
        if len(visible[pid]) != len(program["turns"]):
            raise ValueError("Turn count mismatch")
        doc = read_json(ROOT / f"intermediate/{pid}.json.gz")
        for i, turn in enumerate(program["turns"]):
            keep = turn["keep_prefix_tokens"]
            if not 0 <= keep <= len(prompt): raise ValueError("Invalid prefix length")
            prompt = prompt[:keep] + turn["added_token_ids"]
            if len(prompt) != turn["prompt_tokens"] or sha(json.dumps(prompt).encode()) != turn["prompt_sha256"]:
                raise ValueError("Prompt reconstruction mismatch")
            expected = [m for m in doc["messages"][:doc["turns"][i]["message_index"]] if m["role"] != "exit"]
            if visible[pid][i] != expected: raise ValueError("Visible boundary leakage")
            if tokenizer is not None and ids(tokenizer, expected, True) != prompt:
                raise ValueError("Tokenizer mismatch")
            if len(prompt)+turn["output_tokens"] > trace["model"]["max_sequence_tokens"]:
                raise ValueError("Overlength call")
            if i+1 < len(program["turns"]) and turn["gap_after_s"] != doc["turns"][i]["gap_after_s"]:
                raise ValueError("Altered source tool interval")
            calls += 1
    if calls != manifest["calls"] or len(trace["programs"]) != manifest["programs"]:
        raise ValueError("Bundle count mismatch")
    return dict(model=manifest["model"], subset=manifest["subset"], programs=manifest["programs"],
                calls=calls, valid=True, independently_retokenized=tokenizer is not None)

def validate():
    from .verification import verify_sources, compare_models
    verify_sources()
    compare_models()
    results = []
    for model in MODELS:
        tokenizer = load_tokenizer(model)
        for manifest in sorted((ROOT / f"bundles/{model}").glob("*/manifest.json")):
            results.append(validate_bundle(manifest.parent, tokenizer))
    if len(results) != 10: raise ValueError("Expected all ten model/subset bundles")
    write_json(ROOT / "audit/bundle_validation.json", results)
    print(json.dumps(results, indent=2), flush=True)
