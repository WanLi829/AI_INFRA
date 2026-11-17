"""Recompute A's source checks; reads JSON as data, never executes tool commands."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sources"
OUT = ROOT / "audit"


def stats(values):
    v = sorted(values)
    def q(p):
        x = (len(v)-1)*p
        lo = math.floor(x)
        return v[lo] + (v[math.ceil(x)]-v[lo])*(x-lo)
    return dict(n=len(v), mean=statistics.mean(v), std_population=statistics.pstdev(v),
                min=v[0], p50=q(.5), p95=q(.95), max=v[-1])


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    trace = json.loads((SRC / "Kairos/data/workload/trace.json").read_text())
    rows, programs, raw_gaps, replay_gaps, outputs, contexts, prompts, keeps = [], [], [], [], [], [], [], []
    accepted = []
    for p in trace["programs"]:
        path = SRC / "teacher_samples" / (p["id"] + ".traj.json")
        raw = json.loads(path.read_text())
        messages = raw["messages"]
        assistants = [(i, m) for i, m in enumerate(messages) if m["role"] == "assistant"]
        assert len(assistants) == len(p["turns"])
        exit_status = raw["info"].get("exit_status")
        complete = exit_status == "Submitted"
        if complete:
            accepted.append(p["id"])
        reconstructed = []
        for j, turn in enumerate(p["turns"]):
            assert 0 <= turn["keep_prefix_tokens"] <= len(reconstructed)
            reconstructed = reconstructed[:turn["keep_prefix_tokens"]] + turn["added_token_ids"]
            assert len(reconstructed) == turn["prompt_tokens"]
            original = turn["original_round"]
            i, assistant = assistants[original]
            stamp = assistant["extra"]["timestamp"]
            assert abs(stamp-turn["original_assistant_timestamp"]) < 1e-5
            assert turn["output_tokens"] > 0
            outputs.append(turn["output_tokens"])
            prompts.append(turn["prompt_tokens"])
            keeps.append(turn["keep_prefix_tokens"])
            contexts.append(turn["prompt_tokens"] + turn["output_tokens"])
            if j == len(p["turns"])-1:
                continue
            end = assistants[original+1][0]
            tools = [m for m in messages[i+1:end] if m["role"] == "tool"]
            assert tools, (p["id"], j, "missing tool feedback")
            gap = max(m["extra"]["timestamp"] for m in tools)-stamp
            assert gap >= 0
            raw_gaps.append(gap)
            replay_gaps.append(turn["gap_after_s"])
            rows.append(dict(program_id=p["id"], turn=j, raw_feedback_gap_s=gap,
                             kairos_gap_s=turn["gap_after_s"],
                             changed=abs(gap-turn["gap_after_s"]) > 1e-5,
                             raw_exit_status=exit_status))
        programs.append(dict(program_id=p["id"], turns=len(p["turns"]),
                             raw_exit_status=exit_status, complete_by_exit_status=complete,
                             final_prompt_tokens=p["turns"][-1]["prompt_tokens"],
                             final_prompt_plus_output_tokens=p["turns"][-1]["prompt_tokens"]+p["turns"][-1]["output_tokens"],
                             source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    bfcl_path = SRC / "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_web_search.json"
    bfcl = [json.loads(s) for s in bfcl_path.read_text().splitlines() if s.strip()]
    models = {}
    for model in ("Qwen3-4B", "Qwen3-8B"):
        c = json.loads((SRC / (model+"_config.json")).read_text())
        per_token = 2*c["num_hidden_layers"]*c["num_key_value_heads"]*c["head_dim"]*2
        models[model] = dict(kv_bf16_bytes_per_token=per_token,
                             kv_8192_tokens_GiB=per_token*8192/2**30,
                             kv_16384_tokens_GiB=per_token*16384/2**30)
    result = dict(
        status="CPU source audit only; no GPU experiments",
        programs=len(programs), calls=len(outputs), raw_tool_intervals=stats(raw_gaps),
        kairos_tool_intervals=stats(replay_gaps), changed_intervals=sum(r["changed"] for r in rows),
        raw_exit_status_counts={s: sum(p["raw_exit_status"] == s for p in programs)
                                for s in sorted(set(p["raw_exit_status"] for p in programs))},
        submitted_programs=len(accepted),
        submitted_calls=sum(p["turns"] for p in programs if p["complete_by_exit_status"]),
        output_budget_tokens=stats(outputs), max_prompt_plus_output=max(contexts),
        prompt_tokens=stats(prompts),
        final_prompt_tokens=stats([p["final_prompt_tokens"] for p in programs]),
        final_prompt_plus_output_tokens=stats([p["final_prompt_plus_output_tokens"] for p in programs]),
        declared_reusable_prompt_tokens=sum(keeps),
        total_prompt_tokens=sum(prompts),
        declared_prefix_reuse_fraction=sum(keeps)/sum(prompts),
        bfcl_web_search_questions=len(bfcl), bfcl_first_row_keys=list(bfcl[0]),
        bfcl_contains_recorded_tool_timestamps=any("timestamp" in r for r in bfcl),
        models=models,
        limitations=["16 preselected Kairos IDs; not a random sample of 499 source files",
                     "Submitted is completion status, not SWE-bench correctness",
                     "tool gap is observed feedback timestamp minus assistant completion timestamp; includes client/sandbox overhead",
                     "No current-model tokenizer verification was run",
                     "BFCL check covers the downloaded question file, not every upstream result directory"])
    (OUT / "source_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    (OUT / "submitted_candidate_ids.json").write_text(json.dumps(accepted, ensure_ascii=False, indent=2)+"\n")
    write_csv("programs.csv", programs)
    write_csv("tool_gap_comparison.csv", rows)
    manifest = []
    for p in sorted(SRC.rglob("*")):
        if p.is_file():
            manifest.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                 sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (OUT / "source_checksums.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
