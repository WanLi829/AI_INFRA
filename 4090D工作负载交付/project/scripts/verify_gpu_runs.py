"""Verify completed pilot output IDs, fixed inputs, dependencies and resource records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def verify(run, project):
    state = read(run / "state.json")
    summary = read(run / "b_replay/summary.json")
    manifest = read(run / "b_input_provenance.json")["manifest"]
    bundle = project / "bundles/Qwen3-8B" / ("pilot_" + str(manifest["programs"]))
    trace = read(bundle / "trace.json")
    expected = {(p["id"], i): t for p in trace["programs"] for i, t in enumerate(p["turns"])}
    assert state["status"] == "complete" and summary["strict_complete"] is True
    assert state["verified_model"]["weights_verified"] is True and state["gpu"]
    assert state["trace_sha256"] == hashlib.sha256((bundle / "trace.json").read_bytes()).hexdigest()
    records = [json.loads(x) for x in (run / "b_replay/requests.jsonl").read_text().splitlines()]
    actual = {(r["program_id"], r["turn_id"]): r for r in records}
    assert len(records) == len(actual) == len(expected) == manifest["calls"]
    assert actual.keys() == expected.keys()
    for key, record in actual.items():
        teacher = expected[key]
        assert record["success"] and record["error"] is None
        assert record["prompt_sha256"] == teacher["prompt_sha256"]
        assert record["prompt_tokens"] == teacher["prompt_tokens"]
        assert len(record["output_ids"]) == record["output_tokens"] == teacher["output_tokens"]
        assert record["planned_output_tokens"] == teacher["output_tokens"]
        assert record["cont_tool_sent"] == "unclassified-gap"
        if key[1]:
            previous = actual[(key[0], key[1] - 1)]
            assert record["start_s"] + .002 >= previous["end_s"] + previous["gap_after_s"]
    fixtures = read(run / "gapped-correctness.json")
    assert len(fixtures) == 5 and all(x["same_greedy_tokens"] for x in fixtures)
    samples = [x for x in read(run / "resources.json") if "memory" in x]
    assert samples
    limit = state["memory_before_start"]["effective_limit_bytes"]
    charges = [g["current_bytes"] for x in samples for g in x["memory"]["cgroups"]]
    return dict(run=str(run), programs=manifest["programs"], calls=len(records),
                output_tokens=sum(r["output_tokens"] for r in records),
                strict_complete=True, fixed_inputs_and_output_counts_verified=True,
                continuation_waits_verified=True, correctness_layouts=len(fixtures),
                correctness_mode=state.get("correctness_mode"), gpu=state["gpu"],
                container_limit_bytes=limit, sampled_peak_cgroup_bytes=max(charges),
                minimum_available_ram_bytes=min(x["available_ram_bytes"] for x in samples),
                sampled_peak_gpu_mib=max(int(x["gpu"]["used"]) for x in samples),
                gpu_pid_policy=state.get("gpu_pid_policy", "strict"),
                gpu_ownership_verified=state.get("gpu_ownership_verified"),
                packages=state["packages"], trace_sha256=state["trace_sha256"],
                arrival_manifest_sha256=state["b_arrival_manifest_sha256"],
                performance_claim_supported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [verify(path.resolve(), args.project.resolve()) for path in args.run]
    report = dict(status="passed", gpu_input_replay_validated=True, runs=results,
                  total_recorded_calls=sum(x["calls"] for x in results),
                  total_output_tokens=sum(x["output_tokens"] for x in results),
                  performance_claim_supported=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
