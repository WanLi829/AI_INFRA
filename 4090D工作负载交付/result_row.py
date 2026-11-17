"""Print one Markdown row after checking a completed run against the frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT / "scripts"))
from verify_gpu_runs import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    state = json.loads((run / "state.json").read_bytes())
    summary = json.loads((run / "b_replay/summary.json").read_bytes())
    data = verify(run, PROJECT)
    manifest = PROJECT / "bundles/Qwen3-8B/pilot_14/manifest.json"
    expected = json.loads((ROOT / "input_hashes.json").read_bytes())
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if (state["b_bundle_manifest_sha256"] != manifest_hash or manifest_hash != expected[manifest.name]
            or state["method"] != "kairos" or state["server_profile"] != "ram30"):
        raise ValueError("Run differs from the chosen workload, method or resource profile.")
    rates = {expected[f"pilot_14_jps{rate}_seed20261030.json"]: rate for rate in ("0.05","0.1","0.2")}
    rate = rates[state["b_arrival_manifest_sha256"]]
    if (data["programs"], data["calls"], data["output_tokens"]) != (14, 161, 24182):
        raise ValueError("Unexpected program, call or output count.")
    print(f"| {run.name} | {rate} | {data['programs']}/14 | {data['calls']}/161 | "
          f"{summary['jct_s']['p95']:.2f} | {summary['request_ttft_s']['p95']:.3f} | "
          f"{summary['output_tokens_per_s']:.2f} | {data['sampled_peak_gpu_mib']} | "
          f"{data['sampled_peak_cgroup_bytes'] / 2**30:.2f} | {data['gpu_pid_policy']} |")


if __name__ == "__main__":
    main()
