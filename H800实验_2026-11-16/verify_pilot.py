"""Recheck the two archived pilot runs on CPU using the project validator."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "4090D工作负载交付/project"
sys.path.insert(0, str(PROJECT))

from b_replay.comparison_results import inspect


def main():
    reference = json.loads((HERE / "H800_pilot_comparison.json").read_text(encoding="utf-8"))
    bundle = PROJECT / "bundles/Qwen3-8B/pilot_14"
    runs = {}
    identities = set()
    for method, expected in reference["runs"].items():
        filename = {"native": "原生vLLM_原始记录.tar.gz", "continuum": "Continuum_原始记录.tar.gz"}[method]
        archive = HERE / filename
        if hashlib.sha256(archive.read_bytes()).hexdigest() != reference["source_sha256"][filename]:
            raise ValueError(f"Archive checksum mismatch: {filename}")
        with TemporaryDirectory() as temp, tarfile.open(archive, "r:gz") as tar:
            tar.extractall(temp, filter="data")
            row, identity = inspect(Path(temp) / expected["run"], bundle)
        for field in ("method", "status", "completed_programs", "completed_calls", "output_tokens",
                      "speedup_eligible", "jct_mean_s", "jct_p95_s", "output_tokens_per_s", "ttft_p95_s"):
            if row[field] != expected[field]:
                raise ValueError(f"{method}: mismatch in {field}")
        if identity is None:
            raise ValueError(f"No complete run for {method}")
        identities.add(identity)
        runs[method] = row
        print(f"{method}: {row['completed_programs']} programs, {row['completed_calls']} calls, "
              f"mean JCT {row['jct_mean_s']:.4f} s")
    ratio = runs["native"]["jct_mean_s"] / runs["continuum"]["jct_mean_s"]
    if len(identities) != 1 or not math.isclose(ratio, reference["observed_mean_jct_ratio"]):
        raise ValueError("Pilot identity or comparison mismatch")
    print(f"Descriptive mean JCT ratio: {ratio:.4f} (single run, GPU ownership unverified)")


if __name__ == "__main__":
    main()
