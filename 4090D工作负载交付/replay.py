"""Plan or launch one fixed pilot_14 run. The existing server environment stays in place."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
REV = "b968826d9c46dd6066d109eabc6255188de91218"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("jps", choices=["0.05", "0.1", "0.2"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", type=Path, default=PROJECT,
                        help="Existing B project next to its Linux env/")
    args = parser.parse_args()
    project = args.project.resolve()
    out = args.output.resolve()
    if out.exists():
        parser.error("Choose a new output directory.")
    # Bind the adapter and the frozen runtime manifest to this delivery.
    for src in [*sorted((PROJECT / "b_replay").glob("*.py")),
                PROJECT / "scripts/gpu_python.py", PROJECT / "runtime/Kairos/provenance/release_manifest.json"]:
        candidate = project / src.relative_to(PROJECT)
        if not candidate.is_file() or candidate.read_bytes() != src.read_bytes():
            parser.error(f"Source mismatch: {candidate}")
    bundle = PROJECT / "bundles/Qwen3-8B/pilot_14"
    arrivals = PROJECT / "arrivals" / f"pilot_14_jps{args.jps}_seed20261030.json"
    expected = json.loads((ROOT / "input_hashes.json").read_text(encoding="utf-8"))
    for file in [arrivals, bundle / "manifest.json"]:
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected[file.name]:
            parser.error(f"Input mismatch: {file.name}")
    interpreter = [sys.executable, "-B", "-X", "utf8"]
    model = args.model or Path("/models/snapshots") / REV
    if args.action == "run":
        if sys.platform != "linux" or not args.model or not model.is_dir():
            parser.error("GPU run requires Linux and --model pointing to the pinned snapshot.")
        if len(os.fsencode(out / "control-xxxxxxxx/control.sock")) >= 108:
            parser.error("Choose a shorter output path for the Unix socket.")
        interpreter = [str(project.parent / "env/bin/python"), str(project / "scripts/gpu_python.py")]
        if not Path(interpreter[0]).is_file():
            parser.error("Missing B's private env/bin/python next to --project.")
    subprocess.run([*interpreter, "-m", "b_replay", "server-" + args.action,
                    "--server-profile", "ram30", "--gpu-pid-policy", "strict",
                    "--method", "kairos", "--bundle", str(bundle), "--arrivals", str(arrivals),
                    "--model", str(model.resolve()), "--output", str(out)], cwd=project, check=True)


if __name__ == "__main__":
    main()
