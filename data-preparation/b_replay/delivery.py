"""Package and verify the assignment's code, data, report and evidence."""
from __future__ import annotations
import json
import zipfile
from pathlib import Path
from .common import ROOT, read_json, write_json, sha

def gpu_evidence():
    path=ROOT/"verification/gpu_verification.json"
    if not path.exists():return None
    proof=read_json(path)
    expected={1:(10,2437),4:(44,7311),14:(161,24182)}
    rows=proof.get("runs",[])
    if (proof.get("status")!="passed" or len(rows)!=3
        or {r.get("programs") for r in rows}!=set(expected)
        or any(not r.get("strict_complete") or not r.get("fixed_inputs_and_output_counts_verified")
               or (r.get("calls"),r.get("output_tokens"))!=expected[r["programs"]] for r in rows)):
        raise ValueError("GPU evidence must cover complete 1/4/14 pilot runs")
    return proof

def excluded(path):
    return (path.parts[0] in {"runs","tmp",".venv",".git",".ssh"}
        or any(p in {"__pycache__",".pytest_cache",".DS_Store"} for p in path.parts)
        or path.name.endswith((".partial",".pyc",".bak")))

def release_files():
    return [p for p in sorted(ROOT.rglob("*")) if p.is_file()
        and not excluded(p.relative_to(ROOT)) and p!=ROOT/"release_manifest.json"]

def verify_release():
    manifest=read_json(ROOT/"release_manifest.json")
    for row in manifest["files"]:
        path=ROOT/row["path"]
        if not path.resolve().is_relative_to(ROOT.resolve()) or path.stat().st_size!=row["bytes"] or sha(path.read_bytes())!=row["sha256"]:
            raise ValueError("Submission checksum mismatch: "+row["path"])
    return dict(valid=True,files=len(manifest["files"]))

def package():
    proof=gpu_evidence()
    files=release_files()
    manifest=dict(title="数据分析与Replay数据准备",author="杨智涵",student_id="redacted",
        gpu_input_replay_verified=bool(proof),performance_claim_supported=False,
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in files])
    write_json(ROOT/"release_manifest.json",manifest)
    target=ROOT.parent/(ROOT.name+".zip")
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files+[ROOT/"release_manifest.json"]:
            info=zipfile.ZipInfo(str(Path(ROOT.name)/path.relative_to(ROOT)),date_time=(2026,11,3,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            archive.writestr(info,path.read_bytes())
    checksum=sha(target.read_bytes())
    target.with_suffix(".zip.sha256").write_text(checksum+"  "+target.name+"\n")
    print(json.dumps(dict(path=str(target),bytes=target.stat().st_size,sha256=checksum,
                         verification=verify_release()),ensure_ascii=False,indent=2))
