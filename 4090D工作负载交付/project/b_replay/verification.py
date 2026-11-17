from __future__ import annotations
import contextlib
import io
from pathlib import Path
import tempfile
from .common import ROOT, MODELS, sha, read_json, write_json
from .acquire import blob_hash

def verify_sources():
    manifest=read_json(ROOT/"sources/download_manifest.json")
    tree=read_json(ROOT/"sources/a_handoff/sources/teacher_tree.json")
    expected={"sources/teacher/"+r["path"]:r for r in tree if r["type"]=="file"}
    raw=0
    lfs=0
    for row in manifest:
        data=(ROOT/row["path"]).read_bytes()
        if len(data)!=row["bytes"] or sha(data)!=row["sha256"]:
            raise ValueError("Source changed: "+row["path"])
        if row["path"] in expected:
            source=expected[row["path"]]
            if len(data)!=source["size"]:raise ValueError("Source size mismatch")
            if source.get("lfs"):
                assert sha(data)==source["lfs"]["oid"]
                lfs+=1
            else:
                assert blob_hash(data)==source["oid"]
            raw+=row["path"].endswith(".traj.json")
    assert raw==499
    tokenizer_files=list((ROOT/"sources/tokenizers").rglob("*"))
    assert not any(p.suffix in {".safetensors",".bin",".pt",".pth"} for p in tokenizer_files)
    result=dict(valid=True,downloaded_files=len(manifest),raw_trajectories=raw,
                lfs_files_checked=lfs,weights_present=False,network_used_for_verification=False)
    write_json(ROOT/"verification/source_integrity.json",result)
    return result

def compare_models():
    data={m:read_json(ROOT/f"encoded/{m}/turns.json.gz") for m in MODELS}
    columns=("prompt_hash","prompt_tokens","output_budget_tokens","previous_prompt_lcp","gap_after_s")
    indexed={m:{(r["program_id"],r["turn_id"]):tuple(r[k] for k in columns) for r in rows}
             for m,rows in data.items()}
    a,b=MODELS
    mismatches=[list(k) for k in sorted(set(indexed[a])|set(indexed[b])) if indexed[a].get(k)!=indexed[b].get(k)]
    accepted={m:[p["program_id"] for p in read_json(ROOT/f"encoded/{m}/programs.json") if p["accepted"]] for m in MODELS}
    result=dict(checked_calls=len(indexed[a]),mismatches=mismatches,identical_request_signatures=not mismatches,
        accepted_ids_identical=accepted[a]==accepted[b],
        template_hashes={m:read_json(ROOT/f"encoded/{m}/tokenizer_manifest.json")["chat_template_sha256"] for m in MODELS},
        note="Both models independently re-tokenized all source calls; equality does not establish 4B runtime support")
    write_json(ROOT/"verification/cross_model_comparison.json",result)
    return result

def verify_determinism():
    from .pipeline import export
    from .client import make_arrivals
    files=[p for p in sorted((ROOT/"bundles").rglob("*")) if p.is_file()]
    before={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}
    with contextlib.redirect_stdout(io.StringIO()):
        for model in MODELS:export(model)
    after={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}
    if before!=after:raise ValueError("Repeated export changed bytes")
    with tempfile.TemporaryDirectory() as temporary:
        for model in MODELS:
            a,b=Path(temporary)/"a.json",Path(temporary)/"b.json"
            bundle=ROOT/f"bundles/{model}/pilot_14"
            make_arrivals(bundle,.1,20261030,a);make_arrivals(bundle,.1,20261030,b)
            assert a.read_bytes()==b.read_bytes()
    result=dict(bundle_files=len(files),all_reexports_byte_identical=True,arrival_manifests_identical=True,
        scope="Repeated full bundle exports and arrival generation; independent tokenizer validation recorded separately")
    write_json(ROOT/"verification/determinism.json",result)
    return result
