"""Pinned downloads; no trajectory content is ever executed."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from .common import ROOT, DATASET, REVISION, KAIROS, MODELS, sha, read_json, write_json

def fetch(url, attempts=5):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "B-Replay-Preparation/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (OSError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 12))

def blob_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def bootstrap(a_zip):
    destination = ROOT / "sources/a_handoff"
    with zipfile.ZipFile(a_zip) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            relative = Path(*Path(entry.filename).parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe ZIP member")
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(entry))
    if (ROOT.parent / "分工计划.md").exists():
        shutil.copyfile(ROOT.parent / "分工计划.md", ROOT / "sources/分工计划.md")
    elif not (ROOT / "sources/分工计划.md").exists():
        raise FileNotFoundError("Keep the supplied 分工计划.md alongside the B directory for first bootstrap")
    write_json(ROOT / "sources/input_package.json", {
        "a_package_name": Path(a_zip).name, "sha256": sha(Path(a_zip).read_bytes()),
        "kairos_commit": KAIROS, "teacher_revision": REVISION,
    })

def download(workers=8):
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
    tree = read_json(ROOT / "sources/a_handoff/sources/teacher_tree.json")
    rows = [row for row in tree if row["type"] == "file" and
            (row["path"].endswith(".traj.json") or row["path"] == "README.md")]
    assert sum(row["path"].endswith(".traj.json") for row in rows) == 499
    jobs = []
    for row in rows:
        relative = "sources/teacher/" + row["path"]
        path = ROOT / relative
        sample = ROOT / "sources/a_handoff/sources/teacher_samples" / Path(row["path"]).name
        if not path.exists() and sample.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sample, path)
        jobs.append((relative, base + "/" + row["path"], row))
    kairos_files = ["data/training/causal_states.json.gz", "kairos/run.py",
                    "kairos/_runtime/teacher_replay.py", "kairos/_runtime/activation.py", "kairos/verify.py",
                    "kairos/train.py", "docs/METHOD.md", "provenance/release_manifest.json",
                    "provenance/tokenizer-verification.json"]
    for name in kairos_files:
        jobs.append(("sources/kairos/" + name,
                     f"https://raw.githubusercontent.com/MaxizYB/Kairos/{KAIROS}/{name}", None))
    for model, revision in MODELS.items():
        metadata_url = f"https://huggingface.co/api/models/Qwen/{model}/revision/{revision}"
        metadata = json.loads(fetch(metadata_url))
        write_json(ROOT / f"sources/tokenizers/{model}/upstream_metadata.json", metadata)
        allow = {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json", "chat_template.jinja", "LICENSE"}
        for item in metadata["siblings"]:
            name = item["rfilename"]
            if name in allow:
                jobs.append((f"sources/tokenizers/{model}/{name}",
                             f"https://huggingface.co/Qwen/{model}/resolve/{revision}/{name}", None))
    def obtain(job):
        relative, url, source = job
        path = ROOT / relative
        data = path.read_bytes() if path.exists() else fetch(url)
        if source:
            identity_ok = sha(data) == source["lfs"]["oid"] if source.get("lfs") else blob_hash(data) == source["oid"]
            if len(data) != source["size"] or not identity_ok:
                raise ValueError(f"Pinned source mismatch: {relative}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
        return dict(path=relative, url=url, bytes=len(data), sha256=sha(data),
                    upstream_git_blob=source["oid"] if source else None,
                    upstream_lfs_sha256=source.get("lfs", {}).get("oid") if source else None)
    manifest = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(obtain, job) for job in jobs]
        for future in as_completed(futures):
            manifest.append(future.result())
            if len(manifest) % 50 == 0:
                print(f"Downloaded/verified {len(manifest)}/{len(jobs)} files", flush=True)
    write_json(ROOT / "sources/download_manifest.json", sorted(manifest, key=lambda x: x["path"]))
    print(f"Source acquisition complete: {len(manifest)} files", flush=True)
