from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATASET = "tarsur385/qwen3-30b-a3b-instruct-2507-swebench-verified-mini-swe-agent"
REVISION = "58389eb5426db5a3ae6159faa37498072b4735ff"
KAIROS = "4ddff8786254c8c5ed8ac18deb22486e7a0594fa"
MODELS = {
    "Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
}
# Exact descriptor translation in the pinned Kairos activation.py, without importing upstream code.
RUNTIME_ACTIVATION = {
    "kairos": "v21-eager-a~r5callcommitment~progressivewindow~jointmean~localeta~provenanceprior~onlinefitresidual~knownoutput~finishlogical~kairos",
    "continuum": "continuum-index-partial-eager-a",
}

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def json_bytes(value, pretty=True):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       indent=2 if pretty else None,
                       separators=None if pretty else (",", ":")) + "\n").encode()

def write_json(path, value, pretty=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(value, pretty)
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.write_bytes(data)

def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def stats(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"n": 0, "mean": None, "min": None, "p50": None, "p95": None, "max": None}
    def q(p):
        x = (len(values) - 1) * p
        return values[math.floor(x)] + (values[math.ceil(x)] - values[math.floor(x)]) * (x % 1)
    return dict(n=len(values), mean=statistics.fmean(values), min=values[0],
                p50=q(.5), p95=q(.95), max=values[-1])

def lcp(a, b):
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i
