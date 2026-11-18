"""Conservative raw Prometheus deltas and scheduler-event evidence.

Only same-name/same-label samples are subtracted. Missing/reset counters are
not zero. Layer-specific hit/query ratios retain their metric names.
"""
from collections import Counter
import json
import math
from pathlib import Path
import re
from .common import stats

SAMPLE=re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+\S+)?$')


def samples(path):
    result={}
    path=Path(path)
    if not path.is_file(): return result
    for line in path.read_text(encoding='utf-8').splitlines():
        match=SAMPLE.match(line)
        if match:
            value=float(match[3])
            if math.isfinite(value): result[match[1]+(match[2] or '')]=value
    return result


def collect(run):
    run=Path(run)
    before,after=samples(run/'metrics-before.txt'),samples(run/'metrics-after.txt')
    deltas={k: after[k]-before[k] if after[k]>=before[k] else None for k in before.keys() & after.keys()}
    ratios={}
    for layer in ('gpu_prefix_cache','cpu_prefix_cache','prefix_cache','external_prefix_cache'):
        hits=f'vllm:{layer}_hits_total'; queries=f'vllm:{layer}_queries_total'
        h={k[len(hits):]:v for k,v in deltas.items() if k==hits or k.startswith(hits+'{')}
        q={k[len(queries):]:v for k,v in deltas.items() if k==queries or k.startswith(queries+'{')}
        valid=h and h.keys()==q.keys() and all(v is not None for v in [*h.values(),*q.values()])
        numerator=sum(h.values()) if valid else None
        denominator=sum(q.values()) if valid else None
        ratios[layer]=dict(hits=numerator,queries=denominator,
            ratio=numerator/denominator if denominator and numerator<=denominator else None,
            definition=f'{hits} / {queries}; engine-defined token query population; never added across layers')
    events=[]
    path=run/'scheduler.jsonl'
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip(): events.append(json.loads(line))
    counts=dict(Counter(e.get('event','unknown') for e in events))
    observed=[]
    if (run/'experiment_metrics.jsonl').is_file():
        observed=[json.loads(s) for s in (run/'experiment_metrics.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    arrivals={e['request']:e['time'] for e in observed if e['event']=='request_arrival'}
    queue=stats([e['time']-arrivals[e['request']] for e in observed
                 if e['event']=='first_admit' and e['request'] in arrivals])
    helper_before=samples(run/'lmcache-metrics-before.txt');helper_after=samples(run/'lmcache-metrics-after.txt')
    helper_deltas={k:helper_after[k]-helper_before[k] if helper_after[k]>=helper_before[k] else None
                   for k in helper_before.keys() & helper_after.keys()}
    return dict(counter_or_gauge_deltas=deltas,cache_ratios=ratios,scheduler_event_counts=counts,
        helper_counter_or_gauge_deltas=helper_deltas,first_admit_wait_s=queue,
        queue_kv_samples=[e for e in observed if e['event']=='sample'],
        scheduler_events_available=path.is_file(),
        note='Raw deltas include gauges; only named counter ratios are interpreted. No inferred CPU hits or unique pin occupancy.')
