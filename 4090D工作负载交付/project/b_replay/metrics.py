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
    queue_samples=[e for e in observed if e['event']=='sample']
    transfers=[]
    for name in ('transfers.jsonl','replicas.jsonl'):
        if (run/name).is_file():
            transfers.extend(dict(json.loads(s),source_log=name) for s in (run/name).read_text().splitlines() if s.strip())
    rewind=[e['logical_frontier_loss_tokens'] for e in observed if e['event']=='computed_frontier_rewound']
    return dict(counter_or_gauge_deltas=deltas,cache_ratios=ratios,scheduler_event_counts=counts,
        helper_counter_or_gauge_deltas=helper_deltas,first_admit_wait_s=queue,
        queue_kv_samples=queue_samples,
        sampled_queue_size=stats([e['waiting'] for e in queue_samples]),
        sampled_running_size=stats([e['running'] for e in queue_samples]),
        sampled_pin_ratio=stats([e.get('unique_pin_ratio') for e in queue_samples]),
        sampled_kv_used_ratio=stats([1-e['free_blocks']/e['total_blocks'] for e in queue_samples if e.get('total_blocks')]),
        actual_kv_page_counts=sorted({e['total_blocks'] for e in queue_samples if e.get('total_blocks') is not None}),
        sampled_schedule_cost_sum_s=sum(e.get('schedule_cost_sum_s',0) for e in queue_samples) if queue_samples else None,
        sampled_controller_cost_sum_s=sum(e.get('controller_cost_sum_s',0) for e in queue_samples) if queue_samples else None,
        logical_frontier_loss_tokens=sum(rewind) if observed else None,
        actual_recomputed_tokens=None,
        transfer_event_counts=dict(Counter(e.get('event','unknown') for e in transfers)),
        transfer_events=transfers,
        scheduler_events_available=path.is_file(),
        note='Raw deltas include gauges. Pin occupancy is sampled and physical-ID deduplicated. Logical frontier loss is not actual recomputation. Controller costs are nested in scheduler costs; do not add. Transfer events do not prove wire bytes.')
