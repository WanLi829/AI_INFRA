"""Memory-bounded metrics extraction; frozen experiment sources stay unchanged."""
from collections import Counter
import gzip
import json
from pathlib import Path
import re

from b_replay.common import stats
from b_replay.metrics import samples

EVENT=re.compile(r'^\s*\{\s*"event"\s*:\s*"([^"\\]+)"')


def log_path(path):
    path=Path(path)
    return path if path.is_file() else path.with_name(path.name+'.gz')


def lines(path):
    path=log_path(path)
    if not path.is_file():return
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield line


def event_counts(path):
    counts=Counter();ttls=[]
    for line in lines(path):
        match=EVENT.match(line);event=match[1] if match else json.loads(line).get('event','unknown')
        counts[event]+=1
        if event=='pin':
            value=json.loads(line).get('ttl')
            if value is not None:ttls.append(value)
    return dict(counts),stats(ttls)


def collect(run):
    run=Path(run);before=samples(run/'metrics-before.txt');after=samples(run/'metrics-after.txt')
    deltas={k:after[k]-before[k] if after[k]>=before[k] else None for k in before.keys()&after.keys()}
    ratios={}
    for layer in ('gpu_prefix_cache','cpu_prefix_cache','prefix_cache','external_prefix_cache'):
        hits=f'vllm:{layer}_hits_total';queries=f'vllm:{layer}_queries_total'
        h={k[len(hits):]:v for k,v in deltas.items() if k==hits or k.startswith(hits+'{')}
        q={k[len(queries):]:v for k,v in deltas.items() if k==queries or k.startswith(queries+'{')}
        valid=h and h.keys()==q.keys() and all(v is not None for v in [*h.values(),*q.values()])
        numerator=sum(h.values()) if valid else None;denominator=sum(q.values()) if valid else None
        ratios[layer]=dict(hits=numerator,queries=denominator,
            ratio=numerator/denominator if denominator and numerator<=denominator else None,
            definition=f'{hits} / {queries}; engine-defined token query population; never added across layers')
    counts,ttls=event_counts(run/'scheduler.jsonl')
    observed=[json.loads(line) for line in lines(run/'experiment_metrics.jsonl')]
    arrivals={e['request']:e['time'] for e in observed if e['event']=='request_arrival'}
    queue=stats([e['time']-arrivals[e['request']] for e in observed if e['event']=='first_admit' and e['request'] in arrivals])
    samples_= [e for e in observed if e['event']=='sample']
    hb=samples(run/'lmcache-metrics-before.txt');ha=samples(run/'lmcache-metrics-after.txt')
    transfers=Counter();transfer_paths=[]
    for name in ('transfers.jsonl','replicas.jsonl'):
        path=log_path(run/name)
        if path.is_file():
            counted,_=event_counts(path);transfers.update(counted);transfer_paths.append(str(path))
    return dict(counter_or_gauge_deltas=deltas,cache_ratios=ratios,scheduler_event_counts=counts,
        helper_counter_or_gauge_deltas={k:ha[k]-hb[k] if ha[k]>=hb[k] else None for k in hb.keys()&ha.keys()},
        first_admit_wait_s=queue,queue_kv_samples=samples_,
        sampled_queue_size=stats([e['waiting'] for e in samples_]),sampled_running_size=stats([e['running'] for e in samples_]),
        sampled_pin_ratio=stats([e.get('unique_pin_ratio') for e in samples_]),
        sampled_kv_used_ratio=stats([1-e['free_blocks']/e['total_blocks'] for e in samples_ if e.get('total_blocks')]),
        actual_kv_page_counts=sorted({e['total_blocks'] for e in samples_ if e.get('total_blocks') is not None}),
        sampled_schedule_cost_sum_s=sum(e.get('schedule_cost_sum_s',0) for e in samples_) if samples_ else None,
        sampled_controller_cost_sum_s=sum(e.get('controller_cost_sum_s',0) for e in samples_) if samples_ else None,
        logical_frontier_loss_tokens=sum(e['logical_frontier_loss_tokens'] for e in observed if e['event']=='computed_frontier_rewound') if observed else None,
        actual_recomputed_tokens=None,transfer_event_counts=dict(transfers),transfer_events=None,
        transfer_events_preserved_in=transfer_paths,pin_ttl_s=ttls,
        scheduler_events_available=log_path(run/'scheduler.jsonl').is_file(),
        note='Same counter/queue definitions as frozen metrics.collect; JSONL is streamed, including verified gzip archives. Full transfer records remain in raw logs. Controller cost is nested in scheduler; never add. Logical frontier loss is not actual recomputation.')
