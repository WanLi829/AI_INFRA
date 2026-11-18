"""Finite closed-loop workloads and event-based concurrency, without GPU imports."""
from collections import defaultdict
import math
import random
from .common import read_json, write_json, stats


def make_closed(bundle, active_programs, seed, output):
    from .client import load_bundle
    manifest, _, _ = load_bundle(bundle)
    if type(active_programs) is not int or not 1 <= active_programs <= manifest['programs']:
        raise ValueError('active_programs must be between 1 and bundle program count')
    order = list(manifest['program_ids'])
    random.Random(seed).shuffle(order)
    write_json(output, dict(schema_version='b-replay-closed/1.0', mode='closed',
        active_programs=active_programs, seed=seed, program_order=order,
        trace_sha256=manifest['files']['trace.json'], failure_policy='continue-and-mark-failed'))


def validate_workload(manifest, payload):
    if payload.get('mode', 'open') == 'open':
        from .client import validate_arrivals
        validate_arrivals(manifest, payload)
        return 'open'
    if payload.get('mode') != 'closed' or payload.get('schema_version') != 'b-replay-closed/1.0':
        raise ValueError('Unknown workload protocol')
    k = payload.get('active_programs')
    order = payload.get('program_order', [])
    if type(k) is not int or not 1 <= k <= manifest['programs']:
        raise ValueError('Invalid active program count')
    if len(order) != manifest['programs'] or set(order) != set(manifest['program_ids']):
        raise ValueError('Closed workload must contain each program exactly once')
    if payload.get('trace_sha256') != manifest['files']['trace.json']:
        raise ValueError('Closed workload targets another trace')
    if payload.get('failure_policy') != 'continue-and-mark-failed':
        raise ValueError('Unsupported closed workload failure policy')
    return 'closed'


def concurrency(jobs, target=None):
    """Integrate [arrival,end) events. Canceled jobs use their observed end."""
    intervals = []
    for job in jobs:
        a, b = job.get('actual_arrival_s'), job.get('end_s')
        if a is None:
            continue
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)) or b < a:
            raise ValueError('Invalid program interval')
        intervals.append((a, b))
    if not intervals:
        return dict(peak=0, mean=None, target=target, segments=[], target_window=None)
    events = defaultdict(int)
    for a, b in intervals:
        events[a] += 1
        events[b] -= 1
    active, area, peak = 0, 0., 0
    segments = []
    points = sorted(events)
    for a, b in zip(points, points[1:]):
        active += events[a]
        peak = max(peak, active)
        area += active * (b-a)
        segments.append(dict(start_s=a, end_s=b, active=active))
    duration = points[-1]-points[0]
    window = None
    if target and len(intervals) > target:
        starts = sorted(a for a, _ in intervals)
        lo, hi = starts[target-1], starts[-1]
        seconds = hi-lo
        weights = [(max(0., min(s['end_s'], hi)-max(s['start_s'], lo)), s['active']) for s in segments]
        window = dict(start_s=lo, end_s=hi, duration_s=seconds,
            mean=sum(d*n for d,n in weights)/seconds if seconds else None,
            target_fraction=sum(d for d,n in weights if n == target)/seconds if seconds else None,
            below_s=sum(d for d,n in weights if n < target),
            above_s=sum(d for d,n in weights if n > target))
    return dict(peak=peak, mean=area/duration if duration else None, target=target,
        start_s=points[0], end_s=points[-1], segments=segments, target_window=window,
        refill_delay_s=stats([j.get('refill_delay_s') for j in jobs]))
