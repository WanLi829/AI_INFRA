"""Prepare selectable study sizes, estimate only from H800 measurements, run serially."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

from .common import ROOT, read_json, write_json, sha
from .hardware import HARDWARE, validate_gpu
from .suite import MAIN_METHODS, METHOD_LABELS, create, execute
from .comparison_results import inspect

SEEDS = [2026111801, 2026111802, 2026111803]
TIERS = {
    'smoke': dict(jps=[.5], dataset_jps=[], k=[], repeats=1,
                  note='5 single-program protocol smoke runs; no performance claim'),
    'quick': dict(jps=[.1,2], dataset_jps=[.1,2], k=[8,32,80], repeats=1,
                  note='Single-repeat exploration; all five methods, no repeated formal result'),
    'minimal': dict(jps=[.1,2], dataset_jps=[.5], k=[8,80], repeats=3,
                    note='Three-repeat subset; dataset low/high comparison and intermediate K omitted'),
    'compact': dict(jps=[.1,2], dataset_jps=[.1,2], k=[8,32,80], repeats=3,
                    note='Three-repeat subset; intermediate JPS/K omitted'),
    'full': dict(jps=[.1,.5,1,2], dataset_jps=[.1,2], k=[8,16,32,50,64,80], repeats=3,
                 note='Full task-book grid: 60/60/90, excluding smoke, calibration, failures'),
}


def options():
    rows=[]
    for name,t in TIERS.items():
        a=len(t['jps'])*5*t['repeats']; b=len(t['dataset_jps'])*2*5*t['repeats']; c=len(t['k'])*5*t['repeats']
        rows.append(dict(tier=name, jps_runs=a,dataset_runs=b,concurrency_runs=c,total=a+b+c,
                         repetitions=t['repeats'],note=t['note']))
    return dict(hardware=HARDWARE,main_methods=MAIN_METHODS,auxiliary_methods=['native'],method_labels=METHOD_LABELS,tiers=rows,
        caution='Counts describe requested grids, not GPU-ready runs. Second-source timing, external weights and H800 checks remain separate gates.')


def configure(args):
    """Write portable input configs; build available matrices, retain blocked tasks."""
    out=args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    tier=TIERS[args.tier]
    base=dict(model=args.model,joint_assets=str(args.joint_assets.resolve()) if args.joint_assets else None,
        methods=MAIN_METHODS,seeds=SEEDS[:tier['repeats']],cpu_kv_gib=args.cpu_kv_gib,
        gpu_kv_bytes=args.gpu_kv_bytes,resource_group=args.resource_group,
        engine_cpus='auto',helper_cpus='auto',start_timeout=1200,run_timeout=7200,
        request_timeout=600,static_ttl_seconds=2,allow_exploratory=args.allow_exploratory,
        tier=args.tier,hardware=HARDWARE,selection_note=tier['note'])
    for k in ('profile','reload_profile','calibration_evidence','capacity_evidence','cache_isolation_evidence'):
        path=getattr(args,k)
        if path: base[k]=str(path.resolve())
    prepared=ROOT/'inputs/prepared'
    specs={
        'jps': dict(axis='jps',jps=tier['jps'],datasets=[dict(name='mini-swe-verified',
            bundle=str(prepared/('pilot_1-v2' if args.tier=='smoke' else 'heldout-v2')))]),
        'dataset': dict(axis='dataset',jps=tier['dataset_jps'],dataset_protocol='observed-tool-time',
            datasets=[dict(name='mini-swe-verified',bundle=str(args.first_dataset.resolve() if args.first_dataset else prepared/'heldout-v2')),
                      dict(name='second-source-observed',bundle=str(args.second_dataset.resolve() if args.second_dataset else ROOT/'inputs/SECOND_SOURCE_NOT_DELIVERED'))]),
        'concurrency': dict(axis='concurrency',active_programs=tier['k'],
            datasets=[dict(name='mini-swe-36sources-400instances',bundle=str(prepared/'instances400'))]),
    }
    if args.zero_wait_sensitivity:
        specs['dataset'].update(dataset_protocol='explicit-zero-wait-sensitivity',datasets=[
            dict(name='mini-swe-32-zero-wait',bundle=str(prepared/'swe_iso32')),
            dict(name='nebius-32-zero-wait',bundle=str(prepared/'nebius_iso32'))])
    inventory=[]
    for task in args.tasks:
        spec=specs[task]
        if (task=='dataset' and not tier['dataset_jps']) or (task=='concurrency' and not tier['k']):
            inventory.append(dict(task=task,status='omitted-by-tier')); continue
        config={**base,**spec}
        write_json(out/(task+'.json'),config)
        try:
            create(out/(task+'.json'),out/task)
            m=read_json(out/task/'matrix.json')
            inventory.append(dict(task=task,status='planned',runs=len(m['runs']),
                matrix=str(out/task/'matrix.json'),readiness_blockers=sorted({s for r in m['runs'] for s in r['plan']['readiness_blockers']})))
        except (ValueError,FileNotFoundError) as error:
            inventory.append(dict(task=task,status='blocked',reason=str(error),config=str(out/(task+'.json'))))
    result=dict(schema='h800-preparation/1',gpu_started=False,tier=args.tier,tasks=inventory,
        requested_options=options(),rule='Generate execution matrices on the Linux host after final path/resource freeze')
    write_json(out/'preparation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def timing_key(row):
    """Allow new seeds, never scale a tiny pilot into an unmeasured 400-instance run."""
    p=row['plan']
    command=p['engine_command']
    flags=('max-model-len','max-num-seqs','max-num-batched-tokens','gpu-memory-utilization','block-size','dtype')
    limits={k:command[command.index('--'+k)+1] for k in flags if '--'+k in command}
    value=dict(axis=row['axis'],value=row['value'],method=row['method'],
        input=p['bundle_manifest_sha256'],model=p['model'],runtime=p['runtime'],
        code=p['adapter_sha256'],resource_group=p.get('resource_group'),limits=limits,
        cpu_kv_gib=p['cpu_kv_gib'],gpu_kv_bytes=p['gpu_kv_bytes'],
        engine_cpus=command[2],helper_cpus=p['helper_command'][2] if p.get('helper_command') else None,
        profile=p['profile_sha256'],reload=p['reload_profile_sha256'],
        joint_release=p.get('joint_release_sha256'),ttl=p['ttl_seconds'])
    return sha(json.dumps(value,sort_keys=True).encode())


def measured_timings(paths):
    samples=defaultdict(list)
    for path in paths:
        matrix=read_json(path)
        for row in matrix['runs']:
            state_path=Path(row['result'])/'state.json'
            attempt=Path(path).parent/'attempts'/(row['run_id']+'.json')
            if not state_path.is_file() or not attempt.is_file(): continue
            state,attempt_row=read_json(state_path),read_json(attempt)
            if state.get('status')!='complete' or attempt_row.get('status')!='complete': continue
            validate_gpu(state['gpu'])
            if (state.get('protocol') != row['plan']['protocol'] or state.get('plan') != row['plan']
                    or not state.get('summary',{}).get('strict_complete')):
                raise ValueError('Timing sample is not the frozen complete workload: '+row['run_id'])
            measured,_=inspect(Path(row['result']),Path(row['bundle']))
            if measured['status']!='complete':
                raise ValueError('Timing sample failed replay/resource verification')
            elapsed=attempt_row.get('wall_s')
            if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<=0:
                raise ValueError('Timing sample lacks the full build/start/replay/cleanup duration')
            samples[timing_key(row)].append(dict(wall_s=elapsed,gpu_uuid=state['gpu']['uuid'],
                row=row['run_id'],matrix=str(Path(path).resolve()),packages=state['packages']))
    identities={(r['gpu_uuid'],json.dumps(r['packages'],sort_keys=True)) for values in samples.values() for r in values}
    if len(identities)>1: raise ValueError('Mixed H800 UUID/software timing samples; estimate each environment separately')
    return samples


def estimate(matrix_paths, pilot_paths, walltime_minutes, reserve_minutes=15, factor=1.3):
    if not all(math.isfinite(x) for x in (walltime_minutes,reserve_minutes,factor)) or walltime_minutes<=reserve_minutes or reserve_minutes<1 or factor<1:
        raise ValueError('Require positive usable time, >=1 minute reserve and safety factor >=1')
    samples=measured_timings(pilot_paths)
    conditions=defaultdict(list);unknown=[];all_rows=[]
    identities=[]
    for path in matrix_paths:
        path=Path(path).resolve(); identities.append(dict(path=str(path),sha256=sha(path.read_bytes())))
        for row in read_json(path)['runs']:
            item=dict(matrix=str(path),index=row['index'],run_id=row['run_id'],method=row['method'],repetition=row['repetition'])
            matched=samples.get(timing_key(row),[])
            item['estimated_s']=max(r['wall_s'] for r in matched)*factor if matched else None
            item['timing_evidence']=matched
            key=(str(path),row['dataset'],row['axis'],str(row['value']))
            conditions[key].append(item);all_rows.append(item)
            if not matched: unknown.append(item)
    available=(walltime_minutes-reserve_minutes)*60; used=0.;selected=[];condition_rows=[]
    # Complete every pre-registered repetition of a condition before adding one.
    for key,rows in conditions.items():
        by_repeat=defaultdict(set)
        for r in rows: by_repeat[r['repetition']].add(r['method'])
        if any(methods!=set(MAIN_METHODS) for methods in by_repeat.values()):
            raise ValueError('Budget selection requires all five main methods in every repetition')
        cost=sum(r['estimated_s'] for r in rows) if all(r['estimated_s'] is not None for r in rows) else None
        fits=cost is not None and used+cost<=available
        condition_rows.append(dict(condition=key,runs=len(rows),repetitions=len(by_repeat),estimated_s=cost,selected=fits,
            status='selected' if fits else 'unmeasured' if cost is None else 'outside-budget'))
        if fits:
            used+=cost; selected.extend(rows)
    return dict(schema='h800-budget/1',created_at=time.time(),matrices=identities,
        walltime_minutes=walltime_minutes,reserve_minutes=reserve_minutes,safety_factor=factor,
        selected_runs=len(selected),selected_conditions=sum(c['selected'] for c in condition_rows),
        estimated_selected_minutes=used/60,unknown_runs=len(unknown),conditions=condition_rows,
        schedule=selected,sample_keys=sorted(samples),gpu_started=False,
        note='Empirical estimate, not a completion guarantee. No seed-time scaling, no tiny-pilot extrapolation. Include all registered repeats of each selected condition. Unmeasured cells remain unestimated.')


def run_budget(path, log_output):
    budget=read_json(path)
    if budget.get('schema')!='h800-budget/1' or not budget['schedule']:
        raise ValueError('A nonempty measured budget selection is required')
    for m in budget['matrices']:
        if sha(Path(m['path']).read_bytes())!=m['sha256']: raise ValueError('Matrix changed after budget selection')
    for item in budget['schedule']:
        matrix=read_json(item['matrix']); row=matrix['runs'][item['index']]
        if Path(row['result']).exists() or (Path(item['matrix']).parent/'attempts'/(row['run_id']+'.json')).exists():
            raise FileExistsError('Existing attempt; preserve it and generate a new continuation plan: '+row['run_id'])
    log_output=Path(log_output)
    if log_output.exists(): raise FileExistsError(log_output)
    deadline=time.time()+(budget['walltime_minutes']-budget['reserve_minutes'])*60
    state=dict(schema='h800-budget-execution/1',budget_sha256=sha(Path(path).read_bytes()),
        began_at=time.time(),execution_deadline_epoch=deadline,attempts=[],status='running')
    write_json(log_output,state)
    try:
        schedule=budget['schedule']
        for offset,item in enumerate(schedule):
            # Test the entire remaining condition before starting its first row.
            row=read_json(item['matrix'])['runs'][item['index']]
            condition=(item['matrix'],row['dataset'],row['axis'],row['value'])
            if offset==0 or condition!=last:
                remaining=[]
                for candidate in schedule[offset:]:
                    other=read_json(candidate['matrix'])['runs'][candidate['index']]
                    if (candidate['matrix'],other['dataset'],other['axis'],other['value'])!=condition: break
                    remaining.append(candidate['estimated_s'])
                if time.time()+sum(remaining)>deadline:
                    state['status']='stopped-before-condition'; break
            began=time.monotonic()
            state['current_run']=item
            write_json(log_output,state)
            execute(item['matrix'],item['index'],deadline_epoch=deadline)
            state['attempts'].append(dict(**item,actual_wall_s=time.monotonic()-began,status='complete'))
            write_json(log_output,state); last=condition
        else: state['status']='complete'
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        state['ended_at']=time.time();write_json(log_output,state)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    o=sub.add_parser('options');o.add_argument('--output',type=Path)
    c=sub.add_parser('plan');c.add_argument('--tier',choices=TIERS,default='quick')
    c.add_argument('--tasks',nargs='+',choices=['jps','dataset','concurrency'],default=['jps','dataset','concurrency'])
    c.add_argument('--output',type=Path,required=True);c.add_argument('--model',required=True)
    c.add_argument('--joint-assets',type=Path);c.add_argument('--first-dataset',type=Path);c.add_argument('--second-dataset',type=Path)
    c.add_argument('--profile',type=Path);c.add_argument('--reload-profile',type=Path);c.add_argument('--calibration-evidence',type=Path)
    c.add_argument('--capacity-evidence',type=Path)
    c.add_argument('--cache-isolation-evidence',type=Path)
    c.add_argument('--gpu-kv-bytes',type=int);c.add_argument('--cpu-kv-gib',type=float,default=64)
    c.add_argument('--resource-group',choices=['pilot','main','kv-pressure'],default='pilot')
    c.add_argument('--allow-exploratory',action='store_true');c.add_argument('--zero-wait-sensitivity',action='store_true')
    e=sub.add_parser('estimate');e.add_argument('--matrix',nargs='+',type=Path,required=True)
    e.add_argument('--pilot-matrix',nargs='+',type=Path,required=True);e.add_argument('--walltime-minutes',type=float,required=True)
    e.add_argument('--reserve-minutes',type=float,default=15);e.add_argument('--factor',type=float,default=1.3);e.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run-budget');r.add_argument('--budget',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='options':
        result=options()
        if a.output: write_json(a.output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif a.command=='plan': configure(a)
    elif a.command=='estimate':
        result=estimate(a.matrix,a.pilot_matrix,a.walltime_minutes,a.reserve_minutes,a.factor)
        if a.output.exists(): raise FileExistsError(a.output)
        write_json(a.output,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('schedule','sample_keys')},ensure_ascii=False,indent=2))
    else: run_budget(a.budget,a.output)


if __name__=='__main__': main()
