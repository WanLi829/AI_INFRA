"""Freeze a reproducible experiment matrix; execute only explicitly selected rows."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from .client import load_bundle, make_arrivals
from .common import ROOT, read_json, write_json, sha
from .comparison import METHODS, arguments, build, run
from .workload import make_closed
from .hardware import deadline_guard

MAIN_METHODS = ['lmcache', 'zero_ttl', 'static_ttl', 'continuum', 'kairos']
METHOD_LABELS = dict(lmcache='vLLM + standard LMCache', zero_ttl='Continuum Zero TTL',
    static_ttl='Continuum Static TTL (2 s)', continuum='Continuum Dynamic TTL', kairos='Full Kairos CG1',
    native='Auxiliary native vLLM')


def create(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read_json(config_path)
    if output.exists():
        raise FileExistsError(output)
    axis = config['axis']
    if axis not in ('jps','dataset','concurrency'):
        raise ValueError('axis must be jps, dataset, or concurrency')
    methods=config.get('methods',MAIN_METHODS)
    if len(set(methods)) != len(methods) or not methods or not set(methods) <= set(METHODS):
        raise ValueError('Invalid method list')
    datasets=config['datasets']
    if not datasets or (axis != 'dataset' and len(datasets) != 1):
        raise ValueError('JPS/concurrency scans require exactly one dataset')
    if axis == 'dataset' and len(datasets) < 2:
        raise ValueError('Dataset comparison requires at least two actual datasets')
    seeds=config['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Provide unique integer repetition seeds')
    values=config['active_programs'] if axis=='concurrency' else config['jps']
    if not values or len(set(values)) != len(values):
        raise ValueError('Empty or duplicate axis values')
    validated=[]
    for dataset in datasets:
        bundle=(config_path.parent/dataset['bundle']).resolve()
        manifest,_,_=load_bundle(bundle)
        if axis=='concurrency' and any(type(v) is not int or not 1 <= v <= manifest['programs'] for v in values):
            raise ValueError('Active program counts exceed available instances')
        validated.append((dataset,bundle,manifest))
    if axis=='dataset':
        ids=[m.get('dataset_id') for _,_,m in validated]
        if None in ids or len(set(ids))!=len(ids):
            raise ValueError('Use registered distinct dataset_id values; subsets are not distinct datasets')
        families=[m.get('source_family') for _,_,m in validated]
        if None in families or len(set(families)) != len(families):
            raise ValueError('Cross-dataset comparison requires distinct audited source families')
        if config.get('dataset_protocol','observed-tool-time') == 'observed-tool-time':
            if any(m.get('timing_kind') != 'observed_tool_feedback' for _,_,m in validated):
                raise ValueError('Main cross-dataset results require observed gaps; use the explicit zero-wait sensitivity protocol separately')
        elif config['dataset_protocol'] == 'explicit-zero-wait-sensitivity':
            if any(m.get('timing_kind') != 'explicit_zero_wait_control' for _,_,m in validated):
                raise ValueError('Both sensitivity sources must use the same explicit zero-wait protocol')
        else:
            raise ValueError('Unknown cross-dataset timing protocol')
    output.mkdir(parents=True)
    rows=[]
    frozen_code=None
    for dataset,bundle,manifest in validated:
        for value in values:
            for rep,seed in enumerate(seeds,1):
                tag=f"d{len(rows):04d}-{axis}{value}-r{rep}"
                workload=output/'workloads'/(tag+'.json')
                if axis=='concurrency':
                    make_closed(bundle,value,seed,workload)
                else:
                    make_arrivals(bundle,value,seed,workload)
                order=methods[(rep-1)%len(methods):]+methods[:(rep-1)%len(methods)]
                for method in order:
                    run_id=f'{len(rows):04d}-{method}-{axis}{value}-r{rep}'
                    argv=['run','--method',method,'--bundle',str(bundle),'--arrivals',str(workload),
                          '--model',config['model'],'--output',str(output/'runs'/run_id)]
                    mapping={'joint_assets':'--joint-assets','profile':'--profile','reload_profile':'--reload-profile',
                             'cpu_kv_gib':'--cpu-kv-gib','gpu_kv_bytes':'--gpu-kv-bytes','gpu':'--gpu',
                             'gpu_pid_policy':'--gpu-pid-policy',
                             'engine_cpus':'--engine-cpus','helper_cpus':'--helper-cpus',
                             'start_timeout':'--start-timeout','run_timeout':'--run-timeout','request_timeout':'--request-timeout',
                             'resource_group':'--resource-group','calibration_evidence':'--calibration-evidence',
                             'capacity_evidence':'--capacity-evidence','cache_isolation_evidence':'--cache-isolation-evidence'}
                    for key,flag in mapping.items():
                        if config.get(key) is not None:
                            argv += [flag,str(config[key])]
                    if method=='static_ttl':
                        argv += ['--ttl-seconds',str(config.get('static_ttl_seconds',2))]
                    if config.get('allow_exploratory',False):
                        argv += ['--allow-exploratory']
                    # Build every plan now: no GPU starts; bad input fails early.
                    selected=arguments(['plan',*argv[1:]])
                    plan,_,_,_=build(selected)
                    code_identity=(plan['runtime'],plan['adapter_sha256'])
                    if frozen_code is None: frozen_code=code_identity
                    if code_identity!=frozen_code:
                        raise RuntimeError('Code changed while generating the matrix; create a fresh matrix after edits finish')
                    write_json(output/'plans'/(run_id+'.json'),plan)
                    rows.append(dict(index=len(rows),run_id=run_id,method=method,repetition=rep,seed=seed,
                        axis=axis,value=value,dataset=dataset['name'],bundle=str(bundle),
                        bundle_manifest_sha256=sha((bundle/'manifest.json').read_bytes()),
                        workload_sha256=sha(workload.read_bytes()), argv=argv,
                        plan=plan, result=str(output/'runs'/run_id)))
    provenance=ROOT.parent/'upstream-project-manifest.json'
    matrix=dict(schema='h800-suite/1',config=config,config_sha256=sha(config_path.read_bytes()),
        source_commit=read_json(provenance)['commit'] if provenance.is_file() else None,
        method_labels=METHOD_LABELS,
        runs=rows, note='CPU plans are not H800 validation; execute explicit indices only')
    write_json(output/'matrix.json',matrix)
    print(json.dumps(dict(matrix=str(output/'matrix.json'),runs=len(rows),gpu_started=False)))


def execute(matrix_path,index,deadline_epoch=None):
    with deadline_guard(deadline_epoch):
        return _execute(matrix_path,index)


def _execute(matrix_path,index):
    began_wall=time.time(); began=time.monotonic()
    matrix=read_json(matrix_path)
    if type(index) is not int or not 0<=index<len(matrix['runs']):
        raise ValueError('Run index out of range')
    row=matrix['runs'][index]
    attempt=Path(matrix_path).parent/'attempts'/(row['run_id']+'.json')
    if attempt.exists():
        raise FileExistsError('Preserve the previous attempt and create a continuation matrix: '+str(attempt))
    try:
        args=arguments(row['argv'])
        plan,upstream,native,env=build(args)
        if plan['adapter_sha256']!=row['plan']['adapter_sha256'] or plan['runtime']!=row['plan']['runtime']:
            raise ValueError('Code changed after matrix freeze; create a new matrix')
        for key in ('bundle_manifest_sha256','arrivals_sha256','profile_sha256','reload_profile_sha256',
                    'cpu_kv_gib','gpu_kv_bytes','resource_group','calibration_evidence','capacity_evidence','cache_isolation_evidence'):
            if plan[key]!=row['plan'][key]:
                raise ValueError('Frozen plan changed: '+key)
        run(args,plan,upstream,native,env)
    except BaseException as error:
        write_json(Path(matrix_path).parent/'attempts'/(row['run_id']+'.json'),
            dict(status='failed',started_at=began_wall,time=time.time(),wall_s=time.monotonic()-began,
                 error=repr(error),result=row['result']))
        raise
    write_json(Path(matrix_path).parent/'attempts'/(row['run_id']+'.json'),
        dict(status='complete',started_at=began_wall,time=time.time(),wall_s=time.monotonic()-began,result=row['result']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan'); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run'); p.add_argument('--matrix',type=Path,required=True); p.add_argument('--index',type=int,required=True)
    a=parser.parse_args()
    if a.command=='plan': create(a.config,a.output)
    else: execute(a.matrix,a.index)


if __name__=='__main__': main()
