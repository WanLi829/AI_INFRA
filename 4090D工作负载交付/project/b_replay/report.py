"""Offline suite report: retain failures, validate successful runs, pair by seed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .common import read_json,write_json,write_csv,sha
from .comparison_results import inspect
from .metrics import collect


def report(matrix_path,output,plots=True):
    matrix=read_json(matrix_path)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows=[]; pairs=defaultdict(dict)
    for item in matrix['runs']:
        row=dict(run_id=item['run_id'],axis=item['axis'],axis_value=item['value'],
            dataset=item['dataset'],repetition=item['repetition'],seed=item['seed'],method=item['method'],
            ttl_seconds=item['plan']['ttl_seconds'],status='not_run',result=item['result'])
        root=Path(item['result'])
        try:
            if (root/'state.json').is_file():
                actual_plan=read_json(root/'state.json')['plan']
                for key in ('method','ttl_seconds','protocol','arrivals_sha256','bundle_manifest_sha256',
                            'adapter_sha256','runtime','cpu_kv_gib','gpu_kv_bytes','profile_sha256','reload_profile_sha256'):
                    if actual_plan.get(key)!=item['plan'].get(key):
                        raise ValueError('Run differs from frozen matrix: '+key)
                measured,identity=inspect(root,Path(item['bundle']))
                row.update(measured)
                evidence=collect(root)
                write_json(output/'evidence'/(item['run_id']+'.json'),evidence)
                for layer,value in evidence['cache_ratios'].items():
                    row[layer+'_ratio']=value['ratio']
                if identity:
                    environment=json.loads(identity)
                    environment.pop('arrivals',None)
                    row['comparison_environment_sha256']=sha(json.dumps(environment,sort_keys=True).encode())
                    pairs[identity][item['method']]=row
            else:
                attempt=Path(matrix_path).parent/'attempts'/(item['run_id']+'.json')
                if attempt.is_file(): row.update(read_json(attempt))
        except Exception as error:
            row.update(status='invalid',validation_error=repr(error),speedup_eligible=False)
        rows.append(row)
    for members in pairs.values():
        for row in members.values():
            baseline=members.get('lmcache')
            if baseline and row.get('speedup_eligible') and baseline.get('speedup_eligible'):
                row['jct_reduction_vs_lmcache']=1-row['jct_mean_s']/baseline['jct_mean_s']
                row['jct_speedup_vs_lmcache']=baseline['jct_mean_s']/row['jct_mean_s']
            dynamic=members.get('continuum')
            if dynamic and row.get('speedup_eligible') and dynamic.get('speedup_eligible'):
                row['jct_reduction_vs_dynamic']=1-row['jct_mean_s']/dynamic['jct_mean_s']
    groups=defaultdict(list)
    for row in rows:
        groups[(row['dataset'],row['axis_value'],row['method'],row['ttl_seconds'])].append(row)
    aggregate=[]
    metrics=['jct_mean_s','jct_p95_s','ttft_mean_s','ttft_p95_s','programs_per_s','output_tokens_per_s',
             'active_mean','active_peak','jct_reduction_vs_lmcache','jct_reduction_vs_dynamic']
    for (dataset,value,method,ttl),values in groups.items():
        environments={r['comparison_environment_sha256'] for r in values if r.get('comparison_environment_sha256')}
        if len(environments)>1:
            raise ValueError(f'Mixed hardware/software/input environments in {dataset}/{value}/{method}; split the matrix before aggregating')
        item=dict(dataset=dataset,axis_value=value,method=method,ttl_seconds=ttl,
            planned=len(values),complete=sum(r['status']=='complete' for r in values),
            failed=sum(r['status'] not in ('complete','not_run') for r in values),
            not_run=sum(r['status']=='not_run' for r in values),
            formal_runs=sum(bool(r.get('speedup_eligible')) for r in values))
        for metric in metrics:
            observed=[r[metric] for r in values if r['status']=='complete' and r.get(metric) is not None]
            item[metric+'_n']=len(observed)
            item[metric+'_mean']=statistics.fmean(observed) if observed else None
            item[metric+'_min']=min(observed) if observed else None
            item[metric+'_max']=max(observed) if observed else None
        item['repeated_formal_pair']=item.get('jct_reduction_vs_lmcache_n',0)>=3 and item['failed']==0
        aggregate.append(item)
    write_json(output/'report.json',dict(runs=rows,groups=aggregate,
        notes=['Raw latency aggregates describe completed runs only; failures and missing runs are explicit.',
               'Speedups require both formally eligible paired runs. Range bars are min/max across repeats, not confidence intervals.']))
    write_csv(output/'runs.csv',rows); write_csv(output/'summary.csv',aggregate)
    if plots: plot(rows,output)
    return rows,aggregate


def plot(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for dataset in sorted({r['dataset'] for r in rows}):
        selected=[r for r in rows if r['dataset']==dataset and r['status']=='complete']
        for metric in ('jct_mean_s','jct_p95_s','ttft_mean_s','ttft_p95_s','output_tokens_per_s','active_mean'):
            fig,ax=plt.subplots(figsize=(7,4))
            any_points=False
            for method in sorted({r['method'] for r in selected}):
                points=defaultdict(list)
                for r in selected:
                    if r['method']==method and r.get(metric) is not None: points[float(r['axis_value'])].append(r[metric])
                if not points: continue
                xs=sorted(points); ys=[statistics.fmean(points[x]) for x in xs]
                ax.errorbar(xs,ys,yerr=[[y-min(points[x]) for x,y in zip(xs,ys)],
                    [max(points[x])-y for x,y in zip(xs,ys)]],label=method,marker='o',capsize=3)
                any_points=True
            if any_points:
                ax.set(xlabel=selected[0]['axis'],ylabel=metric,title=f'{dataset} (completed runs; see failure table)')
                ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
                # Dataset names never become filesystem paths.
                number=sorted({r['dataset'] for r in rows}).index(dataset)
                for ext in ('png','svg'): fig.savefig(Path(output)/f'dataset{number}-{metric}.{ext}',dpi=180)
            plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args(); report(a.matrix,a.output,not a.no_plots)


if __name__=='__main__': main()
