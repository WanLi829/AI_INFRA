"""Rebuild local, source-checked reports from completed H800 run snapshots."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

BASE=Path(__file__).resolve().parent
PROJECT=BASE/'project'
REMOTE=Path('/root/bayes-tmp/h8run')
sys.path[:0]=[str(PROJECT),str(PROJECT/'runtime/Kairos')]
from b_replay.common import read_json,write_json,write_csv
from b_replay import report as reporting
from streaming_evidence import collect

LABELS={'lmcache':'vLLM + LMCache','zero_ttl':'Zero TTL','static_ttl':'Static TTL (2s)',
        'continuum':'Dynamic TTL','kairos':'Kairos (retrained)'}
COLORS=dict(zip(LABELS,['#2563a6','#d18727','#8a68ac','#249074','#c74343']))


def matrices(raw,tail,available_only=False):
    result=defaultdict(list);missing=[]
    metadata=BASE/'success_metadata' if (BASE/'success_metadata/fast05/progress.json').exists() else BASE/'live_results'
    for batch in ('fast05',tail):
        progress=read_json(metadata/batch/'progress.json')
        schedule={r['index']:r for r in read_json(metadata/batch/'schedule.json')['runs']}
        for row in progress['completed']:
            planned=schedule[row['index']];local=raw/Path(row['result']).relative_to(REMOTE)
            if not (local/'state.json').exists() and available_only:
                missing.append(row['result']);continue
            state=read_json(local/'state.json');assert state['status']=='complete'
            bundle=PROJECT/'inputs/prepared'/Path(planned['bundle']).name
            result[row['axis']].append(dict(run_id=batch+'-'+row['run_id'],source_run_id=row['run_id'],
                axis=row['axis'],value=row['value'],dataset=row['dataset'],repetition=row['repetition'],seed=row['seed'],
                method=row['method'],result=str(local),bundle=str(bundle),plan=state['plan']))
    return result,missing


def pair_signature(state):
    p=state['plan'];command=p['engine_command']
    limits={flag:command[command.index(flag)+1] for flag in
        ('--max-model-len','--max-num-seqs','--max-num-batched-tokens','--gpu-memory-utilization')}
    return dict(gpu_uuid=state['gpu']['uuid'],packages=state['packages'],model=p['model'],
        limits=limits,engine_cpus=command[2],runtime=p['runtime'],
        **{k:p.get(k) for k in ('bundle_manifest_sha256','arrivals_sha256','adapter_sha256',
            'gpu_kv_bytes','cpu_kv_gib','profile_sha256','reload_profile_sha256','resource_group')})


def exploratory_pairs(rows):
    groups=defaultdict(dict)
    for r in rows:
        if r['status']=='complete':groups[(r['axis'],r['axis_value'],r['dataset'],r['repetition'],r['seed'])][r['method']]=r
    result=[]
    for members in groups.values():
        for baseline_name in ('lmcache','continuum'):
            baseline=members.get(baseline_name)
            if baseline is None:continue
            base_state=read_json(Path(baseline['result'])/'state.json')
            for r in members.values():
                state=read_json(Path(r['result'])/'state.json')
                if pair_signature(state)!=pair_signature(base_state):
                    raise ValueError('Descriptive pair differs in input/hardware/resource/software: '+r['run_id'])
                result.append(dict(run_id=r['run_id'],axis=r['axis'],axis_value=r['axis_value'],dataset=r['dataset'],
                    repetition=r['repetition'],seed=r['seed'],method=r['method'],baseline=baseline_name,
                    jct_reduction_fraction=1-r['jct_mean_s']/baseline['jct_mean_s'],
                    jct_speedup=baseline['jct_mean_s']/r['jct_mean_s'],
                    classification='exploratory_descriptive_pair',formal_pair=False,
                    note='Completed successful runs only. GPU PID ownership is record-only; Kairos is retrained and in a later temporal block, with a 7200-second collection deadline.'))
    return result


def bottom_tables(rows,folder):
    datasets=sorted({r['dataset'] for r in rows})
    for png in folder.glob('dataset*-*.png'):
        prefix,metric=png.stem.split('-',1);dataset=datasets[int(prefix[7:])]
        values=[dict(run_id=r['run_id'],method=r['method'],label=LABELS[r['method']],axis=r['axis'],
            axis_value=r['axis_value'],dataset=dataset,repetition=r['repetition'],seed=r['seed'],
            metric=metric,value=r.get(metric),status=r['status']) for r in rows if r['dataset']==dataset]
        write_csv(png.with_suffix('.csv'),values)
    for number,row in enumerate(rows):
        curve=Path(row['result'])/'b_replay/concurrency.json'
        if curve.is_file():write_csv(folder/f'run{number:04d}-active-programs.csv',read_json(curve).get('segments',[]))


def additional_plots(rows,pairs,folder):
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True,exist_ok=True)
    selected=[r for r in rows if r['axis']=='dataset' and r['status']=='complete']
    datasets=sorted({r['dataset'] for r in selected})
    if datasets:
        metrics=('jct_mean_s','jct_p95_s','ttft_mean_s','ttft_p95_s')
        fig,axes=plt.subplots(2,2,figsize=(11,7));width=.15
        for ax,metric in zip(axes.ravel(),metrics):
            for j,method in enumerate(LABELS):
                values=[[r[metric] for r in selected if r['dataset']==d and r['method']==method] for d in datasets]
                for i,v in enumerate(values):
                    if not v:continue
                    mean=statistics.fmean(v)
                    ax.bar(i+(j-2)*width,mean,width,color=COLORS[method],label=LABELS[method] if i==0 else None,
                        yerr=[[mean-min(v)],[max(v)-mean]],capsize=2)
            ax.set_xticks(range(len(datasets)),datasets);ax.set_ylabel(metric+' (seconds)');ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle('Source comparison: explicit zero tool wait, 0.5 JPS')
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(folder/f'dataset-comparison.{ext}',dpi=180)
        plt.close(fig);write_csv(folder/'dataset-comparison.csv',selected)
    for axis in ('jps','dataset','concurrency'):
        values=[r for r in pairs if r['axis']==axis and r['baseline']=='lmcache']
        if not values:continue
        groups=sorted({r['dataset'] for r in values});fig,axes=plt.subplots(1,len(groups),figsize=(7*len(groups),4),squeeze=False)
        for ax,dataset in zip(axes[0],groups):
            for method in LABELS:
                points=defaultdict(list)
                for r in values:
                    if r['dataset']==dataset and r['method']==method:points[float(r['axis_value'])].append(100*r['jct_reduction_fraction'])
                if not points:continue
                xs=sorted(points);means=[statistics.fmean(points[x]) for x in xs]
                ax.errorbar(xs,means,yerr=[[m-min(points[x]) for x,m in zip(xs,means)],
                    [max(points[x])-m for x,m in zip(xs,means)]],marker='o',capsize=3,label=LABELS[method],color=COLORS[method])
            ax.axhline(0,color='grey',lw=.8);ax.set(title=dataset,xlabel='JPS' if axis!='concurrency' else 'Target active Programs K (S=8)',
                ylabel='JCT reduction vs vLLM + LMCache (%)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('Exploratory paired differences; range across repeats')
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(folder/f'{axis}-exploratory-jct-change.{ext}',dpi=180)
        plt.close(fig);write_csv(folder/f'{axis}-exploratory-jct-change.csv',values)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=BASE/'live_results/raw')
    p.add_argument('--tail');p.add_argument('--output',type=Path,required=True);p.add_argument('--expected',type=int,default=90)
    p.add_argument('--no-plots',action='store_true');args=p.parse_args()
    scope=read_json(BASE/'current_scope.json');tail=args.tail or Path(scope['tail_progress']).parent.name
    matrix,missing=matrices(args.raw,tail,available_only=args.expected==0);n=sum(map(len,matrix.values()))
    if args.expected and n!=args.expected:raise ValueError(f'Expected {args.expected} complete rows, found {n}')
    args.output.mkdir(parents=True,exist_ok=False);reporting.collect=collect
    all_rows=[];all_groups=[]
    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        original=Figure.savefig
        def annotated(figure,*a,**kw):
            if not getattr(figure,'h800_caption',False):
                repeat_note=('range=min/max; '+scope.get('repeats_note_en','n in CSV')
                    if n==scope.get('target_runs') else 'partial repeats (see CSV)')+'; per-run curves n=1'
                figure.text(.5,-.025,'Exploratory | H800 80GB, Qwen3-8B BF16, S=8, CPU KV=64GiB | '+repeat_note+' | Kairos retrained',
                    ha='center',fontsize=6.5);figure.h800_caption=True
            kw.setdefault('bbox_inches','tight');return original(figure,*a,**kw)
        Figure.savefig=annotated
    for axis,runs in matrix.items():
        path=args.output/'matrices'/f'{axis}.json';write_json(path,dict(schema='local-real-run-matrix/1',runs=runs))
        rows,groups=reporting.report(path,args.output/axis,plots=False)
        invalid=[r for r in rows if r['status']!='complete']
        if invalid:raise ValueError('Source validation failed: '+json.dumps(invalid,ensure_ascii=False)[:3000])
        all_rows+=rows;all_groups+=groups
        if not args.no_plots:
            figures=args.output/axis/'figures';figures.mkdir()
            pretty=[dict(r,method=LABELS[r['method']],axis=('Target active Programs K (S=8)' if axis=='concurrency' else 'Program arrival rate (JPS)')) for r in rows]
            reporting.plot(pretty,figures);bottom_tables(rows,figures)
        print(json.dumps(dict(axis=axis,complete=len(rows))),flush=True)
    pairs=exploratory_pairs(all_rows)
    write_csv(args.output/'runs.csv',all_rows);write_csv(args.output/'summary.csv',all_groups)
    write_csv(args.output/'exploratory-pairs.csv',pairs)
    write_json(args.output/'coverage.json',dict(valid_completed_runs=n,target_runs=scope.get('target_runs',90),
        original_target_runs=scope.get('original_target_runs',90),closure_status=scope.get('closure_status'),
        not_yet_downloaded=missing,classification='exploratory',
        per_method_runs={m:sum(r['method']==m for r in all_rows) for m in LABELS},
        repetition_counts=scope.get('accepted_repetition_counts'),
        matrix_scope='JPS 0.1/2; SWE/Nebius explicit-zero-wait at 0.5 JPS; K 8/80 with 400 instances from 36 sources; '+scope.get('repeats_note_en','counts in CSV'),
        predictor='Kairos retrained with frozen pretrained MiniLM; held-out source IDs; not official CG1 weights',
        collection_note='Successful runs only. Kairos collection deadline is 7200 seconds; other methods use 600 seconds. Failure rate and stability are not estimated.',
        archival_note='Completed large logs are losslessly compressed with decompressed SHA proofs; streaming evidence loader preserves metric definitions',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if not args.no_plots:additional_plots(all_rows,pairs,args.output/'comparison-figures')
    print(json.dumps({'status':'complete','runs':n,'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
