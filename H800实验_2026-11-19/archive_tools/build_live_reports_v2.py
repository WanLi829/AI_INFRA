"""Extended offline report: explicit units/ticks, long tails, and predictor timings."""
import csv
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

import build_live_reports as base
from b_replay.common import read_json,write_json,write_csv,stats
from streaming_evidence_v2 import collect

METRICS={
    'jct_mean_s':'Mean Program JCT (s)','jct_p95_s':'P95 Program JCT (s)',
    'ttft_mean_s':'Mean request TTFT (s)','ttft_p95_s':'P95 request TTFT (s)',
    'ttft_p99_s':'P99 request TTFT (s)','ttft_max_s':'Maximum request TTFT (s)',
    'output_tokens_per_s':'Output throughput (token/s)','programs_per_s':'Completed Programs / s',
    'program_completion_fraction':'Main-run Program completion fraction',
    'active_mean':'Time-weighted active Programs',
    'prefix_cache_ratio':'Engine prefix hit/query (fraction)',
    'external_prefix_cache_ratio':'External prefix hit/query (fraction)',
    'cpu_prefix_cache_ratio':'CPU prefix hit/query (fraction; NA if unavailable)',
    'pin_ratio_mean':'Sampled unique physical pin / KV budget',
    'pin_ratio_max':'Sampled maximum unique pin / KV budget',
    'kv_used_ratio_max':'Sampled maximum nonfree KV / KV budget',
    'queue_size_mean':'Sampled mean engine waiting requests',
    'active_target_fraction':'Fraction of target window at configured K',
    'first_admit_wait_p95_s':'P95 first engine admission wait (s)',
    'schedule_cost_sum_s':'Sampled scheduling time sum (s)',
    'controller_cost_sum_s':'Sampled controller time (s; nested in scheduling)',
    'gpu_peak_gib':'Sampled whole-device GPU memory peak (GiB)',
    'ram_peak_gib':'Sampled cgroup memory peak (GiB)',
    'gpu_util_mean_percent':'Time-weighted device GPU utilization (%; resource window)',
    'gpu_busy_ge90_fraction':'Resource-window time at GPU utilization >=90% (fraction)',
}
DATASETS={'heldout-v2':'mini-SWE: 36 source Programs','instances400':'400 instances from 36 mini-SWE sources',
    'swe_iso32':'mini-SWE: 32 Programs, zero wait','nebius_iso32':'Nebius: 32 Programs, zero wait'}
REVERSE={v:k for k,v in base.LABELS.items()}
original_report=base.reporting.report
original_signature=base.pair_signature


def resource_metrics(run):
    """Observed adjacent-sample intervals; includes startup and post-replay work."""
    path=Path(run)/'resources.json'
    samples=read_json(path) if path.is_file() else []
    samples=sorted((s for s in samples if isinstance(s.get('time'),(float,int))),key=lambda s:s['time'])
    covered=weighted=busy=0.0;count=0;peak=None
    for left,right in zip(samples,samples[1:]):
        interval=right['time']-left['time']
        try:util=float(left['gpu']['util'])
        except (TypeError,ValueError,KeyError):continue
        if not 0<=util<=100 or not 0<interval<=5:continue
        covered+=interval;weighted+=interval*util;busy+=interval*(util>=90);count+=1
        peak=util if peak is None else max(peak,util)
    span=samples[-1]['time']-samples[0]['time'] if len(samples)>1 else 0
    return dict(gpu_util_mean_percent=weighted/covered if covered else None,
        gpu_busy_ge90_fraction=busy/covered if covered else None,
        gpu_util_peak_percent=peak,gpu_util_interval_count=count,
        gpu_util_covered_s=covered,gpu_resource_observed_span_s=span,
        gpu_util_measured_fraction=covered/span if span else None)


def signature(state):
    result=original_signature(state)
    result.update(memory_limit_bytes=state['memory_before_start']['effective_limit_bytes'],
        cpu_quota_cores=state.get('cpu_inventory',{}).get('quota_cores'))
    return result


def expanded_report(matrix,output,plots=True):
    rows,groups=original_report(matrix,output,plots=False);output=Path(output)
    phases=defaultdict(lambda:dict(first_call=[],returning_call=[],errors=0))
    with (output/'requests.csv').open(encoding='utf-8-sig',newline='') as stream:
        for r in csv.DictReader(stream):
            if r.get('ttft_s') not in (None,''):
                phases[r['run_id']]['first_call' if int(r['turn_id'])==0 else 'returning_call'].append(float(r['ttft_s']))
            phases[r['run_id']]['errors']+=int(bool(r.get('error')))
    extra=set()
    for r in rows:
        summary=read_json(Path(r['result'])/'b_replay/summary.json')
        evidence=read_json(output/'evidence'/(r['run_id']+'.json'))
        r['expected_programs']=summary['expected_programs'];r['expected_calls']=summary['expected_calls']
        values=dict(ttft_max_s=summary['request_ttft_s']['max'],request_e2e_max_s=summary['request_e2e_s']['max'],
            tpot_max_s=summary['request_tpot_s']['max'],
            program_completion_fraction=r['completed_programs']/r['expected_programs'],
            call_completion_fraction=r['completed_calls']/r['expected_calls'],
            gpu_peak_gib=r['sampled_gpu_peak_mib']/1024,
            ram_peak_gib=r['sampled_cgroup_peak_bytes']/1024**3 if r.get('sampled_cgroup_peak_bytes') is not None else None)
        values.update(resource_metrics(r['result']))
        for phase in ('first_call','returning_call'):
            observed=stats(phases[r['run_id']][phase])
            for metric in ('mean','p95','p99','max'):values[phase+'_ttft_'+metric+'_s']=observed[metric]
        for phase,observed in evidence.get('predictor_host_time_s',{}).items():
            for metric in ('mean','p95','max'):values['predictor_'+phase+'_'+metric+'_s']=observed[metric]
        reuse=evidence.get('posterior_reuse',{})
        values['posterior_quote_hit_fraction']=reuse.get('quote_hits',0)/reuse['quotes'] if reuse.get('quotes') else None
        r.update(values);extra.update(values)
    for g in groups:
        members=[r for r in rows if r['status']=='complete' and r['dataset']==g['dataset']
            and r['axis_value']==g['axis_value'] and r['method']==g['method'] and r['ttl_seconds']==g['ttl_seconds']]
        for metric in extra:
            observed=[r[metric] for r in members if r.get(metric) is not None]
            summary=stats(observed)
            g[metric+'_n']=len(observed)
            for name in ('mean','min','max'):g[metric+'_'+name]=summary[name]
    write_csv(output/'runs.csv',rows);write_csv(output/'summary.csv',groups)
    report=read_json(output/'report.json');report.update(runs=rows,groups=groups)
    report['notes'].append('Extended long-tail, first/returning-call, memory, and predictor host-time metrics come from original raw records; timing parents/children are not added.')
    report['notes'].append('GPU utilization is whole-device, time-weighted over observed adjacent resource samples (interval <=5s). The resource window includes startup and post-replay work; missing/long gaps are excluded and coverage is reported. This is not useful kernel throughput or strict process-exclusive utilization.')
    write_json(output/'report.json',report)
    return rows,groups


def plot(rows,output):
    import matplotlib.pyplot as plt
    output=Path(output)
    for number,dataset in enumerate(sorted({r['dataset'] for r in rows})):
        selected=[r for r in rows if r['dataset']==dataset and r['status']=='complete']
        xs_all=sorted({float(r['axis_value']) for r in selected})
        for metric,label in METRICS.items():
            fig,ax=plt.subplots(figsize=(7.5,4.3));present=False
            for method in base.LABELS.values():
                points=defaultdict(list)
                for r in selected:
                    if r['method']==method and r.get(metric) is not None:points[float(r['axis_value'])].append(r[metric])
                if not points:continue
                xs=sorted(points);ys=[sum(points[x])/len(points[x]) for x in xs]
                ax.errorbar(xs,ys,yerr=[[y-min(points[x]) for x,y in zip(xs,ys)],
                    [max(points[x])-y for x,y in zip(xs,ys)]],label=method,
                    color=base.COLORS[REVERSE[method]],marker='o',capsize=3);present=True
            ax.set(xlabel=selected[0]['axis'],ylabel=label,title=DATASETS.get(dataset,dataset)+' (exploratory)')
            ax.set_xticks(xs_all,[f'{x:g}' for x in xs_all]);ax.grid(alpha=.2)
            if present:ax.legend(fontsize=8)
            else:ax.text(.5,.5,'NA: no matching measurements exposed',transform=ax.transAxes,ha='center')
            fig.tight_layout()
            for ext in ('png','svg'):fig.savefig(output/f'dataset{number}-{metric}.{ext}',dpi=180)
            plt.close(fig)
    for number,r in enumerate(rows):
        path=Path(r['result'])/'b_replay/concurrency.json'
        if not path.is_file():continue
        curve=read_json(path);segments=curve.get('segments',[])
        if not segments:continue
        fig,ax=plt.subplots(figsize=(8,3))
        ax.step([s['start_s'] for s in segments]+[segments[-1]['end_s']],
            [s['active'] for s in segments]+[0],where='post',label='Observed active Programs')
        if curve.get('target'):ax.axhline(curve['target'],color='grey',ls='--',label='Target K')
        if curve.get('target_window'):
            window=curve['target_window'];ax.axvspan(window['start_s'],window['end_s'],alpha=.12,color='green',label='Target window')
        ax.set(xlabel='Seconds from replay clock origin',ylabel='Active Programs',title=r['run_id'])
        ax.legend(fontsize=8);fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(output/f'run{number:04d}-active-programs.{ext}',dpi=180)
        plt.close(fig)


def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    save=Figure.savefig
    scope=read_json(base.BASE/'current_scope.json')
    def captioned(figure,*a,**kw):
        for text in figure.texts:
            old=text.get_text()
            if old.startswith('Exploratory | H800'):
                repeats='partial repeats; n in CSV' if 'partial repeats' in old else 'range=min/max; '+scope.get('repeats_note_en','n in CSV')
                text.set_text('Exploratory | H800 80GB, Qwen3-8B BF16, 27 CPU / 230GiB RAM, S=8, GPU KV=55.3GiB / CPU KV=64GiB\n'
                    +repeats+'; individual curves: one named run | Kairos retrained | successful runs only')
        return save(figure,*a,**kw)
    Figure.savefig=captioned
    base.collect=collect;base.pair_signature=signature;base.reporting.report=expanded_report;base.reporting.plot=plot
    base.main()
    output=Path(sys.argv[sys.argv.index('--output')+1]);coverage=read_json(output/'coverage.json')
    coverage.update(extension_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric_loader_sha256=hashlib.sha256((base.BASE/'streaming_evidence_v2.py').read_bytes()).hexdigest(),
        archival_interference_note='During later Kairos runs, lossless archival/backup used low-priority helper CPU 26 to prevent disk exhaustion. CPU/I/O interference is not ruled out. The first K80 Kairos run ended before archival began.')
    write_json(output/'coverage.json',coverage)


if __name__=='__main__':main()
