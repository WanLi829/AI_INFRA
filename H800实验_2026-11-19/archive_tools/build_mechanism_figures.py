"""Plot one named paired repetition per condition from scheduler samples and K(t)."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'project'),str(BASE/'project/runtime/Kairos')]
from b_replay.common import write_csv
from build_live_reports import LABELS,COLORS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();output=args.report/'mechanism';output.mkdir(parents=True,exist_ok=False)
    with (args.report/'runs.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    groups=defaultdict(list)
    for r in rows:
        if r['repetition']=='1':groups[(r['axis'],r['dataset'],float(r['axis_value']))].append(r)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index=[]
    for (axis,dataset,value),members in sorted(groups.items()):
        assert {r['method'] for r in members}==set(LABELS)
        fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
        samples=[];active=[];ids=[]
        for r in sorted(members,key=lambda r:list(LABELS).index(r['method'])):
            run=Path(r['result']);client=json.loads((run/'b_replay/state.json').read_bytes())
            summary=json.loads((run/'b_replay/summary.json').read_bytes())
            origin=client['clock_origin_monotonic_s'];end=summary['wall_from_clock_zero_s']
            evidence=json.loads((args.report/axis/'evidence'/(r['run_id']+'.json')).read_bytes())
            observed=[]
            for s in evidence['queue_kv_samples']:
                t=s['time']-origin
                if t<0 or t>end:continue
                observed.append(dict(run_id=r['run_id'],method=r['method'],time_s=t,
                    engine_waiting=s['waiting'],engine_running=s['running'],pin_ratio=s.get('unique_pin_ratio'),
                    kv_nonfree_ratio=1-s['free_blocks']/s['total_blocks'] if s.get('total_blocks') else None))
            assert observed,'No matched monotonic samples for '+r['run_id']
            samples+=observed;ids.append(r['run_id'])
            for ax,metric in zip(axes.ravel()[:3],('engine_waiting','pin_ratio','kv_nonfree_ratio')):
                points=[s for s in observed if s[metric] is not None]
                ax.plot([s['time_s'] for s in points],[s[metric] for s in points],
                    color=COLORS[r['method']],label=LABELS[r['method']],lw=1,alpha=.85)
            curve=json.loads((run/'b_replay/concurrency.json').read_bytes())
            segments=curve['segments']
            axes[1,1].step([s['start_s'] for s in segments]+[segments[-1]['end_s']],
                [s['active'] for s in segments]+[0],where='post',label=LABELS[r['method']],
                color=COLORS[r['method']],lw=1,alpha=.85)
            active += [dict(run_id=r['run_id'],method=r['method'],**s) for s in segments]
        for ax,label in zip(axes.ravel(),('Engine waiting requests','Unique physical pin / KV budget',
                                        'Nonfree physical KV / KV budget','Active Programs (tool wait included)')):
            ax.set_ylabel(label);ax.set_xlabel('Seconds from replay clock origin');ax.grid(alpha=.2)
        for ax in (axes[0,1],axes[1,0]):ax.set_ylim(-.02,1.02)
        if axis=='concurrency':axes[1,1].axhline(value,color='grey',ls='--',lw=.8,label=f'Target K={value:g}')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{axis} | {dataset} | value={value:g} | paired repetition 1')
        fig.text(.5,.005,'Exploratory | H800 80GB, Qwen3-8B BF16, 27 CPU / 230GiB RAM, S=8, GPU KV=55.3GiB, CPU KV=64GiB\n'
            'Kairos retrained | individual sampled traces, no error bars | missing counters remain NA | full run IDs in CSV/index',
            ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.055,1,.96));stem=f'trace-{axis}-{dataset}-{value:g}-r1'
        for extension in ('png','svg'):fig.savefig(output/(stem+'.'+extension),dpi=180)
        plt.close(fig);write_csv(output/(stem+'-samples.csv'),samples);write_csv(output/(stem+'-active.csv'),active)
        index.append(dict(figure=stem,axis=axis,dataset=dataset,value=value,repetition=1,
            run_ids=ids,samples=len(samples),active_segments=len(active)))
    (output/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(figures=len(index),output=str(output)),ensure_ascii=False))


if __name__=='__main__':main()
