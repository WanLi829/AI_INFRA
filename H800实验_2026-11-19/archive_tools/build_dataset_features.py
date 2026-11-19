"""Describe the actual two-source, zero-wait inputs without editing frozen bundles."""
import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import sys

BASE=Path(__file__).resolve().parent;PROJECT=BASE/'project'
sys.path[:0]=[str(PROJECT),str(PROJECT/'runtime/Kairos')]
from b_replay.common import read_json,write_json,write_csv
from b_replay.datasets import describe


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=read_json(BASE/'live_results/kairos02/assets/RETRAINED_MANIFEST.json')
    fitting=set(manifest['training_ids'])|set(manifest['validation_ids'])
    summaries=[];calls=[];programs=[]
    for dataset in ('swe_iso32','nebius_iso32'):
        original=PROJECT/'inputs/prepared'/dataset;copied=args.output/'inputs'/dataset
        shutil.copytree(original,copied)
        summary=describe(copied,args.output/(dataset+'-calls'))
        rows=list(csv.DictReader((args.output/(dataset+'-calls.csv')).open()))
        progs=list(csv.DictReader((args.output/(dataset+'-calls-programs.csv')).open()))
        assert not fitting&{r['source_task'] for r in progs},'Predictor fitting/evaluation overlap'
        for r in rows:r['dataset']=dataset
        for r in progs:r.update(dataset=dataset,retrained_predictor_test=True)
        calls+=rows;programs+=progs
        summaries.append(dict(dataset=dataset,programs=summary['programs'],independent_sources=summary['independent_sources'],
            calls=summary['calls'],turns_mean=summary['turns']['mean'],turns_p95=summary['turns']['p95'],
            input_tokens_mean=summary['input_tokens']['mean'],input_tokens_p95=summary['input_tokens']['p95'],
            output_tokens_mean=summary['output_tokens']['mean'],output_tokens_p95=summary['output_tokens']['p95'],
            tool_gap_mean_s=summary['tool_gap_s']['mean'],tool_gap_p95_s=summary['tool_gap_s']['p95'],
            timing_protocol='explicit_zero_tool_wait',retrained_predictor_overlap=0,
            input_manifest_sha256=hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest()))
    write_csv(args.output/'dataset_features.csv',summaries);write_csv(args.output/'feature_calls.csv',calls)
    write_csv(args.output/'feature_programs.csv',programs)
    write_json(args.output/'dataset_features.json',dict(datasets=summaries,
        note='Only these frozen test subsets; all tool gaps are deliberately zero. LCP measures input reuse opportunity, not actual cache hits. Predictor fitting IDs were excluded.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['SWE (32 Programs)','Nebius (32 Programs)'];datasets=['swe_iso32','nebius_iso32']
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    specifications=[(programs,'turns','LLM calls per Program'),(calls,'input_tokens','Input tokens per call'),
        (calls,'output_tokens','Output tokens per call'),(calls,'prior_input_lcp_ratio','Consecutive-input LCP / input length')]
    bottom=[]
    for ax,(rows,key,title) in zip(axes.ravel(),specifications):
        arrays=[[float(r[key]) for r in rows if r['dataset']==d and r.get(key) not in (None,'')] for d in datasets]
        ax.boxplot(arrays,tick_labels=names,showmeans=True);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
        for d,values in zip(datasets,arrays):bottom.extend(dict(dataset=d,metric=key,value=value) for value in values)
    fig.suptitle('Frozen test workload features; Qwen3-8B tokenizer')
    fig.text(.5,-.025,'Per-call panels: SWE n=352, Nebius n=385. Boxes=Q1-Q3; orange=median; green=mean; whiskers=1.5 IQR.\n'
        'Both sources use explicit zero tool wait. LCP is reuse opportunity, not a measured hit rate.',ha='center',fontsize=7)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(args.output/('dataset-features.'+ext),dpi=180,bbox_inches='tight')
    write_csv(args.output/'dataset-features.csv',bottom)
    print(args.output)


if __name__=='__main__':main()
