"""Offline source-aware inputs; missing feedback timestamps stay missing."""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys

from .common import ROOT,read_json,write_json,sha
from .client import load_bundle
from .datasets import bootstrap,expand,save_bundle,describe,split_audit
from .server import RUNTIME_ROOT


def nebius_actions(source):
    trajectory=source['trajectory']
    if isinstance(trajectory,str): trajectory=json.loads(trajectory)
    actions=[]
    for step in trajectory:
        if step['role']!='ai': continue
        message=dict(role='assistant',content=step.get('text') or step.get('system_prompt'))
        if not isinstance(message['content'],str): raise ValueError('Missing recorded assistant body')
        blocks=re.findall(r'```(?:bash|sh)?\s*\n(.*?)\n```',message['content'],re.S)
        try: words=shlex.split(blocks[-1].strip()) if blocks else []
        except ValueError: words=[]
        terminal=words==['submit']
        actions.append(dict(tool=words[0] if words else None,terminal=terminal,
            evidence='explicit-submit-action' if terminal else 'current-tool-action' if words else 'assistant-only',
            assistant_sha256=sha(json.dumps(message,sort_keys=True).encode())))
    if not actions or not actions[-1]['terminal']:
        raise ValueError('A natural submitted trajectory is required')
    return actions


def prepare_zero_wait(name,output):
    manifest,trace,visible=load_bundle(ROOT.parent/'upstream/zero-wait-bundles'/name)
    if trace['timing_provenance']['kind']!='explicit_zero_wait_control':
        raise ValueError('The sensitivity must be explicitly labeled zero-wait')
    sys.path.insert(0,str(RUNTIME_ROOT))
    from kairos.replay.completion import recorded_action
    actions={};audit=[]
    nebius=name=='nebius_iso32'
    raw_hashes={r['program_id']:r['sha256'] for r in read_json(ROOT.parent/'upstream/raw-teacher-manifest.json')['files']}
    for p in trace['programs']:
        pid=p['id'];source_id=p.get('source_task_id',pid)
        raw=ROOT.parent/'upstream'/('nebius/selected' if nebius else 'raw_teacher_selected')/(pid+('.json' if nebius else '.traj.json'))
        expected=p.get('source_sha256') if nebius else raw_hashes[pid]
        if sha(raw.read_bytes())!=expected: raise ValueError('Raw action source hash mismatch: '+pid)
        source=read_json(raw)
        actions[pid]=nebius_actions(source) if nebius else [recorded_action(m) for m in source['messages'] if m['role']=='assistant']
        if len(actions[pid])!=len(p['turns']): raise ValueError('Action/trace mismatch')
        if any(t['gap_after_s']!=0 for t in p['turns']): raise ValueError('Nonzero wait in isolation control')
        p['source_id']=source_id
        audit.append(dict(instance_id=pid,source_id=source_id,source_revision='68195a1450865274106246d0d0296a1d6807b88e' if nebius else '58389eb5426db5a3ae6159faa37498072b4735ff',
            source_sha256=expected,split='service-isolation-development',exit_status='Submitted',
            gap_evidence='Explicit zero wait; original tool timestamps unavailable for Nebius'))
    manifest.update(dataset_id='nebius-swe-agent-zero-wait' if nebius else 'mini-swe-verified-zero-wait',
        source_family='nebius/SWE-agent-trajectories' if nebius else 'mini-swe-verified',
        timing_kind='explicit_zero_wait_control',predictor_split_audit=split_audit(trace,None))
    save_bundle(output,manifest,trace,visible,dict(schema='recorded-current-actions-v1',actions=actions),dict(programs=audit))
    describe(output,Path(output)/'features')


def prepare(output, training_ids=None):
    output=Path(output)
    bootstrap(output,training_ids)
    expand(output/'heldout-v2',400,2026111800,output/'instances400')
    for name in ('swe_iso32','nebius_iso32'):
        prepare_zero_wait(name,output/name)
    status=dict(schema='h800-input-status/1',gpu_validated=False,
        real_feedback_source='mini-swe-verified',independent_sources=36,parallel_instances=400,
        cross_dataset_observed_timing_ready=False,cross_source_zero_wait_sensitivity_ready=True,
        current_predictor_independence='unknown' if training_ids is None else 'audited from supplied exclusion manifest',
        note='36 sources and 400 isolated instances are different quantities. Cache-salt GPU/LMCache validation remains pending.')
    write_json(output/'status.json',status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'inputs/prepared')
    p.add_argument('--training-ids',type=Path);a=p.parse_args()
    print(json.dumps(prepare(a.output,a.training_ids),ensure_ascii=False,indent=2))
