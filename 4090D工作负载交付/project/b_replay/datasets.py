"""Prepare audited replay bundles, isolate repeated instances, and summarize features."""
import argparse
import copy
from pathlib import Path
import random
import sys
import tempfile
import zipfile

from .client import load_bundle
from .common import read_json, write_json, write_csv, sha, stats, lcp
from .server import RUNTIME_ROOT


def save_bundle(output, manifest, trace, visible, actions, audit):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {'trace.json': trace, 'visible_messages.json': visible,
             'completed_actions.json': actions, 'data_audit.json': audit}
    for name, content in files.items():
        write_json(output/name, content)
    manifest = copy.deepcopy(manifest)
    manifest.update(schema_version='b-replay-bundle/2.0', gpu_validated=False,
        programs=len(trace['programs']), program_ids=[p['id'] for p in trace['programs']],
        independent_sources=len({p.get('source_id',p['id']) for p in trace['programs']}),
        calls=sum(len(p['turns']) for p in trace['programs']),
        total_output_tokens=sum(t['output_tokens'] for p in trace['programs'] for t in p['turns']),
        files={n: sha((output/n).read_bytes()) for n in files})
    write_json(output/'manifest.json', manifest)
    load_bundle(output)


def prepare(bundle, intermediate, output, dataset_id, training_ids=None):
    """Enrich B's mini-SWE bundles using the archived current assistant evidence."""
    manifest, trace, visible = load_bundle(bundle)
    if not Path(intermediate).is_dir():
        from .common import ROOT
        intermediate = ROOT.parent/'upstream/intermediate'
    sys.path.insert(0, str(RUNTIME_ROOT))
    from kairos.replay.completion import recorded_action
    actions, rows = {}, []
    for p in trace['programs']:
        source_id = p.get('source_id', p['id'])
        source_path = Path(intermediate)/(source_id+'.json.gz')
        source = read_json(source_path)
        if len(source['turns']) != len(p['turns']):
            raise ValueError('Source turn count differs: '+source_id)
        actions[p['id']] = []
        for i, turn in enumerate(p['turns']):
            recorded = source['turns'][i]
            # Compare observed messages and timing; never infer a final action
            # from the fact that the recorded trace ends.
            if source['messages'][:recorded['message_index']] != visible[p['id']][i]:
                raise ValueError('Visible history differs from source: '+source_id)
            if i+1 < len(p['turns']) and (recorded['gap_after_s'] is None or abs(recorded['gap_after_s']-turn['gap_after_s']) > 1e-8):
                raise ValueError('Tool gap differs from source: '+source_id)
            actions[p['id']].append(recorded_action(source['messages'][recorded['message_index']]))
        p['source_id'] = source_id
        p['cache_salt'] = p['id']
        rows.append(dict(instance_id=p['id'], source_id=source_id, split=source.get('split'),
            exit_status=source.get('exit_status'), source_revision=source.get('source_revision'),
            source_sha256=source.get('source_sha256'), intermediate_sha256=sha(source_path.read_bytes()),
            gaps='observed; checked against source', calls=len(p['turns'])))
    manifest['dataset_id'] = dataset_id
    manifest['source_family'] = trace.get('source_dataset', dataset_id)
    manifest['timing_kind'] = 'observed_tool_feedback'
    manifest['predictor_split_audit'] = split_audit(trace, training_ids)
    save_bundle(output, manifest, trace, visible,
        dict(schema='recorded-current-actions-v1', actions=actions), dict(programs=rows))
    describe(output, Path(output)/'features')


def split_audit(trace, path):
    if path is None:
        return dict(independent_test=False, status='unknown-current-joint-training-split')
    payload = read_json(path)
    if payload.get('runtime_commit') != 'a3dbac12eac80d6444222ca3e2e9592fef029158':
        raise ValueError('Training exclusion manifest must identify the frozen runtime')
    if not isinstance(payload.get('training_ids'), list) or not isinstance(payload.get('validation_ids'), list):
        raise ValueError('Provide both training_ids and validation_ids')
    if not payload['training_ids'] or not all(isinstance(s,str) and s for s in payload['training_ids']+payload['validation_ids']):
        raise ValueError('Provide real nonempty source task training identities; empty placeholders are not an audit')
    excluded = set(payload['training_ids']) | set(payload['validation_ids'])
    sources = {p.get('source_id',p['id']) for p in trace['programs']}
    overlap = sorted(sources & excluded)
    return dict(independent_test=not overlap, status='audited', overlap=overlap,
        exclusion_manifest_sha256=sha(Path(path).read_bytes()),
        source_identity_rule=payload.get('source_identity_rule','exact source task IDs'))


def expand(bundle, instances, seed, output):
    manifest, trace, visible = load_bundle(bundle)
    if type(instances) is not int or instances < len(trace['programs']):
        raise ValueError('instances must be at least the source bundle size')
    actions = read_json(Path(bundle)/'completed_actions.json')['actions']
    source = list(trace['programs'])
    rng = random.Random(seed)
    programs, new_visible, new_actions, mapping = [], {}, {}, []
    while len(programs) < instances:
        rng.shuffle(source)
        for p in source:
            if len(programs) == instances:
                break
            item = copy.deepcopy(p)
            identity = f"instance-{len(programs):05d}-{p['id']}"
            item.update(id=identity, source_id=p.get('source_id', p['id']), cache_salt=identity)
            programs.append(item)
            new_visible[identity] = visible[p['id']]
            new_actions[identity] = actions[p['id']]
            mapping.append(dict(instance_id=identity, source_id=item['source_id'], source_instance=p['id']))
    trace = dict(trace, programs=programs)
    manifest = dict(manifest, subset='expanded-independent-cache-instances', expansion_seed=seed,
        source_bundle_sha256=sha((Path(bundle)/'manifest.json').read_bytes()))
    save_bundle(output, manifest, trace, new_visible,
        dict(schema='recorded-current-actions-v1',actions=new_actions),
        dict(instances=mapping, programs=read_json(Path(bundle)/'data_audit.json').get('programs',[]),
             independent_sources=len({p['source_id'] for p in programs}),
             cross_instance_cache='unique salt; no automatic sharing of copied trajectories'))
    describe(output, Path(output)/'features')


def describe(bundle, output):
    manifest, trace, _ = load_bundle(bundle)
    rows, program_rows = [], []
    actions = read_json(Path(bundle)/'completed_actions.json')['actions']
    audit = read_json(Path(bundle)/'data_audit.json')
    provenance = {r['source_id']:r for r in audit.get('programs',[]) if 'source_id' in r}
    for p in trace['programs']:
        previous = []
        for i,t in enumerate(p['turns']):
            prompt = previous[:t['keep_prefix_tokens']] + t['added_token_ids']
            rows.append(dict(program_id=p['id'],source_id=p.get('source_id',p['id']),turn=i,
                input_tokens=len(prompt),output_tokens=t['output_tokens'],
                gap_after_s=t['gap_after_s'] if i+1 < len(p['turns']) else None,
                prior_input_lcp_ratio=lcp(previous,prompt)/len(prompt),
                prior_input_lcp_tokens=lcp(previous,prompt),
                input_growth_tokens=len(prompt)-len(previous) if i else None,
                action_category=actions[p['id']][i]['tool']))
            previous=prompt
        calls=[r for r in rows if r['program_id']==p['id']]
        identity=provenance.get(p.get('source_id',p['id']),{})
        program_rows.append(dict(dataset_id=manifest.get('dataset_id'),
            source_revision=identity.get('source_revision',trace.get('source_revision')),
            source_task=p.get('source_id',p['id']), source_trajectory=identity.get('source_trajectory',p.get('source_id',p['id'])),
            instance_id=p['id'],split=identity.get('split',p.get('split','unknown-current-joint-split')),
            end_type=identity.get('exit_status',p.get('exit_status','unknown')),
            turns=len(calls),input_tokens=sum(r['input_tokens'] for r in calls),
            output_tokens=sum(r['output_tokens'] for r in calls),
            gap_total_s=sum(r['gap_after_s'] for r in calls if r['gap_after_s'] is not None),
            input_growth_tokens=calls[-1]['input_tokens']-calls[0]['input_tokens'],
            input_prefix_opportunity=sum(r['prior_input_lcp_tokens'] for r in calls)/sum(r['input_tokens'] for r in calls),
            source_sha256=identity.get('source_sha256',p.get('source_sha256')),
            timing_kind=manifest.get('timing_kind','unknown'),
            tokenizer=manifest['model'],tokenizer_revision=manifest['model_revision'],
            valid=True,exclusion_reason=None,
            independent_current_predictor_test=manifest.get('predictor_split_audit',{}).get('independent_test',False)))
    summary=dict(dataset_id=manifest.get('dataset_id','unregistered'),
        programs=manifest['programs'], independent_sources=len({p.get('source_id',p['id']) for p in trace['programs']}),
        calls=manifest['calls'], turns=stats([len(p['turns']) for p in trace['programs']]),
        input_tokens=stats([r['input_tokens'] for r in rows]), output_tokens=stats([r['output_tokens'] for r in rows]),
        tool_gap_s=stats([r['gap_after_s'] for r in rows]),
        prefix_note='LCP of consecutive INPUT sequences, excludes preceding generated output; potential reuse, not measured cache hits',
        split_audit=manifest.get('predictor_split_audit'))
    output=Path(output)
    write_json(output.with_suffix('.json'),summary)
    write_csv(output.with_suffix('.csv'),rows)
    write_csv(Path(bundle)/'dataset_manifest.csv',program_rows)
    write_csv(output.with_name(output.name+'-programs').with_suffix('.csv'),program_rows)
    write_json(output.with_name(output.name+'-programs').with_suffix('.json'),
        {k:stats([r[k] for r in program_rows]) for k in ('turns','input_tokens','output_tokens','gap_total_s','input_growth_tokens','input_prefix_opportunity')})
    return summary


def bootstrap(output, training_ids=None):
    """Rebuild the three supplied mini-SWE subsets; no network or GPU."""
    from .common import ROOT
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    intermediate=ROOT.parent/'upstream/intermediate'
    for subset in ('pilot_1','pilot_14'):
        prepare(ROOT/'bundles/Qwen3-8B'/subset,intermediate,output/(subset+'-v2'),
                'mini-swe-verified',training_ids)
    prefix='bundles/Qwen3-8B/heldout_candidates/'
    with tempfile.TemporaryDirectory() as temp:
        source=Path(temp)
        with zipfile.ZipFile(ROOT.parent/'upstream/bundles.zip') as archive:
            for entry in archive.namelist():
                if entry.startswith(prefix):
                    name=entry[len(prefix):]
                    if not name: continue
                    if Path(name).name!=name: raise ValueError('Unexpected nested archive member')
                    (source/name).write_bytes(archive.read(entry))
        prepare(source,intermediate,output/'heldout-v2','mini-swe-verified',training_ids)


def audit_split(bundle, training_ids, output):
    manifest,trace,visible=load_bundle(bundle)
    manifest['predictor_split_audit']=split_audit(trace,training_ids)
    if not manifest['predictor_split_audit']['independent_test']:
        raise ValueError('Overlap with the current predictor; select an independent subset before running')
    save_bundle(output,manifest,trace,visible,read_json(Path(bundle)/'completed_actions.json'),
                read_json(Path(bundle)/'data_audit.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('bootstrap')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training-ids',type=Path)
    p=sub.add_parser('audit-split')
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--training-ids',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--intermediate',type=Path,required=True)
    p.add_argument('--dataset-id',required=True)
    p.add_argument('--training-ids',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('expand')
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--instances',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('describe')
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args()); command=args.pop('command').replace('-','_')
    globals()[command](**args)


if __name__=='__main__':
    main()
