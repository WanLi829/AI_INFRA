"""Read-only CPU checks and extraction of one archived interface example."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--implementation', type=Path, required=True)
    ap.add_argument('--kairos', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    impl, kairos, out = a.implementation.resolve(), a.kairos.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    checks = []
    def run(name, args, cwd, extra_env=None):
        result = subprocess.run(args, cwd=cwd, env={**env, **(extra_env or {})},
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
        (out / (name + '.log')).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
        checks.append(dict(name=name, command=args, cwd=str(cwd), exit_code=result.returncode))
        print(name, result.returncode)
    py = [sys.executable, '-B', '-X', 'utf8']
    run('implementation_ttl', py + ['tasks/continuum/test_ttl.py'], impl)
    run('simulation_smoke', py + ['tasks/continuum/run.py', '--load', 'mixed', '--n', '4',
        '--policy', 'all', '--seed', '0', '--out', str(out/'simulation.json')], impl)
    run('kairos_verify', py + ['-m', 'kairos.verify'], kairos)
    run('kairos_ttl_regression', py + [str(impl/'tasks/continuum/test_ttl_policy_kairos.py')],
        kairos, {'PYTHONPATH': str(kairos)})
    run('continuum_dry_run', py + ['-m', 'kairos.run', '--method', 'continuum',
        '--model', '/MODEL_PATH_NOT_VERIFIED', '--output', str(out/'not_executed'), '--dry-run'], kairos)
    trace = json.loads((kairos/'data/workload/trace.json').read_bytes())
    visible = json.loads((kairos/'data/workload/visible_messages.json').read_bytes())
    protocol = json.loads((kairos/'data/workload/visible_protocol.json').read_bytes())
    program = next(p for p in trace['programs'] if not p.get('predecessor'))
    pid = program['id']
    signatures = {(r['program'], r['stage']): r for r in protocol['request_signatures']}
    tokens, lengths = [], []
    for stage, turn in enumerate(program['turns']):
        tokens = tokens[:turn.get('keep_prefix_tokens', len(tokens))] + turn['added_token_ids']
        digest = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
        assert digest == signatures[(pid, stage)]['prompt_sha256']
        lengths.append(len(tokens))
    assert len(visible[pid]) == len(program['turns'])
    sample = out/'interface_example'
    sample.mkdir()
    def dump(path, obj):
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    dump(sample/'trace.sample.json', {'programs': [program]})
    dump(sample/'visible_messages.sample.json', {pid: visible[pid]})
    dump(sample/'signatures.reference.json', [r for r in protocol['request_signatures'] if r['program'] == pid])
    revisions = {name: subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
                 for name, root in [('implementation', impl), ('kairos', kairos)]}
    dump(out/'check_results.json', dict(python=sys.version, revisions=revisions, checks=checks,
         sample=dict(program=pid, turns=len(program['turns']), prompt_lengths=lengths,
                     program_keys=list(program), turn_keys=list(program['turns'][0]),
                     signature_check='passed; no tokenizer/model execution',
                     status='interface example only; not a standalone runnable release')))
    print('Sample:', pid, len(lengths), 'turns; signature checks passed')
    print('All commands passed:', all(c['exit_code'] == 0 for c in checks))


if __name__ == '__main__':
    main()
