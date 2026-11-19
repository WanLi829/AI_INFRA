"""Propose a common natural H800 KV capacity, then verify an explicit five-method smoke."""
import argparse
from pathlib import Path
from .common import read_json,write_json,sha
from .comparison_results import inspect
from .hardware import validate_gpu
from .metrics import collect
from .suite import MAIN_METHODS

QWEN8B_BF16_BYTES_PER_TOKEN=2*36*8*128*2


def capacity(matrix_path,output,headroom_blocks=64,proposal=None):
    if type(headroom_blocks) is not int or headroom_blocks<0: raise ValueError('Nonnegative block headroom required')
    matrix=read_json(matrix_path); members={};identity=None
    for row in matrix['runs']:
        if row['method'] not in MAIN_METHODS: continue
        if row['method'] in members: raise ValueError('Use a single-condition, single-repetition five-method smoke matrix')
        measured,key=inspect(Path(row['result']),Path(row['bundle']))
        if measured['status']!='complete': raise ValueError('All five smoke runs must complete')
        if identity is None: identity=key
        if key!=identity: raise ValueError('The five smoke runs do not share one hardware/software/input/resource environment')
        state=read_json(Path(row['result'])/'state.json');validate_gpu(state['gpu'])
        pages=collect(Path(row['result']))['actual_kv_page_counts']
        if len(pages)!=1: raise ValueError('Missing or changing actual GPU KV page count')
        command=row['plan']['engine_command']; block_size=int(command[command.index('--block-size')+1])
        members[row['method']]=dict(gpu=state['gpu'],packages=state['packages'],plan=row['plan'],pages=pages[0],block_size=block_size)
    if set(members)!=set(MAIN_METHODS): raise ValueError('A complete five-method smoke is required')
    values=list(members.values());first=values[0]; budgets={v['plan']['gpu_kv_bytes'] for v in values}
    if len(budgets)!=1: raise ValueError('Mixed explicit KV budgets')
    explicit=next(iter(budgets)); total_pages=min(v['pages'] for v in values)
    if explicit is not None and len({v['pages'] for v in values})!=1:
        raise ValueError('Same explicit bytes did not produce the same actual KV page capacity')
    candidate=(total_pages-headroom_blocks)*first['block_size']*QWEN8B_BF16_BYTES_PER_TOKEN
    if candidate<=0:raise ValueError('Insufficient natural capacity')
    proposal_hash=None
    if explicit is not None:
        if proposal is None: raise ValueError('Explicit main-capacity smoke must reference the preceding automatic-capacity proposal')
        natural=read_json(proposal)
        for field,expected in (('schema','h800-common-kv/1'),('verified_explicit_five_method_smoke',False),
                ('candidate_gpu_kv_bytes',explicit),('adapter_sha256',first['plan']['adapter_sha256']),
                ('runtime',first['plan']['runtime']),('model',first['plan']['model']),('packages',first['packages'])):
            if natural.get(field)!=expected:raise ValueError('Explicit smoke does not match natural-capacity proposal: '+field)
        if natural['gpu']['uuid']!=first['gpu']['uuid']: raise ValueError('Capacity proposal used another GPU')
        proposal_hash=sha(Path(proposal).read_bytes())
    result=dict(schema='h800-common-kv/1',gpu=first['gpu'],packages=first['packages'],
        model=first['plan']['model'],runtime=first['plan']['runtime'],adapter_sha256=first['plan']['adapter_sha256'],
        cpu_kv_gib=first['plan']['cpu_kv_gib'],verified_explicit_five_method_smoke=explicit is not None,
        common_gpu_kv_bytes=explicit,candidate_gpu_kv_bytes=candidate if explicit is None else explicit,
        actual_pages_by_method={k:v['pages'] for k,v in members.items()},
        headroom_blocks=headroom_blocks if explicit is None else 0,
        matrix_sha256=sha(Path(matrix_path).read_bytes()),
        natural_proposal_sha256=proposal_hash,
        note='Automatic sizing proposes the minimum observed natural capacity minus named headroom. Verify it in a fresh explicit-capacity five-method smoke before main results; artificial smaller budgets belong to kv-pressure.')
    if Path(output).exists(): raise FileExistsError(output)
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--headroom-blocks',type=int,default=64)
    p.add_argument('--proposal',type=Path)
    a=p.parse_args();capacity(a.matrix,a.output,a.headroom_blocks,a.proposal)
