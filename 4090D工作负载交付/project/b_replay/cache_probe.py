"""Independent cache-salt isolation probe; inconclusive layer counters fail closed."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from .common import read_json,write_json,sha
from .hardware import validate_gpu
from .metrics import samples
from .suite import MAIN_METHODS
from .client import cache_namespace,replay_metadata

PROTOCOL='h800-diagnostic-cache-salt/1'


def counter_delta(before,after,name):
    a={k:v for k,v in before.items() if k==name or k.startswith(name+'{')}
    b={k:v for k,v in after.items() if k==name or k.startswith(name+'{')}
    if not a or a.keys()!=b.keys() or any(b[k]<a[k] for k in a):return None
    return sum(b[k]-a[k] for k in a)


def assess(rows):
    by={r['phase']:r for r in rows}
    if set(by)!={'gpu_first_A','gpu_repeat_A','gpu_first_B','gpu_repeat_B','cpu_repeat_A','cpu_new_C'}:
        raise ValueError('Cache probe phases are incomplete')
    for phase in ('gpu_repeat_A','gpu_repeat_B'):
        row=by[phase]
        # vLLM 0.28 can omit prompt_tokens_details. The independently sampled
        # GPU *token* counter still proves reuse; never substitute missing=0.
        if (row.get('gpu_hits') is None or row['gpu_hits']<row['prompt_tokens']-32
                or (row.get('cached_tokens') is not None and row['cached_tokens']<row['prompt_tokens']-32)):
            raise ValueError('GPU same-instance reuse is absent or unobservable: '+phase)
    first=by['gpu_first_B']
    if first.get('gpu_hits')!=0 or (first.get('cached_tokens') is not None and first['cached_tokens']>16):
        raise ValueError('Copied input unexpectedly hit another instance GPU cache, or counters are missing')
    warm,cold=by['cpu_repeat_A'],by['cpu_new_C']
    for row in (warm,cold):
        if row.get('external_queries') is None or row['external_queries']<=0 or row.get('gpu_hits')!=0:
            raise ValueError('Cannot identify an external-cache lookup after GPU prefix reset')
    if warm.get('external_hits') is None or warm['external_hits']<=0 or cold.get('external_hits')!=0:
        raise ValueError('LMCache same-salt reuse / different-salt isolation is absent or unobservable')
    return dict(gpu_same_instance_reuse=True,gpu_cross_instance_isolated=True,
                lmcache_same_instance_reuse=True,lmcache_cross_instance_isolated=True,
                verified_both_layers=True)


async def _measure(args):
    import httpx
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];prefix='independent-isolation-'+str(time.monotonic_ns())
    salts={key:prefix+'-'+key for key in 'ABC'}
    async with httpx.AsyncClient(timeout=args.request_timeout,trust_env=False) as client:
        async def snapshot(label):
            response=await client.get(args.endpoint+'/metrics');response.raise_for_status()
            path=out/(label+'.prom');path.write_text(response.text);return samples(path)
        async def reset_gpu():
            response=await client.post(args.endpoint+'/reset_prefix_cache',params=dict(reset_connector='false'))
            response.raise_for_status()
            await asyncio.sleep(.2)
        async def request(phase,salt):
            before=await snapshot(phase+'-before');began=time.monotonic()
            metadata=replay_metadata(args.method,salts[salt],0,
                [dict(role='user',content='Independent synthetic cache-isolation probe')])
            # Activate the real method/STORE path, while leaving synthetic
            # one-token work outside the private recorded-output executor.
            metadata.pop('cont_replay_key')
            response=await client.post(args.endpoint+'/v1/completions',json=dict(model=args.model,
                prompt=[100+i%900 for i in range(4096)],max_tokens=1,ignore_eos=True,
                temperature=0,seed=1,stream=False,return_token_ids=True,vllm_xargs=metadata,
                **cache_namespace(salts[salt])))
            response.raise_for_status();body=response.json();usage=body.get('usage',{})
            if usage.get('completion_tokens')!=1:raise ValueError('Isolation probe did not complete its fixed one-token work')
            await asyncio.sleep(.5)
            after=await snapshot(phase+'-after')
            # Keep the actual selected metric name, never combine cache layers.
            gpu_name='vllm:gpu_prefix_cache_hits_total'
            if not any(k==gpu_name or k.startswith(gpu_name+'{') for k in after):
                gpu_name='vllm:prefix_cache_hits_total'
            row=dict(phase=phase,salt=salts[salt],prompt_tokens=4096,
                cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                gpu_hits=counter_delta(before,after,gpu_name),gpu_metric=gpu_name,
                external_hits=counter_delta(before,after,'vllm:external_prefix_cache_hits_total'),
                external_queries=counter_delta(before,after,'vllm:external_prefix_cache_queries_total'),
                wall_s=time.monotonic()-began,usage=usage)
            rows.append(row);write_json(out/'observations.json',rows)
        for phase,salt in [('gpu_first_A','A'),('gpu_repeat_A','A'),('gpu_first_B','B'),('gpu_repeat_B','B')]:
            await request(phase,salt)
        await reset_gpu();await request('cpu_repeat_A','A')
        await reset_gpu();await request('cpu_new_C','C')
    verdict=assess(rows);write_json(out/'validation.json',verdict)
    return dict(strict_complete=True,diagnostic='cache-salt',performance_claim_supported=False,**verdict)


async def measure(args):
    return await asyncio.wait_for(_measure(args),args.run_timeout)


def freeze(runs,output):
    members={};identity=None
    for run in runs:
        run=Path(run);state=read_json(run/'state.json');p=state['plan'];method=state['method']
        if method not in MAIN_METHODS or method in members:raise ValueError('Supply exactly one successful probe per main method')
        if state.get('protocol')!=PROTOCOL or state.get('status')!='complete' or not state.get('summary',{}).get('verified_both_layers'):
            raise ValueError('Missing complete two-layer diagnostic: '+str(run))
        validate_gpu(state['gpu'])
        if not state.get('gpu_ownership_verified') or not state.get('verified_model',{}).get('weights_verified'):
            raise ValueError('Isolation proof lacks GPU/model ownership validation')
        resources=read_json(run/'resources.json')
        observed=[r for r in resources if 'gpu' in r]
        if not observed or any(r.get('foreign_gpu_pids') or 'guard_error' in r or 'cleanup_error' in r for r in resources):
            raise ValueError('Isolation probe resource guard failed')
        rows=read_json(run/'b_replay/observations.json')
        for row in rows:
            before=samples(run/'b_replay'/(row['phase']+'-before.prom'))
            after=samples(run/'b_replay'/(row['phase']+'-after.prom'))
            for key,name in [('gpu_hits',row['gpu_metric']),('external_hits','vllm:external_prefix_cache_hits_total'),
                             ('external_queries','vllm:external_prefix_cache_queries_total')]:
                if row[key]!=counter_delta(before,after,name):raise ValueError('Probe counter evidence changed')
        assess(rows)
        key=dict(gpu_uuid=state['gpu']['uuid'],packages=state['packages'],model=p['model'],
                 runtime=p['runtime'],adapter_sha256=p['adapter_sha256'],cpu_kv_gib=p['cpu_kv_gib'])
        if identity is None:identity=key
        if key!=identity:raise ValueError('Mixed isolation-probe environments')
        members[method]=dict(state_sha256=sha((run/'state.json').read_bytes()),run=str(run.resolve()))
    if set(members)!=set(MAIN_METHODS):raise ValueError('All five configurations must pass the two-layer probe')
    result=dict(schema='h800-cache-isolation/1',verified_both_layers=True,methods=members,**identity)
    if Path(output).exists():raise FileExistsError(output)
    write_json(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a.runs,a.output)
