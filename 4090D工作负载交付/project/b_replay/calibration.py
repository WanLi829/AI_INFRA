"""Independent synthetic service/DMA probes; never auto-replace a cost profile."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from .common import ROOT,read_json,write_json,stats

PROMPTS=(512,2048,4096,8192,12000)


def fit(records):
    import numpy as np
    from scipy.optimize import nnls
    rows=[r for r in records if r['kind']=='prefill' and r['output_tokens']==1 and r['success']]
    if len({r['prompt_tokens'] for r in rows})<4:raise ValueError('At least four different prompt sizes required')
    # Client TTFT also contains the first decode and transport/control overhead.
    # This schema is intentionally not accepted as an automatic deployment profile.
    x=np.array([[1,r['prompt_tokens'],r['prompt_tokens']*(r['prompt_tokens']+1)/2] for r in rows],float)
    y=np.array([r['ttft_s'] for r in rows]);coef,_=nnls(x,y)
    predictions=x@coef
    groups={}
    for concurrency in (1,2,4,8):
        values=[r['observed_decode_interval_s'] for r in records if r['kind']=='decode' and r['success']
                and r['client_concurrency']==concurrency and r['observed_decode_interval_s'] is not None]
        groups[str(concurrency)]=stats(values)
    return dict(schema='endpoint-observed-service-profile/1',prefill_ttft_coefficients=coef.tolist(),
        prefill_ttft_fit_mae_s=float(np.abs(predictions-y).mean()),decode_interval_by_client_concurrency=groups,
        deployment_profile_ready=False,reference_profile_unchanged=True,
        caveat='Host endpoint clocks include first-token/transport overhead; long prefill may be chunked. Client concurrency does not prove GPU batch size. Validate against native timing before translating these observations into a deployment model.')


async def _measure(args):
    import httpx
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);records=[]
    async with httpx.AsyncClient(timeout=120,trust_env=False) as client:
        async def request(n,o,kind,concurrency,repeat):
            payload=dict(model=args.model,prompt=[100+i%900 for i in range(n)],max_tokens=o,
                ignore_eos=True,temperature=0,seed=1,return_token_ids=True,stream=True,
                cache_salt=f'calibration-{kind}-{n}-{o}-{concurrency}-{repeat}-{time.monotonic_ns()}')
            started=time.perf_counter();chunks=[];tokens=0;error=None
            try:
                async with client.stream('POST',args.endpoint+'/v1/completions',json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'):continue
                        data=line[5:].strip()
                        if data=='[DONE]':break
                        body=json.loads(data)
                        for choice in body.get('choices',[]):
                            ids=choice.get('token_ids') or []
                            if ids:chunks.append((time.perf_counter()-started,len(ids)));tokens+=len(ids)
                if tokens!=o:raise ValueError('Calibration output count mismatch')
            except Exception as exc:error=repr(exc)
            row=dict(kind=kind,prompt_tokens=n,output_tokens=o,actual_output_tokens=tokens,
                client_concurrency=concurrency,repeat=repeat,success=error is None,error=error,
                ttft_s=chunks[0][0] if chunks else None,e2e_s=time.perf_counter()-started,
                observed_decode_interval_s=(chunks[-1][0]-chunks[0][0])/(tokens-1) if len(chunks)>1 and tokens>1 else None,
                stream_chunks=chunks)
            records.append(row)
            with (out/'probe-requests.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
            if error:raise RuntimeError(error)
        await request(256,4,'warmup',1,0)
        for n in PROMPTS:
            for repeat in range(3):await request(n,1,'prefill',1,repeat)
        for concurrency in (1,2,4,8):
            for repeat in range(3):
                await asyncio.gather(*(request(1024,129,'decode',concurrency,repeat) for _ in range(concurrency)))
    write_json(out/'observations.json',records);write_json(out/'candidate-profile.json',fit(records))
    summary=dict(strict_complete=True,evidence='Independent synthetic service calibration; not experiment performance')
    write_json(out/'summary.json',summary)
    return summary


async def measure(args):
    return await asyncio.wait_for(_measure(args),timeout=args.run_timeout)


def serving(output,model):
    from . import comparison
    comparison.COMMON_PROTOCOL='independent-hardware-calibration/1'
    from .client import make_arrivals
    bundle=ROOT/'inputs/prepared/pilot_1-v2'
    arrivals=Path(output).parent/(Path(output).name+'-placeholder-arrivals.json');make_arrivals(bundle,.1,1,arrivals)
    args=comparison.arguments(['run','--method','native','--bundle',str(bundle),'--arrivals',str(arrivals),
        '--model',model,'--output',str(output),'--run-timeout','1800','--allow-exploratory'])
    plan,upstream,native,env=comparison.build(args)
    plan.update(protocol='independent-hardware-calibration/1',actual_replay='synthetic isolated service probes',
                input_bundle_used_for_gpu_work=False,synthetic_calibration=True)
    # Reuse ownership/model/RAM/timeouts/cleanup; only the workload callback changes.
    comparison.replay=measure
    comparison.run(args,plan,upstream,native,env)


def dma(output):
    if sys.platform!='linux':raise RuntimeError('DMA probes require the dedicated Linux GPU')
    from . import server
    upstream=server.load_upstream()
    gpu=server.require_idle_before_model_import(upstream,0)
    from .hardware import validate_gpu
    validate_gpu(gpu)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    import torch
    rows=[]
    for size in (16,64,256):
        host=torch.empty(size*1024**2,dtype=torch.uint8,pin_memory=True)
        host.zero_()
        device=torch.empty_like(host,device='cuda:0')
        for direction in ('H2D','D2H'):
            values=[]
            for repeat in range(23):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record()
                if direction=='H2D':device.copy_(host,non_blocking=True)
                else:host.copy_(device,non_blocking=True)
                end.record();end.synchronize()
                if repeat>=3:values.append(begin.elapsed_time(end)/1000)
            rows.append(dict(direction=direction,payload_mib=size,seconds=stats(values),
                             gib_per_s=stats([size/1024/v for v in values])))
        del host,device
    write_json(output/'dma.json',dict(gpu=gpu,torch=torch.__version__,rows=rows,
        scope='Pinned contiguous synthetic tensor copies; not LMCache wire traffic or whole-system transfer time',
        deployment_profile_ready=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','serving','dma','fit'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--observations',type=Path)
    parser.add_argument('--model',default='/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
    args=parser.parse_args()
    if args.action=='plan':write_json(args.output,dict(protocol='independent-hardware-calibration/1',gpu_executed=False,
        prompt_sizes=list(PROMPTS),prefill_output_tokens=1,decode_prompt_tokens=1024,decode_output_tokens=129,
        client_concurrency=[1,2,4,8],service_repeats=3,dma_sizes_mib=[16,64,256],dma_measured_repeats=20,
        note='Run serving and dma explicitly only on the dedicated GPU; fit never overwrites reference/deployment profiles'))
    elif args.action=='serving':serving(args.output,args.model)
    elif args.action=='dma':dma(args.output)
    else:write_json(args.output,fit(read_json(args.observations)))
