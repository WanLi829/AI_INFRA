"""Real recorded input + waits, fake transport: CPU protocol verification only."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from b_replay.client import load_bundle,replay
from b_replay.common import ROOT,write_json,read_json,sha
from b_replay.workload import make_closed


async def exercise(bundle,k,output):
    code={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/'b_replay/client.py',ROOT/'b_replay/workload.py',ROOT/'b_replay/datasets.py',Path(__file__).resolve()]}
    manifest,trace,_=load_bundle(bundle)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    workload=output/'workload.json';make_closed(bundle,k,2026111801,workload)
    programs={p['id']:p for p in trace['programs']};cursor=dict.fromkeys(programs,0);salts=set()
    async def handler(request):
        body=json.loads(request.content);meta=body['vllm_xargs'];pid=meta['cont_program_id']
        assert body['cache_salt']==pid
        salts.add(body['cache_salt'])
        assert meta['cont_tool']=='unknown' and 'cont_final' not in meta and 'cont_oracle_gap' not in meta
        turn=programs[pid]['turns'][cursor[pid]];cursor[pid]+=1
        assert body['max_tokens']==16384-turn['prompt_tokens']
        return httpx.Response(200,text='data: '+json.dumps({'id':f"mock-{pid}-{cursor[pid]}",
            'choices':[{'token_ids':[1]*turn['output_tokens']}]})+'\n\ndata: [DONE]\n\n')
    args=SimpleNamespace(bundle=bundle,arrivals=workload,method='kairos',model='CPU-MOCK',
        endpoint='http://mock.local',output=output/'client',mock=True,request_timeout=120,run_timeout=300)
    result=await replay(args,httpx.MockTransport(handler))
    assert len(salts)==manifest['programs']
    curve=read_json(output/'client/concurrency.json')
    assert curve['peak']<=k and (curve['target_window'] or {}).get('above_s',0)==0
    write_json(output/'validation.json',dict(evidence='CPU protocol test only; no model/predictor/cache execution',
        programs=manifest['programs'],calls=result['completed_calls'],unique_instance_salts=len(salts),
        target=k,observed_peak=curve['peak'],strict_complete=result['strict_complete'],gpu_started=False))
    validation=read_json(output/'validation.json')
    validation.update(code_sha256=code,input_manifest_sha256=sha((Path(bundle)/'manifest.json').read_bytes()))
    write_json(output/'validation.json',validation)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=ROOT/'inputs/prepared/instances400')
    p.add_argument('--active-programs',type=int,default=80);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();asyncio.run(exercise(a.bundle,a.active_programs,a.output))
