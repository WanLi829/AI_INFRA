"""Exercise actual delivered pilot bundles through the real client, with an in-memory mock engine."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import httpx
from b_replay.client import make_arrivals, replay, load_bundle
from b_replay.common import ROOT, write_json, sha

async def main(output_root, result_path):
    results=[]
    for subset in ("pilot_1", "pilot_4", "pilot_14"):
        source=ROOT/f"bundles/Qwen3-8B/{subset}"
        _,trace,visible=load_bundle(source)
        programs={p["id"]:p for p in trace["programs"]}
        cursor={pid:0 for pid in programs}
        inputs={pid:[] for pid in programs}
        async def handler(request):
            body=json.loads(request.content);meta=body["vllm_xargs"]
            pid=meta["cont_program_id"];i=cursor[pid];turn=programs[pid]["turns"][i]
            prompt=inputs[pid][:turn["keep_prefix_tokens"]]+turn["added_token_ids"]
            assert body["prompt"]==prompt
            assert json.loads(meta["cont_visible_messages"])==visible[pid][i]
            assert body["max_tokens"]==turn["output_tokens"] and body["ignore_eos"]
            assert meta["cont_final"]==(i==len(programs[pid]["turns"])-1)
            assert meta["cont_oracle_gap"]==0
            assert meta["cont_tool"]=="unclassified-gap"
            assert "gap_after_s" not in meta and "source_usage" not in meta
            inputs[pid]=prompt;cursor[pid]+=1
            payloads=[dict(id=f"mock-{pid}-{i}",choices=[dict(token_ids=[0]*body["max_tokens"]) ]),
                      dict(choices=[],usage=dict(completion_tokens=body["max_tokens"]))]
            data="".join("data: "+json.dumps(p)+"\n\n" for p in payloads)+"data: [DONE]\n\n"
            return httpx.Response(200,content=data,headers={"content-type":"text/event-stream"})
        arrival=output_root/f"arrivals/{subset}.json"
        make_arrivals(source,1000,20261030,arrival)
        args=SimpleNamespace(bundle=source,arrivals=arrival,endpoint="http://mock.local",model="CPU-MOCK",
            method="kairos",output=output_root/f"runs/{subset}",
            request_timeout=10,run_timeout=120,mock=True)
        summary=await replay(args,httpx.MockTransport(handler))
        results.append(dict(subset=subset,programs=summary["completed_programs"],calls=summary["completed_calls"],
            strict_complete=summary["strict_complete"],evidence=summary["evidence"],
            online_tool_label="unclassified-gap",future_tool_annotation_sent=False,
            wait_policy="Unscaled original gaps",arrival_jps=1000,
            arrival_rate_purpose="Protocol test only; not a GPU recommendation"))
        print(subset,summary["completed_programs"],summary["completed_calls"],flush=True)
    write_json(result_path,results)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--results",type=Path)
    args=parser.parse_args()
    asyncio.run(main(args.output_root,args.results or args.output_root/"results.json"))
