import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import time
import httpx
import pytest
from b_replay.common import write_json, read_json, sha, json_bytes
from b_replay.client import make_arrivals, load_bundle, replay

def bundle(tmp_path, programs=1):
    root=tmp_path/"bundle";root.mkdir()
    rows=[];visible={}
    for n in range(programs):
        pid=f"p{n}"; turns=[]
        messages=[[{"role":"user","content":"current"}],
                  [{"role":"user","content":"current"},{"role":"assistant","content":"teacher"},
                   {"role":"tool","content":"feedback","tool_call_id":"x"}]]
        for i,prompt in enumerate(([1,2],[1,2,3,4])):
            turns.append(dict(keep_prefix_tokens=0 if i==0 else 2,
                added_token_ids=prompt if i==0 else [3,4], prompt_tokens=len(prompt),
                prompt_sha256=sha(json.dumps(prompt).encode()),
                visible_messages_sha256=sha(json_bytes(messages[i],False)),
                output_tokens=3,gap_after_s=.025 if i==0 else 0,tool="cat"))
        rows.append(dict(id=pid,turns=turns,cache_salt=pid));visible[pid]=messages
    write_json(root/"trace.json",dict(model={"max_sequence_tokens":16384},programs=rows),False)
    write_json(root/"visible_messages.json",visible,False)
    write_json(root/"manifest.json",dict(program_ids=[r["id"] for r in rows], programs=programs,
        calls=programs*2,files={p.name:sha(p.read_bytes()) for p in root.iterdir()}))
    return root

def args_for(tmp_path, source):
    arrivals=tmp_path/"arrivals.json";make_arrivals(source,100000,20261030,arrivals)
    return SimpleNamespace(bundle=source,arrivals=arrivals,endpoint="http://mock.local",model="mock",
        output=tmp_path/"run",request_timeout=2,run_timeout=2,mock=True,method="kairos")

def stream_response(count):
    parts=[{"id":"test","choices":[{"token_ids":[999]*count}]},
           {"choices":[],"usage":{"completion_tokens":count}}]
    data="".join("data: "+json.dumps(p)+"\n\n" for p in parts)+"data: [DONE]\n\n"
    return httpx.Response(200,content=data,headers={"content-type":"text/event-stream"})

def test_arrivals_determinism_jps_and_identity(tmp_path):
    source=bundle(tmp_path,3); a,b=tmp_path/"a.json",tmp_path/"b.json"
    make_arrivals(source,1,20261030,a);make_arrivals(source,1,20261030,b)
    assert a.read_bytes()==b.read_bytes()
    assert read_json(a)["arrivals"][0]["arrival_s"]>0
    with pytest.raises(ValueError):make_arrivals(source,0,1,b)

def test_real_client_dependency_and_teacher_prompt(tmp_path):
    source=bundle(tmp_path);args=args_for(tmp_path,source);requests=[]
    async def handler(request):
        payload=json.loads(request.content);requests.append((time.monotonic(),payload))
        assert payload["ignore_eos"] and payload["max_tokens"]==3
        assert set(payload["vllm_xargs"])=={"cont_program_id","cont_tool","cont_final","cont_oracle_gap","cont_visible_messages","cont_experiment"}
        return stream_response(3)
    result=asyncio.run(replay(args,httpx.MockTransport(handler)))
    assert result["strict_complete"] and not result["gpu_validated"]
    assert requests[1][0]-requests[0][0]>=.025
    assert requests[1][1]["prompt"]==[1,2,3,4]  # excludes actual generated 999 tokens
    assert requests[0][1]["vllm_xargs"]["cont_final"] is False
    assert requests[1][1]["vllm_xargs"]["cont_final"] is True

def test_future_teacher_tool_labels_never_enter_online_requests(tmp_path):
    source=bundle(tmp_path)
    trace=read_json(source/"trace.json")
    for i,turn in enumerate(trace["programs"][0]["turns"]):
        turn["tool"]=f"FUTURE_TEACHER_ACTION_{i}"
    write_json(source/"trace.json",trace,False)
    manifest=read_json(source/"manifest.json")
    manifest["files"]["trace.json"]=sha((source/"trace.json").read_bytes())
    write_json(source/"manifest.json",manifest)
    args=args_for(tmp_path,source)
    async def handler(request):
        payload=json.loads(request.content)
        assert "FUTURE_TEACHER_ACTION" not in request.content.decode()
        assert payload["vllm_xargs"]["cont_tool"]=="unclassified-gap"
        assert payload["max_tokens"]==3
        return stream_response(3)
    result=asyncio.run(replay(args,httpx.MockTransport(handler)))
    assert result["strict_complete"]
    assert read_json(args.output/"state.json")["future_tool_annotation_sent"] is False

def test_early_stop_is_failure_other_program_drains(tmp_path):
    source=bundle(tmp_path,2);args=args_for(tmp_path,source)
    async def handler(request):
        pid=json.loads(request.content)["vllm_xargs"]["cont_program_id"]
        return stream_response(1 if pid=="p0" else 3)
    with pytest.raises(RuntimeError):asyncio.run(replay(args,httpx.MockTransport(handler)))
    summary=read_json(args.output/"summary.json")
    assert summary["failed_programs"]==1 and summary["completed_programs"]==1
    assert summary["completed_calls"]==2 and summary["unattempted_calls"]==1

def test_timeout_and_no_overwrite(tmp_path):
    source=bundle(tmp_path);args=args_for(tmp_path,source);args.run_timeout=.01
    async def handler(request):
        await asyncio.sleep(.1);return stream_response(3)
    with pytest.raises(RuntimeError):asyncio.run(replay(args,httpx.MockTransport(handler)))
    assert read_json(args.output/"summary.json")["timed_out"]
    with pytest.raises(FileExistsError):asyncio.run(replay(args,httpx.MockTransport(handler)))

def test_tamper_and_wrong_arrival_rejected(tmp_path):
    source=bundle(tmp_path);args=args_for(tmp_path,source)
    arrivals=read_json(args.arrivals);arrivals["trace_sha256"]="wrong";write_json(args.arrivals,arrivals)
    with pytest.raises(ValueError):asyncio.run(replay(args))
    assert not args.output.exists()
    (source/"trace.json").write_text("{}")
    with pytest.raises(ValueError):load_bundle(source)

def test_multitoken_chunks_are_not_reported_as_token_itl(tmp_path):
    source=bundle(tmp_path);args=args_for(tmp_path,source)
    async def handler(request):
        parts=[{"choices":[{"token_ids":[5,6]}]}, {"choices":[{"token_ids":[7]}]},
               {"choices":[],"usage":{"completion_tokens":3}}]
        return httpx.Response(200,content="".join("data: "+json.dumps(p)+"\n\n" for p in parts))
    asyncio.run(replay(args,httpx.MockTransport(handler)))
    records=[json.loads(s) for s in (args.output/"requests.jsonl").read_text().splitlines()]
    assert all(r["stream_chunk_sizes"]==[2,1] and r["single_token_chunk_itl_s"]==[] for r in records)

@pytest.mark.parametrize("mode",["http_error","usage_mismatch"])
def test_response_errors_are_saved_as_failure(tmp_path,mode):
    source=bundle(tmp_path);args=args_for(tmp_path,source)
    async def handler(request):
        if mode=="http_error":return httpx.Response(500,content="test failure")
        return httpx.Response(200,content='data: {"choices":[{"token_ids":[1,2,3]}],"usage":{"completion_tokens":2}}\n\n')
    with pytest.raises(RuntimeError):asyncio.run(replay(args,httpx.MockTransport(handler)))
    assert read_json(args.output/"summary.json")["failed_calls"]==1
