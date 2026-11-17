import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
from b_replay import server
from b_replay.common import ROOT,MODELS,read_json
from b_replay.common import write_json
from b_replay.client import make_arrivals

def args_for(tmp_path,subset="pilot_1",model="Qwen3-8B"):
    bundle=ROOT/f"bundles/{model}/{subset}"
    arrivals=tmp_path/"arrivals.json"
    make_arrivals(bundle,.1,20261030,arrivals)
    return SimpleNamespace(bundle=bundle,arrivals=arrivals,model="/models/models--Qwen--Qwen3-8B/snapshots/"+MODELS["Qwen3-8B"],
        output=tmp_path/"gpu-run",method="kairos",gpu=0,engine_cpus="0-1",helper_cpus="2-3",
        start_timeout=2400,request_timeout=120,run_timeout=900)

def test_frozen_runtime_verifies_and_plan_does_not_start(tmp_path,monkeypatch):
    args=args_for(tmp_path)
    monkeypatch.setenv("B_REPLAY_SECRET_PROBE","never-export-this")
    plan=server.make_plan(args)
    assert plan["runtime"]["files_verified"]==179
    assert not plan["gpu_started"] and plan["programs"]==1 and plan["calls"]==10
    assert "B_REPLAY_SECRET_PROBE" not in plan["environment"]
    assert plan["engine_command"][0:3]==["taskset","-c","0-1"]
    assert not args.output.exists()

def test_training_overlap_and_unsupported_4b_rejected(tmp_path):
    with pytest.raises(ValueError,match="training identities"):
        server.validate_inputs(args_for(tmp_path,"all_eligible"))
    with pytest.raises(ValueError,match="only the pinned Qwen3-8B"):
        server.validate_inputs(args_for(tmp_path,"pilot_1","Qwen3-4B"))
    assert server.validate_inputs(args_for(tmp_path,"heldout_candidates"))["programs"]==36

def test_client_hook_uses_b_inputs_instead_of_archived_16(tmp_path,monkeypatch):
    args=args_for(tmp_path)
    manifest=server.validate_inputs(args)
    seen={}
    async def fake_replay(selected):
        seen.update(vars(selected))
        return {"strict_complete":True}
    monkeypatch.setattr(server,"replay",fake_replay)
    callback=server.make_adapter(args,manifest)
    result=asyncio.run(callback(SimpleNamespace(port=18765,model=args.model),
                               [{"id":"ARCHIVED_SHOULD_NOT_EXECUTE"}]*16,tmp_path/"native"))
    assert seen["bundle"]==args.bundle.resolve()
    assert seen["output"]==tmp_path/"native/b_replay"
    assert seen["method"]=="kairos" and not seen["mock"]
    assert result["archived_replay_executed"] is False

def test_cpu_auto_partition_and_minimum(monkeypatch):
    monkeypatch.setattr(server.os,"sched_getaffinity",lambda pid:{4,5,10,11,12},raising=False)
    assert server.cpu_sets("auto","auto",actual=True)==("4,5","10,11",False)
    with pytest.raises(ValueError):server.cpu_sets("auto","2-3",actual=True)
    monkeypatch.setattr(server.os,"sched_getaffinity",lambda pid:{4,5},raising=False)
    with pytest.raises(RuntimeError):server.cpu_sets("auto","auto",actual=True)

@pytest.mark.parametrize("fail",[False,True])
def test_native_lifecycle_hook_restoration_and_provenance(tmp_path,monkeypatch,fail):
    args=args_for(tmp_path)
    upstream=server.load_upstream()
    original=upstream.teacher_replay
    async def fake_client(selected):return {"strict_complete":True,"expected_calls":10}
    monkeypatch.setattr(server,"replay",fake_client)
    monkeypatch.setattr(server,"require_idle_before_model_import",lambda upstream,gpu:{"uuid":"CPU_TEST_STUB"})
    monkeypatch.setattr(server.sys,"platform","linux")
    def fake_native(native):
        assert upstream.teacher_replay is not original
        native.output.mkdir()
        result=asyncio.run(upstream.teacher_replay(SimpleNamespace(port=18765,model=native.model),
                                                  [{"id":"ORIGINAL_16"}]*16,native.output))
        write_json(native.output/"state.json",dict(status="failed" if fail else "complete",
            trace_sha256="ORIGINAL_16_HASH",verified_model={"weights_verified":True},
            gpu={"uuid":"CPU_TEST_STUB"},summary=result))
        if fail:raise RuntimeError("test native failure")
    monkeypatch.setattr(upstream,"run",fake_native)
    if fail:
        with pytest.raises(RuntimeError,match="test native failure"):server.run(args)
    else:server.run(args)
    assert upstream.teacher_replay is original
    state=read_json(args.output/"state.json")
    assert state["trace_sha256"]==read_json(args.bundle/"manifest.json")["files"]["trace.json"]
    assert state["upstream_archived_trace_sha256"]=="ORIGINAL_16_HASH"
    assert state["gpu_input_smoke_validated"] is (not fail)

@pytest.mark.parametrize("used,util,pids",[(22873,0,set()),(0,94,set()),(0,0,{123})])
def test_busy_gpu_blocked_before_torch_or_vllm_import(used,util,pids):
    harness=SimpleNamespace(smi=lambda:[{"index":"0","uuid":"test","used":str(used),"util":str(util)}],
                            compute_apps=lambda uuid:pids)
    with pytest.raises(RuntimeError,match="before model import"):
        server.require_idle_before_model_import(SimpleNamespace(harness=harness),0)
