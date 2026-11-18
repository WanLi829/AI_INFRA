import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from b_replay.client import load_bundle, replay, replay_metadata, make_arrivals
from b_replay.common import ROOT, sha, write_json, read_json
from b_replay.datasets import save_bundle, expand, prepare
from b_replay.workload import make_closed, concurrency, validate_workload
from b_replay.metrics import collect
from b_replay.comparison import arguments, build, METHODS


def tiny_bundle(path, n=6, gap=.003):
    """Synthetic, explicitly marked data with two dependent calls per program."""
    visible={}; programs=[]; actions={}
    for i in range(n):
        pid=f'synthetic-{i}'
        messages=[[{'role':'user','content':'synthetic protocol test'}],
                  [{'role':'user','content':'synthetic protocol test'}, {'role':'assistant','content':'done'},
                   {'role':'user','content':'continue'}]]
        visible[pid]=messages
        turns=[]
        for j,tokens in enumerate(([11,12],[11,12,13,14])):
            from b_replay.common import json_bytes
            turns.append(dict(keep_prefix_tokens=0,added_token_ids=tokens,prompt_tokens=len(tokens),
                prompt_sha256=sha(json.dumps(tokens).encode()),output_tokens=2,gap_after_s=gap,
                visible_messages_sha256=sha(json_bytes(messages[j],pretty=False))))
        programs.append(dict(id=pid,source_id=pid,cache_salt=pid,turns=turns))
        actions[pid]=[dict(tool=None,terminal=False,evidence='assistant-only',assistant_sha256='a'*64)]*2
    from b_replay.common import MODELS
    manifest=dict(model='Qwen3-8B',model_revision=MODELS['Qwen3-8B'],dataset_id='synthetic-test',
                  evidence='CPU synthetic test only')
    trace=dict(model={'max_sequence_tokens':16384},programs=programs)
    save_bundle(path,manifest,trace,visible,dict(schema='recorded-current-actions-v1',actions=actions),{})
    return path


class MidtermTests(unittest.TestCase):
    def test_closed_loop_keeps_bound_and_refills_after_completion(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp); bundle=tiny_bundle(root/'bundle')
                workload=root/'closed.json'; make_closed(bundle,2,17,workload)
                async def handler(request):
                    body=json.loads(request.content)
                    self.assertEqual(body['max_tokens'],16384-len(body['prompt']))
                    self.assertNotIn('cont_final',body['vllm_xargs'])
                    await asyncio.sleep(.004)
                    return httpx.Response(200,text='data: '+json.dumps({'choices':[{'token_ids':[1,2]}]})+'\n\ndata: [DONE]\n\n')
                args=SimpleNamespace(bundle=bundle,arrivals=workload,method='zero_ttl',model='synthetic',
                    endpoint='http://mock',output=root/'out',mock=True,request_timeout=1,run_timeout=3)
                result=await replay(args,httpx.MockTransport(handler))
                self.assertTrue(result['strict_complete'])
                jobs=read_json(root/'out/jobs.json')
                curve=read_json(root/'out/concurrency.json')
                self.assertEqual(curve['peak'],2)
                self.assertEqual(curve['target_window']['above_s'],0)
                ordered=sorted(jobs,key=lambda j:j['actual_arrival_s'])
                self.assertEqual([j['program_id'] for j in ordered],read_json(workload)['program_order'])
                for i,job in enumerate(ordered[2:],2):
                    self.assertGreaterEqual(job['actual_arrival_s'],sorted(j['end_s'] for j in ordered[:i])[i-2])
                records=[json.loads(s) for s in (root/'out/requests.jsonl').read_text().splitlines()]
                for job in jobs:
                    req=sorted((r for r in records if r['program_id']==job['program_id']),key=lambda r:r['turn_id'])
                    self.assertGreaterEqual(req[1]['start_s']-req[0]['end_s'],.002)
        asyncio.run(exercise())

    def test_closed_timeout_records_unattempted_programs(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp); bundle=tiny_bundle(root/'bundle')
                workload=root/'closed.json'; make_closed(bundle,2,1,workload)
                async def stall(request):
                    await asyncio.sleep(10)
                args=SimpleNamespace(bundle=bundle,arrivals=workload,method='native',model='synthetic',
                    endpoint='http://mock',output=root/'out',mock=True,request_timeout=1,run_timeout=.03)
                with self.assertRaises(RuntimeError): await replay(args,httpx.MockTransport(stall))
                summary=read_json(root/'out/summary.json')
                self.assertTrue(summary['timed_out'])
                self.assertEqual(summary['unattempted_programs'],4)
                self.assertEqual(summary['failed_programs'],2)
        asyncio.run(exercise())

    def test_event_intervals_are_half_open_and_weighted(self):
        jobs=[dict(actual_arrival_s=0,end_s=2),dict(actual_arrival_s=1,end_s=3),dict(actual_arrival_s=2,end_s=4)]
        value=concurrency(jobs,2)
        self.assertEqual(value['peak'],2)
        self.assertEqual(value['mean'],1.5)
        self.assertEqual(value['target_window']['target_fraction'],1)

    def test_instance_expansion_preserves_source_and_isolates_salt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); bundle=tiny_bundle(root/'bundle',2)
            expand(bundle,7,21,root/'expanded')
            manifest,trace,_=load_bundle(root/'expanded')
            self.assertEqual(manifest['programs'],7)
            self.assertEqual(len({p['cache_salt'] for p in trace['programs']}),7)
            self.assertEqual(len({p['source_id'] for p in trace['programs']}),2)
            with self.assertRaises(ValueError): make_closed(root/'expanded',8,1,root/'bad.json')

    def test_real_source_enrichment_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'enriched'
            prepare(ROOT/'bundles/Qwen3-8B/pilot_1',ROOT.parents[1]/'data-preparation/intermediate',output,'mini-swe')
            manifest,trace,_=load_bundle(output)
            self.assertEqual(manifest['calls'],10)
            actions=read_json(output/'completed_actions.json')['actions']
            self.assertEqual(len(actions[trace['programs'][0]['id']]),10)
            self.assertFalse(manifest['predictor_split_audit']['independent_test'])
            (output/'completed_actions.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'): load_bundle(output)

    def test_zero_policy_survives_parent_reset(self):
        class Parent:
            def __init__(self): self.experiment=None; self.events=[]
            def _switch_experiment(self,name): self.experiment=name; self.policy_model=SimpleNamespace(mode='paper',fixed=9)
            def _emit(self,event,**kw): self.events.append((event,kw))
        with patch.dict(sys.modules,{'kairos._runtime.compat.pipelined_prefix':SimpleNamespace(PipelinedPrefixScheduler=Parent)}):
            spec=importlib.util.spec_from_file_location('zero_test',ROOT/'b_replay/static_scheduler.py')
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with patch.dict('os.environ',{'CONTINUUM_TTL_SECONDS':'0'}):
            scheduler=module.StaticTTLScheduler(); scheduler._switch_experiment('continuum-index~heterogeneous')
            self.assertEqual(scheduler.policy_model.fixed,0)
            self.assertEqual(scheduler.events[0][0],'zero_ttl_active')

    def test_private_execution_never_exposes_recorded_length(self):
        sys.path.insert(0,str(ROOT/'runtime/Kairos'))
        from kairos.replay.boundary import validate_arrived_metadata,validate_online_method
        from kairos.replay.execution import prepare_manifest,update_output
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); bundle=tiny_bundle(root/'bundle',1)
            _,trace,visible=load_bundle(bundle); p=trace['programs'][0]
            meta=replay_metadata('kairos',p['id'],0,visible[p['id']][0])
            validate_arrived_metadata(meta); validate_online_method(meta['cont_experiment'])
            path=root/'private.json'; prepare_manifest([(meta['cont_experiment'],trace['programs'])],path,16384,16384)
            request=SimpleNamespace(sampling_params=SimpleNamespace(extra_args=meta,ignore_eos=True),
                output_token_ids=[],max_tokens=16382,status=None)
            def native(r,ids): r.output_token_ids.extend(ids); return ids,False
            with patch.dict('os.environ',{'CONTINUUM_REPLAY_EXECUTION_MANIFEST':str(path)}):
                output,stopped=update_output(request,[7,8,9],native,'FINISHED')
            self.assertEqual(output,[7,8]); self.assertTrue(stopped)
            self.assertEqual(request.max_tokens,16382)

    def test_metrics_missing_and_reset_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertIsNone(collect(root)['cache_ratios']['prefix_cache']['ratio'])
            (root/'metrics-before.txt').write_text('vllm:prefix_cache_hits_total{model="x"} 2\nvllm:prefix_cache_queries_total{model="x"} 5\n')
            (root/'metrics-after.txt').write_text('vllm:prefix_cache_hits_total{model="x"} 5\nvllm:prefix_cache_queries_total{model="x"} 15\n')
            self.assertEqual(collect(root)['cache_ratios']['prefix_cache']['ratio'],.3)
            (root/'metrics-after.txt').write_text('vllm:prefix_cache_hits_total{model="x"} 1\nvllm:prefix_cache_queries_total{model="x"} 15\n')
            self.assertIsNone(collect(root)['cache_ratios']['prefix_cache']['ratio'])

    def test_six_plans_share_resource_budget_and_fail_closed_on_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); bundle=tiny_bundle(root/'bundle',2)
            workload=root/'closed.json';make_closed(bundle,2,1,workload)
            for method in METHODS:
                argv=['plan','--method',method,'--bundle',str(bundle),'--arrivals',str(workload),
                      '--model','/model','--output',str(root/method),'--gpu-kv-bytes','6000000000','--cpu-kv-gib','8']
                if method=='static_ttl': argv+=['--ttl-seconds','2']
                plan,_,_,env=build(arguments(argv))
                cmd=plan['engine_command']
                self.assertEqual(cmd[cmd.index('--kv-cache-memory-bytes')+1],'6000000000')
                self.assertEqual(env['LMCACHE_MAX_LOCAL_CPU_SIZE'],'8.0') if method!='native' else None
                self.assertIn('--async-scheduling',cmd)
                if method=='kairos': self.assertTrue(any('asset' in s for s in plan['readiness_blockers']))

    def test_suite_freezes_inputs_and_report_preserves_not_run(self):
        from b_replay.suite import create
        from b_replay.report import report
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=tiny_bundle(root/'bundle',2)
            config=dict(axis='jps',datasets=[dict(name='synthetic',bundle=str(bundle))],model='/model',
                jps=[.1,1],seeds=[21,22],methods=['lmcache','zero_ttl'])
            write_json(root/'config.json',config)
            create(root/'config.json',root/'matrix')
            matrix=read_json(root/'matrix/matrix.json')
            self.assertEqual(len(matrix['runs']),8)
            for offset in (0,2,4,6):
                self.assertEqual(matrix['runs'][offset]['workload_sha256'],matrix['runs'][offset+1]['workload_sha256'])
            rows,summary=report(root/'matrix/matrix.json',root/'report',False)
            self.assertTrue(all(r['status']=='not_run' for r in rows))
            self.assertTrue(all(r['jct_mean_s_mean'] is None for r in summary))

    def test_observability_deduplicates_pins_without_changing_schedule(self):
        from b_replay.observability import MetricsMixin
        class Parent:
            def schedule(self): return 'unchanged'
            def add_request(self,r): return 'accepted'
        class Observed(MetricsMixin,Parent): pass
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'events.jsonl'
            observed=Observed()
            block=SimpleNamespace(block_id=3,is_null=False)
            observed.pins={'a':SimpleNamespace(blocks=[block]),'b':SimpleNamespace(blocks=[block])}
            observed.running=[SimpleNamespace(request_id='r')];observed.waiting=[];observed.skipped_waiting=[]
            observed.kv_cache_manager=SimpleNamespace(block_pool=SimpleNamespace(num_gpu_blocks=10,get_num_free_blocks=lambda:7))
            with patch.dict('os.environ',{'B_REPLAY_METRICS_PATH':str(path)}):
                self.assertEqual(observed.add_request(observed.running[0]),'accepted')
                self.assertEqual(observed.schedule(),'unchanged')
            events=[json.loads(s) for s in path.read_text().splitlines()]
            sample=next(e for e in events if e['event']=='sample')
            self.assertEqual(sample['unique_pin_blocks'],1)
            self.assertEqual(sample['pin_programs'],2)

    def test_report_pairs_seeds_and_never_counts_missing_baseline(self):
        from b_replay.report import report
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);items=[];measured={}
            for rep in range(3):
                for method,jct in [('lmcache',20),('kairos',10)]:
                    run=root/f'{method}-{rep}'; plan={'method':method,'ttl_seconds':None}
                    write_json(run/'state.json',dict(plan=plan))
                    items.append(dict(run_id=run.name,axis='jps',value=1,dataset='synthetic',
                        repetition=rep,seed=rep,method=method,plan=plan,result=str(run),bundle=str(root)))
                    measured[str(run)]=(dict(method=method,status='complete',speedup_eligible=True,
                        jct_mean_s=jct,ttft_mean_s=1),json.dumps(dict(arrivals=f'seed-{rep}',gpu='same')))
            write_json(root/'matrix.json',dict(runs=items))
            with patch('b_replay.report.inspect',side_effect=lambda run,bundle:measured[str(run)]):
                rows,summary=report(root/'matrix.json',root/'report',False)
            kairos=next(r for r in summary if r['method']=='kairos')
            self.assertEqual(kairos['jct_reduction_vs_lmcache_mean'],.5)
            self.assertEqual(kairos['jct_reduction_vs_lmcache_n'],3)
            self.assertTrue(kairos['repeated_formal_pair'])
            measured[str(root/'lmcache-2')]=(dict(status='failed',speedup_eligible=False),None)
            with patch('b_replay.report.inspect',side_effect=lambda run,bundle:measured[str(run)]):
                _,summary=report(root/'matrix.json',root/'report-missing',False)
            kairos=next(r for r in summary if r['method']=='kairos')
            self.assertEqual(kairos['jct_reduction_vs_lmcache_n'],2)
            self.assertFalse(kairos['repeated_formal_pair'])


if __name__=='__main__': unittest.main()
