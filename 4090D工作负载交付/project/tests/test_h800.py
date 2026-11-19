"""Control/budget integrity tests. Every timing fixture is synthetic and CPU-only."""
import copy
import asyncio
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,AsyncMock
from types import SimpleNamespace

from b_replay.common import read_json,write_json,ROOT,sha
from b_replay.hardware import validate_gpu,deadline_guard
from b_replay.h800 import options,estimate,SEEDS,MAIN_METHODS
from b_replay.suite import create,execute
from b_replay.comparison import build,arguments
from b_replay.server import cpu_sets
from b_replay.datasets import expand,split_audit
from b_replay.workload import make_closed,concurrency
from tests.test_midterm import tiny_bundle


class H800Tests(unittest.TestCase):
    def test_hardware_and_full_27_core_affinity(self):
        validate_gpu(dict(name='NVIDIA H800 PCIe',total=81559))
        with self.assertRaises(RuntimeError):validate_gpu(dict(name='RTX 4090D',total=24576))
        with patch('os.sched_getaffinity',return_value=set(range(10,37)),create=True):
            engine,helper,deferred=cpu_sets('auto','auto',actual=True)
        self.assertEqual(engine,','.join(map(str,range(10,33))));self.assertEqual(helper,'33,34,35,36')
        self.assertFalse(deferred)

    def test_tiers_preserve_five_methods_and_pre_registered_seeds(self):
        counts={r['tier']:(r['jps_runs'],r['dataset_runs'],r['concurrency_runs']) for r in options()['tiers']}
        self.assertEqual(counts['full'],(60,60,90));self.assertEqual(counts['quick'],(10,20,15))
        self.assertEqual(counts['minimal'],(30,30,30));self.assertEqual(counts['compact'],(30,60,45))
        self.assertEqual(len(set(SEEDS)),3)

    def test_auto_kv_removes_archived_limit_and_kairos_selects_full_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=tiny_bundle(root/'b',2);make_closed(bundle,1,SEEDS[0],root/'k.json')
            a=arguments(['plan','--method','kairos','--bundle',str(bundle),'--arrivals',str(root/'k.json'),
                         '--output',str(root/'run'),'--model','/model'])
            p,_,_,env=build(a)
            self.assertNotIn('--kv-cache-memory-bytes',p['engine_command'])
            self.assertEqual(env['CONTINUUM_Kairos_BACKEND'],'1')
            self.assertIn('scripts/gpu_python.py',p['adapter_sha256'])
            self.assertEqual(p['hardware']['cpu_cores'],27)

    def test_upstream_smi_is_enriched_before_hardware_validation(self):
        from b_replay.server import require_idle_before_model_import
        upstream=SimpleNamespace(harness=SimpleNamespace(smi=lambda:[dict(index='0',uuid='GPU-CPU-TEST',total='81559',used='0',util='0')],
            compute_apps=lambda uuid:[]))
        with patch('b_replay.server.subprocess.check_output',return_value='NVIDIA H800, 580.65.06\n'):
            gpu=require_idle_before_model_import(upstream,0)
        self.assertEqual(gpu['name'],'NVIDIA H800');validate_gpu(gpu)

    def test_replica_instance_names_cannot_hide_training_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=tiny_bundle(root/'b',2);expand(bundle,10,5,root/'expanded')
            from b_replay.client import load_bundle
            _,trace,_=load_bundle(root/'expanded')
            write_json(root/'train.json',dict(runtime_commit='a3dbac12eac80d6444222ca3e2e9592fef029158',
                training_ids=['synthetic-0'],validation_ids=[]))
            self.assertFalse(split_audit(trace,root/'train.json')['independent_test'])
            write_json(root/'empty.json',dict(runtime_commit='a3dbac12eac80d6444222ca3e2e9592fef029158',training_ids=[],validation_ids=[]))
            with self.assertRaises(ValueError):split_audit(trace,root/'empty.json')

    def test_process_query_timeout_is_missing_observation_only_in_record_mode(self):
        from b_replay.low_memory import guard
        upstream=SimpleNamespace(harness=SimpleNamespace(
            smi=lambda:[dict(index='0',used='1',total='81559')],
            compute_apps=lambda uuid:(_ for _ in ()).throw(subprocess.TimeoutExpired('nvidia-smi',10)),
            gpu_process_scope=lambda apps,pids:(set(),set())))
        with patch('b_replay.low_memory.memory_snapshot',return_value=dict(available_bytes=32*1024**3)):
            row=guard(upstream,SimpleNamespace(gpu=0,gpu_pid_policy='record-only'),[],'GPU-CPU-TEST')
            self.assertIsNone(row['unattributed_gpu_pids'])
            self.assertFalse(row['process_query_available'])
            self.assertFalse(row['gpu_ownership_verified'])
            with self.assertRaises(subprocess.TimeoutExpired):
                guard(upstream,SimpleNamespace(gpu=0,gpu_pid_policy='strict'),[],'GPU-CPU-TEST')

    def test_suite_preserves_explicit_container_pid_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=tiny_bundle(root/'b',2)
            write_json(root/'c.json',dict(axis='jps',datasets=[dict(name='synthetic',bundle=str(bundle))],
                model='/model',jps=[1],seeds=[1],methods=['lmcache'],gpu_pid_policy='record-only'))
            create(root/'c.json',root/'m')
            row=read_json(root/'m/matrix.json')['runs'][0]
            self.assertEqual(row['plan']['gpu_pid_policy'],'record-only')
            self.assertEqual(row['argv'][row['argv'].index('--gpu-pid-policy')+1],'record-only')

    def _matrix(self,root):
        bundle=tiny_bundle(root/'b',2)
        write_json(root/'config.json',dict(axis='jps',datasets=[dict(name='synthetic',bundle=str(bundle))],
            model='/synthetic-model',jps=[1],seeds=SEEDS,allow_exploratory=True))
        create(root/'config.json',root/'m')
        return root/'m/matrix.json'

    def _timings(self,path,exclude=None):
        matrix=read_json(path)
        for row in matrix['runs']:
            if row['repetition']!=1 or row['method']==exclude: continue
            # This in-memory acceptance fixture bypasses raw GPU inspect below.
            write_json(Path(row['result'])/'state.json',dict(status='complete',plan=row['plan'],
                protocol=row['plan']['protocol'],gpu=dict(name='NVIDIA H800',total=81559,uuid='SYNTHETIC-CPU-TEST'),
                packages=dict(test_fixture=True),summary=dict(strict_complete=True)))
            write_json(path.parent/'attempts'/(row['run_id']+'.json'),dict(status='complete',wall_s=10))

    def test_budget_uses_full_conditions_three_seeds_and_includes_load_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix=self._matrix(root);self._timings(matrix)
            with patch('b_replay.h800.inspect',return_value=(dict(status='complete'),None)):
                small=estimate([matrix],[matrix],3,1,1.3)
                enough=estimate([matrix],[matrix],5,1,1.3)
            self.assertEqual(small['selected_runs'],0)
            self.assertEqual(enough['selected_runs'],15)
            self.assertAlmostEqual(enough['estimated_selected_minutes'],195/60)
            self.assertEqual({r['repetition'] for r in enough['schedule']},{1,2,3})

    def test_missing_kairos_timing_does_not_budget_only_four_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix=self._matrix(root);self._timings(matrix,'kairos')
            with patch('b_replay.h800.inspect',return_value=(dict(status='complete'),None)):
                result=estimate([matrix],[matrix],100,1)
            self.assertEqual(result['selected_runs'],0);self.assertEqual(result['unknown_runs'],3)

    def test_small_dataset_timing_is_not_extrapolated_to_different_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix=self._matrix(root);self._timings(matrix)
            target=read_json(matrix)
            for row in target['runs']:row['plan']['bundle_manifest_sha256']='different-400-instance-input'
            write_json(root/'different.json',target)
            with patch('b_replay.h800.inspect',return_value=(dict(status='complete'),None)):
                result=estimate([root/'different.json'],[matrix],1000,1)
            self.assertEqual(result['selected_runs'],0);self.assertEqual(result['unknown_runs'],15)

    def test_prelaunch_failure_is_retained_and_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix=self._matrix(root)
            with patch('b_replay.suite.build',side_effect=ValueError('simulated setup rejection')):
                with self.assertRaises(ValueError):execute(matrix,0)
            row=read_json(matrix)['runs'][0]
            path=matrix.parent/'attempts'/(row['run_id']+'.json')
            self.assertEqual(read_json(path)['status'],'failed')
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):execute(matrix,0)
            self.assertEqual(path.read_bytes(),before)

    def test_deadline_refuses_starting_after_deadline(self):
        with self.assertRaises(TimeoutError):
            with deadline_guard(1):self.fail('Must not enter')

    def test_target_window_fill_refill_and_drain_are_separate(self):
        jobs=[dict(actual_arrival_s=0,end_s=2),dict(actual_arrival_s=.1,end_s=3),
              dict(actual_arrival_s=2.2,end_s=4,refill_delay_s=.2)]
        c=concurrency(jobs,2)
        self.assertAlmostEqual(c['initial_fill_s'],.1);self.assertAlmostEqual(c['drain_s'],1.8)
        self.assertGreater(c['target_window']['below_fraction'],0)
        self.assertEqual(c['target_window']['above_fraction'],0)
        self.assertAlmostEqual(c['refill_delay_s']['p95'],.2)

    def test_zero_wait_sources_are_rejected_from_main_dataset_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);datasets=[]
            for i in range(2):
                b=tiny_bundle(root/str(i),2);m=read_json(b/'manifest.json')
                m.update(dataset_id=str(i),source_family=str(i),timing_kind='explicit_zero_wait_control')
                write_json(b/'manifest.json',m);datasets.append(dict(name=str(i),bundle=str(b)))
            write_json(root/'c.json',dict(axis='dataset',datasets=datasets,jps=[1],seeds=SEEDS,model='/model'))
            with self.assertRaisesRegex(ValueError,'observed gaps'):create(root/'c.json',root/'m')

    def test_cache_isolation_needs_reuse_and_separate_observed_cpu_counters(self):
        from b_replay.cache_probe import assess
        rows=[]
        for phase in ('gpu_first_A','gpu_repeat_A','gpu_first_B','gpu_repeat_B','cpu_repeat_A','cpu_new_C'):
            reuse=phase in ('gpu_repeat_A','gpu_repeat_B')
            rows.append(dict(phase=phase,prompt_tokens=4096,cached_tokens=4080 if reuse else 0,
                gpu_hits=4080 if reuse else 0,external_hits=4000 if phase=='cpu_repeat_A' else 0,external_queries=4096))
        self.assertTrue(assess(rows)['verified_both_layers'])
        omitted_usage=copy.deepcopy(rows)
        for row in omitted_usage:row['cached_tokens']=None
        self.assertTrue(assess(omitted_usage)['verified_both_layers'])
        missing_gpu=copy.deepcopy(omitted_usage);missing_gpu[1]['gpu_hits']=None
        with self.assertRaises(ValueError):assess(missing_gpu)
        missing=copy.deepcopy(rows);missing[-1]['external_hits']=None
        with self.assertRaises(ValueError):assess(missing)
        shared=copy.deepcopy(rows);shared[2]['cached_tokens']=4080
        with self.assertRaises(ValueError):assess(shared)

    def test_probe_activates_method_and_keeps_namespaces_across_gpu_reset(self):
        import httpx
        from b_replay.cache_probe import _measure
        original_client=httpx.AsyncClient
        async def exercise():
            with tempfile.TemporaryDirectory() as temp:
                gpu=set();cpu=set();seen=[]
                counters={'prefix_cache_hits_total':0,'external_prefix_cache_hits_total':0,
                          'external_prefix_cache_queries_total':0}
                def handler(request):
                    if request.url.path=='/metrics':
                        return httpx.Response(200,text='\n'.join('vllm:'+k+' '+str(v) for k,v in counters.items())+'\n')
                    if request.url.path=='/reset_prefix_cache':
                        self.assertEqual(request.url.params['reset_connector'],'false')
                        gpu.clear();return httpx.Response(200,json=True)
                    body=json.loads(request.content);seen.append(body)
                    salt=body['cache_salt'];meta=body['vllm_xargs']
                    self.assertEqual(meta['cont_program_id'],salt)
                    self.assertIn('continuum-index-partial-eager-a',meta['cont_experiment'])
                    self.assertNotIn('cont_replay_key',meta)
                    self.assertEqual(body['kv_transfer_params'],{'lmcache.tag.instance':salt})
                    hit=4080 if salt in gpu else 0
                    counters['prefix_cache_hits_total']+=hit
                    counters['external_prefix_cache_queries_total']+=4096-hit
                    counters['external_prefix_cache_hits_total']+=(4095-hit) if salt in cpu else 0
                    gpu.add(salt);cpu.add(salt)
                    return httpx.Response(200,json=dict(usage=dict(completion_tokens=1,prompt_tokens=4096)))
                def client(*args,**kwargs):
                    return original_client(*args,transport=httpx.MockTransport(handler),**kwargs)
                args=SimpleNamespace(output=Path(temp)/'probe',endpoint='http://synthetic',
                    method='zero_ttl',model='CPU-MOCK',request_timeout=1)
                with patch('httpx.AsyncClient',client),patch('b_replay.cache_probe.asyncio.sleep',new_callable=AsyncMock):
                    result=await _measure(args)
                self.assertTrue(result['verified_both_layers']);self.assertEqual(len(seen),6)
                self.assertFalse(result['performance_claim_supported'])
        asyncio.run(exercise())

    def test_natural_capacity_proposal_requires_all_five_and_preceding_auto_probe(self):
        from b_replay.h800_capacity import capacity
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);matrix=self._matrix(root);payload=read_json(matrix);payload['runs']=payload['runs'][:5]
            write_json(matrix,payload)
            for row in payload['runs']:
                write_json(Path(row['result'])/'state.json',dict(gpu=dict(name='H800',total=81559,uuid='CPU-FIXTURE'),packages={}))
            with patch('b_replay.h800_capacity.inspect',return_value=(dict(status='complete'),'same-fixture-environment')),\
                 patch('b_replay.h800_capacity.collect',return_value=dict(actual_kv_page_counts=[20000])):
                proposed=capacity(matrix,root/'proposal.json')
                self.assertFalse(proposed['verified_explicit_five_method_smoke'])
                for row in payload['runs']:row['plan']['gpu_kv_bytes']=8*1024**3
                write_json(matrix,payload)
                with self.assertRaisesRegex(ValueError,'preceding automatic'):capacity(matrix,root/'bad.json')

    def test_diagnostics_have_a_separate_nonperformance_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=tiny_bundle(root/'b',2);make_closed(bundle,1,1,root/'k.json')
            p,_,_,_=build(arguments(['plan','--method','lmcache','--bundle',str(bundle),'--arrivals',str(root/'k.json'),
                '--output',str(root/'run'),'--model','/model','--diagnostic','cache-salt']))
            from b_replay.cache_probe import PROTOCOL
            from b_replay.comparison import COMMON_PROTOCOL
            self.assertEqual(p['protocol'],PROTOCOL);self.assertNotEqual(p['protocol'],COMMON_PROTOCOL)

    def test_open_timeout_does_not_label_unarrived_programs_as_failed(self):
        from b_replay.client import make_arrivals,replay
        import httpx
        async def exercise():
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);bundle=tiny_bundle(root/'b',2);make_arrivals(bundle,.000001,1,root/'a.json')
                async def unexpected(request):self.fail('No arrival is due')
                args=SimpleNamespace(bundle=bundle,arrivals=root/'a.json',method='lmcache',model='CPU-MOCK',
                    endpoint='http://mock',output=root/'out',mock=True,request_timeout=1,run_timeout=.02)
                with self.assertRaises(RuntimeError):await replay(args,httpx.MockTransport(unexpected))
                result=read_json(root/'out/summary.json')
                self.assertEqual(result['unattempted_programs'],2);self.assertEqual(result['failed_programs'],0)
        asyncio.run(exercise())

    def test_report_rejects_resource_group_change_after_freeze(self):
        from b_replay.report import report
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run'
            expected=dict(method='lmcache',ttl_seconds=None,resource_group='main')
            write_json(run/'state.json',dict(plan={**expected,'resource_group':'kv-pressure'}))
            write_json(root/'m.json',dict(runs=[dict(run_id='synthetic',axis='jps',value=1,dataset='CPU-only',
                repetition=1,seed=1,method='lmcache',plan=expected,result=str(run),bundle=str(root))]))
            rows,_=report(root/'m.json',root/'report',False)
            self.assertEqual(rows[0]['status'],'invalid')
            self.assertIn('resource_group',rows[0]['validation_error'])


if __name__=='__main__': unittest.main()
