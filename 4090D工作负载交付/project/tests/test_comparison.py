import asyncio
import json
import importlib.util
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from b_replay.client import load_bundle, make_arrivals, replay
from b_replay.comparison import arguments, build, METHODS
from b_replay.comparison_results import aggregate, inspect
from b_replay.common import ROOT, read_json, write_json, sha


class ComparisonTests(unittest.TestCase):
    def test_all_fifteen_plans(self):
        with tempfile.TemporaryDirectory() as temp:
            for method in METHODS:
                for rate in ("0.05", "0.1", "0.2"):
                    args = arguments(["plan", "--method", method,
                        "--bundle", str(ROOT / "bundles/Qwen3-8B/pilot_14"),
                        "--arrivals", str(ROOT / f"arrivals/pilot_14_jps{rate}_seed20261030.json"),
                        "--model", "/models/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
                        "--output", str(Path(temp) / "run"),
                        *(["--ttl-seconds", "2"] if method == "static_ttl" else [])])
                    with patch.dict("os.environ", {"CONTINUUM_TTL_SECONDS": "999", "LMCACHE_CONFIG_FILE": "/bad.yaml"}):
                        plan, _, _, env = build(args)
                    self.assertEqual(plan["calls"], 161)
                    self.assertEqual(plan["runtime"]["files_verified"], 179)
                    self.assertNotIn("LMCACHE_CONFIG_FILE", env)
                    command = plan["engine_command"]
                    if method in ("native", "lmcache"):
                        self.assertIsNone(plan["helper_command"])
                        self.assertNotIn("--scheduler-cls", command)
                        self.assertNotIn("--worker-cls", command)
                        self.assertEqual(command[command.index("--scheduling-policy")+1], "fcfs")
                        self.assertEqual("--kv-transfer-config" in command, method == "lmcache")
                    if method == "static_ttl":
                        self.assertEqual(env["CONTINUUM_TTL_SECONDS"], "2.0")
                        self.assertEqual(env["CONTINUUM_EXPECTED_SCHEDULER"], "b_replay.static_scheduler.StaticTTLScheduler")

    def test_five_clients_with_actual_pilot_14(self):
        asyncio.run(self.exercise_clients())

    async def exercise_clients(self):
        bundle = ROOT / "bundles/Qwen3-8B/pilot_14"
        manifest, trace, visible = load_bundle(bundle)
        with tempfile.TemporaryDirectory() as temp:
            arrivals = Path(temp) / "arrivals.json"
            make_arrivals(bundle, 1000, 20261030, arrivals)
            for method in METHODS:
                cursor = {p["id"]: 0 for p in trace["programs"]}
                prompts = {p["id"]: [] for p in trace["programs"]}
                programs = {p["id"]: p for p in trace["programs"]}
                salt_ids = {p.get("cache_salt", p["id"]): p["id"] for p in trace["programs"]}

                async def handler(request):
                    body = json.loads(request.content)
                    pid = salt_ids[body["cache_salt"]]
                    i = cursor[pid]
                    turn = programs[pid]["turns"][i]
                    prompt = prompts[pid][:turn["keep_prefix_tokens"]] + turn["added_token_ids"]
                    self.assertEqual(body["prompt"], prompt)
                    self.assertEqual(body["max_tokens"], turn["output_tokens"])
                    self.assertTrue(body["return_token_ids"] and body["ignore_eos"])
                    if method in ("native", "lmcache"):
                        self.assertNotIn("vllm_xargs", body)
                    else:
                        meta = body["vllm_xargs"]
                        self.assertEqual(meta["cont_program_id"], pid)
                        self.assertEqual(json.loads(meta["cont_visible_messages"]), visible[pid][i])
                        self.assertEqual(meta["cont_oracle_gap"], 0)
                    prompts[pid] = prompt
                    cursor[pid] += 1
                    chunks = [{"choices": [{"token_ids": [1] * turn["output_tokens"]}]},
                              {"choices": [], "usage": {"completion_tokens": turn["output_tokens"]}}]
                    stream = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
                    return httpx.Response(200, text=stream)

                args = SimpleNamespace(bundle=bundle, arrivals=arrivals, endpoint="http://mock.local",
                                       method=method, model="CPU-MOCK", output=Path(temp)/method,
                                       mock=True, request_timeout=10, run_timeout=120)
                result = await replay(args, httpx.MockTransport(handler))
                self.assertTrue(result["strict_complete"])
                self.assertEqual(result["completed_calls"], manifest["calls"])
                self.assertIn("mock", result["evidence"].lower())
                print(method, result["completed_programs"], result["completed_calls"], flush=True)

    def test_failure_is_preserved(self):
        async def failure():
            with tempfile.TemporaryDirectory() as temp:
                bundle = ROOT / "bundles/Qwen3-8B/pilot_1"
                arrival = Path(temp) / "arrival.json"
                make_arrivals(bundle, 1000, 20261030, arrival)
                args = SimpleNamespace(bundle=bundle, arrivals=arrival, endpoint="http://mock.local",
                                       method="native", model="CPU-MOCK", output=Path(temp)/"failure",
                                       mock=True, request_timeout=1, run_timeout=5)
                with self.assertRaises(RuntimeError):
                    await replay(args, httpx.MockTransport(lambda r: httpx.Response(500, text="intentional")))
                result = read_json(args.output/"summary.json")
                self.assertFalse(result["strict_complete"])
                self.assertEqual(result["failed_calls"], 1)
                self.assertEqual(result["unattempted_calls"], 9)
        asyncio.run(failure())

    def test_ratios_require_matching_baseline_and_repeats(self):
        def row(method, jct, run):
            return dict(method=method, jct_mean_s=jct, output_tokens_per_s=100/jct,
                        speedup_eligible=True, jps=.1, run=run, ttl_seconds=None)
        isolated = aggregate([(row("kairos", 40, "k"), "different-input")])
        self.assertIsNone(isolated[0]["jct_speedup"])
        rows = [(row("native", 60, f"n{i}"), "same") for i in range(3)]
        rows += [(row("kairos", 40, f"k{i}"), "same") for i in range(3)]
        result = next(r for r in aggregate(rows) if r["method"] == "kairos")
        self.assertEqual(result["jct_speedup"], 1.5)
        self.assertTrue(result["repeated_comparison"])

    def test_fixed_ttl_survives_experiment_switch(self):
        # The real engine parent is unavailable on this Windows CPU host.
        # Emulate its experiment switch resetting the policy to dynamic mode.
        class EngineParent:
            def __init__(self):
                self.experiment = None
                self.events = []
            def _switch_experiment(self, name):
                self.experiment = name
                self.policy_model = SimpleNamespace(mode="paper", fixed=999)
            def _emit(self, event, **fields):
                self.events.append((event, fields))
        module_name = "kairos._runtime.compat.pipelined_prefix"
        with patch.dict(sys.modules, {module_name: SimpleNamespace(PipelinedPrefixScheduler=EngineParent)}):
            spec = importlib.util.spec_from_file_location("fixed_test", ROOT/"b_replay/static_scheduler.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with patch.dict("os.environ", {"CONTINUUM_TTL_SECONDS": "2"}):
            scheduler = module.StaticTTLScheduler()
            scheduler._switch_experiment("continuum-index-partial-eager-a~heterogeneous")
            self.assertEqual(scheduler.policy_model.mode, "fixed")
            self.assertEqual(scheduler.policy_model.fixed, 2)
            scheduler._switch_experiment("continuum-index-partial-eager-a~heterogeneous")
            self.assertEqual(len(scheduler.events), 1)

    def test_raw_result_validation(self):
        # Synthetic provenance in a temporary directory tests the collector.
        # Archived measurements are never promoted into new GPU evidence.
        bundle = ROOT/"bundles/Qwen3-8B/pilot_14"
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/"synthetic"
            args = arguments(["plan", "--method", "kairos", "--bundle", str(bundle),
                "--arrivals", str(ROOT/"arrivals/pilot_14_jps0.1_seed20261030.json"),
                "--model", "/model", "--output", str(run)])
            plan, _, _, _ = build(args)
            shutil.copytree(ROOT/"verification/pilot_14", run)
            state = read_json(run/"state.json")
            state.update(schema="comparison/1", protocol="comparison-v1-cold-process",
                         plan=plan, gpu_pid_policy="strict", gpu_ownership_verified=True)
            write_json(run/"state.json", state)
            client = dict(status="complete", method="kairos", evidence="Synthetic validation fixture",
                          bundle_manifest_sha256=plan["bundle_manifest_sha256"],
                          trace_sha256=plan["trace_sha256"], arrivals_sha256=plan["arrivals_sha256"])
            write_json(run/"b_replay/state.json", client)
            resources = read_json(run/"resources.json")
            for sample in resources:
                if "memory" in sample:
                    sample["gpu_pid_policy"] = "strict"
                    sample["foreign_gpu_pids"] = []
            write_json(run/"resources.json", resources)
            row, identity = inspect(run, bundle)
            self.assertEqual(row["output_tokens"], 24182)
            self.assertAlmostEqual(row["jct_mean_s"], 48.84048797, places=5)
            state.update(gpu_pid_policy="record-only", gpu_ownership_verified=False)
            state["plan"]["gpu_pid_policy"] = "record-only"
            for sample in resources:
                if "memory" in sample:
                    sample.update(gpu_pid_policy="record-only", gpu_ownership_verified=False)
            write_json(run/"resources.json", resources)
            write_json(run/"state.json", state)
            pilot_row = inspect(run, bundle)
            self.assertEqual(pilot_row[0]["output_tokens"], 24182)
            self.assertFalse(pilot_row[0]["speedup_eligible"])
            self.assertEqual(aggregate([pilot_row]), [])
            client["arrivals_sha256"] = "wrong"
            write_json(run/"b_replay/state.json", client)
            with self.assertRaisesRegex(ValueError, "provenance"):
                inspect(run, bundle)
            state["status"] = "failed"
            state["error"] = "Intentional failed run"
            write_json(run/"state.json", state)
            row, identity = inspect(run, bundle)
            self.assertFalse(row["speedup_eligible"])
            self.assertIsNone(identity)


if __name__ == "__main__":
    unittest.main()
