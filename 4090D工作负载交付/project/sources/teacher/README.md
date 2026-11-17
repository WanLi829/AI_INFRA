---
license: mit
language:
- en
pretty_name: SWE-bench Verified — Qwen3-30B-A3B-Instruct-2507 with mini-swe-agent (T=0)
tags:
- swe-bench
- agent
- qwen3
- mini-swe-agent
- code-generation
size_categories:
- n<1K
---

# SWE-bench Verified — Qwen3-30B-A3B-Instruct-2507 rollouts with mini-swe-agent (T=0)

End-to-end agent rollouts and grading on [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) (test split, 500 instances) for [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), using [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) as the scaffold.

## Headline result

**Pass@1: 44 / 500 = 8.8%**

| Category | Count |
|---|---|
| Resolved | 44 |
| Unresolved (patch applied, tests failed) | 119 |
| Empty patch (LimitsExceeded) | 284 |
| Patch failed to apply | 53 |
| Pass rate among applied patches | 27% (44/163) |

## Run configuration

**vLLM server (4× H100 single endpoint):**
- `--data-parallel-size 4` + `--enable-expert-parallel`
- `--max-model-len 131072`
- `--max-num-seqs 256`
- `--dtype bfloat16`
- `--tool-call-parser hermes`
- `temperature: 0.0`
- `max_tokens: 2048` per turn

**Agent (mini-swe-agent v2.2.8):**
- Scaffold: `mini-extra swebench` with the default `swebench.yaml` prompts
- `step_limit: 50`
- `temperature: 0.0`
- `max_tokens: 2048`
- 100 concurrent workers
- Execution: custom `swerex_modal_fast` env class — Modal sandboxes with `swe-rex` pre-installed via `pipx` on top of the standard `docker.io/swebench/sweb.eval.x86_64.<instance_id>` images

**Grading:** `swebench.harness.run_evaluation --modal true --max_workers 100` (official harness in Modal mode).

## File layout

```
.
├── preds.json              # dict[instance_id] = {model_name_or_path, instance_id, model_patch}
├── preds.jsonl             # same data, one JSON per line (swebench harness format)
├── grading_report.json     # canonical SWE-bench harness output: resolved_ids, error_ids, etc.
└── trajectories/
    ├── astropy__astropy-12907.traj.json
    ├── ...                  # 499 files (one instance failed at sandbox-spawn)
    └── sympy__sympy-24661.traj.json
```

## Trajectory format

Each `<instance_id>.traj.json` contains the full agent conversation produced by mini-swe-agent. Top-level keys:

- `info`:
  - `exit_status`: one of `Submitted`, `LimitsExceeded`, `ContextWindowExceededError`
  - `submission`: the model's final patch text (empty if not Submitted)
  - `model_stats`: `{api_calls: int, instance_cost: float}`
  - `config`: snapshot of the agent + model + environment config
- `messages`: ordered list of `{role, content, tool_calls?, extra?}` — the full LLM conversation including system prompt, instance prompt, agent reasoning, tool calls, and observations
- `trajectory_format`: usually `mini-swe-agent`
- `instance_id`

## Grading report fields

`grading_report.json` is the schema_version=2 output from `swebench.harness.run_evaluation`. Key fields:

| Field | Meaning |
|---|---|
| `total_instances` | 500 |
| `submitted_instances` | 500 |
| `resolved_instances` | passed FAIL_TO_PASS + maintained PASS_TO_PASS |
| `unresolved_instances` | patch applied, tests didn't pass |
| `empty_patch_instances` | preds had empty model_patch (mostly LimitsExceeded) |
| `error_instances` | patch failed to apply (malformed diff) |
| `resolved_ids`, `unresolved_ids`, `empty_patch_ids`, `error_ids` | corresponding instance ID lists |

## Reproducibility

Components:
- vLLM 0.20.1 (serves Qwen3-30B-A3B-Instruct-2507 with `--tool-call-parser hermes`)
- mini-swe-agent 2.2.8 + a small custom env class (`swerex_modal_fast`)
- `swebench==4.1.0` for grading
- Modal account for execution sandboxes

## Reference baselines

| Source | Model | Scaffold | Pass@1 |
|---|---|---|---|
| Nebius / SWE-rebench | Qwen3-30B-A3B-Instruct-2507 | OpenHands (100 turns) | 25.2% |
| Nebius | Qwen3-Coder-30B-A3B-Instruct | OpenHands | 51.9% |
| **This dataset** | **Qwen3-30B-A3B-Instruct-2507** | **mini-swe-agent (step_limit=50)** | **8.8%** |

The gap between this dataset's 8.8% and Nebius's 25.2% on the same model is attributable to mini-swe-agent's bash-only diff generation (vs OpenHands' structured editing tools) and the tighter step limit (50 vs 100). A/B testing in our setup found `temperature=0.7` produced no improvement over `temperature=0` (both gave 44/500).
