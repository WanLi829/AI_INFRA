# 运行与交接说明

## CPU 数据检查

从本目录运行，使用独立 Python 环境。CPU 转换不需要模型权重或 GPU。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m b_replay verify-release
.venv/bin/python -m pytest
.venv/bin/python -m b_replay validate
```

离线复算数据处理：

```bash
.venv/bin/python -m b_replay audit
.venv/bin/python -m b_replay convert --model Qwen3-8B
.venv/bin/python -m b_replay convert --model Qwen3-4B
.venv/bin/python -m b_replay export --model Qwen3-8B
.venv/bin/python -m b_replay export --model Qwen3-4B
.venv/bin/python -m b_replay validate
```

已有来源文件可离线复算。重新下载并核验固定来源：

```bash
.venv/bin/python -c "from b_replay.acquire import download; download()"
```

若重新导入完整 A 侧资料，可使用 `acquire --a-zip`。随包的 A 侧输入仅保留转换和审计需要的资料。

CPU 模拟协议检查：

```bash
.venv/bin/python -m tests.smoke_real_bundles --output-root runs/cpu_check
```

模拟服务只检查提交顺序、等待、预算和失败处理；其耗时不作为 GPU 性能。

## 数据格式与到达表

`bundles/<模型>/<子集>/trace.json` 保存增量输入与每轮输出预算；`visible_messages.json` 保存当前轮可见消息；`manifest.json` 与 `visible_protocol.json` 保存身份和签名。

```bash
.venv/bin/python -m b_replay arrivals \
  --bundle bundles/Qwen3-8B/pilot_1 \
  --jps 0.1 --seed 20261030 --output runs/pilot_1_arrivals.json
```

JPS 必须显式指定。示例用于输入联调，不是测出的最优压力。两种方法比较应使用同一份 arrival manifest；换包后重新生成到达表。续轮等待和输出预算不缩放。

## 对接已有服务

服务须提供 `/v1/completions`，接受整数 prompt IDs、流式返回 token IDs、ignore_eos、cache_salt 和 Kairos 的 vllm_xargs 扩展。模型目录须对应固定 Qwen3-8B snapshot。将下列模型目录和地址替换为实际值：

```bash
.venv/bin/python -m b_replay replay \
  --bundle bundles/Qwen3-8B/pilot_1 \
  --arrivals runs/pilot_1_arrivals.json \
  --endpoint http://127.0.0.1:18765 \
  --model /models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --method kairos --output runs/pilot_1_replay
```

检查 summary.json 的 strict_complete、调用数量和输出数量。结果目录必须不存在。数据包中的 gpu_validated 为导出阶段标识；实际输入验证独立保存在 verification/gpu_verification.json。

## 随包服务启动入口

GPU 服务使用独立环境；CPU 转换锁与 GPU 服务锁分别安装。`requirements-gpu-linux-py312-hashed.txt` 对应 Linux x86_64、CPython 3.12 的固定 wheel 选择。模型权重另行准备。

项目目录旁的 `env/` 为服务环境；`scripts/gpu_python.py` 设置私有 CUDA Toolkit、驱动库发现和工作区缓存。先查看启动计划，再启动：

```bash
../env/bin/python scripts/gpu_python.py -m b_replay server-plan \
  --server-profile ram30 --gpu-pid-policy record-only \
  --bundle bundles/Qwen3-8B/pilot_1 \
  --arrivals examples/smoke_arrivals/pilot_1_jps0.1_seed20261030.json \
  --model /models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --method kairos --output runs/pilot_1_gpu
```

核实 GPU 空闲后，将 `server-plan` 改为 `server-run`。4、14 条包替换对应 bundle 与 arrival 文件，分别使用新的短输出目录。`ram30` 保留 8 个生成槽位、8GiB CPU KV 容量、BF16 和 16K 限制；`record-only` 用于容器 PID 与驱动 PID 无法对应的环境，只记录未映射 PID。默认 strict 策略保留。容器内存、OOM 监测与本次进程组清理保持启用。

冻结服务仅支持 Qwen3-8B。Kairos 会拒绝包含训练身份的数据；`all_eligible` 用于数据清点，扩展输入可选择 `heldout_candidates`。原始 `python -m kairos.run` 固定读取旧归档，应使用 B 的独立入口。

## 校验已保存的先导记录

```bash
python3 scripts/verify_gpu_runs.py \
  --run verification/pilot_1 --run verification/pilot_4 --run verification/pilot_14 \
  --output runs/checked_gpu_inputs.json
```

验证材料保留输入签名、完整输出 ID、续轮时序、资源样本和基础布局检查结果。接口验证不替代正式性能对比。

## 重新打包

```bash
.venv/bin/python -m b_replay package
```

打包代码更新文件校验清单并生成同名 ZIP。sources 中保留原始轨迹和 tokenizer；数据转换不执行轨迹命令。
