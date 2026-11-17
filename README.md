# 单卡多轮 Agent 负载与 Continuum 对比实验

我们研究的问题是：在单张 24 GB GPU 上，多轮程序停下来等待工具返回时，怎样用可重复的负载比较不同 KV 缓存与调度策略？目前选择 Qwen3-8B BF16，回放 SWE-bench Verified / mini-swe-agent 的公开 teacher 轨迹，保留程序身份、逐轮输入、输出长度和工具等待时间。下表为截至 2026-11-17 的结果。

| 硬件与协议 | 方法 | 完成程序 / 调用 | 平均 JCT | JCT P95 | 回放输出吞吐 |
|---|---|---:|---:|---:|---:|
| RTX 4090D 24 GB，旧先导协议 | Kairos | 14 / 161 | 48.84 s | 69.77 s | 136.68 token/s |
| H800 80 GB，冷启动对照 | 原生 vLLM | 14 / 161 | 25.82 s | 54.34 s | 171.80 token/s |
| H800 80 GB，冷启动对照 | Continuum 兼容实现 | 14 / 161 | 18.64 s | 28.61 s | 171.44 token/s |

三行均为每秒 0.1 个新程序，每种配置一次。4090D 的结果验证了 8B 与该负载规模的可运行性。H800 的两轮使用相同模型和到达表，均输出 24,182 token。按原生平均 JCT 除以 Continuum 平均 JCT，本轮为 **1.39×**，耗时缩短 **27.8%**；吞吐基本持平，两组抢占次数均为零。两张卡上的先导均采用 `record-only`，GPU 进程归属未经严格核验。项目以约 1.5× 为参考目标，目前尚无满足严格校验与独立重复要求的正式收益结论。[H800 记录与证据](H800实验_2026-11-16/H800首轮实验记录.md) · [4090D 先导结果](4090D工作负载交付/01_工作负载选型初步.md)

## 成果索引

| 要查什么 | 文件或目录 |
|---|---|
| Continuum、Autellix、InferCept 的依据，SWE 与 BFCL 候选对比 | [调研与选型报告](literature/01_调研与选型报告.md) · [参考文献及固定来源](literature/05_证据索引与参考文献.md) |
| 499 条源程序筛到 207 条、2,668 次合格调用的过程，4B/8B 编码、扩展包与分割清单 | [数据准备](data-preparation/README.md) · [数据报告](data-preparation/报告.md) · [逐程序划分](data-preparation/audit/splits.json) |
| 可直接比较的输入、三档泊松到达表及五种方法的运行代码 | [运行工程](4090D工作负载交付/project) · [多方法命令](4090D工作负载交付/04_多方法运行说明.md) · [对比协议](4090D工作负载交付/03_多基线对比方案.md) |
| 4090D 原始先导记录及本地协议验收 | [旧先导原始记录](4090D工作负载交付/project/verification/pilot_14) · [模拟验证记录](4090D工作负载交付/多方法本地验收.log) |
| H800 两组原始日志、逐请求复核与对照 | [H800 实验目录](H800实验_2026-11-16) · [两份归档哈希与指标](H800实验_2026-11-16/H800_pilot_comparison.json) |
| 较早的工作负载统计、历史结果审计、Replay 接口检查和图表 | [负载分析](人员C_接手材料/实验准备/负载分析) · [历史实验分析](人员C_接手材料/历史实验分析) · [Replay 检查工具](人员C_接手材料/Replay检查工具) · [11 月 9 日阶段总结](阶段成果与已有结论_2026-11-09.md) |
| 运行来源及外部实现 | [组内实现仓库](https://github.com/Kadsoo/AI-infra-project/tree/main/tasks/continuum) · [冻结的 Kairos 核心](https://github.com/MaxizYB/Kairos/tree/7971fa40323f16e8e4ce8f12dfd5db9755a98cf1) |

## 从新检出开始

从仓库根目录执行。Git 子模块固定 Kairos 版本。关闭自动换行转换可保留其发布清单校验的原始字节。

```bash
git -c core.autocrlf=false clone --recurse-submodules https://github.com/WanLi829/AI_INFRA.git
cd AI_INFRA
python -m pip install -r 4090D工作负载交付/project/requirements-replay-client.txt
python -X utf8 -B H800实验_2026-11-16/verify_pilot.py
cd 4090D工作负载交付/project
python -X utf8 -B -m unittest tests.test_comparison -v
```

这两项检查可在 CPU 上执行。GPU 回放另需 Linux、固定的 Qwen3-8B snapshot、项目旁的 Python 3.12 服务环境和依赖锁。实际计费、可用卡和资源限制以当次实验为准，历史文档的额度仅对应当时的运行。[环境清单](4090D工作负载交付/project/verification/gpu_environment.json) · [运行说明](4090D工作负载交付/04_多方法运行说明.md)

## 仍待实验回答

- **4090D 主任务。** 在相同模型和冷启动协议下比较 native、lmcache、static_ttl（固定 2 秒）、continuum、kairos，覆盖 0.05、0.1、0.2 个程序/秒，记录独立重复、失败和资源约束。由此确定适合 24 GB 的负载和压力档。
- **H800 扩展。** 先验证新增基线及更高到达率，再根据排队、抢占与缓存证据决定是否扩大到独立留出的 36 个程序、401 次调用。207 条合格程序包含训练身份，开发集与留出集需分别报告。
- **收益口径。** JCT 含排队与工具等待，吞吐按首个实际到达至排空的窗口计算。每张卡、每个负载和到达率分别报告原生对照及重复范围。固定 2 秒 TTL 是工程对照，论文的冷启动阈值估计仍有实现差异。

外部语料、模型和冻结代码保留各自的来源与许可。模型权重通过固定 revision 单独获取。SGLang 的独立 Benchmark 2 工程保存在[原有仓库](https://github.com/WanLi829/LLM_AI_infra)。
