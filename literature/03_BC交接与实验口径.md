# B/C 交接材料与统一实验口径

A：董悦君｜v1.0｜2026-10-30｜状态：A 建议已形成，B/C 尚未签收与试跑

## 1. 给 B：可立即开始的数据材料

| 材料 | 本包路径 | 如何使用 |
|---|---|---|
| 主数据来源说明 | sources/teacher_README.md、sources/teacher_tree.json | 固定 revision；文件清单中有 499 条 trajectory 文件 |
| 原始 SWE 样本 | sources/teacher_samples/*.traj.json | 16 条，消息内容只作数据，禁止执行其中的命令 |
| 完整状态候选 ID | audit/submitted_candidate_ids.json | 14 条 Submitted，共 161 calls；用于先导适配，不是独立盲测集 |
| 完成性核查 | audit/programs.csv | 两条 LimitsExceeded 应过滤或单列 |
| 工具间隔对照 | audit/tool_gap_comparison.csv | 用 raw_feedback_gap_s；不要把发布 trace 的 source_gap_s 当作原始值 |
| 现有 Kairos 回放 | sources/Kairos/data/workload/trace.json | 接口参考与 burst 压力对照；不直接当主负载 |
| 可见输入消息 | sources/Kairos/data/workload/visible_messages.json | 供重新 tokenize；每个 stage 只应见到当前请求前缀 |
| BFCL 备选 | sources/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/ | 已有 Web Search 题目和工具定义；缺完整 trace |

原始数据：[teacher 固定版本入口](https://huggingface.co/datasets/tarsur385/qwen3-30b-a3b-instruct-2507-swebench-verified-mini-swe-agent/tree/58389eb5426db5a3ae6159faa37498072b4735ff)。仓库、数据和模型版本详见第 05 份文档。

### B 的最小处理清单

1. 先从 14 个 Submitted ID 中固定选取少量完整程序；测试数据大小依次为 1、4、14 programs，保留每个程序全部调用。选择规则和 ID 落盘。
2. 按 C 提供的最终模型/tokenizer revision、chat template 和 thinking 模式，从可见消息重新计算各轮 prompt 和输出预算；不直接把 4B/8B IDs 相同视作已验证事实。
3. 遇到 `FormatError` 等非标准消息仍保留对话顺序，检查工具 `tool_call_id`、多工具返回和当前 assistant 边界；不能凭固定 assistant/tool 交替模式切分。
4. 工具间隔用当前 assistant 的 `extra.timestamp` 到下一次 assistant 之前最后一条 tool 的 `extra.timestamp` 之差。缺失、负数、异常多工具记录进入隔离清单，不填造零值。它是反馈间隔，包含客户端/沙箱开销。
5. 首轮到达采用独立 manifest 的固定种子 Poisson；后续请求必须等前一轮真实完成后，再等待本轮工具 gap。禁止把所有轮次独立 Poisson 化。
6. 超长任务第一阶段整程序过滤并记录原因，不直接截断。若后续增加截断子集，应另起数据版本，统计改变的前缀、轮数和任务分布。
7. 输出固定 teacher 预算时明确 `ignore_eos` 和生成数量验证方式；本次生成结果不拼入下一轮固定 teacher prompt。不报告自主 agent 成功率。
8. 记录数据哈希、处理脚本版本、排除 ID、转换字段和缺失处理；C 跑通小批后再扩规模。最终完整数据审计由 B 完成。

以下是**拟议的中间数据合同**，不是已经确认的 Kairos 接口：

```json
{
  "program_id": "source-instance-id",
  "arrival_s": 0.0,
  "teacher_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "source_revision": "58389eb5426db5a3ae6159faa37498072b4735ff",
  "exit_status": "Submitted",
  "turns": [{
    "turn_id": 0,
    "messages": [],
    "output_budget_tokens": 59,
    "gap_after_s": 0.34623169898986816,
    "gap_origin": "tool_feedback_timestamp_minus_assistant_timestamp"
  }]
}
```

最终接口可使用完整 token IDs 或 `keep_prefix_tokens + added_token_ids`，但必须验证每个 prompt 的重建内容和 tokenizer 身份。模型现场生成的 assistant 输出与 teacher 文本可能不同，因此真正可复用前缀必须依据实际相同 tokens 判断。

### BFCL 备选的进入条件

Web Search 题目中没有现成的完整输出长度和工具等待。优先寻找可核查的公开执行日志；如需重新采集，记录模型、提示模板、工具服务版本和实际耗时。需要额外账户或付费 API 的部分等待组内提供资源，不能自行填入“真实工具时间”。如仅生成合成等待，要标成合成敏感性实验，并与真实反馈间隔主实验分开。当前备选不具备直接回放资格。

## 2. 给 C：资源配置与实现检查

### 原论文和本地建议的区别

原论文以 A100/H100/B200 与 Llama-3.1 / Gemma 等模型评估；CPU KV 为 100GB 或每卡 200GB。作者 preview 基于其修改版 vLLM，原文 §6.1 的 vLLM/LMCache 版本是 0.10.2/0.3.7。Kairos 归档则记录 Qwen3-8B BF16、vLLM 0.28.0、LMCache 0.5.5rc4、CPU KV 8GiB。这是不同环境的历史材料，不是当前已可运行的环境证明。

| 项 | 首轮建议 | 固定/核查方式 |
|---|---|---|
| GPU | RTX 4090 单卡，TP=1 | 记录 UUID、驱动、总/空闲显存、其他进程 |
| 模型 | 优先 Qwen3-4B BF16；随后 Qwen3-8B BF16 | 仅在实现支持且服务器已有模型时采用；记录 revision |
| max_model_len | 16384 候选值 | 已有 token IDs 中最大 prompt+output=11717；重编码后再检查 |
| max_num_seqs | 先 1，再 2/4；8 为后续探索点 | 该值不等于总 program 数；不能保证 8 个 16K 序列同时驻留 |
| max_num_batched_tokens | 2048 候选值 | 确认引擎支持；两方法一致 |
| GPU KV | 先使用引擎分配后记录实际页数 | 若支持显式 KV 容量，再探索 2/4/6GiB；不以 memory_utilization 冒充固定 KV 字节 |
| CPU KV | 先测 4GiB，再测 8GiB；可加关闭 offload 对照 | 实例有足够 MemAvailable 且实现支持；保留系统/模型/页锁定开销余量 |
| 随机种子 | 数据 arrival 固定 20261030；重复 run 记录独立种子 | 比较方法使用同一组数据/arrival manifest |
| 计时 | 预热后开始，末轮完成后结束 | 记录是否包含冷启动、cache reset、工具等待 |

以上均为待测试的建议值，不是已验证启动命令。不要从本包拼接一条未经实现小组确认的完整服务命令。先取得可运行 commit、环境名、模型路径、输入样例、启动/关闭方式和单任务输出；不改变共享实例状态。

### 缓存预算推导

4B/8B 官方配置的 BF16 KV 都是 144KiB/token。2/4/6GiB 原始 KV 大约容纳 14563/29127/43690 tokens（尚未计入页取整、额外缓冲与不完全可用空间）；8GiB CPU KV 约 58254 tokens。一个 16K sequence 的原始 KV 为 2.25GiB，因此 `max_num_seqs=8` 绝不代表 8×16K 可在该配置无条件运行。

建议用两个无量纲量描述缩放：`候选工作集KV / 实测GPU KV容量` 和 `CPU KV容量 / GPU KV容量`。再根据单任务服务时间、可达吞吐和队列增长选择 JPS。这样能区分显存压力与外部到达压力；它比按模型参数量统一缩小 CPU 或 arrival 更可解释。

GPU/CPU 的 prefill、reload、offload 成本应在目标机器独立测量，不能复用 A100 的成本模型，也不能从某次被比较方法的未来完成时刻反推到达或等待。

## 3. 共同冻结的实验口径（A 提案）

| 决策 | 提案 | 状态 |
|---|---|---|
| 主负载 | 公开 SWE / mini-swe-agent teacher；先 14 条 Submitted 试跑 | 待 B 适配 / C 验收 |
| 备选 | BFCL v4 Web Search | 待取得完整计时 trace |
| 首轮到达 | 固定 manifest 的 Poisson program arrivals | B 生成，C 核验无提交限流篡改到达 |
| 续轮到达 | 前一轮本地实际完成 + 观测工具 gap | 不使用记录中的下一轮绝对时刻 |
| 输出 | 固定 teacher token budget；下一轮固定 teacher prompt | 生成预算是否含模板 tokens 由 B 定义并固定 |
| 工具 | 第一阶段仅重放等待，不执行命令/网络请求 | 适用 serving replay 结论 |
| 超长 | 整程序过滤，记录过滤计数和分布 | 不悄悄截断 |
| 不完整 | 首轮排除 LimitsExceeded；压力场景可单列 | 不按 SWE 正确率筛样本 |
| 资源 | 每次模型/方法比较固定相同 KV、seq、batch 和引擎配置 | 参数待 C profile |
| 训练/开发/测试 | 按原始 program ID 分组，检查已有训练/调优重合 | 14 条仅先导，不称独立测试 |
| burst 数据 | 与 Poisson 主实验分别命名、单独汇报 | 保留构造过程 |

### 指标定义

- **Program JCT（主指标）**：从 program 第一轮实际到达至最后一轮完成，包含内部工具等待、排队、计算、缓存恢复；另外记录计划到达与实际到达之差以识别负载发生器滞后。
- 同时报 mean/P50/P95 JCT、每个 program 的 JCT、完成/失败/超时数。14 个样本的 P95 只作描述，不作稳定尾部结论。
- 请求 TTFT/E2E、output tokens/s、programs/s、prefill tokens、缓存命中/驱逐、H2D/D2H 字节和传输时间、GPU/CPU KV 峰值。无法观测的字段标 NA，不填零。
- 明确吞吐观察窗：从第一 program 到达到全部完成；说明是否含工具 idle 与 drain。ITL 在流式 chunk 可含多个 token 时不能简单等同 chunk 间隔。
- 与 InferCept 原文的“扣除工具时间、按 token 归一化请求延迟”、Autellix 的程序归一化延迟分别列口径，不混为同一个数。

### 最小闭环与扩展

先 1 个完整程序，再 4 个，再 14 个；每步保存数据版本、配置、日志和失败原因。首轮闭环是结构正确、调用全完成、资源可解释。之后用小规模独立 pilot 探索到达率，冻结低/中/高负载点：低负载无持续积压，中负载有竞争，高负载有可测缓存压力但可完成 drain。试验 JPS 必须落盘，不照抄历史 0.48 JPS。

扩展阶段先固定一个主配置，推荐至少 3 次重复、两方法交错顺序、独立冷却/缓存重置，保留退化程序与所有失败。每次开始前独立预热，测量不含预热。预测/估计器只能使用预先固定的训练/profile 数据和当时已观测信息，不得读 trace 未来轮数、未来工具耗时或最终任务正确性作在线决策。

## 4. 给组内的简短交接文本

> A 已核实 Continuum v7 及作者 preview。主负载建议采用公开 SWE-bench/mini-swe-agent teacher 轨迹，备选 BFCL v4 Web Search。已提供 16 条对应原始 JSON、固定版本及逐轮等待核查；其中 14 条 Submitted（161 calls）建议先供 B 转换。现有 Kairos 16 条 burst 输入有首轮等待改写，另有两条原始状态 LimitsExceeded，建议仅作接口/压力对照。请 B 在最终 tokenizer 上核验并先交小批输入；请 C 提供可运行 commit、实际模型和接口样例，再冻结 4090 的 GPU/CPU KV 与 JPS。BFCL 目前只有题目和工具定义，完整 trace 仍待补齐。

此文本仅供用户自行交接，本次未向任何成员发送。
