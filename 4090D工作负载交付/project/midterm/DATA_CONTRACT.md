# 第二数据集与输入交付合同

本仓库提供 mini-SWE 的现成转换/审计路径。Nebius 的准确测试包尚未给到，不能根据名称臆造其字段或合成时间。本文件用于对接实际提供方；拿到源数据后按其 schema 编写转换器，并通过统一 `load_bundle` 验收。

## 每个目录包含

- `manifest.json`：`schema_version=b-replay-bundle/2.0`、`model=Qwen3-8B`、`model_revision`、唯一 `dataset_id`、有序 `program_ids`、`programs`、`calls`、`total_output_tokens`、`files`（以下各文件 SHA256）、`predictor_split_audit`。
- `trace.json`：`model.max_sequence_tokens=16384`；`programs` 数组。每程序含 `id`、稳定 `source_id`、`cache_salt`、`turns`。
- `visible_messages.json`：对象，键是实例 ID，值是逐轮调用前的消息数组；不含当前 assistant 回答或未来反馈。
- `completed_actions.json`：`schema=recorded-current-actions-v1`，`actions` 是实例 ID 到逐轮已完成动作的映射。
- `data_audit.json`：源地址/revision、源任务与轨迹身份、许可信息、数据生成模型、终止类型、逐间隔时间来源、过滤记录、转换版本和训练重叠审计。

每轮 turn 包含：

| 字段 | 定义 |
|---|---|
| keep_prefix_tokens | 从上一轮输入 token 序列保留的前缀长度；首轮为 0 |
| added_token_ids | 后续追加的非负整数 token 列表 |
| prompt_tokens | 还原后输入长度 |
| prompt_sha256 | `sha256(json.dumps(prompt_ids).encode())`，使用 Python 默认空格分隔 |
| visible_messages_sha256 | 对本仓库 `json_bytes(messages, pretty=False)` 取 SHA256（排序键、紧凑分隔、末尾换行） |
| output_tokens | 本轮记录输出工作量，正整数，仅私有执行器读取 |
| gap_after_s | 从该轮实际完成到下一轮可提交的真实返回间隔；末轮不使用，可记 0 |

当前模型/tokenizer 必须使用固定 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`。即使源轨迹由别的模型生成，也需按回放 tokenizer 重新定义固定 token 工作量；转换规则和源输出文本须可追溯。输入+输出不能超上下文，工具时间缺失不能静默填零。

每轮 completed action 只能有 `tool`、`terminal`、`evidence`、`assistant_sha256`。使用上游 `kairos.replay.completion.recorded_action` 从当前已完成 assistant 生成；其 `terminal` 必须有显式提交动作证据，不能因为“这是最后一轮”就设为 true。当前解析器针对已支持的动作 schema；对 Nebius 新工具格式，需要真实样例和兼容性核验，不能无声推断。如果其结束证据不符合现有 schema，先向总负责人反馈，不能伪造 terminal。

复制实例需保留 source_id、改独立实例 ID/cache_salt；当前复制协议仍需实机检验 CPU 层隔离。`predictor_split_audit` 未审计时明确 `independent_test=false`。模型训练/验证排除清单须覆盖当前联合预测器，不能只给辅助森林清单。

## 验收过程

1. 先交 1–2 个完整 Program 的源轨迹、转换包、间隔来源和可重建脚本。
2. 运行 `python -m b_replay.datasets describe --bundle /path/to/bundle --output /path/to/features`，检查结构、哈希、长度和动作数量。
3. 各方法生成 plan，核查缺项与公共输出边界；进行 CPU 模拟协议测试。
4. H800 上进行小样本五配置验收，检查真实输出量、工具依赖、completion 事件及缓存隔离。
5. 扩大独立测试样本，冻结输入哈希，才进入跨数据集主矩阵。

没有第二套真实数据时，可以报告单源特征分组，但不能把两套同源子集更名为两个数据集。矩阵工具会拒绝同一 `dataset_id` 的跨数据集对比；真实来源审计仍由人工负责。
