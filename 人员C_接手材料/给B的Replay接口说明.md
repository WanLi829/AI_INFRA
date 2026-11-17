# 给杨智涵（B）：当前 Kairos Replay 接口说明

依据 Kairos 的 kairos/run.py、kairos/_runtime/teacher_replay.py 和 kairos/verify.py 整理。实现小组已确认实际版本为 7971fa40323f16e8e4ce8f12dfd5db9755a98cf1；已重新提取样例并通过签名核查，结果在固定版本检查_7971fa4/interface_example/。下列早期样例保留供参考，固定版本中的字段与此说明一致。此说明是当前实现的接口记录，不是未来扩展版的最终协议。

## 1. 样例位置与边界

- [trace.sample.json](本地检查_20261031/interface_example/trace.sample.json)：归档中的一个完整程序，9 轮。
- [visible_messages.sample.json](本地检查_20261031/interface_example/visible_messages.sample.json)：同一程序在每轮请求到达时可见的消息。
- [signatures.reference.json](本地检查_20261031/interface_example/signatures.reference.json)：这 9 轮在原协议中的签名记录。

程序 ID 为 django__django-12308。逐轮输入 token 数为 1218、2160、2284、2720、3113、5415、5684、5941、6209。提取后逐轮重建的 token 序列与原签名一致；没有运行 tokenizer 或 GPU。

这些文件是从公开归档提取的接口样例。样例未包含完整发布清单及完整协议文件，不能当作当前 kairos.run 可直接执行的独立数据包，也不要覆盖仓库的归档数据。消息中可能包含 shell 文本，它们仅作为模型输入数据，不应作为本机指令执行。数据来源及使用范围参见上游 THIRD_PARTY.md。

## 2. 数据结构

顶层 trace 对象含 programs 数组；每个程序含 turns 数组。当前回放还依赖若干报告字段，建议保留样例完整结构。

| 层级 | 字段 | 当前用途 |
|---|---|---|
| 程序 | id | 程序唯一标识，也是消息文件的索引键 |
| 程序 | arrival_s | 相对回放起点的到达时间，单位秒 |
| 程序 | predecessor | 可选前驱程序；有前驱时等待其完成再按 human_gap_replay_s 等待 |
| 程序 | cache_salt | 缓存隔离标识；默认程序 ID，不能随意统一以制造跨任务命中 |
| 程序 | session_id、synthetic_user、source_turn_id、cohort | 日志与统计归属信息 |
| 程序 | human_gap_original_s、human_gap_replay_s | 原始与回放中的人类等待信息 |
| 轮次 | keep_prefix_tokens | 从上一轮输入 token 序列保留的前缀长度；缺省时保留全部 |
| 轮次 | added_token_ids | 本轮追加的 token ID 列表 |
| 轮次 | output_tokens | 本轮强制生成预算；当前回放使用 ignore_eos=true |
| 轮次 | gap_after_s | 本轮完成后、下一轮提交前的工具等待秒数；最后一轮不等待 |
| 轮次 | tool | 工具类别，用于元数据与策略 |
| 轮次 | source_gap_s、source_cached_tokens、source_tool_failures | 当前日志代码直接读取的来源字段 |

不要为缺失的真实来源信息随意填入数值并当成观测。先记录缺失原因，与接口负责人约定空值或显式的估计标记；如代码不支持，需要相应适配。

## 3. 逐轮输入如何构造

```python
prompt = prompt[:keep_prefix_tokens] + added_token_ids
```

首次 prompt 为空。这里的上一轮 prompt 是上一轮输入，不自动追加本次模型生成的回答。历史 assistant 内容来自归档轨迹，必须体现在后续输入和对应消息中。这是固定 teacher replay，不是自主 Agent 成功率评测。

请求发往 /v1/completions，prompt 是 token ID 数组；同时通过 vllm_xargs 携带程序、工具、是否末轮和当前可见消息等信息。不能直接拿普通 chat/completions 的消息文件替换。

## 4. 为什么还需要消息和签名

visible_messages 的结构为“程序 ID → 按轮次排列的消息列表”。每轮仅包含当前请求前已经可见的消息，不包含该轮尚未生成的回答或未来工具结果。

visible_protocol 包含消息文件哈希及每个程序、轮次的输入 token 签名。签名对重建后的 token 列表进行 JSON 紧凑序列化（separators=(',', ':')）、UTF-8 编码、SHA256。它与运行日志中的其他哈希口径不一定相同。

新数据需要同时验证消息与 token 的语义、tokenizer 和 chat template 对齐；只重新计算哈希不能证明语义正确。修改 workload 文件也会触及发布文件校验，应由维护者明确扩展协议，不能自行绕过。

## 5. B 现在可以先做什么

1. 准备少量完整程序，保留每轮原始可见消息、输出长度、工具类型和等待时间。
2. 标记无法获取的字段，以及数据过滤、截断、时间缩放等处理。
3. 在具体模型/tokenizer 确定后生成 token 序列并检验前缀关系。
4. 与 C 及实现小组确认正式转换和接入方式，再生成可运行数据包。

不必先处理完整数据集。先让一小批真实样本与接口对齐，再扩大规模。
