# 数据分析与 Replay 数据准备

B 部分任务报告    杨智涵    学号：redacted

## 一、任务与交付范围

本任务负责候选轨迹获取、数据完整性核查、特征统计、格式转换、小规模回放数据和接口适配，为后续系统测试提供可追溯输入。交付包括原始来源、审计表、过滤清单、消息中间格式、增量 token 数据、回放客户端及运行说明。

## 二、数据来源与完整性

主负载为公开 SWE-bench Verified / mini-swe-agent teacher 轨迹，来源模型为 Qwen3-30B-A3B-Instruct-2507。每个原始轨迹对应一个 program，assistant 消息对应一轮已记录调用。原始文件按固定修订获取，并保留逐文件 SHA256。

| 项目 | 结果 |
| --- | --- |
| 原始程序 / 已记录 assistant 调用 | 499 / 13,884 |
| Submitted / LimitsExceeded | 223 / 276 |
| 合格程序 / 已记录调用 | 207 / 2,668 |
| 排除 / 隔离 | 292 / 0 |
| 已记录调用缺失 usage | 0 |
| 未记录完整 assistant 消息的 API 尝试 | 3,517，涉及 479 个程序 |

已记录消息的角色、先后顺序、工具调用关联和时间戳检查通过。3,517 次额外 API 尝试未保存完整 assistant 消息，其差额逐程序与 FormatError 反馈数一致；相关反馈仍保留在后续可见历史中，不补造输出或耗时。因此回放覆盖已记录调用序列，不代表完整 agent 的全部执行成本。Submitted 仅表示提交结束，不等于任务正确。

## 既有 Kairos 归档核查

既有归档包含 16 个程序、178 个调用，其中 14 个 Submitted、2 个 LimitsExceeded。162 个非终止间隔中有 16 处被改写，均位于各程序首轮；原始平均间隔为 0.3568s，归档为 0.5626s。本交付沿用原始工具反馈间隔，不沿用归档的首轮改写。

## 三、特征统计与筛选

以下统计以 Qwen3-8B tokenizer 下的合格候选为口径；长度单位为 token，间隔单位为秒。全量逐程序、逐轮统计分别保存在 audit/ 与 encoded/ 中。

| 指标（合格候选） | 均值 | P50 | P95 | 最大值 |
| --- | --- | --- | --- | --- |
| 轮数 | 12.9 | 11.0 | 26.0 | 38.0 |
| Prompt token | 5,344.8 | 4,643.5 | 11,731.8 | 16,070.0 |
| 输出预算 token | 149.6 | 110.0 | 364.3 | 1,503.0 |
| 上下文增长 token | 484.9 | 289.0 | 1,529.0 | 9,031.0 |
| 非终止反馈间隔 / 秒 | 0.4372 | 0.1896 | 0.9190 | 60.0459 |

合格程序最终上下文平均长度为 7,271.8 token，P95 为 13,513.5 token。相邻 teacher prompt 的前缀重复量占总 prompt token 的 89.38%；该比例描述输入一致性，不作为实际 GPU 缓存命中率。

## 筛选规则

主候选要求 Submitted、已记录轮次结构完整、反馈时间有效，且整程序每轮 prompt 加输出预算不超过 16,384 token。超限按整程序排除，不截断、不拼接其他任务；不按任务正确率筛选。原始数据均保留，过滤原因逐程序记录。

499 个程序中接受 207 个、排除 292 个、隔离 0 个。非 Submitted 原因为 276 个，超长原因为 187 个，两类原因重叠 171 个；其中 16 个 Submitted 程序因超长排除。多标签原因数不直接相加。

| 划分 | 合格程序 | 已记录调用 |
| --- | --- | --- |
| train | 157 | 2106 |
| development | 14 | 161 |
| blind | 36 | 401 |

划分沿用 Kairos 原始 program ID。all_eligible 包含 157 个训练身份，供数据清点与明确标注的压力准备；Kairos 启动入口会拒绝这些训练身份。heldout_candidates 的 36 个程序与 train/development 不重合，但曾参与历史离线评价，不称为全新盲测集。

## 四、转换与回放规则

分别固定 Qwen3-8B 与 Qwen3-4B tokenizer，仅下载 tokenizer 文件；固定 enable_thinking=False 和 chat template。每轮仅编码当前 assistant 生成前已可见的消息，原始 usage 独立保留，目标模型长度重新计算。

增量输入由相邻完整 prompt 的最长公共前缀生成：current_ids = previous_ids[:keep_prefix_tokens] + added_token_ids。每轮保留完整 prompt 的 SHA256，并逐轮验证重建与完整 tokenize 一致。输出预算采用完成模板与生成前模板的差分，包含模板边界 token。

只有 program 首轮使用 Poisson 到达，seed 固定为 20261030，JPS 由命令显式指定并单独保存 arrival manifest。续轮在上一轮实际完成后，再等待原始工具反馈间隔；间隔定义为 assistant 完成到下一轮前最后工具反馈的时间差。缺失或负间隔进入隔离清单。

客户端使用固定输出预算、ignore_eos=True，并核对实际输出 ID 数量。实际生成文本不进入后续 teacher prompt。工具命令不执行；多工具与异常反馈按原顺序保留。在线工具类别采用未知标签 unclassified-gap，最终 patch、退出状态和未来工具类别不传入在线特征。

## 五、回放数据包

| 数据包 | 程序数 | 调用数 | 输出预算 token |
| --- | --- | --- | --- |
| pilot_1 | 1 | 10 | 2,437 |
| pilot_4 | 4 | 44 | 7,311 |
| pilot_14 | 14 | 161 | 24,182 |
| all_eligible | 207 | 2668 | 399,134 |
| heldout_candidates | 36 | 401 | 58,588 |

先导包按 program ID 排序构成嵌套的 1、4、14-program 子集，对应原 development 的合格程序。两套模型均提供相同格式的转换结果；冻结服务已支持 Qwen3-8B，4B 的交付范围为数据转换。

每个包包含 trace.json、visible_messages.json、visible_protocol.json、manifest.json；统计与过滤清单位于 encoded/，完整消息中间格式位于 intermediate/。到达表与具体 trace 的 SHA256 绑定，不能跨数据包混用。

## 六、接口验证与交接

采用 Qwen3-8B/BF16，在 RTX 4090 D（24,564MiB 显存）、30GiB 容器内完成先导输入联调。保持 8 个生成槽位、8GiB CPU KV 容量及 16K 长度限制；使用 0.1 JPS、seed 20261030 的到达表。

| 先导包 | 成功 / 计划调用 | 实际输出 token | 采样容器占用峰值 / GiB |
| --- | --- | --- | --- |
| pilot_1 | 10 / 10 | 2,437 | 13.741 |
| pilot_4 | 44 / 44 | 7,311 | 13.782 |
| pilot_14 | 161 / 161 | 24,182 | 14.007 |

逐调用核对 program/turn、prompt 签名、输出预算和续轮等待，全部计划调用完成。三包各通过五类 prefix-only 基础布局检查。CPU 数据重编码、增量重建、消息边界和模拟客户端检查均有验证记录。以上结果用于确认输入与接口可用，不作为系统性能提升结论。

GPU 环境为 Python 3.12.4、torch 2.13.0+cu130、vLLM 0.28.0、LMCache 0.5.5rc4、Transformers 5.16.1。实际 CUDA 构建与原归档 cu132 存在差异。运行期未映射的宿主机 PID 仅记录；内存与 OOM 监测保持启用。

## 七、适用边界与备选负载

正式方法对比、压力与资源扫描、重复性能实验及系统设计修复由 C 和实现组开展。本交付不补写未测性能，也不把固定 teacher 回放解释为自主 agent 成功率。

BFCL Web Search 已核查 100 道题目、工具定义及相关公开仓库日志线索；现有材料缺少完整逐轮可见消息、assistant 输出和工具反馈时间戳，尚不能导出真实计时 Replay。第二类真实负载仍需补齐合格执行日志。

## 固定来源

| 来源 | 固定身份 |
| --- | --- |
| SWE teacher 数据 | 58389eb5426db5a3ae6159faa37498072b4735ff |
| Kairos 接口 | 4ddff8786254c8c5ed8ac18deb22486e7a0594fa |
| Kairos 冻结执行代码 | 7971fa40323f16e8e4ce8f12dfd5db9755a98cf1 |
| Qwen3-8B tokenizer | b968826d9c46dd6066d109eabc6255188de91218 |
| Qwen3-4B tokenizer | 1cfa9a7208912126459214e8b04321603b3df60c |

原始链接、许可与逐文件身份见 sources/来源清单.json、sources/download_manifest.json 及各 tokenizer 的 LICENSE。运行与复算入口见 README.md、运行说明.md。
