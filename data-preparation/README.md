# 数据、处理代码与先导验证

**直接做 GPU 实验请使用[当前运行工程](../4090D工作负载交付/project)。** 本目录保留数据准备阶段的代码、审计、逐模型统计、生成的输入和历史验收。两份代码有各自的版本，当前五方法入口位于运行工程。

- 从本目录执行 `python -m zipfile -e bundles.zip .` 还原两种模型的全部五档输入。正式开发先导 `pilot_14` 有 14 个程序、161 次调用；`heldout_candidates` 有 36 个程序、401 次调用；`all_eligible` 有 207 个程序，包含训练身份。
- `sources/download_manifest.json` 记录上游原始轨迹与 tokenizer 的固定来源、大小和校验值。复算原始抓取和转换时，可在装好 CPU 依赖后执行 `python -c "from b_replay.acquire import download; download()"`。处理完成的 `intermediate/`、`encoded/`、`audit/` 已收录。
- 原始交付的 `runtime/Kairos` 与当前运行工程共用同一固定提交。在 Linux 上如需运行本目录的旧版本测试，可从本目录执行 `mkdir -p runtime && ln -s ../../4090D工作负载交付/project/runtime/Kairos runtime/Kairos`。
- 原交付清单位于 `release_manifest.json`，记录历史文件身份。仓库版汇集了数据包、引用上游原文、隐去了学号和内部链接，因此旧包整体验签只适用于原交付包。回放时使用各实验包自己的 manifest。


**数据准备原始交付说明**

提交内容对应数据获取与审计、特征统计、转换脚本、小规模回放数据、数据处理规则和接口适配。

| 内容 | 文件位置 |
|---|---|
| 任务报告 | `报告.md` |
| 运行和交接说明 | `运行说明.md` |
| 原始来源、身份与许可 | `sources/` |
| 全量审计与缺失清单 | `audit/` |
| 消息中间格式 | `intermediate/` |
| 两套 tokenizer 转换、统计和过滤 | `encoded/` |
| 先导与扩展回放数据 | `bundles/`、`examples/` |
| 处理、回放与打包代码 | `b_replay/`、`scripts/`、`tests/` |
| 固定运行依赖代码 | `runtime/Kairos/` |
| 数据校验及先导输入验证 | `verification/` |

原始来源共 499 个程序；合格候选 207 个、2,668 个已记录调用。先导包包含 1、4、14 个程序，分别对应 10、44、161 个调用，Qwen3-8B 输入联调通过。Qwen3-4B 提供转换数据。BFCL 当前仅完成可用性审计，缺少完整计时轨迹。

数据格式、处理规则和结论边界详见报告。所有统计可追溯到逐程序清单；回放不执行轨迹中的工具命令。`release_manifest.json` 保存提交文件的大小和 SHA256。
