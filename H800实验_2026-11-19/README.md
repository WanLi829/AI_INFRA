# 2026-11-19 H800 实机代码同步

本目录将成功实验归档中可取得的源码同步到 AI_INFRA。基础提交为 ca3e687f；来源归档 SHA-256 见 provenance/sync-manifest.json。本次不向项目总仓库或 Kairos 上游提交。

## 内容

- `../4090D工作负载交付/project/b_replay/`：实机回放、硬件检查、容量/缓存校验、H800 输入准备、调度观测和矩阵管理。31 个适配文件与实机记录逐项校验。
- 同一 project 的 `scripts/`、`tests/` 和先导配置：归档中的辅助脚本与测试。
- `archive_tools/`：原样保存的离线流式指标、图表、数据特征和归档校验工具；不改其路径语义。
- `provenance/`：原始代码身份与版本记录、归档文件到仓库路径的 SHA-256 映射。

保留冻结的 Kairos 子模块，未修改其 340 个核心发布文件。数据包、原始大日志、模型和训练 checkpoint 不在本次同步中；已有文件不删除。

## 验证

在仓库根目录执行：

```bash
python -X utf8 -B H800实验_2026-11-19/verify_sync.py
cd 4090D工作负载交付/project
python -X utf8 -B -m unittest discover -s tests -v
python -X utf8 -B -m b_replay.h800 options
```

本次已执行：57 个归档文件哈希/语法校验、31 个实机适配文件校验通过；38 项 CPU 单元测试通过（122.764 秒）。这些检查不启动 GPU。实机服务仍需 Linux、冻结模型、服务依赖和完整输入；本次同步没有重新运行 H800。

## 离线重算

原归档按如下布局提供 `project/`、`live_results/`、`success_metadata/`、`release_manifest.json`。本目录 archive_tools 的 Python 脚本必须在该完整归档根目录运行，不能直接从仓库的 archive_tools 目录重算：其 BASE/project 等路径有明确语义。原包已有相同脚本，通常无需覆盖。

```bash
# 在完整解压的 H800_success85_20261119 根目录
python build_live_reports_v2.py --expected 85 --output reports/recomputed85
```

此原始入口处理归档全部 85 轮，并保留原始五配置口径。当前中期报告采用四配置 72 轮，筛选时必须同时筛选统计分母、图表和配对比较，不能把 85 轮总量写进 72 轮报告。数据集实验的工具等待均为 0；结果是探索性成功轮次对比。

## 未入包的实机启动源码

状态日志记录了 `fast_minimal_driver_v4` 和 `kairos_tail_driver_v2` 等临时运行驱动的名字/哈希，但上传包没有这些驱动的完整源码；不能凭日志伪造还原。本次同步了包内可获得的实验源码和统计工具，不声称补齐所有服务器临时文件。额外的重训资产、临时 bootstrap 和机器配置不应被当成通用启动入口。本次不重新训练，也不上传权重。

此前 midterm README 属于上卡前指南；归档的同名 README 引用归档根路径。仓库中已改为正确链接，原字节保存在 archive_tools/midterm-README.archived.md。
