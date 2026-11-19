# H800 实验入口

当前代码已同步 2026-11-19 成功实验归档中的实机适配版本。请先看 [本轮同步与复现说明](../../../H800实验_2026-11-19/README.md)。早期 CPU 验收记录保留在 VALIDATION.md，不代表本次实机状态。

从本 project 目录执行：

```bash
python -m b_replay.h800 options
python -m b_replay.h800 plan --help
python -m b_replay.suite --help
python -m b_replay.h800_inputs --help
```

options/plan 只说明档位或生成计划，不会启动 GPU。先导与正式性能轮次不可混用。CPU KV 默认 64 GiB；正式运行需冻结共同 GPU KV 容量。三个 example.json 是先导模板，包含多于本次实际完成的条件；不是成功轮次的逐条重放清单。

本次实际条件：JPS=0.1/2；SWE32/Nebius32 在 0.5 JPS 下统一零工具等待；400 实例的 K=8/80，S=8。数据和外部资产需另行放置到实际路径。当前中期报告只分析 lmcache、zero_ttl、static_ttl、continuum 的 72 轮；代码保留已有其他方法接口。
