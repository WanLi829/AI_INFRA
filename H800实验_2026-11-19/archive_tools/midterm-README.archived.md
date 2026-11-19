# H800 本地集成入口

当前集成代码请先读 [根 README](../../README.md)、[执行顺序](../../GPU_RUNBOOK.md) 和 [实验规格](../../EXPERIMENT_SPEC.md)。原上游说明保存在 `../../upstream/AI_INFRA-midterm-README.md`，其历史命令/路径不直接作为本包启动命令。

本包固定单 H800 80GB、27 核 230GB；Linux auto 分为 engine 23/helper 4 核。五主方法为 lmcache、zero_ttl、static_ttl、continuum（Dynamic）、kairos。示例均为先导配置，GPU KV 自动分配只作先导，正式需冻结共同自然容量与 H800 成本。

`python -m b_replay.h800 options` 查看轮数档位；`plan` 只生成矩阵，`suite run` 显式选择一轮，`h800 estimate` 在本次 H800 实测后按时限选完整条件。

真实跨来源第二包仍缺工具计时。已有 Nebius32 只属于显式零等待服务隔离对照；原交付合同见 DATA_CONTRACT.md。缺联合权重不能绕过。
