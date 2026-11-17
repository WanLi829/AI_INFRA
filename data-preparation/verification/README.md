# 验证材料

- audit/bundle_validation.json：全包结构与独立重编码结果。
- gpu_verification.json：先导输入验证汇总。
- pilot_1、pilot_4、pilot_14：必要运行状态、完整输出ID、时序和资源样本。
- gpu_environment.json：GPU环境依赖。
- runtime_integrity.json：冻结代码与原始输入校验。
- environment.json、determinism.json、source_integrity.json：CPU转换环境、确定性和来源身份。

记录按数据包名称组织，仅保留任务验收所需字段。数据包导出标识与GPU输入验证分别保存；本材料不提供正式性能对比结论。
