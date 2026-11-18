# H800 中期实验：统一入口与本地准备

本目录对应三位成员的 JPS、不同数据集、并行 Program 实验。实验仓库是 **WanLi829/AI_INFRA**；Kairos 作为只读上游子模块固定到 `a3dbac12eac80d6444222ca3e2e9592fef029158`。本轮只修改实验仓库，不向项目总仓库或 Kairos 上游推送。

## 1. 当前能做什么、还差什么

**本地已提供的能力：** 六个运行入口（五种主配置加原生辅助）、新版 Kairos 接入、Zero/Static TTL、开放 JPS 与固定活跃数回放、私有输出停止、数据校验与独立实例扩充、矩阵生成、逐行执行、失败保留、日志采集、离线统计和绘图。

**不等于 H800 已验收。** 当前没有启动 H800；模拟回放不能证明 GPU 引擎兼容、LMCache 实际命中、预测器 GPU 运行或方法性能。正式上卡前仍需要：

| 条件 | 当前状态与处理 |
|---|---|
| 四个 CG1 权重文件 | 上游不提交 Git，本仓库也没有；按第 4 节获取并校验，缺失时 Kairos 启动明确失败 |
| 当前预测器训练/验证身份 | 旧的森林划分不能替代新联合预测器划分；未取得时标探索性 |
| H800 成本标定 | 上游携带 RTX4090 标定；先导可明确使用旧标定，正式需独立 H800 标定及来源记录 |
| 共同 GPU KV 容量 | 自动分配仅作先导；MiniLM 的 GPU 占用可能改变剩余容量，正式需冻结共同显式容量 |
| 第二套数据 | mini-SWE 有可重建包；Nebius 实际测试包及可靠时间证据尚未交付，不拿同源子集冒充 |
| 复制实例的缓存隔离 | 代码生成唯一实例与 cache_salt；必须在 GPU 和 LMCache 两层验证生效，目前复制输入限定探索性 |
| 五配置 H800 smoke | 待执行；新原生入口有共同的私有停止适配和只读观测，不声称完全未修改的 vLLM |

`plan` 和 `doctor` 会展示缺项。示例配置中的 `allow_exploratory: true` 是为先导准备，结果会被标为探索性，报告不输出正式加速比。它不能绕过缺失模型权重、数据损坏、输出数量错误或策略激活失败。

## 2. 新检出与 CPU 验证

需要 Python 3.12+。以下为终端命令，路径中的中文请保留引号。只安装回放依赖，不升级现有 GPU 的 torch/vLLM 环境。

```bash
git -c core.autocrlf=false clone --recurse-submodules https://github.com/WanLi829/AI_INFRA.git
cd AI_INFRA
git -c core.autocrlf=false submodule update --init --recursive
cd "4090D工作负载交付/project"
python -m pip install -r requirements-replay-client.txt
python -X utf8 -B -m unittest discover -s tests -v
```

如果使用发布分支或标签，请先在根目录切换到负责人提供的版本，再更新子模块。不要对 Kairos 子模块执行 `git pull`：那会改变已验证版本。

可另在 `runtime/Kairos` 下运行 `python -m kairos.verify` 核验 340 个发布文件；上游全套科学计算测试另需 NumPy/SciPy，参考其 README。生成图表安装 `requirements-analysis.txt`。本地 CPU 检查不需要 torch、CUDA、vLLM 或四个权重文件。

## 3. 一次准备好现有数据

在 `project` 目录执行：

```bash
python -X utf8 -B -m b_replay.datasets bootstrap --output runs/prepared
python -X utf8 -B -m b_replay.doctor --bundle runs/prepared/heldout-v2 --output runs/readiness.json
```

这会从仓库已存的数据重建：

- `runs/prepared/pilot_1-v2`：1 Program、10 调用，小样本入口检查。
- `runs/prepared/pilot_14-v2`：14 Program、161 调用，开发先导。
- `runs/prepared/heldout-v2`：36 Program、401 调用，历史留出候选；与新联合预测器的独立性仍待核验。

工具逐轮比较源消息和工具间隔，生成当前 assistant 完成后的动作证据 `completed_actions.json`，保存来源审计、哈希和特征 CSV/JSON。末轮无后续等待不补成虚假真实时间。生成目录不能预先存在；已有结果不会覆盖。数据较大，生成结果位于被 Git 忽略的 `runs/`；同事从同一冻结仓库执行同命令即可重建，不需要将数百份重复正文上传 Git。

已获得新预测器排除身份时，用真实清单（不要用空清单占位）重新审计：

```bash
python -m b_replay.datasets audit-split --bundle runs/prepared/heldout-v2 --training-ids /data/current-predictor-exclusions.json --output runs/prepared/heldout-audited
```

清单为 JSON，必须含 `runtime_commit`（上面的完整提交）、`training_ids`、`validation_ids` 两个源任务 ID 列表，可附 `source_identity_rule`。存在交集会拒绝输出；同源多轨迹要在提供清单前统一身份。审计只核对提供的清单，真实性由训练产物提供方确认。

用于较大 K 的探索性实例扩充：

```bash
python -m b_replay.datasets expand --bundle runs/prepared/heldout-v2 --instances 400 --seed 2026111801 --output runs/prepared/instances400
```

会保留 36 个独立源身份，产生 400 个实例，使用唯一 `cache_salt`，保持每个实例内的多轮复用；不修改 token、输出工作量或工具等待。**不能把它写成 400 条独立真实轨迹，也不能在未上卡校验前保证 LMCache 已按 salt 隔离。** 正式独立大规模测试优先扩充新轨迹。

## 4. 新版 Kairos 权重与环境

从实现组获取上游 `runtime/Kairos/assets/joint/README.md` 指定的四个原始文件：

```text
/data/kairos-joint/
  encoder/                      # 从子模块 assets/joint/encoder 复制
  checkpoint/
    SELECTION_LOCK.json         # 从子模块复制
    expanded.pt                 # 外部
    distribution.pt             # 外部
  study/
    SELECTION_LOCK.json         # 从子模块复制
    coupled.pt                  # 外部
  density.npz                   # 外部
```

准确 SHA-256 和原产物名称见 `runtime/Kairos/provenance/joint_release.json`，不可用随机初始化或旧森林冒充。可以先复制整份 `assets/joint` 小文件目录，再补四个二进制。校验：

```bash
python -m b_replay.doctor --joint-assets /data/kairos-joint --bundle runs/prepared/pilot_1-v2 --require-ready
```

该命令只做文件级检查；`require-ready` 是所选文件检查通过，不证明 H800 全部准备完毕。GPU 环境版本见 `runtime/Kairos/requirements-server.txt`。启动器核对 vLLM 0.28.0、模型权重、GPU 空闲与进程归属、内存限制、CPU 亲和性及端口；不要在已有 GPU 环境中随意安装整个训练依赖集。

`--engine-cpus` 和 `--helper-cpus` 必须同时设置或都用 auto。auto 只选择前 2+2 个可用 CPU，方便验通；正式建议根据现场配额冻结更充分的 CPU 资源，并对所有方法一致，不能将它误认为自动复现历史 12+2 配置。

## 5. 五配置的共同回放边界

| 参数 method | 含义 |
|---|---|
| lmcache | vLLM 异步 FCFS + 标准 LMCache connector + 共同私有停止/观测适配 |
| zero_ttl | Continuum 的程序调度保留，固定 TTL=0；禁止出现额外 pin |
| static_ttl | Continuum 程序调度 + `--ttl-seconds 2`；启动后检查固定策略及 pin 阈值 |
| continuum | 上游冻结的动态 TTL 兼容实现 |
| kairos | 新版 MiniLM 连续联合预测与几何 J 完整方案 |
| native | 辅助基线，无 CPU KV connector；仍有共同私有停止/观测适配 |

所有方法公开 `max_tokens` 都是剩余上下文容量，不是该轮真实输出标签。离线目标输出量存入 `private-execution.json`，仅停止适配器使用；程序是否终止和当前工具动作在本轮完成后才经私有 completion 记录进入方法。请求不发送 `cont_final`、`cont_oracle_gap`、未来轮数或 known-output 标签。原生基线的子类仅提供私有停止与只读指标，保留原生 FCFS；所有方法采用同一异步执行选项。

输出 token 数、消息、程序依赖、工具间隔必须一致。以记录轨迹为下一轮输入，不把本次随机生成内容追加到轨迹；不实际执行工具命令，也不评估解题成功率。

## 6. 单格计划与 H800 小样本（当前不在本地执行 GPU）

先产生到达表：

```bash
python -m b_replay arrivals --bundle runs/prepared/pilot_1-v2 --jps 1 --seed 2026111801 --output runs/p1-arrivals.json
python -m b_replay.comparison plan --method zero_ttl --bundle runs/prepared/pilot_1-v2 --arrivals runs/p1-arrivals.json --model /data/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --output /data/exp/z01 --allow-exploratory
```

`plan` 不启动 GPU，模型路径可暂为现场预定路径。H800 上使用真实环境和路径，将 `plan` 改为 `run`。目录应短且全新，Unix socket 路径不能超过限制。Static 追加 `--ttl-seconds 2`，Zero 不传该参数，Kairos 追加 `--joint-assets /data/kairos-joint`。

先导通过后确定共同 `--gpu-kv-bytes`；不要直接沿用上游 4090 的约 5.9GB 人工预算冒充 H800 默认容量。CPU KV 默认各缓存方法均为 8 GiB，可用 `--cpu-kv-gib` 统一修改。H800 profile 与 reload profile 通过对应参数传入，记录标定方法、设备和文件哈希。缺标定时只能明确解释为旧成本模型的探索性评测。

正式入口去掉 `--allow-exploratory` 后，会拒绝准备条件不完整的计划。即便成功，单次结果仍不是重复实验结论。

## 7. 三位成员各自生成矩阵

示例配置均在本目录：`jps.example.json`、`datasets.example.json`、`concurrency.example.json`。复制为自己的配置，填真实路径、时限、CPU 集和预算；不要修改共享代码。数据包路径相对于配置文件目录，模型/权重/profile 建议用服务器绝对路径。

配置中的 `jps`、`active_programs`、`seeds` 冻结后，各方法共享相同输入/种子；方法顺序按重复号轮换。默认五种主配置，原生辅助可显式加入 `methods`。

```bash
python -m b_replay.suite plan --config midterm/jps.example.json --output runs/jps-matrix
python -m b_replay.suite plan --config midterm/concurrency.example.json --output runs/k-matrix
```

数据集成员先取得第二来源的完整包，再使用 `datasets.example.json`。它包含一个明确未交付的 Nebius 路径，当前直接执行会失败；不得将其改成 mini-SWE 的另一个子集来伪造跨数据集实验。第二套数据接口见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。

每个矩阵生成 `matrix.json`、共享 workload 文件和每轮计划。**请在目标主机上生成要实际执行的矩阵**：路径和 CPU 配额与主机相关；Windows 生成的矩阵用于 CPU 检查，不直接复制到 Linux 执行。

运行时显式指定一个索引，不自动启动数十轮昂贵实验：

```bash
python -m b_replay.suite run --matrix runs/jps-matrix/matrix.json --index 0
```

`matrix.json` 列出每行的 method、档位、种子、重复号、输出位置。先执行五方法 pilot，再按排期选择下一格；一张 GPU 串行运行。执行会重新核对代码、数据、工作负载和成本文件哈希，变更后必须生成新矩阵。已存在结果目录不覆盖，失败尝试也保存；补跑新建矩阵/输出组并在报告登记关联，不能删失败历史。

默认 4 JPS×5×3=60 轮，2 数据×2 JPS×5×3=60 轮，6 K×5×3=90 轮。还要计入加载、工具等待和排空，不要一次性启动全矩阵。整轮超时必须覆盖到达跨度；0.1 JPS 的 400 实例光平均到达跨度约 3990 秒，示例 3600 秒不适用于这种组合，需要在冻结前调整。

## 8. 固定活跃数的具体语义

```bash
python -m b_replay closed-workload --bundle runs/prepared/instances400 --active-programs 50 --seed 2026111801 --output runs/k50.json
```

该文件通过同一个 `--arrivals` 参数交给 comparison，参数名沿用但内容协议明确为 closed。前 K 个程序激活，完成一个补一个，工具等待期间仍占活跃名额；列表耗尽后排空。引擎最大序列数固定为 8，不能把它当 K 修改。

不同方法的补入时间随完成时间变化，因此不要求实际到达表一致；要求源实例列表、顺序、K 和工作量一致。失败按“继续收集并标整轮失败”规则处理；全局超时保存已尝试与未尝试数量。结果 `b_replay/concurrency.json` 以 `[arrival,end)` 精确积分，提供 K(t)、目标区间、达到比例与补入延迟，目标区间不自动等于稳态。

## 9. 结果、指标与离线汇总

每次成功或开始后的失败保留 `launch.json`、`state.json`、资源采样、服务日志、客户端 `requests.jsonl/jobs.json/summary.json/concurrency.json`，研究方法保留 scheduler/transfer/replica 日志。启动前失败由 suite 的 `attempts/` 保存。

共同观测层每约 0.25 秒写 `experiment_metrics.jsonl` 的运行/等待数、空闲/总 KV 页、去重 pin 页和 pin 程序数，另写请求到达与首次准入事件。它不改变排序或容量决策，但有统一采样开销，正式报告应注明。原生方法无额外 pin，观察为零不表示没有普通缓存。调度时钟为单调时钟；客户端 `state.json` 保存其相对时间起点。

引擎与独立 LMCache helper 分别保存前后 metrics 快照。无法取得 CPU 命中或某项传输指标时输出缺失，不推测为零。`metrics.py` 保留原指标名、标签、分母，多个层次的命中不能相加。抢占、页驱逐、重算 token 分开解释。

```bash
python -m pip install -r requirements-analysis.txt
python -m b_replay.report --matrix runs/jps-matrix/matrix.json --output reports/jps-r1
```

没有 matplotlib 可先加 `--no-plots`。报告包含逐轮 CSV、汇总 CSV/JSON、机制证据、PNG/SVG。`not_run`、`failed`、`invalid`、完成但探索性均可区分。正式相对比要求同输入、同资源、同种子的 vLLM+LMCache 配对；Kairos 对 Dynamic 另列。汇总跨重复但配对身份保留，各点误差条是重复最小/最大值，不是置信区间。若缺共同基线或仅探索性，对应正式比例保持空值。

JCT 从实际到达/激活到程序结束，TTFT 从调用提交到首 token；主 JCT 按 Program 等权。成功轮原始延迟图仅描述完成样本，失败表必须同时报告，不能据此忽视失败宣称优势。N 不同的数据集不能直接用 makespan 排名。旧 `comparison_results` 为低级单包核验，三人交付统一使用本节 suite report。

## 10. 最终交付和旧入口

三份任务书见仓库根目录 `中期检查_三人实验任务/`。分别交原始归档、配置/输入身份、逐轮结果、图表底表、原因分析和 1–2 页讲稿素材；最终 PPT 由总负责人制作。所有数字应能在不上 GPU 的情况下从日志复算。

旧文档和 4090/H800 历史结果保留用于追溯，不代表本轮新版实测。`server-plan/server-run` 已停止使用，避免旧 warmup/回放路径错接新版私有停止；当前统一用 `b_replay.comparison` 和 `b_replay.suite`。旧协议结果不能混入 v2 报告。

共同问题由总负责人修复，成员不要各自改底层后混合结果。费用、登录信息和最晚关机时间在每批上卡前约定，不将凭据上传仓库。
