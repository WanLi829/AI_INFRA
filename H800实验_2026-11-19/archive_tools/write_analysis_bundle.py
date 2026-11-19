"""Write Chinese analysis and slide materials for the user-accepted complete runs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'project'))
from b_replay.common import write_csv

LABELS = {'lmcache': 'vLLM+LMCache', 'zero_ttl': 'Zero TTL',
          'static_ttl': 'Static TTL 2s', 'continuum': 'Dynamic TTL',
          'kairos': 'Kairos 重训版'}
CONDITIONS = [('jps', 'heldout-v2', .1), ('jps', 'heldout-v2', 2),
              ('dataset', 'swe_iso32', .5), ('dataset', 'nebius_iso32', .5),
              ('concurrency', 'instances400', 8), ('concurrency', 'instances400', 80)]
NAMES = {('jps', 'heldout-v2', .1): 'JPS=0.1', ('jps', 'heldout-v2', 2): 'JPS=2',
         ('dataset', 'swe_iso32', .5): 'SWE32，0.5 JPS，零等待',
         ('dataset', 'nebius_iso32', .5): 'Nebius32，0.5 JPS，零等待',
         ('concurrency', 'instances400', 8): 'K=8',
         ('concurrency', 'instances400', 80): 'K=80'}


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join('---' for _ in headers) + '|'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--features', type=Path, default=BASE / 'reports/data_features_20261119_v2')
    p.add_argument('--failures', type=Path)
    p.add_argument('--success-only',action='store_true')
    p.add_argument('--expected',type=int)
    args = p.parse_args()
    report = args.report.resolve()
    rows = load_csv(report / 'runs.csv')
    scope=json.loads((BASE/'current_scope.json').read_bytes())
    expected=args.expected or scope.get('target_runs',90)
    assert len(rows) == expected and all(r['status'] == 'complete' for r in rows)
    groups = defaultdict(list)
    for r in rows:
        key = (r['axis'], r['dataset'], float(r['axis_value']))
        groups[key, r['method']].append(r)
        assert r['program_completion_fraction'] == r['call_completion_fraction'] == '1.0'
    assert set(groups) == {(c, m) for c in CONDITIONS for m in LABELS}
    for (c,method),value in groups.items():
        key=f'{c[0]}|{c[1]}|{c[2]:g}|{method}'
        count=scope.get('accepted_repetition_counts',{}).get(key,3)
        assert len(value)==count
        assert {r['repetition'] for r in value}=={str(i) for i in range(1,count+1)}
        assert {r['seed'] for r in value}=={str(2026111800+i) for i in range(1,count+1)}
    repeat_note=scope.get('repeat_note_zh','五配置各3次')
    axis_n={a:sum(r['axis']==a for r in rows) for a in ('jps','dataset','concurrency')}

    def values(c, m, metric):
        return [float(r[metric]) for r in groups[c, m] if r.get(metric) not in ('', None)]

    def avg(c, m, metric):
        v = values(c, m, metric)
        return statistics.fmean(v) if v else None

    def fmt(c, m, metric, digits=2, percent=False, spread=False):
        v = values(c, m, metric)
        if not v:
            return 'NA'
        mul = 100 if percent else 1
        result = f'{statistics.fmean(v)*mul:.{digits}f}' + ('%' if percent else '')
        if spread:
            result += f' [{min(v)*mul:.{digits}f}, {max(v)*mul:.{digits}f}]'
        return result

    def change(c, m, baseline='lmcache'):
        baseline_rows = {r['seed']: r for r in groups[c, baseline]}
        return statistics.fmean(100 * (1 - float(r['jct_mean_s']) /
            float(baseline_rows[r['seed']]['jct_mean_s'])) for r in groups[c, m])

    def delta(c, m, baseline='lmcache'):
        return f'{change(c,m,baseline):+.2f}%'

    common = ('H800 80GB，Qwen3-8B BF16，16K 上下文；27 核 CPU 配额/230GiB RAM，'
        'S=8、batch tokens=2048，GPU KV=59,402,354,688 bytes（55.32GiB），CPU KV=64GiB。'
        '每轮新服务/缓存，加载与回放分开；'+repeat_note+'，种子为2026111801/02/03的已完成前缀。')
    caveat = ('本批为探索性工程结果：GPU PID 归属仅 record-only、成本 profile 未在本机独立重标定；'
        'Kairos 是冻结 MiniLM 后重训数值层/联合密度头的变体，安排在后置时间块；'
        '后段低优先级归档可能产生 CPU/I/O 扰动。三次运行不自动构成统计显著性，'
        '也不是原版 Kairos CG1/原论文完整复现。')
    definition = ('JCT 从 Program 实际激活/到达到最后完成，含排队、执行与工具等待；'
        'TTFT 从调用提交到首 token。每轮先计算 Program 等权 JCT、调用级 TTFT，'
        '再对各格已有重复的统计量取均值；方括号和图上误差线为已有重复的 min–max，不是置信区间。'
        'JCT 降低比例按相同种子配对计算后再平均，正值表示更快，负值表示更慢。'
        'Kairos只有两次的条件，配对收益仅使用基线相同的前两次；各方法均值表使用各自全部已完成重复。'
        '本次采用冻结轨迹和固定输出预算进行服务回放，工具文本作为数据读取、不实际执行；不评价Agent解题正确率。')

    def performance(conditions):
        return table(['条件', '方法', '重复n', '平均 JCT s [范围]', 'P95 JCT s',
            '平均 TTFT s', 'P95 TTFT s', '输出 token/s', 'JCT降低/LMCache'],
            [[NAMES[c], LABELS[m],len(groups[c,m]),fmt(c,m,'jct_mean_s',spread=True),
              fmt(c,m,'jct_p95_s'), fmt(c,m,'ttft_mean_s'), fmt(c,m,'ttft_p95_s'),
              fmt(c,m,'output_tokens_per_s'), delta(c,m)] for c in conditions for m in LABELS])

    def mechanism(conditions):
        return table(['条件', '方法', '引擎prefix hit/query', 'external hit/query',
                      '平均pin/预算', '最大KV使用/预算', '平均等待队列', 'P95请求首次准入等待 s',
                      '采样scheduler累计 s', '其中controller s'],
            [[NAMES[c], LABELS[m], fmt(c,m,'prefix_cache_ratio',percent=True),
              fmt(c,m,'external_prefix_cache_ratio',percent=True),
              fmt(c,m,'pin_ratio_mean',percent=True), fmt(c,m,'kv_used_ratio_max',percent=True),
              fmt(c,m,'queue_size_mean'), fmt(c,m,'first_admit_wait_p95_s'),
              fmt(c,m,'schedule_cost_sum_s'), fmt(c,m,'controller_cost_sum_s')]
             for c in conditions for m in LABELS])

    def refs(conditions):
        return '\n'.join(f"- {NAMES[c]} / {LABELS[m]}：" + '、'.join(r['run_id'] for r in groups[c,m])
                         for c in conditions for m in LABELS)

    def ttl(conditions):
        return table(['条件','Dynamic相对Static的JCT降低','Zero相对LMCache的JCT降低',
                      'Dynamic相对LMCache的JCT降低','Kairos相对Dynamic的JCT降低'],
                     [[NAMES[c],delta(c,'continuum','static_ttl'),delta(c,'zero_ttl'),
                       delta(c,'continuum'),delta(c,'kairos','continuum')] for c in conditions])

    jps = CONDITIONS[:2]
    datasets = CONDITIONS[2:4]
    concurrency = CONDITIONS[4:]
    jct_growth = table(['方法','JPS 2 / 0.1：平均JCT倍数','K80 / K8：平均JCT倍数'],
        [[LABELS[m],f"{avg(jps[1],m,'jct_mean_s')/avg(jps[0],m,'jct_mean_s'):.2f}×",
          f"{avg(concurrency[1],m,'jct_mean_s')/avg(concurrency[0],m,'jct_mean_s'):.2f}×"] for m in LABELS])

    jps_doc = f'''# JPS 负载压力实验分析

{common}

本次仅取0.1和2 JPS，共{axis_n['jps']}个完整轮次；重复数见各方法表格。首轮使用保存的泊松到达表；续轮必须等待上轮完成和冻结工具间隔。36条源Program/401次调用限制了最大在途规模，不能据两端点定位完整转折点或外推到更高压力。

{definition}

## 实测性能

{performance(jps)}

![平均JCT及重复范围](jps/figures/dataset0-jct_mean_s.png)

平均/P95 TTFT、P95 JCT及长尾见同目录对应指标图；[相对变化图](comparison-figures/jps-exploratory-jct-change.png)。

## TTL对照与相对收益

{ttl(jps)}

Dynamic相对LMCache在两个实测档分别为{delta(jps[0],'continuum')}和{delta(jps[1],'continuum')}；这只说明两个端点的差异，不定位未测区间的反转位置。Dynamic相对Static的配对结果直接列在上表；固定2s是工程对照，不代表Static经调参后的最优值。Kairos在本表的收益或代价都保留，不预设它优胜。

## 机制证据及解释边界

{mechanism(jps)}

引擎prefix hit/query保留引擎原始分母，不能当作纯GPU token命中率。CPU层缺可比计数器时为NA，external与引擎比例不相加。首次准入等待只覆盖首次准入，不能代表整个Program的全部排队。pin/KV仅为采样；scheduler累计包含controller，不能叠加。高pin本身不能证明阻碍准入；重算token、完整DMA时钟等未直接观测的项目不由代理量推断。低JPS组的整轮输出吞吐受到达表和排空影响，不等于饱和GPU吞吐。

代表时间序列：[低JPS的第一次配对](mechanism/trace-jps-heldout-v2-0.1-r1.png)、[高JPS的第一次配对](mechanism/trace-jps-heldout-v2-2-r1.png)。每图同时展示等待队列、物理pin、非空闲KV和真实K(t)；这是指定单轮采样，不是三次均值。

{caveat}

## 对应原始轮次

{refs(jps)}
'''
    (report/'JPS实验分析.md').write_text(jps_doc)

    feature_data=json.loads((args.features/'dataset_features.json').read_text())['datasets']
    features=table(['数据','独立源/实例','调用数','平均轮数','平均输入token/调用','平均输出token/调用',
                    '工具等待','重训源ID重叠'],[[f['dataset'],f"{f['independent_sources']}/{f['programs']}",
                    f['calls'],f"{f['turns_mean']:.2f}",f"{f['input_tokens_mean']:.2f}",
                    f"{f['output_tokens_mean']:.2f}",'明确固定0s',f['retrained_predictor_overlap']]
                   for f in feature_data])
    features_relative = Path(__import__('os').path.relpath(args.features.resolve(), report))
    dataset_doc=f'''# 不同数据集对比实验分析

{common}

双方均在0.5 JPS、显式零工具等待下运行；SWE32与Nebius32，五配置共{axis_n['dataset']}个完整轮次，各格重复数见表。这是两来源的零等待敏感性对照。Nebius缺可靠逐轮工具时间，不能冒充真实工具等待的跨数据集主实验，也无法验证对工具等待分布的适配收益。

## 数据特征

{features}

![输入数据特征]({features_relative.as_posix()}/dataset-features.png)

Nebius在这些冻结子集中每调用输入更长、Program轮数更多、输出更短，几个因素共同变化。完整轮数、输入/输出、上下文增长、动作及LCP特征底表见特征目录。LCP使用实际token序列，仅代表潜在复用机会，不是实测cache hit；长Program会在调用级统计中权重更高，因此也保留Program级特征。纳入/排除与不可变来源见project/inputs和upstream证据。

## 相同到达率内的性能对照

{definition}

{performance(datasets)}

![数据来源对照](comparison-figures/dataset-comparison.png)

{ttl(datasets)}

Dynamic相对LMCache的JCT降低在SWE为{delta(datasets[0],'continuum')}，Nebius为{delta(datasets[1],'continuum')}；Kairos相对Dynamic分别为{delta(datasets[0],'kairos','continuum')}和{delta(datasets[1],'kairos','continuum')}。比较相对收益时保留各自基线，不用两个数据集的绝对JCT差代替算法收益。

## 缓存、排队与局限

{mechanism(datasets)}

输入量、输出量、轮数和工作集均可能影响结果；本实验没有逐一隔离它们，不能据相关性认定单一因果。由于工具等待被固定为0，数据来源差异也不能归因于真实工具时间。CPU命中与实际重算token等缺可靠计数器的指标保留NA。训练排除了全部评测源ID，来源间是否存在语义近重复仍不是仅凭ID检查就能完全排除。

两来源第一次配对的队列/pin/KV/K(t)见mechanism/trace-dataset-*-0.5-r1.png，图中时间原点为客户端回放时钟，未将较慢方法的长尾截去。

{caveat}

## 对应原始轮次

{refs(datasets)}
'''
    (report/'数据集实验分析.md').write_text(dataset_doc)

    activity=table(['目标K','方法','目标窗平均K','等于目标时间占比','低于目标占比','高于目标占比',
                    '补入延迟P95 s','排空时间 s'],
        [[str(c[2]),LABELS[m],fmt(c,m,'active_target_mean'),fmt(c,m,'active_target_fraction',percent=True),
          fmt(c,m,'active_below_fraction',percent=True),fmt(c,m,'active_above_fraction',percent=True),
          fmt(c,m,'refill_delay_p95_s'),fmt(c,m,'drain_s')] for c in concurrency for m in LABELS])
    tail=table(['目标K','方法','P99 TTFT s','max TTFT s：跨轮均值 [范围]',
                '首轮平均TTFT s','续轮平均TTFT s','平均GPU峰值 GiB','平均RAM峰值 GiB',
                '资源窗平均GPU利用率 %','GPU>=90%时间占比'],
        [[str(c[2]),LABELS[m],fmt(c,m,'ttft_p99_s'),fmt(c,m,'ttft_max_s',spread=True),
          fmt(c,m,'first_call_ttft_mean_s'),fmt(c,m,'returning_call_ttft_mean_s'),
          fmt(c,m,'gpu_peak_gib'),fmt(c,m,'ram_peak_gib'),fmt(c,m,'gpu_util_mean_percent'),
          fmt(c,m,'gpu_busy_ge90_fraction',percent=True)] for c in concurrency for m in LABELS])
    representative=next(r for r in groups[concurrency[1],'kairos'] if r['repetition']=='1')
    representative_note=(f"指定的K80第一次Kairos轮中，平均JCT={float(representative['jct_mean_s']):.2f}s，"
        f"采样平均等待队列={float(representative['queue_size_mean']):.2f}个请求，"
        f"非空闲KV采样峰值={100*float(representative['kv_used_ratio_max']):.2f}%，"
        f"物理pin峰值={100*float(representative['pin_ratio_max']):.2f}%，"
        f"资源窗整卡GPU平均利用率={float(representative['gpu_util_mean_percent']):.2f}%。"
        f"采样scheduler累计={float(representative['schedule_cost_sum_s']):.2f}s，其中controller="
        f"{float(representative['controller_cost_sum_s']):.2f}s。"
        '观测呈现长时间排队与较高控制路径代价；采样未显示非空闲/pin页把预算耗尽，'
        '因此不能仅用“显存满了”或“pin更多”解释退化。空闲队列可含可回收的前缀缓存页，'
        '该比例也不能证明完全没有缓存驱逐；瞬时未采到的行为和各控制开销的独立贡献仍待隔离。')
    concurrency_doc=f'''# 并行 Program 规模实验分析

{common}

固定工作量为同一400实例/4456调用，源于36条独立轨迹；K=8/80，五配置共{axis_n['concurrency']}个完整轮次，各格重复数见表。K为在途Program数，包含工具等待；S=8为引擎生成槽位。Program结束后按冻结顺序补入，末尾单独排空；封闭负载不同时声称固定外部JPS。实例复制用于增加规模和隔离缓存，不把400实例当400个独立源。

## 实测性能

{definition}

{performance(concurrency)}

![K与平均JCT](concurrency/figures/dataset0-jct_mean_s.png)

{ttl(concurrency)}

Dynamic相对LMCache在K8为{delta(concurrency[0],'continuum')}，K80为{delta(concurrency[1],'continuum')}；Kairos相对Dynamic为{delta(concurrency[0],'kairos','continuum')}和{delta(concurrency[1],'kairos','continuum')}。这是两个目标规模下的实测对照，不能定位中间拐点或宣称已找到极限并发。所有方法变慢与某方法相对退化分别看绝对延迟和配对差值。

## 真实活跃数及补入

{activity}

每轮K(t)图及底表位于concurrency/figures/run*-active-programs。目标窗口与末尾排空已分开，完整回放平均K不要求等于目标。首次准入等待和引擎等待队列可证明存在排队；只有它们不能完整分解“执行/工具等待/所有中途排队”的时间占比，未观测项不猜测。

## 机制及长尾

{mechanism(concurrency)}

{tail}

{representative_note}

高负载Kairos的端到端代价保留在JCT、首轮/续轮TTFT、最大TTFT和采样控制开销中。预测器模型调用、条件编译、arrival controller的分阶段host耗时另见runs.csv，阶段可能嵌套。scheduler/controller累计并非独立项，不可相加；pin和可用KV只是采样，不能单独把退化归因于缓存保护。排队、控制成本、pin策略和日志/I/O开销尚未做隔离消融，不能用一项解释全部差距。

代表时间序列：[K8第一次配对](mechanism/trace-concurrency-instances400-8-r1.png)、[K80第一次配对](mechanism/trace-concurrency-instances400-80-r1.png)。五方法都保留完整时间跨度，等待请求数与活跃Program数分图，不能混为同一数量。

GPU预分配显存约满不等于有效吞吐饱和，GPU利用率也不是端到端效率。GPU均值为资源观测窗口内相邻采样间隔的时间加权值，包含加载和回放后工作；仅用不超过5s的有效间隔，缺失/长间隔排除，覆盖率见runs.csv。它是整卡观测且PID仅record-only，不是严格自有进程或纯回放kernel利用率。S固定8可能形成执行约束；本实验没有扫描S、batch tokens或KV预算，不能声称H800已实现峰值吞吐。P95/max长尾可提示公平性问题，但不等于已证明任一Program无限饥饿。

{caveat}

## 对应原始轮次

{refs(concurrency)}
'''
    (report/'并行Program实验分析.md').write_text(concurrency_doc)

    index=[]
    for r in rows:
        st=json.loads((Path(r['result'])/'state.json').read_text());plan=st['plan']
        index.append({**r,'variant':('kairos-retrained-frozen-minilm-v1' if r['method']=='kairos' else r['method']),
            'included_in_performance':True,'record_role':'main_complete','diagnostic':False,
            'raw_state_status':st['status'],
            'arrivals_sha256':plan['arrivals_sha256'],'runtime':plan['runtime'],
            'profile_sha256':plan.get('profile_sha256'),'reload_profile_sha256':plan.get('reload_profile_sha256'),
            'started_at':st['started_at'],'ended_at':st['ended_at'],
            'model':plan['model'],'packages':st['packages'],
            'actual_server_result':'/root/bayes-tmp/h8run/'+str(Path(r['result']).relative_to(BASE/'live_results/raw'))})
    if args.success_only:
        failures=[];events=[];interrupted=[]
        write_csv(report/'run_index.csv',index)
    else:
        assert args.failures is not None, '--failures is required unless --success-only'
        snapshot=json.loads((BASE/'live_results/verified_snapshots.json').read_bytes())
        for source in snapshot['source_run_paths']:
            if snapshot['source_run_roles'][source]=='main':continue
            root=BASE/'live_results/raw'/Path(source).relative_to('/root/bayes-tmp/h8run')
            st=json.loads((root/'state.json').read_bytes());plan=st['plan'];diagnostic=bool(plan.get('diagnostic'))
            client_path=root/'b_replay/summary.json'
            client=json.loads(client_path.read_bytes()) if client_path.exists() else {}
            requests=root/'b_replay/requests.jsonl';observed=success=tokens=None;missing_tokens=False
            if requests.is_file():
                observed=success=tokens=0
                with requests.open() as stream:
                    for line in stream:
                        call=json.loads(line);observed+=1;success+=int(bool(call.get('success')))
                        if call.get('output_tokens') is None:missing_tokens=True
                        else:tokens+=call['output_tokens']
                if missing_tokens:tokens=None
            index.append(dict(run_id='extra-'+str(root.relative_to(BASE/'live_results/raw')),
                method=st['method'],variant=('kairos-retrained-frozen-minilm-v1' if st['method']=='kairos' else st['method']),
                included_in_performance=False,diagnostic=diagnostic,status=st['status'],raw_state_status=st['status'],
                record_role=('diagnostic_' if diagnostic else 'attempt_')+st['status'],error=st.get('error'),
                result=str(root),actual_server_result=source,started_at=st.get('started_at'),ended_at=st.get('ended_at'),
                input_bundle_sha256=plan.get('bundle_manifest_sha256'),arrivals_sha256=plan.get('arrivals_sha256'),
                seed=plan.get('arrivals',{}).get('seed'),gpu_uuid=st.get('gpu',{}).get('uuid'),
                gpu_kv_bytes=plan.get('gpu_kv_bytes'),cpu_kv_gib=plan.get('cpu_kv_gib'),
                expected_calls=None if diagnostic else plan.get('calls'),
                expected_programs=None if diagnostic else plan.get('programs'),
                completed_calls=success,observed_calls=observed,output_tokens_observed=tokens,
                completed_programs=client.get('completed_programs'),
                launch_replay_cleanup_s=st.get('timing',{}).get('total_launch_replay_cleanup_s'),
                runtime=plan.get('runtime'),model=plan.get('model'),packages=st.get('packages'),
                source_state_sha256=hashlib.sha256((root/'state.json').read_bytes()).hexdigest()))
        coordination=BASE/'live_results/coordination/server-final'
        assert (coordination/'kairos03/progress.json').is_file(),'Final coordination snapshot is missing'
        observation=json.loads((coordination/'SNAPSHOT_OBSERVATION.json').read_bytes())
        interrupted=[]
        for retained in observation.get('interrupted_runs',[]):
            root=coordination/retained['path'];st=json.loads((root/'state.json').read_bytes());plan=st['plan']
            requests=root/'b_replay/requests.jsonl';count=success=tokens=None;invalid=0
            if requests.exists():
                count=success=tokens=0
                with requests.open() as stream:
                    for line in stream:
                        try:call=json.loads(line)
                        except json.JSONDecodeError:invalid+=1;continue
                        count+=1;success+=int(bool(call.get('success')));tokens+=call.get('output_tokens') or 0
            diagnostic=bool(plan.get('diagnostic'))
            record=dict(run_id='interrupted-'+retained['path'],method=st['method'],
                included_in_performance=False,diagnostic=diagnostic,status='interrupted_unfinalized',
                raw_state_status=st['status'],record_role='interrupted_unfinalized',
                result=str(root),actual_server_result='/root/bayes-tmp/h8run/'+retained['path'],
                started_at=st.get('started_at'),ended_at=st.get('ended_at'),
                expected_calls=None if diagnostic else plan.get('calls'),completed_calls=success,
                observed_calls=count,output_tokens_observed=tokens,invalid_partial_jsonl_records=invalid,
                owned_process_groups_present_at_cpu_backup=retained['owned_groups_present'],
                input_bundle_sha256=plan.get('bundle_manifest_sha256'),arrivals_sha256=plan.get('arrivals_sha256'),
                seed=plan.get('arrivals',{}).get('seed'),gpu_uuid=st.get('gpu',{}).get('uuid'),
                source_state_sha256=hashlib.sha256((root/'state.json').read_bytes()).hexdigest())
            interrupted.append(record);index.append(record)
        write_csv(report/'run_index.csv',index);write_csv(report/'interrupted_runs.csv',interrupted)
        events=[]
        for tag in ('min01','min02','min03','fast01','fast02','fast03','fast04','fast05','kairos01','kairos02','kairos03'):
            path=coordination/tag/'progress.json'
            if not path.exists():continue
            state=json.loads(path.read_bytes());current=state.get('current') or {}
            source=current.get('result') or state.get('current_output');raw_state=None
            if source:
                relative=Path(source).relative_to('/root/bayes-tmp/h8run')
                for candidate in (BASE/'live_results/raw'/relative/'state.json',coordination/relative/'state.json'):
                    if candidate.exists():raw_state=json.loads(candidate.read_bytes());break
            events.append(dict(batch=tag,coordinator_status=state['status'],
                phase=state.get('phase') or state.get('stage'),error=state.get('error'),
                completed_rows=len(state.get('completed',[])),failed_rows=len(state.get('failed',[])),
                current_server_result=source,current_preserved_run_status=raw_state.get('status') if raw_state else None,
                current_run_ended_at=raw_state.get('ended_at') if raw_state else None,
                training_gpu_started=state.get('training_gpu_started'),
                started_at=state.get('started_at'),ended_at=state.get('ended_at'),
                source_progress=str(path.relative_to(BASE)),source_progress_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                separate_performance_repeat=False))
        write_csv(report/'coordination_events.csv',events)
        event_table=table(['批次','协调器结束/保留状态','登记完整/失败行','当前保留run状态','原始错误'],
            [[r['batch'],r['coordinator_status'],f"{r['completed_rows']}/{r['failed_rows']}",
              r['current_preserved_run_status'] or '未创建/无对应run',
              str(r['error'] or '').replace('|','/').replace('\n',' ')] for r in events])
        (report/'协调器事件登记.md').write_text('# 协调器、预启动拒绝与中止记录\n\n'+event_table+
            '\n\n此表是历史控制器快照，不是额外GPU重复。原控制器失败可由新控制器复用同一完整GPU工作，不能把表中failed_rows相加当实际失败轮数。'
            'fast01在GPU忙卡预检时被拒绝，未创建模型轮次；kairos01在使用GPU前被替换。min03的Static诊断被中止且run state未最终刷新，'
            f'原replaying状态、协调器中断错误及原始日志一并保留，未改写为成功。主{expected}轮只来自已下载的completed清单；'
            'run_index含主轮、失败尝试和已结束诊断，included_in_performance字段区分，state.before*恢复记录另表保留，GPU执行不重复计数。\n')

        failures=load_csv(args.failures/'failure_register.csv')
    total_calls=sum(int(r['completed_calls']) for r in rows)
    total_programs=sum(int(r['completed_programs']) for r in rows)
    total_tokens=sum(int(r['output_tokens']) for r in rows)
    total_wall=sum(float(r['launch_replay_cleanup_s']) for r in rows)
    overview=table(['条件','LMCache JCT s','Zero JCT s','Static JCT s','Dynamic JCT s','Kairos重训 JCT s'],
        [[NAMES[c]]+[fmt(c,m,'jct_mean_s') for m in LABELS] for c in CONDITIONS])
    ttft_overview=table(['条件','LMCache','Zero TTL','Static 2s','Dynamic TTL','Kairos重训'],
        [[NAMES[c]]+[fmt(c,m,'ttft_mean_s',3)+' / '+fmt(c,m,'ttft_p95_s',3) for m in LABELS] for c in CONDITIONS])
    overview_doc=f'''# H800 五配置最小覆盖实验结果

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-11-19
- Verification Status: ANALYZED (raw-source hashes and offline metrics checked; GPU rerun not performed)
- Version Label: h800_final{expected}_retrained_v1

## 交付范围与完成状态

原计划90轮；用户停止GPU支出，按已完成数据收尾。最终完整备份并使用{expected}轮，五配置、六条件均有覆盖；{repeat_note}。JPS、数据来源、K三个轴分别{axis_n['jps']}、{axis_n['dataset']}、{axis_n['concurrency']}轮。合计{total_programs:,}次Program实例完成、{total_calls:,}次调用、{total_tokens:,}输出token。完整轮的加载/回放/清理累计{total_wall/3600:.2f}小时；此数不含历史失败、专项诊断、训练及轮间停顿，不等于从开卡到现在的总账单时间。本报告仅统计完整成功轮次的性能，不用于推断全部尝试的失败率或运行稳定性。

{common}

{overview}

数字为各格已有完整重复的平均，逐次范围、重复n与P95/TTFT/吞吐见各轴分析。所有结果按真实完成工作量计算，未使用部分中断轮的幸存者延迟宣称加速。

## TTFT：平均值 / P95（秒）

{ttft_overview}

先逐轮计算调用级TTFT统计量，再跨已有重复取平均；重复数同上。配对相对变化只使用相同种子。

{jct_growth}

## 读取顺序

1. [JPS分析](JPS实验分析.md)：0.1/2 JPS、泊松新Program到达与TTL对照。
2. [不同数据集分析](数据集实验分析.md)：SWE/Nebius各32源、0.5 JPS、显式零等待。
3. [并行Program分析](并行Program实验分析.md)：K8/K80、400实例/36源、S=8与长尾。
4. [PPT素材及讲稿](PPT素材.md)、[变更与局限](实验变更与局限.md)。
5. runs.csv和run_index.csv仅收录{expected}轮完整成功实验。summary.csv为跨轮统计，exploratory-pairs.csv为相同种子的配对描述差异。各图PNG/SVG/底表同时保留。

{definition}

## Kairos训练与版本

冻结nreimers/MiniLM-L6-H384-uncased，revision 3276f0fac9d818781d7a1327b3ff818fc4e643c0；157训练任务/2106状态、14验证任务/161状态，排除全部评测源ID，归一化仅拟合训练集。20 epochs约17.9秒、验证选epoch4。相对训练集拟合的常量混合先验，验证joint NLL 8.9717→8.3564、输出MAE 81.98→57.67 token。9调用真实服务smoke通过。上述是验证集与常量先验的对照，不证明等同或超过原版CG1；原版四个二进制仍缺失。本次权重、训练与manifest已备份。

## 完整性和结论范围

所有{expected}轮经原始输入、完成数量、输出预算、依赖/工具等待和并发事件复核；配对输入/到达SHA、软件与资源签名一致。本交付包按用户要求只收成功实验记录。无卡重连仅执行CPU文件备份与校验，没有重新训练、启动推理或补跑。离线复算通过并不代表重新做了GPU复现。

{caveat}

原任务书建议60/60/90合计210轮，先缩为90轮最小覆盖，最终因GPU预算停止于{expected}个完整轮次。未收齐原90轮的三次重复全矩阵，不据此估计稳定性。没有补0.5/1 JPS、中间K或跨数据集第二JPS。Nebius真实工具时间缺失，因此数据来源对照仅零等待敏感性组。没有验证新模型、多GPU、真实Agent解题成功率或H800饱和极限。
'''
    (report/'00_总报告.md').write_text(overview_doc)
    (report/'实验变更与局限.md').write_text(f'''# 实验变更与局限

1. 硬件由4090D/先前计划的10核30GB，按实际实例改为H800 80GB、27核配额/230GiB RAM；最终资源签名逐轮登记，旧CPU验证和旧卡结果不混入。
2. 先从210轮缩为90轮：JPS0.1/2、来源SWE32/Nebius32在0.5JPS、K8/80；最终用户停止GPU支出，使用{expected}个完整轮次，{repeat_note}。停止点由预算/断开时刻决定，没有按方法性能挑选重复。
3. 第二来源没有可靠逐轮工具时间，双方固定零等待，列为敏感性对照，不冒充真实工具时间主实验。
4. 用户先取消、后重新启用Kairos；四配置72轮先完成，Kairos为后置18轮时间块，不声称完全随机交错五方法。
5. 原版CG1四个权重文件缺失；用户授权用现有轨迹重训。版本为kairos-retrained-frozen-minilm-v1，仅训练数值层和密度头，排除评测源ID。不能声称原版复现或原版性能等价。
6. 原K80 Kairos客户端600秒首token等待超时，失败现场保留。新控制器将客户端收集等待改7200秒，整轮上限仍7200秒；权重/模型/策略/工作量/资源不变，受影响K80从头重跑。不同采集deadline是协议偏差，必须连同原失败呈现。
7. 旧周期nvidia-smi查询超时中断过一轮Zero TTL K80，原部分轮保留并重跑；改为直接NVML可缺失采样，保留RAM/OOM和自有进程保护。
8. 合法实例运输层标签曾被旧离线校验器拒绝；严格匹配Program实例后恢复原完整GPU工作，state.before*保留，未多计重复。重训版通过外部sitecustomize兼容，不改冻结runtime源。
9. 数据盘100GiB，Kairos单轮scheduler日志曾约9.77GB；只对结束且服务退出的日志无损gzip，核对解压SHA/长度和inode/mtime/size后替代，保留archive-proof。后段helper CPU26/nice15归档与备份可能产生CPU/I/O扰动，尚未隔离；首轮K80 Kairos在归档启动前已结束。
10. GPU PID归属为record-only、成本profile为归档版本；未补完整五配置独立标定/全部两层隔离专项。存在局部诊断与中止记录，不说全部正式验收通过。
11. 主表{expected}轮完整不代表不存在失败；失败/诊断/校验恢复/未刷新的中断分表。最后K80第三重复的原始replaying状态与CPU备份时进程组不存在的观测均保留。400实例来自36源；运行是统计重复单位，调用数不能当独立重复数。Kairos部分条件只有两次重复，配对收益只用相同种子基线。
12. scheduler和controller时间嵌套，预测器阶段也可能嵌套；CPU命中、实际重算token和完整DMA耗时缺可靠计数器时保留NA。物理pin/KV为采样；整卡预分配显存和单时刻GPU利用率都不能证明有效吞吐饱和。

当前证据可描述冻结硬件/工作量/参数下的性能与机制相关性，不能证明单一因果、原论文复现、统计显著性或其他模型/资源配置下的普遍规律。
''')

    slides = ['# PPT素材与约一分钟讲稿', '', '以下六页内容对应三项任务；最终PPT排版由汇总者完成。全部图表有PNG/SVG和CSV底表。', '']
    slide_specs = [
        ('1｜新Program到达率的端到端代价', 'jps/figures/dataset0-jct_mean_s.png',
         [f"Dynamic对LMCache：0.1 JPS {delta(jps[0],'continuum')}，2 JPS {delta(jps[1],'continuum')}。",
          '横轴是新Program每秒到达数，续轮保持实际依赖和工具等待。',
          '只测两个端点，不能定位完整拐点；重复范围不是置信区间。'],
         '这一页把新Program的到达压力从0.1提高到2 JPS，五种配置使用相同输入和配对到达表。请先读平均JCT，再结合P95和TTFT判断尾部。误差线是各格已有重复的最小最大范围，重复数在表格和脚注注明。低压力下整组吞吐可能由到达过程决定，因此延迟变化不必表现为同幅度吞吐变化。'),
        ('2｜Static与Dynamic TTL的机制对照', 'mechanism/trace-jps-heldout-v2-2-r1.png',
         [f"Dynamic相对Static：低JPS {delta(jps[0],'continuum','static_ttl')}，高JPS {delta(jps[1],'continuum','static_ttl')}。",
          'Static固定2s；Zero不额外pin轮间KV，普通前缀缓存仍可存在。',
          'pin与等待/可用KV共同解读；没有隔离消融，不给单一因果结论。'],
         '这一页只比较TTL保留策略在两个负载下的差异。Static是固定两秒的工程对照，Dynamic使用冻结在线策略，Zero不额外保留轮间pin。pin更高不必然更快，也不能单独证明阻碍新请求准入。需要结合等待队列、首次准入等待和可用KV看相关性；采样控制时间存在嵌套，不能相加。'),
        ('3｜两来源输入分布不同，工具等待均置零', (features_relative/'dataset-features.png').as_posix(),
         ['SWE：32源、352调用；Nebius：32源、385调用。',
          'Nebius平均输入更长、轮数更多、输出更短，多个因素共同变化。',
          'LCP是潜在复用机会；工具等待固定为0，不能检验真实工具时间适配。'],
         '这两套输入都使用统一Qwen tokenizer和相同筛选口径。SWE与Nebius各有32条源轨迹，调用数并不相同。Nebius输入上下文更长、轮数更多，但输出较短。这里缺少Nebius可靠工具时间，所以双方明确固定零等待。这些特征差异能帮助解释工作量，不能把一个特征和性能差距直接当作因果。'),
        ('4｜相同0.5 JPS下的数据来源对照', 'comparison-figures/dataset-comparison.png',
         [f"Dynamic对LMCache：SWE {delta(datasets[0],'continuum')}，Nebius {delta(datasets[1],'continuum')}。",
          f"Kairos对Dynamic：SWE {delta(datasets[0],'kairos','continuum')}，Nebius {delta(datasets[1],'kairos','continuum')}。",
          '每个来源内比较方法，再比较来源间的相对收益。'],
         '每个来源在同一0.5 JPS下比较五种配置；其他四配置各三次，Kairos各两次，图中同时给出平均和P95的JCT、TTFT。应先比较同一数据内各方法，再看相对基线的变化。输入量、输出量和轮数都不同，来源间绝对延迟差不能直接解释为算法收益。Kairos使用本次重训版，并保留表现较慢的结果。'),
        ('5｜K是活跃Program数，S=8保持固定', 'concurrency/figures/dataset0-jct_mean_s.png',
         ['K8/K80均运行同一400实例/4456调用，来自36条源轨迹。',
          f"Dynamic对LMCache：K8 {delta(concurrency[0],'continuum')}，K80 {delta(concurrency[1],'continuum')}。",
          '工具等待仍计活跃，结束后补入；排空区间独立统计。'],
         '横轴K表示同时在途的Program，包含等待工具的时间；引擎槽位S始终是8。每个规模用同样400个隔离实例，避免工作量不同造成误读。这400个实例来自36条源轨迹，不是400条独立数据。固定活跃数通过完成补入维持，末尾不再补入时进入排空，因此目标窗口和全轮平均活跃数需要分开看。'),
        ('6｜Kairos重训版的长尾与控制成本', 'concurrency/figures/dataset0-ttft_max_s.png',
         [f"Kairos对Dynamic：K8 {delta(concurrency[0],'kairos','continuum')}，K80 {delta(concurrency[1],'kairos','continuum')}。",
          '成功Kairos轮的客户端采集等待上限为7200秒。',
          '同时看首轮/续轮TTFT、最大TTFT、队列和控制成本，不声称重训版必胜。'],
         '最后一页展示成功完成的Kairos重训版实验中的延迟与控制成本。长尾要结合最大TTFT和首轮、续轮差异看。预测器耗时、调度控制、缓存策略和日志都可能参与，但没有隔离实验，不能把全部差距归因于其中一项，也不能把本次重训版结果泛化为原版Kairos。'),
    ]
    for title,fig,claims,speech in slide_specs:
        slides += [f'## {title}', '', f'![主图]({fig})', '',
                   *['- '+c for c in claims], '', '**配置脚注：** '+common+' '+caveat,
                   '', '**讲稿：** '+speech, '']
    (report/'PPT素材.md').write_text('\n'.join(slides))
    if args.success_only:
        (report/'实验变更与局限.md').write_text(f'''# 范围与解释边界

- 用户停止GPU支出，本包仅保留{expected}个完整成功轮次；五配置和六条件都有覆盖。{repeat_note}。原90轮全矩阵未收齐。
- JPS是新Program的泊松到达率；K是含工具等待的活跃Program数。K8/K80使用同一400实例，来自36条独立源轨迹，S=8固定。
- SWE32/Nebius32对照均在0.5 JPS、显式零工具等待下运行，不代替真实工具时间的跨数据集主实验。
- Kairos为冻结MiniLM后的独立重训变体，排除了评测源ID；原版CG1权重未取得，不声称原论文完整复现。Kairos在后置时间块执行。
- 使用归档成本profile，未独立重标定；GPU PID归属为record-only。低优先级日志归档可能带来CPU/I/O扰动，结果按探索性工程对比解读。
- 图上误差为已有重复的min/max范围，不是置信区间；Kairos只有两次的格子，配对收益仅使用相同种子的两次基线。
- 缓存层分母、物理pin与KV页、GPU资源窗利用率均保留原定义；缺观测写NA。调度器与controller耗时存在嵌套，不能相加；机制相关性不证明单一因果。
- 所有性能汇总仅针对本包的成功轮次，不用于推断全部尝试的失败率、系统稳定性、真实Agent解题正确率或H800最优饱和吞吐。
''')
    print(json.dumps(dict(status='complete', runs=len(rows), complete_calls=total_calls,
        complete_programs=total_programs, output_tokens=total_tokens, indexed_attempts=len(index),
        coordination_records=len(events),output=str(report)), ensure_ascii=False))


if __name__ == '__main__':
    main()
