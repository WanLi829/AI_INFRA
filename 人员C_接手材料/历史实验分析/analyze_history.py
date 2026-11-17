"""Summarize frozen historical runs; does not execute archived code or run inference."""
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'source'
IMPL_COMMIT = '4f4c600367b02b82b389689b9a7ece2b2af4c4b9'
RUNS = ['continuum','kairos','continuum_r2','kairos_r2']


def p95(values):
    ordered=sorted(values); pos=(len(values)-1)*.95; low=int(pos); high=min(low+1,len(values)-1)
    return ordered[low]+(ordered[high]-ordered[low])*(pos-low)


def main():
    trace_path=ROOT.parent/'实验准备/负载分析/source/trace.json'
    raw=trace_path.read_bytes(); trace=json.loads(raw)
    expected={p['id']:p for p in trace['programs']}
    calls=sum(len(p['turns']) for p in expected.values())
    outputs=sum(t['output_tokens'] for p in expected.values() for t in p['turns'])
    prompts=sum(t['prompt_tokens'] for p in expected.values() for t in p['turns'])
    rows={}; audits=[]
    def check(run, label, passed):
        audits.append(dict(run=run,check=label,passed=bool(passed)))
    def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6)
    for name in RUNS:
        s=json.loads((SOURCE/(name+'_summary.json')).read_bytes())
        state=json.loads((SOURCE/(name+'_state.json')).read_bytes())
        jobs=json.loads((SOURCE/(name+'_jobs.json')).read_bytes())
        mapping={j['program']:j for j in jobs}
        check(name,'state complete',state['status']=='complete')
        check(name,'reported complete calls / failures / jobs',(s['completed'],s['failed'],s['jobs'])==(calls,0,len(expected)))
        check(name,'state embedded summary equals standalone summary',state.get('summary')==s)
        check(name,'unique complete program ID set',len(jobs)==len(mapping) and set(mapping)==set(expected))
        check(name,'per-program turn counts',all(j['turns']==len(expected[j['program']]['turns']) for j in jobs))
        check(name,'JCT mean recomputation',close(statistics.mean(j['jct_s'] for j in jobs),s['jct_mean_s']))
        check(name,'JCT p95 linear interpolation',close(p95([j['jct_s'] for j in jobs]),s['jct_p95_s']))
        check(name,'job end minus arrival equals JCT',all(close(j['end_s']-j['arrival_s'],j['jct_s']) for j in jobs))
        check(name,'post-initial JCT identity',all(close(j['end_s']-j['first_turn_end_s'],j['post_initial_jct_s']) for j in jobs))
        check(name,'tool waiting sums',all(close(j['tool_dependency_s'],sum(t['gap_after_s'] for t in expected[j['program']]['turns'][:-1])) for j in jobs))
        check(name,'trace SHA256',state['trace_sha256']==hashlib.sha256(raw).hexdigest())
        check(name,'request throughput arithmetic',close(s['request_throughput']*s['wall_s'],calls))
        check(name,'output throughput arithmetic against planned budget',close(s['output_token_throughput']*s['wall_s'],outputs))
        check(name,'total throughput arithmetic against submitted input plus planned output',close(s['total_token_throughput']*s['wall_s'],prompts+outputs))
        check(name,'replay contained within lifecycle',state['started_at']<=s['start_wall_time']<s['end_wall_time']<=state['ended_at'])
        check(name,'method labels agree',s['method']==state['method']==name.split('_')[0])
        check(name,'model verification record',state['verified_model']['weights_verified'] is True and state['verified_model']['revision']==state['expected_model_revision'])
        rows[name]=dict(summary=s,state=state,jobs=mapping,
            start_beijing=datetime.fromtimestamp(s['start_wall_time'],timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
            end_beijing=datetime.fromtimestamp(s['end_wall_time'],timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
            lifecycle_s=state['ended_at']-state['started_at'])
    invariant_keys=['gpu','packages','profile_sha256','reload_sha256','trace_sha256','expected_model_revision','verified_model']
    invariants={k:all(rows[n]['state'][k]==rows[RUNS[0]]['state'][k] for n in RUNS) for k in invariant_keys}
    metrics=[('jct_mean_s','程序 JCT 均值/s','lower'),('jct_p95_s','程序 JCT P95/s','lower'),
        ('ttft_mean_s','请求 TTFT 均值/s','lower'),('ttft_p95_s','请求 TTFT P95/s','lower'),
        ('e2e_mean_s','请求 E2E 均值/s','lower'),('e2e_p95_s','请求 E2E P95/s','lower'),
        ('wall_s','回放耗时/s','lower'),('output_token_throughput','输出吞吐（tokens/s）','higher')]
    deltas={}
    for suffix in ('','_r2'):
        a=rows['continuum'+suffix]['summary'];b=rows['kairos'+suffix]['summary']
        deltas[suffix or 'r1']={k:100*(b[k]/a[k]-1) for k,_,_ in metrics}
    job_changes=[]
    for pid in expected:
        vals={name:rows[name]['jobs'][pid]['jct_s'] for name in RUNS}
        vals['r1_change_pct']=100*(vals['kairos']/vals['continuum']-1)
        vals['r2_change_pct']=100*(vals['kairos_r2']/vals['continuum_r2']-1)
        job_changes.append(dict(program=pid,**vals))
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(SOURCE.iterdir()) if p.is_file()}
    result=dict(analysis_date='2026-11-02',implementation_commit=IMPL_COMMIT,trace_sha256=hashlib.sha256(raw).hexdigest(),
        source_manifest=manifest,audits=audits,recorded_config_equal=invariants,runs=rows,relative_changes_pct=deltas,program_changes=job_changes,
        absent_evidence=['requests.jsonl','resources.json','FOREIGN_LOG','launch.json','scheduler.jsonl','transfers.jsonl'],
        interference_counts_from_SUMMARY_only=dict(continuum=76,continuum_r2=73,kairos=69,kairos_r2=69))
    (ROOT/'历史结果明细.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family':next((f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in fonts),'DejaVu Sans'),
                        'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(key,title,_) in zip(axs.flat,[metrics[0],metrics[3],metrics[6],metrics[7]]):
        for i,method in enumerate(['continuum','kairos']):
            vals=[rows[method+s]['summary'][key] for s in ('','_r2')]
            ax.scatter([i-.08,i+.08],vals,s=75,c=['#2d6ca2','#c76e28'],marker='o',zorder=3)
            for x,v,rep in zip([i-.08,i+.08],vals,['r1','r2']):
                ax.annotate(f'{rep}  {v:.2f}',(x,v),xytext=(-18 if rep=='r1' else 18,10 if rep=='r1' else -20),textcoords='offset points',ha='right' if rep=='r1' else 'left',fontsize=9)
        ax.set_xticks([0,1],['Continuum','Kairos']);ax.set_xlim(-.6,1.6);ax.set_title(title)
        vv=[rows[n]['summary'][key] for n in RUNS];span=max(vv)-min(vv)
        ax.set_ylim(min(vv)-max(span*.5,1),max(vv)+max(span*.5,1));ax.grid(axis='y',alpha=.2)
    fig.suptitle('历史共享 GPU 运行 · 每方法仅 2 次 · 展示单次观测，非显著性结论',fontsize=15)
    fig.savefig(ROOT/'历史指标对照.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,8),layout='constrained')
    ordered=sorted(job_changes,key=lambda j:j['r1_change_pct'])
    yy=list(range(len(ordered)))
    ax.barh([i-.18 for i in yy],[j['r1_change_pct'] for j in ordered],height=.35,label='r1',color='#2d6ca2')
    ax.barh([i+.18 for i in yy],[j['r2_change_pct'] for j in ordered],height=.35,label='r2',color='#c76e28')
    ax.set_yticks(yy,[j['program'] for j in ordered],fontsize=9);ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('100 × (Kairos JCT / Continuum JCT − 1) %；负数表示 Kairos 更短')
    ax.set_title('逐程序 JCT 的描述性变化 · 按 r1 排序，两轮均完整展示');ax.legend();ax.grid(axis='x',alpha=.15)
    fig.savefig(ROOT/'逐程序变化.png',dpi=160,bbox_inches='tight');plt.close(fig)
    main_table='\n'.join('| '+label+' | '+' | '.join(f"{rows[n]['summary'][key]:.3f}" for n in RUNS)+f" | {deltas['r1'][key]:+.2f}% | {deltas['_r2'][key]:+.2f}% |" for key,label,_ in metrics)
    timing_table='\n'.join(f"| {n} | {r['start_beijing']} | {r['end_beijing']} | {r['lifecycle_s']:.2f} |" for n,r in rows.items())
    prog_table='\n'.join(f"| {j['program']} | {j['continuum']:.2f} | {j['kairos']:.2f} | {j['r1_change_pct']:+.1f}% | {j['continuum_r2']:.2f} | {j['kairos_r2']:.2f} | {j['r2_change_pct']:+.1f}% |" for j in ordered)
    comparison_table='\n'.join(f'| {k} | {"一致" if v else "不一致"} |' for k,v in invariants.items())
    range_table='\n'.join(f"| {method} | {label} | {min(rows[method+s]['summary'][key] for s in ('','_r2')):.3f}～{max(rows[method+s]['summary'][key] for s in ('','_r2')):.3f} |" for method in ['continuum','kairos'] for key,label in [('jct_mean_s','单次运行的平均 JCT/s'),('ttft_p95_s','单次运行的 P95 TTFT/s')])
    failures=[a for a in audits if not a['passed']]
    wins1=sum(j['r1_change_pct']<0 for j in job_changes);wins2=sum(j['r2_change_pct']<0 for j in job_changes)
    flips=sum((j['r1_change_pct']<0)!=(j['r2_change_pct']<0) for j in job_changes)
    report=f'''# 历史实验结果核对报告

整理人：万理（人员 C）｜日期：2026-11-02｜本次仅离线读取历史文件，没有运行新 GPU 实验。

## 1. 核心结论

现有材料支持“实现小组的两种方法均有完整回放的历史记录”，并呈现 Kairos 在平均程序完成时间和整段回放吞吐上的改善方向。但两种方法各只有 2 次运行，且实现小组明确记录四次都遇到邻居 GPU 进程；不能据此宣布独占条件下的稳定收益、显著提升或最终最优工作负载。

- 两轮同编号对照中，Kairos 平均程序 JCT 分别变化 {deltas['r1']['jct_mean_s']:+.2f}%、{deltas['_r2']['jct_mean_s']:+.2f}%（负数为时间缩短）。
- 输出吞吐分别变化 {deltas['r1']['output_token_throughput']:+.2f}%、{deltas['_r2']['output_token_throughput']:+.2f}%。
- **代价同样存在：** P95 TTFT 分别变化 {deltas['r1']['ttft_p95_s']:+.2f}%、{deltas['_r2']['ttft_p95_s']:+.2f}%；两轮都更慢。平均指标改善不代表尾部延迟全面改善。
- 本次复算的 {len(audits)} 项算术/记录一致性检查，通过 {len(audits)-len(failures)} 项，失败 {len(failures)} 项。范围是已保存字段，不是完整逐请求或 GPU 资源审计。

## 2. 分析对象和证据层级

来源为实现仓库 `Kadsoo/AI-infra-project` 的本地固定提交 `{IMPL_COMMIT}`，目录 `tasks/continuum/results_cloud/kairos_3way/`。未重新拉取远端，也没有登录服务器找缺失日志。

本地 [source](source) 保存了该目录的 12 个 JSON 和依赖快照，以及 SUMMARY、REPRO、wrapper 原文。文件指纹在 [历史结果明细.json](历史结果明细.json) 中。

| 材料 | 本次是否取得 | 能支持什么 |
|---|---|---|
| 每轮 summary / state / jobs | 是，共 4 轮各 3 个文件 | 运行状态、汇总数字、逐程序 JCT 及部分配置的一致性 |
| 环境 freeze、REPRO | 是 | 历史依赖与启动方法的交接记录，不是本机实时环境 |
| requests.jsonl | 否 | 无法独立复算 TTFT/E2E/ITL，无法逐请求核实无遗漏与实际输出预算 |
| resources.json / FOREIGN_LOG | 否 | 无法独立复算 GPU 峰值或干扰次数、持续时间及时间窗交叠 |
| launch、调度/传输原始日志 | 否 | 无法审计实际命令、CPU 亲和、KV 参数、缓存命中及 TTL 分布 |

目录名虽然是 `kairos_3way`，实际归档只有 continuum 和 kairos 两种方法。SUMMARY 已说明 fixed/zero 的环境变量消融没有生效，不能将这些运行当作三个有效实验组。

此前 Kairos 上游自身的原始归档，以及本仓库更早的 Qwen2.5-3B V0/V1/OFF、CPU 仿真结果，属于不同实验批次和配置，本报告不把它们混为本次重复样本。

## 3. 四次运行的原始指标

r1 指文件名没有 `_r2` 的那次运行。表中百分比为 `100 × (Kairos / Continuum − 1)`，仅按同编号作描述性对照，未证实随机化或严格成对实验设计。

| 指标 | Continuum r1 | Kairos r1 | Continuum r2 | Kairos r2 | r1 相对变化 | r2 相对变化 |
|---|---:|---:|---:|---:|---:|---:|
{main_table}

四次 summary 都记录完成 178 请求、失败 0、完成 16 程序，state 均为 complete。这里只能称“归档记录如此”，不能用 summary 替代缺失的逐请求验证。

![历史指标对照](历史指标对照.png)

程序 JCT 包含程序内工具等待、排队和计算，不含到达之前等待；wall 是回放时间，不含加载、编译和预热。输出吞吐使用整个回放 wall 作为分母，不能当成纯 decode 服务速度。ITL 的上游实现只使用相邻单 token 流块，本报告不把该采样当作完整逐 token 延迟。

没有合并两次运行的 P95，也没有给 n=2 制造可靠置信区间。下面仅展示观测范围；它不是估计出的总体波动范围。

| 方法 | 指标 | 两次观测范围 |
|---|---|---|
{range_table}

## 4. 逐程序是否一致受益

同编号对照中，r1 有 {wins1}/16 个程序 JCT 下降，r2 有 {wins2}/16 个下降；有 {flips} 个程序的改善/退化方向在两轮之间变化。这提示要保留逐程序结果，不能只报告整体均值。

![逐程序变化](逐程序变化.png)

| 程序 | C r1/s | K r1/s | r1 变化 | C r2/s | K r2/s | r2 变化 |
|---|---:|---:|---:|---:|---:|---:|
{prog_table}

此表按程序 ID 对齐，避免把 jobs 文件中的完成顺序误当作相同任务的顺序。16 个程序共享同一运行环境，不应把它们当作 16 次独立硬件重复试验。

## 5. 可复算检查与配置核对

本次核对：state/summary 方法及嵌入摘要一致、16 个程序 ID 无遗漏和重复、每程序轮数与归档 trace 一致；从 jobs 复算 JCT 均值与线性插值 P95；核对时间差、工具等待总和、trace 文件哈希及吞吐算术关系。

吞吐乘 wall 分别与 178 请求、26,164 计划输出 tokens、941,797 输入加计划输出 tokens 一致。这是汇总算术自洽，仍不能代替逐请求核实实际 token 数。

| state 中记录的字段 | 四轮是否一致 |
|---|---|
{comparison_table}

state 中记载相同 GPU UUID、24,564 MiB 总显存、相同模型 revision `b968826d9c46dd6066d109eabc6255188de91218`、相同 trace/profile/reload 哈希与依赖组合。相同配置指纹是可比性的必要证据之一，不足以证明执行时完全公平。

GPU used/util 的 0 是启动时快照，**不能证明整个回放期间无邻居进程，也不是显存峰值**。`weights_verified=true` 是历史校验记录，本次没有读取服务器上的权重重新验证。state 没有提供源码 commit；Kairos `7971fa4` 的版本对应关系来自 REPRO 交接说明。

| 运行 | 回放开始（北京时间） | 回放结束（北京时间） | state 生命周期/s |
|---|---|---|---:|
{timing_table}

时间由文件 Unix 时间戳转换为 UTC+08:00；不使用本机默认时区。生命周期从 state.started_at 到 ended_at，与纯回放 wall 分开记录。

## 6. 干扰证据及缺口

SUMMARY §7 声称回放窗内邻居进程 sightings 分别为 continuum r1 76、r2 73、kairos r1 69、r2 69。**这些是文字报告值，本地没有对应 foreign 原始日志，无法复算。** Sightings 是采样发现次数，不是唯一进程数、GPU 占用率或占用时长。

SUMMARY 写“原始 JSON 与 foreign 日志在目录中”，但本次固定提交中实际只找到前述文件；不能据此假装已核实干扰日志。邻居干扰也不能被精确量化或从总时间中扣除。

REPRO 的 wrapper 在内存中修改进程归属检查，将 foreign 集合并入 own，再返回空 foreign 集合。因此不能凭 wrapper 运行中 resources 的 foreign 字段为空来证明无干扰。每次都需要独立 FOREIGN_LOG 与准确回放窗；本次连 resources 文件也未归档。

从四次观测可描述改善方向，但无法证明差异显著，也无法证明差异完全由干扰造成。两者都需要更完整的证据。

## 7. 供后续复跑使用的问题清单

| 优先级 | 问题 | 现在能做什么 / 后续如何处理 |
|---|---|---|
| 高 | 四次运行有共享 GPU 干扰报告，原始证据未提供 | 当前结果标为探索性；正式比较预约独占时段并保留进程监控 |
| 高 | 缺逐请求日志 | 未来保存 requests.jsonl，并核对唯一请求集合、error 和实际输出预算 |
| 高 | 缺 launch / 资源 / 传输日志 | 每轮保存整个输出目录，运行前记录资源与 CPU 亲和；失败目录也保留 |
| 高 | 当前实例 30GB 标注与原版 40GiB 可用内存要求冲突 | 保留为上机前核查项，离线报告不能解决资源可用性 |
| 中 | 每方法只有 2 次，无随机运行顺序证据 | 新实验至少各 3 次独立运行，预先安排交替/平衡顺序，按实际变异再决定是否增加 |
| 中 | 尾部延迟退化被平均值掩盖 | 同时报告平均 JCT、P95 TTFT/E2E、吞吐和逐程序退化情况 |
| 中 | fixed/zero 消融无效 | 正式实现支持之前不报告三方消融；不沿用无效环境变量设置 |
| 中 | SUMMARY 混有旧 3B/Python3.10 环境段落 | 本轮以 state 依赖与 REPRO 的 Qwen3-8B/Python3.12 记录为准，避免混配 |

这些待补齐项不要求现在开机或联系他人；本报告先固定已有证据和缺口，供下一次上机使用。

## 8. 本次交付与复算

- [历史结果明细.json](历史结果明细.json)：全部数值、逐程序变化、检查结果及输入哈希。
- [analyze_history.py](analyze_history.py)：在本地运行 `python analyze_history.py` 重新生成报告和图表；需要 matplotlib。读取已保存 source 与上一步归档 trace，不运行其中的 wrapper。
- [SUMMARY 原文](source/SUMMARY.md)、[REPRO 原文](source/REPRO.md)、[wrapper 原文](source/run_relaxed.py)。
- [实现仓库固定版本目录](https://github.com/Kadsoo/AI-infra-project/tree/{IMPL_COMMIT}/tasks/continuum/results_cloud/kairos_3way)。

本次完成的是历史资料审查，尚未完成独占复跑、最终工作负载选择或新模型适配。
'''
    (ROOT/'历史实验结果核对报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(checks=len(audits),failed=failures,configuration_equal=invariants,
        deltas=deltas,program_jct_improvements=[wins1,wins2],direction_changes=flips),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
