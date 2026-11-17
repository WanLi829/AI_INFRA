"""Offline descriptive analysis; reads archived trace only, never executes messages."""
import argparse
import hashlib
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def describe(values):
    v = sorted(values)
    return dict(n=len(v), min=v[0], median=st.median(v), mean=st.mean(v),
                p95=v[math.ceil(.95*len(v))-1], max=v[-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, default=Path(__file__).parent/'source'/'trace.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    out=args.output; out.mkdir(parents=True, exist_ok=True)
    raw=args.trace.read_bytes(); trace=json.loads(raw); ps=trace['programs']
    rows=[]; programs=[]; truncations=0; decreases=0; dropped=Counter()
    for p in ps:
        previous=[]; local=[]
        for i,t in enumerate(p['turns']):
            k=t.get('keep_prefix_tokens',len(previous))
            assert isinstance(k,int) and 0<=k<=len(previous), (p['id'],i,'prefix')
            prompt=previous[:k]+t['added_token_ids']
            assert len(prompt)==t['prompt_tokens'], (p['id'],i,'length')
            if i:
                truncations+=k<len(previous); decreases+=len(prompt)<len(previous)
                dropped[len(previous)-k]+=1
            r=dict(program=p['id'],turn=i+1,prompt=len(prompt),keep=k,added=len(t['added_token_ids']),
                   output=t['output_tokens'],gap=t['gap_after_s'],source_gap=t['source_gap_s'],
                   tool=t['tool'],nonfinal=i<len(p['turns'])-1,
                   prefix_fraction=k/len(prompt),growth=len(prompt)-len(previous))
            rows.append(r);local.append(r);previous=prompt
        programs.append(dict(id=p['id'],turns=len(local),arrival=p['arrival_s'],
            initial=local[0]['prompt'],final=local[-1]['prompt'],max_prompt=max(r['prompt'] for r in local),
            growth=local[-1]['prompt']-local[0]['prompt'],
            avg_growth=(local[-1]['prompt']-local[0]['prompt'])/(len(local)-1),
            max_step=max(r['growth'] for r in local[1:]),
            output=sum(r['output'] for r in local),total_gap=sum(r['gap'] for r in local[:-1]),
            prefix_fraction=sum(r['keep'] for r in local[1:])/sum(r['prompt'] for r in local[1:])))
    follow=[r for r in rows if r['turn']>1]; gaps=[r for r in rows if r['nonfinal']]
    first=[r for r in gaps if r['turn']==1]; later=[r for r in gaps if r['turn']>1]
    high=[r for r in rows if r['prompt']>=8192]
    long_counts=Counter(r['program'] for r in high)
    groups={name:[r for r in gaps if test(r['gap'])] for name,test in [
        ('<0.1s',lambda x:x<.1),('[0.1,1)s',lambda x:.1<=x<1),
        ('[1,5)s',lambda x:1<=x<5),('>=5s',lambda x:x>=5)]}
    tools=Counter(r['tool'] for r in gaps)
    changed=[r for r in gaps if abs(r['gap']-r['source_gap'])>1e-9]
    weighted=sum(r['keep'] for r in follow)/sum(r['prompt'] for r in follow)
    stats=dict(trace_sha256=hashlib.sha256(raw).hexdigest(),program_count=len(ps),requests=len(rows),
        prompt=describe([r['prompt'] for r in rows]),output=describe([r['output'] for r in rows]),
        gap=describe([r['gap'] for r in gaps]),first_gap=describe([r['gap'] for r in first]),
        later_gap=describe([r['gap'] for r in later]),prefix=describe([r['prefix_fraction'] for r in follow]),
        weighted_prefix=weighted,high_context_requests=len(high),high_context_programs=len(long_counts),
        high_context_counts=dict(long_counts),gap_bins={k:len(v) for k,v in groups.items()},
        tool_counts=dict(tools),prefix_truncations=truncations,context_decreases=decreases,dropped_suffix_tokens=dict(dropped),
        changed_gap_count=len(changed),changed_first_gap_count=sum(r['turn']==1 for r in changed),
        final_context_total=sum(p['final'] for p in programs),
        max_prompt_plus_output=max(r['prompt']+r['output'] for r in rows),
        programs=programs,requests_detail=rows)
    (out/'analysis.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in fonts),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    blue='#2763a5'; orange='#cf7430'; green='#29836b'
    def save(fig,name):
        fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axs[0,0].hist([r['prompt'] for r in rows],bins=[0,2048,4096,6144,8192,10240,12288],color=blue,rwidth=.9)
    axs[0,0].axvline(8192,color=orange,ls='--',label='8,192（描述性分界）');axs[0,0].legend(fontsize=8)
    axs[0,0].set(title='输入长度：178 次请求',xlabel='输入 tokens',ylabel='请求数')
    edges=[0,64,128,256,512,768]
    axs[0,1].bar([f'{a}–{b-1}' for a,b in zip(edges,edges[1:])],
                 [sum(a<=r['output']<b for r in rows) for a,b in zip(edges,edges[1:])],color=green)
    axs[0,1].set(title='计划输出预算',xlabel='计划输出 tokens',ylabel='请求数')
    counts=[len(v) for v in groups.values()]
    axs[1,0].bar(list(groups),counts,color=orange)
    for i,n in enumerate(counts):axs[1,0].text(i,n+1,str(n),ha='center')
    axs[1,0].set(title='工具等待：162 个非末轮间隔',xlabel='秒（左闭右开）',ylabel='间隔数');axs[1,0].set_ylim(0,max(counts)*1.16)
    edges=[0,50,70,80,90,95,100]
    axs[1,1].bar([f'{a}–{b}' for a,b in zip(edges,edges[1:])],
        [sum(a<=r['prefix_fraction']*100<b for r in follow) for a,b in zip(edges,edges[1:])],color=blue)
    axs[1,1].set(title='非首轮前缀保留比例：162 次请求',xlabel='保留前缀 / 本轮输入（%，左闭右开）',ylabel='请求数')
    fig.suptitle('Kairos 7971fa4 · 归档负载结构（离线统计，非性能测量）',fontsize=16)
    save(fig,'01_负载分布.png')
    fig,axs=plt.subplots(4,4,figsize=(15,11),sharex=True,sharey=True,layout='constrained')
    for ax,p in zip(axs.flat,programs):
        rr=[r for r in rows if r['program']==p['id']]
        ax.plot([r['turn'] for r in rr],[r['prompt'] for r in rr],color=blue,marker='.',lw=1.7)
        ax.axhline(8192,color=orange,ls=':',lw=.8)
        ax.set_title(p['id'].replace('__',' /\n'),fontsize=9)
        ax.set_xticks([1,5,10,15]);ax.set_ylim(0,12500);ax.grid(alpha=.15)
    fig.supxlabel('程序内轮次（不是实际时间）');fig.supylabel('输入 tokens')
    fig.suptitle('全部 16 个程序的上下文增长 · 相同坐标尺度',fontsize=16)
    save(fig,'02_全部程序上下文增长.png')
    fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    sortedps=sorted(programs,key=lambda p:p['arrival'])
    axs[0].scatter([p['arrival'] for p in sortedps],range(len(sortedps)),color=blue)
    axs[0].set_yticks(range(len(sortedps)),[p['id'] for p in sortedps],fontsize=8)
    axs[0].set(xlabel='相对回放开始的秒数',title='首次到达：两批人为构造的突发')
    axs[0].grid(axis='x',alpha=.2)
    for sample,label,c in [(first,'首个等待（16 个）',orange),(later,'后续等待（146 个）',blue)]:
        xx=sorted(r['gap'] for r in sample)
        axs[1].step(xx,[(i+1)/len(xx) for i in range(len(xx))],where='post',label=label,color=c)
    axs[1].set(xlabel='等待时间（秒）',ylabel='累计比例',title='区分被改写的首轮与后续等待',ylim=(0,1.04))
    axs[1].legend();axs[1].grid(alpha=.15)
    save(fig,'03_到达与等待.png')
    longest=max(programs,key=lambda p:p['max_prompt']); fastest=max(programs,key=lambda p:p['avg_growth'])
    jump=max(programs,key=lambda p:p['max_step']); outputmax=max(programs,key=lambda p:p['output'])
    pct=lambda n,d:f'{n/d:.1%}'
    table='\n'.join(f"| {p['id']} | {p['turns']} | {p['initial']:,} → {p['final']:,} | {p['avg_growth']:.0f} | {p['max_step']:,} | {p['output']:,} | {p['total_gap']:.2f} | {p['prefix_fraction']:.1%} |" for p in sorted(programs,key=lambda p:p['max_prompt'],reverse=True))
    tooltable='\n'.join(f'| {k} | {n} | {n/len(gaps):.1%} |' for k,n in tools.most_common())
    disttable='\n'.join(f"| {label} | {s['n']} | {s['min']:.3f} | {s['median']:.3f} | {s['p95']:.3f} | {s['max']:.3f} |" for label,s in [('所有工具等待（秒）',stats['gap']),('首个工具等待（秒）',stats['first_gap']),('后续工具等待（秒）',stats['later_gap'])])
    text=f'''# 现有负载分析报告

分析人：万理（人员 C）｜日期：2026-11-02｜范围：本地静态数据分析，未开机、未运行模型。

## 1. 结论

这份负载适合作为首次接手复跑和缓存竞争机制观察的起点：16 个完整多轮程序、178 次请求，存在较长上下文、较高的前缀保留，以及两批集中到达的程序。但它是经过选择与到达/首个等待改写的 SWE teacher 压力负载，不能独自代表生产流量，也不能据此宣布已确定最适合 24GB 的最终负载。

- **长输入集中在少数程序的后期请求。** 输入至少 8,192 tokens 的请求有 {len(high)}/178（{pct(len(high),178)}），涉及 {len(long_counts)}/16 个程序，其余大多数请求低于此长度。8,192 是本报告的描述性分界，不是硬件容量线。
- **续轮存在明显的复用机会。** 非首轮显式保留前缀的 token 加权比例为 {weighted:.2%}；这不是实测缓存命中率。
- **工具等待以短间隔为主。** 小于 1 秒的等待为 {sum(r['gap']<1 for r in gaps)}/162（{pct(sum(r['gap']<1 for r in gaps),162)}），至少 5 秒的仅 {sum(r['gap']>=5 for r in gaps)} 个；未覆盖分钟级等待。
- **首次到达高度集中。** {sum(p['arrival']<1 for p in programs)} 个程序在 1 秒内到达，其余 {sum(p['arrival']>=1 for p in programs)} 个在约 2.7～2.9 秒到达。这是构造的突发，不能从到达时间推算真实并发或返回时刻。

## 2. 数据版本、来源与重要纠正

分析对象为 Kairos 提交 `7971fa40323f16e8e4ce8f12dfd5db9755a98cf1` 的 `data/workload/trace.json`，SHA256：

`{stats['trace_sha256']}`

本地已保留 [原始 trace 副本](source/trace.json)、[EXPERIMENTS.md](source/EXPERIMENTS.md)、[THIRD_PARTY.md](source/THIRD_PARTY.md) 和 [历史 tokenizer 核验记录](source/tokenizer-verification.json)，便于复查。

据固定版本 THIRD_PARTY，来源为 `tarsur385/qwen3-30b-a3b-instruct-2507-swebench-verified-mini-swe-agent`，revision `58389eb5426db5a3ae6159faa37498072b4735ff`。本次没有重新下载原始数据或验证其全量覆盖范围。

EXPERIMENTS 将实际负载命名为 `dense_return_burst_production_v1`：保留 SWE teacher 多轮内容，改写首次到达和首个工具等待，构造首轮/续轮竞争，目标首个工具返回窗口靠近 4.8 秒。实际返回仍取决于模型服务时间。

**纠正此前准备文档的表述：** 回放器 summary 中通用的 `synthetic token identity` 标签不足以证明当前归档的 token 全为合成。具体归档文档说明使用原 teacher 输入，且历史记录显示 178 次输入在 8B tokenizer 上核对一致。本报告以具体负载数据、构造说明和历史核验记录为依据；本次未重做 tokenizer 验证。

文件头的 Qwen3-4B 路径、Poisson、0.48 program/s、旧资源容量及部分统计是生成阶段遗留信息。以下数字全部从 `programs` 重算，不采用它们。`source_gap_s` 也是归档中的来源字段，不能在未回溯原始轨迹时保证它就是未经处理的真实等待。

## 3. 统计口径

- 逐程序从空输入开始，以 `previous[:keep_prefix_tokens] + added_token_ids` 重建每轮输入；核对声明长度。
- 输出长度是固定生成预算，不是本次实际模型输出；回放不将本次生成的回答自动拼入后续输入。
- 等待仅统计非末轮的 162 个间隔；末轮 gap 不参与实际等待。等待总和不能当作整体墙钟耗时，因为程序之间可能重叠。
- 输入/输出按请求统计，前缀比例排除 16 个首轮；加权前缀比例为保留 token 总和 / 非首轮输入 token 总和。
- P95 为 nearest-rank，中位数为普通样本中位数。此口径与仓库性能报告的插值 P95 不同，不混用。

## 4. 长度、输出与等待分布

![负载分布](01_负载分布.png)

| 项目 | 最小 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|---:|
| 输入 tokens | {stats['prompt']['min']:,} | {stats['prompt']['median']:,.1f} | {stats['prompt']['p95']:,} | {stats['prompt']['max']:,} |
| 计划输出 tokens | {stats['output']['min']:,} | {stats['output']['median']:,.1f} | {stats['output']['p95']:,} | {stats['output']['max']:,} |

输入少于 4,096 tokens 有 {sum(r['prompt']<4096 for r in rows)} 次；4,096～8,191 有 {sum(4096<=r['prompt']<8192 for r in rows)} 次；至少 8,192 有 {len(high)} 次，其中至少 10,240 有 {sum(r['prompt']>=10240 for r in rows)} 次。

总计划输出 {sum(r['output'] for r in rows):,} tokens；输出至少 256 tokens 的请求有 {sum(r['output']>=256 for r in rows)} 次。最长上下文程序为 `{longest['id']}`，最多计划输出的程序为 `{outputmax['id']}`（累计 {outputmax['output']:,} tokens）。两者不必是同一个程序，输入缓存压力和解码工作量应分开讨论。

单请求最大“输入 + 输出预算”为 {stats['max_prompt_plus_output']:,} tokens，低于该固定发布记录的 16,384 上下文上限；这只说明长度合同可容纳，不证明同时并发时显存充足。

| 统计对象 | 样本数 | 最小 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
{disttable}

等待区间计数：{'; '.join(f'{k}: {len(v)}' for k,v in groups.items())}。短等待有利于观察快速返回时缓存是否仍被保留；长等待更能暴露驻留成本与恢复取舍。这里的短等待集中性意味着不能据此概括长工具任务的表现。

## 5. 哪些程序增长快、潜在压力大

![全部程序上下文增长](02_全部程序上下文增长.png)

下表按最大输入排序；平均增长为“末轮输入减首轮输入 / 续轮次数”，不是 tokens/秒。横轴是轮次，不是实际执行时间。

| 程序 | 轮数 | 首轮 → 末轮输入 | 平均每续轮增长 | 最大单轮增长 | 总输出预算 | 工具等待总和/s | 续轮加权前缀比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

- 最大上下文：`{longest['id']}`，{longest['max_prompt']:,} tokens。
- 平均增长最快：`{fastest['id']}`，每个续轮约 {fastest['avg_growth']:.0f} tokens。
- 最大单轮跳增：`{jump['id']}`，新增 {jump['max_step']:,} tokens。
- 观察到显式前缀截短 {truncations} 次，但全部只去掉上轮输入末尾 4 tokens，再追加本轮内容；输入长度下降 {decreases} 次。这是很小的尾部变动，不能称为大量历史截断。未解码这些 token，本报告不推断它们的语义。

输入越长通常需要容纳更多 KV，但真实驻留量还取决于生成、缓存淘汰、页布局、活跃请求及两层缓存策略。以上只是优先观察对象，不是 OOM 风险排序或实测峰值。

16 个末轮输入总和为 {stats['final_context_total']:,} tokens。它不是同一时刻的驻留集合，不可直接与 24GB 比较。不同程序的 cache_salt 共 {len(set(p.get('cache_salt',p['id']) for p in ps))} 个，不能假设程序间共享前缀一定会带来跨程序命中。

## 6. 到达结构、首轮改写与工具类型

![到达与等待](03_到达与等待.png)

所有程序 predecessor 均为空：本负载没有跨程序串行依赖；但每个程序内部必须等上一轮完成并等待工具间隔才能提交下一轮。即使首次到达都在约 3 秒内，也不能当成持续到达的生产流量，更不能把累计到达数当成实际服务并发。

`gap_after_s` 与归档 `source_gap_s` 数值不同的非末轮间隔有 {len(changed)} 个。两字段当前完全相同并不能证明等待未经改写：frontier 与实验说明明确记录了首个等待的改写。当前归档无法仅靠这两个字段恢复改写前的首个等待；还需原始来源数据。

非末轮工具标签（程序如何标记，报告就如何统计；不把标签当作独立执行过的工具）：

| 工具标签 | 间隔数 | 比例 |
|---|---:|---:|
{tooltable}

数据来自软件工程任务，不能因标签多样就认定覆盖检索、数据库、网页或其他 Agent 场景。轨迹中的命令只作为数据读取，本次没有执行。

## 7. 对负载选型的建议与覆盖缺口

| 判断 | 数据依据 | 后续如何验证 |
|---|---|---|
| 保留为接手基线 | 已有固定版本、完整多轮结构和历史运行记录 | 在目标实例复跑并检查完整性 |
| 可用于缓存竞争观察 | 高前缀保留、集中到达、续轮上下文增长 | 采集实际缓存命中、传输、驻留和延迟 |
| 不足以代表自然流量 | 双批到达与首个等待被构造，非盲测 | 后续补充未改写到达/工具等待的可追溯样本 |
| 长等待覆盖不足 | 最长等待不到 7 秒 | 后续补充真实长工具等待；人工缩放另列为合成对照 |
| 前缀变化覆盖需扩展 | 所有续轮仅去掉上轮尾部 4 tokens，输入持续增长 | 后续补充低复用、大量历史截断或重写场景 |
| 模型及场景结论有限 | 单一固定 8B 回放与 SWE 场景，输出预算预先给定 | 新模型须适配；未知输出长度和自主 Agent 另设评测 |

现阶段可提出的推荐是：**先以这份 8B 固定归档作为复跑基线，再选择有明确差异的新负载做补充。** 不能仅凭静态数据选择最终并发、CPU KV 容量或宣称 4B/8B 哪个更优。

仍需上机回答：实际显存峰值、容器内存是否满足要求、稳定负载范围、缓存命中和传输量、延迟与吞吐，以及 Kairos/Continuum 的差异。

## 8. 交付与复算

- [完整统计与逐请求明细](analysis.json)：含逐程序、逐轮数值，便于后续筛选。
- [本地复算脚本](analyze_workload.py)：需要 Python 与 matplotlib；默认读本目录 source/trace.json，输出到本目录。不连接服务器。
- 在此目录运行：`python analyze_workload.py`。脚本按本次已冻结数据结构分析，不替代后续通用 Replay 验收工具。
- [固定版本实验说明](https://github.com/MaxizYB/Kairos/blob/7971fa40323f16e8e4ce8f12dfd5db9755a98cf1/docs/EXPERIMENTS.md)、[来源说明](https://github.com/MaxizYB/Kairos/blob/7971fa40323f16e8e4ce8f12dfd5db9755a98cf1/THIRD_PARTY.md)。本次读取本地固定副本，未重新核实远端最新状态。
'''
    (out/'现有负载分析报告.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in stats.items() if k not in ('requests_detail','programs')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
