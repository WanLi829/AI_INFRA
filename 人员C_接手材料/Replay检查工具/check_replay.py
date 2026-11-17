"""Offline structural validation for the Kairos 7971fa4 replay contract.

Standard library only. Never runs messages, imports Kairos, modifies inputs,
downloads tokenizers, or generates replacement protocol signatures.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def number(value):
    return type(value) in (int, float) and value >= 0 and (type(value) is int or math.isfinite(value))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON key: ' + key)
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValueError('Non-finite JSON number: ' + value)


def validate(trace_path, messages_path=None, protocol_path=None, *, max_context=None, vocab_size=None):
    issues = []
    inputs = {}

    def issue(code, location, detail, level='error'):
        issues.append(dict(level=level, code=code, location=location, detail=detail))

    def read(path, name):
        if path is None:
            issue('NOT_PROVIDED', name, '未提供，相关检查未执行。', 'warning')
            return None
        try:
            raw = Path(path).read_bytes()
            inputs[name] = dict(path=str(Path(path).resolve()), sha256=sha(raw), bytes=len(raw))
            return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique_object,
                              parse_constant=reject_constant)
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            issue('READ_JSON', name, str(exc))
            return None

    trace = read(trace_path, 'trace')
    messages = read(messages_path, 'messages')
    protocol = read(protocol_path, 'protocol')
    if messages_path is not None and messages is None:
        issue('MESSAGES_TYPE', 'messages', '未能读到消息对象（null 也不合法）。')
    if protocol_path is not None and protocol is None:
        issue('PROTOCOL_TYPE', 'protocol', '未能读到完整协议对象（null 也不合法）。')
    programs = trace.get('programs') if isinstance(trace, dict) else None
    if not isinstance(programs, list) or not programs:
        issue('PROGRAMS', 'trace.programs', '需要非空程序数组。')
        programs = []
    ids = {}; expected = {}; token_hashes = {}; lengths = {}; salts = []
    request_count = 0
    for index, p in enumerate(programs):
        loc = f'programs[{index}]'
        if not isinstance(p, dict):
            issue('PROGRAM_TYPE', loc, '程序必须为对象。'); continue
        pid = p.get('id')
        if not isinstance(pid, str) or not pid.strip():
            issue('PROGRAM_ID', loc+'.id', '需要非空字符串 ID。'); continue
        if pid in ids:
            issue('DUPLICATE_ID', loc+'.id', '程序 ID 重复。'); continue
        ids[pid] = p
        for key in ('session_id', 'synthetic_user', 'source_turn_id', 'cohort'):
            if not isinstance(p.get(key), str) or not p[key].strip():
                issue('PROGRAM_FIELD', loc+'.'+key, '当前回放接口需要非空字符串。')
        for key in ('arrival_s', 'human_gap_original_s', 'human_gap_replay_s'):
            if not number(p.get(key)):
                issue('TIME', loc+'.'+key, '时间必须是有限、非负数值（不是布尔值）。')
        parent = p.get('predecessor')
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            issue('PARENT_TYPE', loc+'.predecessor', '前驱必须为非空程序 ID 或 null。')
        salt = p.get('cache_salt', pid)
        if not isinstance(salt, str) or not salt:
            issue('CACHE_SALT', loc+'.cache_salt', '需要非空字符串，或省略以使用程序 ID。')
        else:
            salts.append(salt)
        if p.get('append_generated_outputs', False) is not False:
            issue('REPLAY_MODE', loc+'.append_generated_outputs', '当前 teacher replay 不拼接本次生成结果。')
        turns = p.get('turns')
        if not isinstance(turns, list) or not turns:
            issue('TURNS', loc+'.turns', '需要非空轮次数组。'); continue
        previous = []
        for stage, t in enumerate(turns):
            where = f'{loc}.turns[{stage}]'; identity = (pid, stage)
            request_count += 1; expected[identity] = where
            if not isinstance(t, dict):
                issue('TURN_TYPE', where, '轮次必须为对象。'); previous = None; continue
            output = t.get('output_tokens')
            if not integer(output, 1):
                issue('OUTPUT', where+'.output_tokens', '输出预算必须为正整数。')
            if not number(t.get('gap_after_s')):
                issue('TIME', where+'.gap_after_s', '等待必须为有限非负数。')
            elif stage == len(turns)-1 and t['gap_after_s'] != 0:
                issue('FINAL_GAP', where+'.gap_after_s', '末轮等待不被执行。', 'warning')
            if 'tool' in t and (not isinstance(t['tool'], str) or not t['tool']):
                issue('TOOL', where+'.tool', '工具标签须为非空字符串；省略时使用回放器默认标签。')
            for key in ('source_gap_s', 'source_cached_tokens', 'source_tool_failures'):
                if key not in t:
                    issue('SOURCE_FIELD', where+'.'+key, '当前回放日志直接读取该字段，不能缺省。')
                elif t[key] is None:
                    issue('SOURCE_UNKNOWN', where+'.'+key, '来源信息未知，保留 null，不伪造观测。', 'warning')
                elif not (number(t[key]) if key == 'source_gap_s' else integer(t[key])):
                    issue('SOURCE_FIELD', where+'.'+key, '需要非负数值（计数须为整数），未知可显式标 null。')
            added = t.get('added_token_ids')
            valid_added = isinstance(added, list) and all(integer(v) for v in added)
            if not valid_added:
                issue('TOKEN_IDS', where+'.added_token_ids', '需要非负整数数组，布尔值不是 token ID。')
            elif vocab_size is not None and any(v >= vocab_size for v in added):
                issue('VOCAB_BOUND', where+'.added_token_ids', '存在不小于指定词表上界的 token ID。')
            keep = t.get('keep_prefix_tokens', len(previous) if previous is not None else None)
            valid_keep = integer(keep) and (previous is None or keep <= len(previous))
            if not valid_keep:
                issue('PREFIX', where+'.keep_prefix_tokens', '前缀长度必须为整数且不超过上轮输入；首轮仅能保留 0。')
            if previous is None:
                issue('RECONSTRUCTION_SKIPPED', where, '前轮输入无效，不能可信重建后续输入。', 'warning')
            if previous is None or not valid_added or not valid_keep:
                previous = None; continue
            prompt = previous[:keep] + added
            if not prompt:
                issue('EMPTY_PROMPT', where, '重建输入不能为空。')
            if 'prompt_tokens' in t and (not integer(t['prompt_tokens']) or t['prompt_tokens'] != len(prompt)):
                issue('PROMPT_LENGTH', where+'.prompt_tokens', '声明输入长度与重建长度不符。')
            if max_context is not None and integer(output, 1) and len(prompt)+output > max_context:
                issue('CONTEXT_LIMIT', where, f'输入加输出预算 {len(prompt)+output} 超过上限 {max_context}。')
            token_hashes[identity] = sha(json.dumps(prompt, separators=(',', ':')).encode())
            lengths[identity] = len(prompt); previous = prompt
    # Each node has at most one predecessor. Iterative walk avoids recursion limits.
    done = set()
    for pid in ids:
        chain = set(); cur = pid
        while isinstance(cur, str) and cur in ids and cur not in done:
            if cur in chain:
                issue('DEPENDENCY_CYCLE', 'predecessor', '前驱依赖存在环，涉及程序：'+cur); break
            chain.add(cur); parent = ids[cur].get('predecessor')
            if isinstance(parent, str) and parent not in ids:
                issue('UNKNOWN_PARENT', cur+'.predecessor', '前驱程序不存在：'+parent)
            cur = parent
        done.update(chain)
    if any(n > 1 for n in Counter(salts).values()):
        issue('SHARED_SALT', 'cache_salt', '不同程序共享 cache_salt；需说明跨程序共享是否符合实验设计。', 'warning')
    if messages is not None:
        if not isinstance(messages, dict):
            issue('MESSAGES_TYPE', 'messages', '需要程序 ID 到逐轮消息数组的映射。')
        else:
            if set(messages) != set(ids):
                issue('MESSAGE_PROGRAM_SET', 'messages', '消息中的程序集合与 trace 不一致。')
            for pid, p in ids.items():
                turns = p.get('turns'); stages = messages.get(pid)
                if not isinstance(stages, list) or not isinstance(turns, list) or len(stages) != len(turns):
                    issue('MESSAGE_TURN_COUNT', pid, '消息轮数与 trace 不一致或消息不是数组。'); continue
                for stage, ms in enumerate(stages):
                    where = f'messages[{pid}][{stage}]'
                    if not isinstance(ms, list) or not ms:
                        issue('MESSAGE_LIST', where, '每轮需要非空消息数组。'); continue
                    for j, msg in enumerate(ms):
                        if not isinstance(msg, dict) or not isinstance(msg.get('role'), str) or not msg['role']:
                            issue('MESSAGE_ROLE', where+f'[{j}]', '消息须为对象且含非空 role。'); continue
                        if msg['role'] not in ('system','developer','user','assistant','tool','function'):
                            issue('UNKNOWN_ROLE', where+f'[{j}]', '非标准角色，需确认模板支持。', 'warning')
                        if not ('content' in msg or 'tool_calls' in msg or 'function_call' in msg):
                            issue('MESSAGE_CONTENT', where+f'[{j}]', '消息缺少 content 或工具调用字段。')
                        elif 'content' in msg and msg['content'] is not None and not isinstance(msg['content'], (str,list)):
                            issue('MESSAGE_CONTENT', where+f'[{j}]', 'content 应为字符串、内容块数组或 null。')
    matched = 0
    if protocol is not None:
        if not isinstance(protocol, dict):
            issue('PROTOCOL_TYPE', 'protocol', '需要完整协议对象；签名样例数组不能替代完整协议。')
        else:
            for key in ('current_assistant_included','future_outcomes_included'):
                if protocol.get(key) is not False:
                    issue('CAUSAL_DECLARATION', 'protocol.'+key, '声明必须显式为 false；声明本身不证明没有未来信息。')
            if 'messages' in inputs and protocol.get('visible_messages_sha256') != inputs['messages']['sha256']:
                issue('MESSAGES_HASH', 'protocol.visible_messages_sha256', '消息原始文件字节哈希不一致。')
            # This optional provenance field is not enforced by upstream verify_workload.
            if 'trace' in inputs and 'trace_sha256' in protocol and protocol['trace_sha256'] != inputs['trace']['sha256']:
                issue('TRACE_PROVENANCE_HASH', 'protocol.trace_sha256',
                      '协议记录的 trace 哈希与输入文件不符。上游未校验此来源字段；需维护者解释，工具不改写它。', 'warning')
            signatures = protocol.get('request_signatures'); seen = set()
            if not isinstance(signatures, list):
                issue('SIGNATURES', 'protocol.request_signatures', '需要签名数组。'); signatures = []
            for i, s in enumerate(signatures):
                where = f'protocol.request_signatures[{i}]'
                if not isinstance(s, dict) or not isinstance(s.get('program'), str) or not integer(s.get('stage')):
                    issue('SIGNATURE_ID', where, '需要程序 ID 和从 0 开始的整数 stage。'); continue
                key = (s['program'], s['stage'])
                if key in seen:
                    issue('DUPLICATE_SIGNATURE', where, '同一请求出现重复签名。')
                seen.add(key)
                value = s.get('prompt_sha256')
                if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
                    issue('SIGNATURE_HASH_FORMAT', where, '签名必须为 64 位小写十六进制 SHA256。')
                elif key in token_hashes:
                    if value != token_hashes[key]:
                        issue('PROMPT_HASH', where, '重建 token 序列与协议签名不同。')
                    else:
                        matched += 1
                if key in lengths and 'prompt_tokens' in s and (not integer(s['prompt_tokens']) or s['prompt_tokens'] != lengths[key]):
                    issue('SIGNATURE_LENGTH', where, '签名记录的长度与重建输入不符。')
            if seen != set(expected):
                issue('SIGNATURE_CALL_SET', 'protocol.request_signatures', '签名的程序/轮次集合与输入不一致。')
    if max_context is None:
        issue('CONTEXT_NOT_CHECKED', 'options', '未提供 --max-context，未检查上下文上限。', 'warning')
    if vocab_size is None:
        issue('VOCAB_NOT_CHECKED', 'options', '未提供 --vocab-size，仅检查 token 是非负整数。', 'warning')
    errors = sum(i['level'] == 'error' for i in issues)
    complete = messages_path is not None and protocol_path is not None
    verdict = '不通过' if errors else ('结构与签名检查完成（请查看警告）' if complete else '部分检查完成（材料不齐）')
    return dict(verdict=verdict, exit_code=1 if errors else (0 if complete else 3),
        errors=errors, warnings=len(issues)-errors, inputs=inputs,
        limits=dict(max_context=max_context,vocab_size=vocab_size),
        counts=dict(programs=len(ids),requests=request_count,reconstructed=len(token_hashes),matched_signatures=matched),
        limitations=['未验证 tokenizer/chat template 与消息语义一致性或未来信息泄漏。',
            '未验证模型、发布清单、资源容量、GPU 执行或性能；通过不代表可直接上机。',
            '不修复数据、不重签协议；检查依据 Kairos 7971fa4，其他接口需另行适配。'],issues=issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--messages', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='全新的报告目录，防止覆盖')
    parser.add_argument('--max-context', type=int)
    parser.add_argument('--vocab-size', type=int)
    args = parser.parse_args()
    for key in ('max_context','vocab_size'):
        if getattr(args,key) is not None and getattr(args,key) <= 0:
            parser.error(key+' 必须为正整数')
    if args.output.exists():
        parser.error('输出目录已存在，请使用新目录。')
    result = validate(args.trace,args.messages,args.protocol,max_context=args.max_context,vocab_size=args.vocab_size)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def cell(value):
        return str(value).replace('|','\\|').replace('\n',' ').replace('\r',' ').replace('<','&lt;').replace('>','&gt;')
    lines = ['# Replay 离线检查报告','',f"结论：**{result['verdict']}**",'',
        f"错误 {result['errors']} 项，警告 {result['warnings']} 项。退出码 {result['exit_code']}。",'',
        f"程序 {result['counts']['programs']}；请求 {result['counts']['requests']}；可重建输入 {result['counts']['reconstructed']}；匹配签名 {result['counts']['matched_signatures']}。",'',
        '## 检查边界','']+['- '+v for v in result['limitations']]+['','## 输入指纹','',
        '| 输入 | SHA256 |','|---|---|']+[f"| {k} | {v['sha256']} |" for k,v in result['inputs'].items()]+[
        '', '## 问题明细', '', '| 级别 | 编码 | 位置 | 说明 |','|---|---|---|---|']+[
        '| '+' | '.join(cell(i[k]) for k in ('level','code','location','detail'))+' |' for i in result['issues']]
    (args.output/'检查报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f"{result['verdict']}；错误 {result['errors']}，警告 {result['warnings']}。报告：{args.output.resolve()}")
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
