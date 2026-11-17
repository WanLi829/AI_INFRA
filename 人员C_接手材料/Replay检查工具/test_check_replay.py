"""Adversarial input checks; synthetic fixtures are not experimental workloads."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from check_replay import validate


class ReplayChecks(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parent
        self.tmp = tempfile.TemporaryDirectory(prefix='replay-test-',dir=test_root)
        assert Path(self.tmp.name).resolve().parent == test_root
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trace = {'programs': [dict(id='p',session_id='s',synthetic_user='u',source_turn_id='src',
            cohort='test',arrival_s=0,human_gap_original_s=0,human_gap_replay_s=0,predecessor=None,
            turns=[dict(added_token_ids=[1,2],output_tokens=1,gap_after_s=0,
                source_gap_s=0,source_cached_tokens=0,source_tool_failures=0,prompt_tokens=2)])]}
        self.messages = {'p': [[{'role':'user','content':'Test data, never execute'}]]}

    def write(self, name, obj):
        path = self.root/name
        path.write_text(json.dumps(obj),encoding='utf-8')
        return path

    def run_case(self, trace=None, messages=None, edit_protocol=None, **kw):
        t=self.write('trace.json', self.trace if trace is None else trace)
        m=self.write('messages.json', self.messages if messages is None else messages)
        p=dict(current_assistant_included=False,future_outcomes_included=False,
            trace_sha256=hashlib.sha256(t.read_bytes()).hexdigest(),
            visible_messages_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),
            request_signatures=[dict(program='p',stage=0,prompt_tokens=2,
                prompt_sha256=hashlib.sha256(b'[1,2]').hexdigest())])
        if edit_protocol:edit_protocol(p)
        return validate(t,m,self.write('protocol.json',p),**kw)

    def assert_code(self, report, code):
        self.assertIn(code,[i['code'] for i in report['issues']])
        self.assertEqual(report['exit_code'],1)

    def test_valid_and_read_only(self):
        r=self.run_case(max_context=3,vocab_size=3)
        self.assertEqual((r['exit_code'],r['errors'],r['warnings']), (0,0,0))
        self.assertEqual(r['counts']['matched_signatures'],1)
        before={p:p.read_bytes() for p in self.root.iterdir()}
        validate(self.root/'trace.json',self.root/'messages.json',self.root/'protocol.json')
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.iterdir()})

    def test_dependency_cycle_and_missing(self):
        for parent,code in [('p','DEPENDENCY_CYCLE'),('missing','UNKNOWN_PARENT')]:
            with self.subTest(parent=parent):
                self.trace['programs'][0]['predecessor']=parent
                self.assert_code(self.run_case(),code)

    def test_valid_multiturn_prefix_reconstruction(self):
        turns=self.trace['programs'][0]['turns']
        turns.append(dict(turns[0],keep_prefix_tokens=1,added_token_ids=[3,4],prompt_tokens=3))
        self.messages['p'].append([{'role':'user','content':'Second input'}])
        def extend(p):
            p['request_signatures'].append(dict(program='p',stage=1,prompt_tokens=3,
                prompt_sha256=hashlib.sha256(b'[1,3,4]').hexdigest()))
        r=self.run_case(edit_protocol=extend,max_context=4,vocab_size=5)
        self.assertEqual(r['errors'],0)
        self.assertEqual(r['counts']['matched_signatures'],2)

    def test_invalid_first_turn_does_not_invent_next_prompt(self):
        turns=self.trace['programs'][0]['turns']
        turns.append(copy.deepcopy(turns[0]));turns[0]['added_token_ids']=[False]
        r=self.run_case()
        self.assertEqual(r['counts']['reconstructed'],0)
        self.assertIn('RECONSTRUCTION_SKIPPED',[i['code'] for i in r['issues']])

    def test_multinode_cycle(self):
        p=self.trace['programs'][0];q=copy.deepcopy(p);q['id']='q'
        p['predecessor']='q';q['predecessor']='p';self.trace['programs'].append(q)
        self.assert_code(self.run_case(),'DEPENDENCY_CYCLE')

    def test_duplicate_program(self):
        self.trace['programs'].append(copy.deepcopy(self.trace['programs'][0]))
        self.assert_code(self.run_case(),'DUPLICATE_ID')

    def test_bad_tokens(self):
        for ids in ([True],[-1],[1.5],'1,2'):
            with self.subTest(ids=ids):
                self.trace['programs'][0]['turns'][0]['added_token_ids']=ids
                self.assert_code(self.run_case(),'TOKEN_IDS')

    def test_bad_prefix_and_context(self):
        self.assert_code(self.run_case(max_context=2),'CONTEXT_LIMIT')
        self.assert_code(self.run_case(vocab_size=2),'VOCAB_BOUND')
        self.trace['programs'][0]['turns'][0]['keep_prefix_tokens']=1
        self.assert_code(self.run_case(),'PREFIX')

    def test_bad_length_and_output(self):
        t=self.trace['programs'][0]['turns'][0];t['prompt_tokens']=9;t['output_tokens']=True
        r=self.run_case();self.assert_code(r,'PROMPT_LENGTH');self.assert_code(r,'OUTPUT')

    def test_negative_time_and_nonfinite(self):
        self.trace['programs'][0]['turns'][0]['gap_after_s']=-1
        self.assert_code(self.run_case(),'TIME')
        self.trace['programs'][0]['turns'][0]['gap_after_s']=float('nan')
        self.assert_code(self.run_case(),'READ_JSON')

    def test_missing_metadata(self):
        del self.trace['programs'][0]['turns'][0]['source_cached_tokens']
        self.assert_code(self.run_case(),'SOURCE_FIELD')

    def test_message_shape(self):
        self.assert_code(self.run_case(messages={'p':[]}),'MESSAGE_TURN_COUNT')
        self.assert_code(self.run_case(messages={'p':[[{'content':'x'}]]}),'MESSAGE_ROLE')

    def test_protocol_tampering(self):
        self.assert_code(self.run_case(edit_protocol=lambda p:p.update(visible_messages_sha256='0'*64)),'MESSAGES_HASH')
        self.assert_code(self.run_case(edit_protocol=lambda p:p['request_signatures'][0].update(prompt_sha256='0'*64)),'PROMPT_HASH')
        self.assert_code(self.run_case(edit_protocol=lambda p:p.update(future_outcomes_included=True)),'CAUSAL_DECLARATION')

    def test_missing_extra_duplicate_signatures(self):
        for fn,code in [
            (lambda p:p.update(request_signatures=[]),'SIGNATURE_CALL_SET'),
            (lambda p:p['request_signatures'].append(dict(program='extra',stage=0,prompt_sha256='0'*64)),'SIGNATURE_CALL_SET'),
            (lambda p:p['request_signatures'].append(copy.deepcopy(p['request_signatures'][0])),'DUPLICATE_SIGNATURE')]:
            with self.subTest(code=code):self.assert_code(self.run_case(edit_protocol=fn),code)

    def test_old_trace_provenance_is_warning(self):
        r=self.run_case(edit_protocol=lambda p:p.update(trace_sha256='0'*64))
        self.assertEqual(r['errors'],0)
        self.assertIn('TRACE_PROVENANCE_HASH',[i['code'] for i in r['issues']])

    def test_partial_and_null(self):
        t=self.write('trace.json',self.trace)
        self.assertEqual(validate(t)['exit_code'],3)
        n=self.write('null.json',None)
        self.assert_code(validate(t,n,n),'PROTOCOL_TYPE')

    def test_malformed_and_duplicate_json_keys(self):
        for raw in ('{bad','{"programs":[],"programs":[]}'):
            p=self.root/'broken.json';p.write_text(raw)
            self.assert_code(validate(p),'READ_JSON')

    def test_cli_report_and_overwrite_protection(self):
        self.run_case();out=self.root/'report'
        cmd=[sys.executable,'-B',str(Path(__file__).with_name('check_replay.py')),
             '--trace',str(self.root/'trace.json'),'--messages',str(self.root/'messages.json'),
             '--protocol',str(self.root/'protocol.json'),'--output',str(out)]
        r=subprocess.run(cmd,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertTrue((out/'检查报告.md').exists())
        before=(out/'report.json').read_bytes()
        self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,2)
        self.assertEqual(before,(out/'report.json').read_bytes())


if __name__=='__main__':unittest.main(verbosity=2)
