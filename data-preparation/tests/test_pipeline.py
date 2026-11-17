import copy
import json
import pytest
from b_replay.pipeline import split_rounds, model_message, tool_label
from b_replay.common import write_json, sha
from b_replay.acquire import blob_hash

def document():
    return {"messages": [
        {"role":"system","content":"s"}, {"role":"user","content":"q"},
        {"role":"assistant","content":"a", "extra":{"timestamp":10.0},
         "tool_calls":[{"id":"x","function":{"name":"bash","arguments":'{"command":"cat foo"}'}},
                       {"id":"y","function":{"name":"bash","arguments":'{"command":"sed bar"}'}}]},
        {"role":"tool","content":"x","tool_call_id":"x","extra":{"timestamp":10.2}},
        {"role":"user","content":"FormatError feedback","extra":{"interrupt_type":"FormatError"}},
        {"role":"tool","content":"y","tool_call_id":"y","extra":{"timestamp":10.7}},
        {"role":"assistant","content":"done","extra":{"timestamp":12.0}},
        {"role":"exit","content":"Submitted"}]}

def test_round_boundaries_multiple_tools_and_feedback():
    doc=document(); rows, issues=split_rounds(doc)
    assert issues == []
    assert len(rows)==2 and rows[1]["message_index"]==6
    assert rows[0]["gap_after_s"] == pytest.approx(.7)
    assert rows[0]["feedback_count"]==3
    assert rows[1]["gap_after_s"] is None
    assert tool_label(doc["messages"][2])=="cat+sed"
    assert "extra" not in model_message(doc["messages"][2])

@pytest.mark.parametrize("mutation,reason", [
    (lambda d:d["messages"][3]["extra"].pop("timestamp"), "missing_tool_interval"),
    (lambda d:d["messages"][3]["extra"].update(timestamp=9), "invalid_tool_interval"),
    (lambda d:d["messages"][5].update(tool_call_id="wrong"), "unmatched_tool_feedback"),
    (lambda d:d["messages"][6]["extra"].update(timestamp=10.1), "nonmonotonic_next_assistant"),
])
def test_invalid_intervals_are_not_imputed(mutation, reason):
    doc=document();mutation(doc)
    _,issues=split_rounds(doc)
    assert any(reason in x for x in issues)

def test_gzip_deterministic_and_git_blob(tmp_path):
    a,b=tmp_path/"a.json.gz",tmp_path/"b.json.gz"
    write_json(a,{"a":[1,2]},False);write_json(b,{"a":[1,2]},False)
    assert a.read_bytes()==b.read_bytes()
    assert blob_hash(b"hello\n")=="ce013625030ba8dba906f756967f9e9ca394464a"
