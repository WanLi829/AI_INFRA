import ast
from b_replay.common import ROOT, RUNTIME_ACTIVATION

def test_method_descriptors_match_pinned_source_without_execution():
    tree=ast.parse((ROOT/"sources/kairos/kairos/_runtime/activation.py").read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=="RUNTIME_ACTIVATION" for t in n.targets))
    assert ast.literal_eval(node.value)==RUNTIME_ACTIVATION
