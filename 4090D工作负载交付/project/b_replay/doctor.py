"""Read-only readiness inventory. CPU checks never claim GPU validation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .client import load_bundle
from .common import read_json,write_json
from .server import RUNTIME_ROOT,verify_runtime,load_upstream


def check(bundle=None,joint_assets=None,model=None):
    checks=[]
    def record(name,call):
        try: checks.append(dict(name=name,status='pass',detail=call()))
        except Exception as e: checks.append(dict(name=name,status='blocked',detail=str(e)))
    record('runtime_integrity',verify_runtime)
    record('httpx',lambda:importlib.metadata.version('httpx'))
    if bundle:
        record('bundle',lambda:load_bundle(bundle)[0])
        record('completed_actions',lambda:read_json(Path(bundle)/'completed_actions.json')['schema'])
    load_upstream()
    from kairos.verify import verify_joint_assets,verify_model
    record('external_joint_weights',lambda:verify_joint_assets(joint_assets))
    if model: record('model_weights',lambda:verify_model(model))
    result=dict(schema='midterm-readiness/1',host=platform.system(),python=sys.version,
        gpu_started=False,gpu_validated=False,checks=checks,
        external_requirements=['Actual H800 smoke for all five configurations',
            'H800 service/reload calibration with provenance',
            'Current joint predictor training/validation exclusion IDs',
            'Second dataset with complete input/output/timing provenance'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path);p.add_argument('--joint-assets',type=Path)
    p.add_argument('--model',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--require-ready',action='store_true')
    a=p.parse_args();result=check(a.bundle,a.joint_assets,a.model)
    if a.output: write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if a.require_ready and any(c['status']=='blocked' for c in result['checks']):raise SystemExit(2)


if __name__=='__main__':main()
