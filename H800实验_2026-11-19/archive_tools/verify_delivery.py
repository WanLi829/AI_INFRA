"""Verify a clean extracted payload and every prepared replay input; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skip-inputs',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'release_manifest.json').read_text())
    for row in manifest['files']:
        path=root/row['path']
        if not path.resolve().is_relative_to(root) or not path.is_file():raise ValueError('Missing or unsafe payload path: '+row['path'])
        with path.open('rb') as stream:
            observed_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
        if path.stat().st_size!=row['bytes'] or observed_sha256!=row['sha256']:
            raise ValueError('Payload changed: '+row['path'])
    inputs=[]
    if not a.skip_inputs:
        sys.path.insert(0,str(root/'project'))
        from b_replay.client import load_bundle
        for path in sorted((root/'project/inputs/prepared').iterdir()):
            if not path.is_dir():continue
            data,trace,_=load_bundle(path)
            independent=len({r.get('source_id',r['id']) for r in trace['programs']})
            if independent!=data['independent_sources']:raise ValueError('Independent source count changed')
            inputs.append(dict(bundle=path.name,programs=data['programs'],calls=data['calls'],independent_sources=independent))
        base=root/'project/inputs/prepared'
        expanded=json.loads((base/'instances400/manifest.json').read_text())
        parent=hashlib.sha256((base/'heldout-v2/manifest.json').read_bytes()).hexdigest()
        if expanded['source_bundle_sha256']!=parent:raise ValueError('Instance expansion parent identity changed')
    print(json.dumps(dict(verified_files=len(manifest['files']),inputs=inputs,gpu_started=False),indent=2))


if __name__=='__main__':main()
