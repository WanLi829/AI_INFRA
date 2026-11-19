"""Package only source-checked successful experiments and their analysis assets."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

BASE=Path(__file__).resolve().parent
EXCLUDED={'__pycache__','.codegraph','.git','.pytest_cache','.DS_Store'}
ROOT_FILES=('README.md','RECOMPUTE.md','METRICS.md','REFERENCES.md','SOURCE_VERSIONS.json',
    'current_scope.json','ACTUAL_CODE_IDENTITY_CHECK.json','requirements-analysis-lock.txt',
    'build_live_reports.py','build_live_reports_v2.py','streaming_evidence.py',
    'streaming_evidence_v2.py','build_mechanism_figures.py','build_dataset_features.py',
    'write_analysis_bundle.py','verify_delivery.py','seal_success_archive.py')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    args=p.parse_args();source=json.loads((BASE/'success_metadata/SUCCESSFUL_SOURCE_INDEX.json').read_bytes())
    n=source['complete_main_runs'];scope=json.loads((BASE/'current_scope.json').read_bytes())
    assert n==scope['target_runs']==85
    assert json.loads((args.report/'coverage.json').read_bytes())['valid_completed_runs']==n
    selected={BASE/name for name in ROOT_FILES}
    for root_name in source['source_run_paths']:
        root=BASE/'live_results/raw'/Path(root_name).relative_to(source['source_base'])
        state=json.loads((root/'state.json').read_bytes())
        assert state['status']=='complete' and state['summary']['strict_complete']
    selected.update(BASE/'live_results/raw'/name for name in source['files'])
    for root in [BASE/'project',BASE/'upstream',BASE/'retraining',BASE/'requirements',
                 BASE/'success_metadata',BASE/'live_results/kairos02/assets',
                 args.report.resolve(),BASE/'reports/data_features_20261119_v2']:
        assert root.exists() and root.is_relative_to(BASE)
        for path in root.rglob('*'):
            if set(path.relative_to(BASE).parts)&EXCLUDED:continue
            if path.is_symlink():raise ValueError('Nonportable symlink: '+str(path))
            if path.is_file() and path.name not in ('failures.csv','failures.json','failures.jsonl'):
                selected.add(path)
    records=[]
    for path in sorted(selected):
        before=path.stat();sha=digest(path);after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        records.append(dict(path=str(path.relative_to(BASE)),bytes=before.st_size,sha256=sha))
    observed={r['path']:r for r in records}
    for name,row in source['files'].items():
        actual=observed['live_results/raw/'+name]
        assert (actual['bytes'],actual['sha256'])==(row['bytes'],row['sha256']),name
    assets=BASE/'live_results/kairos02/assets';weights=json.loads((assets/'RETRAINED_MANIFEST.json').read_bytes())
    for name,sha in weights['files'].items():assert observed[str((assets/name).relative_to(BASE))]['sha256']==sha
    metadata=json.loads((BASE/'success_metadata/SOURCE_METADATA_SHA.json').read_bytes())
    for original,name in [('SERVER_ENVIRONMENT.txt','SERVER_ENVIRONMENT.txt'),('fast05/code-identity.json','code-identity.json')]:
        assert observed['success_metadata/'+name]['sha256']==metadata[original]['sha256']
    manifest=dict(schema='h800-successful-experiment-delivery/1',created_at=time.time(),
        complete_main_runs=n,selection='Only completed successful experiments; no failure-rate assessment',
        methods=['lmcache','zero_ttl','static_ttl','continuum','kairos'],
        classification='exploratory',repetition_counts=scope['accepted_repetition_counts'],
        verified_source_files=len(source['files']),verified_trained_asset_files=len(weights['files']),files=records)
    manifest_path=BASE/'release_manifest.json'
    if manifest_path.exists() and not (BASE/'preparation_release_manifest.json').exists():
        (BASE/'preparation_release_manifest.json').write_bytes(manifest_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    selected.add(manifest_path);root_name='H800_success85_20261119'
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(phase='packing',successful_runs=n,files=len(selected),source_bytes=sum(r['bytes'] for r in records))),flush=True)
    with tarfile.open(args.archive,'w:gz',compresslevel=1) as archive:
        for path in sorted(selected):archive.add(path,arcname=root_name+'/'+str(path.relative_to(BASE)),recursive=False)
    hashes={};inside=None
    with tarfile.open(args.archive,'r|gz') as archive:
        for member in archive:
            assert member.isfile() and member.name.startswith(root_name+'/')
            relative=member.name[len(root_name)+1:]
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            with archive.extractfile(member) as stream:
                if relative=='release_manifest.json':inside=json.load(stream);continue
                assert relative not in hashes
                hashes[relative]=dict(path=relative,bytes=member.size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
    assert inside==manifest and hashes==observed
    sha=digest(args.archive)
    args.archive.with_name(args.archive.name+'.sha256').write_text(sha+'  '+args.archive.name+'\n')
    proof=dict(status='verified',successful_runs=n,files=len(hashes),archive_bytes=args.archive.stat().st_size,
        archive_sha256=sha,source_files_verified=len(source['files']),trained_asset_files_verified=len(weights['files']),
        verification='Every archive member was streamed and checked against the sealed file SHA/size manifest')
    args.archive.with_name(args.archive.name+'.verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
