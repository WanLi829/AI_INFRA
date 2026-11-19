"""CPU-only archive source identity and syntax check; no model imports."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    repo = folder.parent
    manifest = json.loads((folder / 'provenance/sync-manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = repo / entry['repository_path']
        if not path.resolve().is_relative_to(repo):
            raise ValueError('Unsafe repository path')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f"Source differs: {entry['repository_path']}")
        if path.suffix == '.py':
            ast.parse(data, filename=str(path))
    identity = json.loads((folder / 'provenance/code-identity.json').read_text(encoding='utf-8'))
    project = repo / '4090D工作负载交付/project'
    for name, digest in identity['adapter_sha256'].items():
        path = project / name if name.startswith('scripts/') else project / 'b_replay' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Runtime adapter differs: {name}')
    print(json.dumps({'archive_files_verified': len(manifest['files']),
                      'runtime_adapters_verified': len(identity['adapter_sha256']),
                      'gpu_started': False}))


if __name__ == '__main__':
    main()
