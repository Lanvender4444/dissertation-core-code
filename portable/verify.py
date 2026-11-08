"""Read-only checksum audit of the submission payload and optional source checkout."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(project=None):
    doc=json.loads((ROOT/'metadata/source_manifest.json').read_text())
    checked=0
    for rel,record in doc['files'].items():
        p=ROOT/record['snapshot']
        if not p.is_file() or sha(p)!=record['sha256']:raise ValueError(f'Snapshot changed: {rel}')
        if project is not None:
            source=project/rel
            if not source.is_file() or sha(source)!=record['sha256']:raise ValueError(f'Project no longer matches snapshot: {rel}')
        checked+=1
    payload=ROOT/'metadata/package_manifest.json'
    if payload.exists():
        for rel,digest in json.loads(payload.read_text())['files'].items():
            if sha(ROOT/rel)!=digest:raise ValueError(f'Package changed: {rel}')
    return dict(complete=True,checked_source_files=checked,package_manifest_checked=payload.exists(),
                source_project_compared=str(project) if project else None)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--against-project',type=Path)
    args=ap.parse_args();print(json.dumps(verify(args.against_project.resolve() if args.against_project else None)))
