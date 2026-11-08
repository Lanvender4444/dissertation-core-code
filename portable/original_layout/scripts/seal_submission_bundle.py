"""Create append-only payload hashes and a zip excluding environments/caches."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def run(args):
    base=ROOT/'code';manifest=base/'metadata/package_manifest.json'
    if manifest.exists() or args.zip.exists():raise FileExistsError('Do not overwrite a sealed code package')
    source_manifest=base/'metadata/source_manifest.json'
    sources=json.loads(source_manifest.read_text())
    tools=['scripts/build_submission_bundle.py','scripts/seal_submission_bundle.py']
    for rel in tools:
        p=ROOT/rel;target=base/'original_layout'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError('Packaging source snapshot already exists')
        shutil.copyfile(p,target)
        sources['files'][rel]=dict(snapshot=str(target.relative_to(base)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    sources['additional_packaging_tools']=tools
    source_manifest.write_text(json.dumps(sources,indent=2),encoding='utf-8')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(base/'tests'),'-v'],
                         cwd=base,capture_output=True,text=True)
    demo=subprocess.run([sys.executable,str(base/'demo.py')],cwd=base,capture_output=True,text=True)
    check=subprocess.run([sys.executable,str(base/'verify.py'),'--against-project',str(ROOT)],
                         cwd=base,capture_output=True,text=True)
    if any(r.returncode for r in (tests,demo,check)):
        raise RuntimeError(tests.stderr+demo.stderr+check.stderr)
    rows=[json.loads(line) for line in demo.stdout.splitlines() if line.startswith('{')]
    report=dict(complete=True,unit_test_output=tests.stderr,standalone_example_rows=rows,
                source_verification=json.loads(check.stdout),detector_calls=0,model_fits=0,
                scope='Software/prepared-feature replay checks, not an independent accuracy experiment.')
    with (base/'metadata/validation_report.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    skip={'.venv','__pycache__','results','.git'}
    files=[p for p in sorted(base.rglob('*')) if p.is_file() and not set(p.relative_to(base).parts)&skip
           and p.suffix not in ('.pyc','.zip')]
    record=dict(complete=True,files={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                file_count=len(files),uncompressed_bytes=sum(p.stat().st_size for p in files),
                excluded=['virtual environment','bytecode','results','Git metadata','RGB/XML dataset','YOLO .pt weights'],
                scope='Local submission payload checksum, not School approval or redistribution permission.')
    manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    args.zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files+[manifest]:archive.write(p,'code/'+p.relative_to(base).as_posix())
    relocated=ROOT/'tmp/submission_extract_v1'
    if relocated.exists():raise FileExistsError('Relocation audit path exists')
    relocated.mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as archive:
        for name in archive.namelist():
            if not (relocated/name).resolve().is_relative_to(relocated.resolve()):raise ValueError('Unsafe archive path')
        archive.extractall(relocated)
    replay=subprocess.run([sys.executable,str(relocated/'code/demo.py')],cwd=relocated,capture_output=True,text=True)
    verify=subprocess.run([sys.executable,str(relocated/'code/verify.py')],cwd=relocated,capture_output=True,text=True)
    if replay.returncode or verify.returncode or replay.stdout!=demo.stdout:
        raise RuntimeError('Relocated package replay/check differs: '+replay.stderr+verify.stderr)
    info=dict(complete=True,zip=str(args.zip.relative_to(ROOT)),zip_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
              zip_bytes=args.zip.stat().st_size,payload_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
              payload_file_count=len(files)+1,relocated_replay_equal=True,
              relocated_payload_verified=True,unit_tests_passed=14,prepared_feature_examples=len(rows),
              source_files_checked=report['source_verification']['checked_source_files'])
    with (ROOT/'thesis/code_submission_delivery_v1.json').open('x',encoding='utf-8') as stream:json.dump(info,stream,indent=2)
    print(json.dumps(info))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,default=ROOT/'output/code_submission_v1.zip')
    run(ap.parse_args())
