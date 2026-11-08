"""Convert the editable English pre-draft to standalone LaTeX, no inference.

Run with pypandoc_binary in an isolated uv environment. PDF compiler is separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pypandoc

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    base=ROOT/'thesis'
    paths=[base/name for name in ('dissertation_en.md','references.bib','references.csl','template/ntu_review.tex','template/review_layout.lua')]
    assets=[base/'figures/native_cases_v3/native_cases.png',base/'figures/native_cases_v3/manifest.json',base/'case_audit_v1.json']
    for directory in ('figures/vector_v3','figures/project_screenshots_v2'):
        assets.extend(sorted((base/directory).glob('*')))
    assets.extend([ROOT/'scripts/build_thesis_figures.py',ROOT/'scripts/render_thesis_screenshots.py'])
    for p in paths:
        if not p.exists():raise FileNotFoundError(p)
    for p in assets:
        if not p.exists():raise FileNotFoundError(p)
    pypandoc.convert_file(str(paths[0]),'latex',outputfile=str(args.out),extra_args=[
        '--standalone','--top-level-division=chapter','--number-sections','--citeproc',
        '--bibliography='+str(paths[1]),'--csl='+str(paths[2]),'--template='+str(paths[3]),
        '--lua-filter='+str(paths[4]),'--syntax-highlighting=none','--wrap=none'])
    manifest=dict(pandoc_version=pypandoc.get_pandoc_version(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+assets},
        latex_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    record=base/f"build_manifest_{manifest['latex_sha256'][:12]}.json"
    if record.exists():
        if json.loads(record.read_text())!=manifest:raise ValueError('Same LaTeX hash with changed dependencies; preserve earlier manifest')
    else:
        with record.open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    print(json.dumps(dict(output=str(args.out),**manifest)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'thesis/dissertation_en.tex')
    run(ap.parse_args())
