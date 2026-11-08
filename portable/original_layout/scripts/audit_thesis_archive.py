"""Compare an archived-source rebuild with a visually reviewed PDF.

PDF container metadata/IDs can differ; require every page's text and raster pixels
to agree instead of presenting successful compilation as sufficient evidence.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import pdfplumber
from PIL import Image,ImageChops


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.out.exists() or args.renders.exists():raise FileExistsError('Append-only archive audit')
    with pdfplumber.open(args.reference) as a,pdfplumber.open(args.rebuilt) as b:
        if len(a.pages)!=len(b.pages):raise ValueError('Rebuilt page count differs')
        for i,(x,y) in enumerate(zip(a.pages,b.pages),1):
            if x.extract_text()!=y.extract_text() or (x.width,x.height)!=(y.width,y.height):
                raise ValueError(f'Rebuilt page text/size differs: {i}')
        pages=len(a.pages)
    args.renders.mkdir(parents=True)
    subprocess.run(['pdftoppm','-r','72','-png',str(args.rebuilt),str(args.renders/'page')],check=True)
    left=sorted(args.reference_renders.glob('page-*.png'));right=sorted(args.renders.glob('page-*.png'))
    if len(left)!=pages or len(right)!=pages:raise ValueError('Not every page rendered')
    for i,(x,y) in enumerate(zip(left,right),1):
        a,b=Image.open(x).convert('RGB'),Image.open(y).convert('RGB')
        if a.size!=b.size or ImageChops.difference(a,b).getbbox() is not None:
            raise ValueError(f'Rebuilt raster differs: {i}')
    result=dict(complete=True,pages=pages,all_page_text_equal=True,all_page_raster_pixels_equal=True,
                reference_pdf_sha256=digest(args.reference),rebuilt_pdf_sha256=digest(args.rebuilt),
                script_sha256=digest(Path(__file__)),
                scope='Archived TeX and supporting image reproduce all reviewed page text and 72dpi pixels; '
                      'PDF container IDs/metadata may differ. Not a School or scientific correctness certificate.')
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name in ('reference','rebuilt','reference-renders','renders','out'):ap.add_argument('--'+name,type=Path,required=True)
    run(ap.parse_args())
