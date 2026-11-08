"""Render every thesis page and create numbered contact sheets for visual QA."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import pdfplumber
from PIL import Image,ImageDraw


def run(args):
    if args.out.exists():raise FileExistsError('Append-only PDF QA')
    args.out.mkdir(parents=True)
    pages=[];key_pages=[];text_all=[];edge=[];replacement=[]
    with pdfplumber.open(args.pdf) as document:
        for index,page in enumerate(document.pages,1):
            text=page.extract_text() or '';text_all.append(text)
            if any(term in text for term in ('Controlled recovery ablation','System architecture','Historical full-image no-BAND','Full-sequence system comparison','References','Review Status','Abstract')):key_pages.append(index)
            for ch in page.chars:
                if ch['text'].strip() and (ch['x0']<-1 or ch['x1']>page.width+1 or ch['top']<-1 or ch['bottom']>page.height+1):
                    edge.append(dict(page=index,text=ch['text'],bounds=[ch['x0'],ch['top'],ch['x1'],ch['bottom']]))
                if '\ufffd' in ch['text']:replacement.append(index)
            pages.append(dict(page=index,width=page.width,height=page.height,characters=len(page.chars),text_start=text[:140]))
    for required in ('10,606','1,120','14,239','24,845','13,473','57.92'):
        if required not in '\n'.join(text_all):raise ValueError(f'Missing PDF result {required}')
    if edge or replacement:raise ValueError(f'Clipped/replacement PDF text: {edge[:5]}, {replacement[:5]}')
    subprocess.run([args.pdftoppm,'-r','72','-png',str(args.pdf),str(args.out/'page')],check=True)
    images=sorted(args.out.glob('page-*.png'))
    if len(images)!=len(pages):raise ValueError('Not every page rendered')
    for first in range(0,len(images),12):
        sheet=Image.new('RGB',(960,1120),'#cbd1d6');draw=ImageDraw.Draw(sheet)
        for j,path in enumerate(images[first:first+12]):
            thumb=Image.open(path).convert('RGB');thumb.thumbnail((225,330))
            x=(j%4)*240+(240-thumb.width)//2;y=(j//4)*373+25
            sheet.paste(thumb,(x,y));draw.text((x,y-18),f'PDF page {first+j+1}',fill='black')
        sheet.save(args.out/f'contact_{first//12+1:02}.png')
    result=dict(complete=True,pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),pages=len(pages),
        every_page_rendered=True,edge_clipped_characters=0,replacement_characters=0,
        key_pages=key_pages,page_details=pages,
        scope='Text/result presence and page rendering only; visual inspection of contact sheets and selected pages remains separate')
    (args.out/'inspection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='page_details'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pdf',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--pdftoppm',default='pdftoppm')
    run(ap.parse_args())
