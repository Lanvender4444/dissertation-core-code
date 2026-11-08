"""Audit written result-table cells, citations and chapter coverage against evidence.

This is a bounded writing check, not proof of originality or thesis acceptance.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def numbers(line):return [s.strip() for s in line.strip().strip('|').split('|')]


def cells_match(cells,metric,fields):
    for cell,field in zip(cells,fields):
        actual=float(cell.replace(',',''))
        expected=metric[field]*100 if field in ('precision','recall','f1') else metric[field]
        tolerance=.0051 if field in ('precision','recall','f1') else 0
        if abs(actual-expected)>tolerance:raise ValueError(f'Written {field}: {actual}, expected {expected}')


def run(args):
    text=(ROOT/'thesis/dissertation_en.md').read_text(encoding='utf-8')
    bibliography=(ROOT/'thesis/references.bib').read_text(encoding='utf-8')
    evidence=json.loads((ROOT/'thesis/evidence_snapshot_v6.json').read_text())
    for rel,digest in evidence['source_sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise ValueError(f'Source changed: {rel}')
    headings=re.findall(r'^# (.+)$',text,re.M)
    chapters=headings[:7]
    if len(chapters)!=7 or chapters[0]!='Introduction' or chapters[-1]!='Conclusions and Future Work':raise ValueError('Chapter coverage')
    cited=set(re.findall(r'(?<!\w)@([A-Za-z0-9_]+)',text))
    available=set(re.findall(r'@\w+\{([^,]+),',bibliography))
    if cited-available:raise ValueError(f'Missing references {cited-available}')
    if available-cited:raise ValueError(f'Unused references {available-cited}')
    field_order=('tp','fp','fn','precision','recall','f1')
    full=evidence['five']['all'];key=evidence['current_key'];three=evidence['three'];checked=[]
    tables={
        'Historical full-image no-BAND':{
            'Cam2 alone':three['960/cam2'],'Cam3 alone':three['960/cam3'],'Cam4 alone':three['960/cam4'],
            'Strict no-BAND fusion, frozen score + E1':three['strict_no_band']},
        'Historical three-camera input':{
            'BAND + depth, without E1':three['BAND/d'],'BAND + depth + E1':three['BAND/d_E1'],
            'Strict no-BAND, current score frozen':three['strict_no_band'],'No-BAND, earlier 960 score + fixed E1':three['adapted_960_score']},
        'Full-sequence system comparison':{
            'Historical strict three-view fusion':evidence['original_three_on_five'],'Current reliable five-view A':full['protected'],
            'A + conditional singleton recovery':full[key],'Same-input five-view raw detection nodes':full['raw']},
        'Controlled recovery ablation':{
            'None: protected A':full['protected'],'Singleton, independent':full[key],
            'Pair, independent':full['pair/independent/tau0.55'],'Shared-state, independent':full['shared/independent/tau0.55']},
        'Fixed-threshold score-factorisation':{
            'Retained singleton':evidence['extensions']['coarse']['baseline'],
            'Two-head score':evidence['extensions']['coarse']['factored/tau0.55'],
            'Two-head plus spatial':evidence['extensions']['coarse']['spatial_factored/tau0.55']},
        'Exact development-threshold':{
            'Original singleton head':evidence['extensions']['exact']['existing_single'],
            'Two-head score':evidence['extensions']['exact']['factored'],
            'Two-head plus spatial':evidence['extensions']['exact']['spatial_factored']},
        'Retained singleton configuration':{
            label:evidence['five'][split][key] for label,split in [('Early fitting','train'),('Development','development'),
                ('Guard','guard'),('Late','late'),('Guard + late','guard_late'),('Full sequence','all')]}
    }
    current=None;seen={name:set() for name in tables}
    for line in text.splitlines():
        if line.startswith('Table: '):current=next((name for name in tables if name in line),None)
        elif current and line.startswith('|'):
            cells=numbers(line);label=cells[0]
            if label not in tables[current]:continue
            fields=('tp','fp','precision','recall','f1') if current=='Controlled recovery ablation' else field_order
            cells_match(cells[1:1+len(fields)],tables[current][label],fields)
            seen[current].add(label);checked.append(dict(table=current,row=label,fields=list(fields)))
    for name,expected in tables.items():
        if seen[name]!=set(expected):raise ValueError(f'Table rows missing in {name}')
    for name,label in (('train','Early'),('development','Development'),('guard','Guard'),('late','Late')):
        meta=evidence['segment_meta'][name]
        expected=f"{int(meta['first'])}-{int(meta['last'])}"
        row=next((numbers(line) for line in text.splitlines() if line.startswith(f'| {label} |') and len(numbers(line))==5),None)
        if not row or expected not in row[1] or int(row[2].replace(',',''))!=meta['frames']:
            raise ValueError(f'Written frame-index range differs for {name}')
    qualitative=json.loads((ROOT/'thesis/case_audit_v1.json').read_text())
    if not qualitative['complete'] or qualitative['panels']!=6 or qualitative['frames']!=6:
        raise ValueError('Incomplete native-image example audit')
    for case in qualitative['checks']:
        if case['independent_iou'] is not None and f"{case['independent_iou']:.3f}" not in text:
            raise ValueError('Written illustrative IoU differs')
    if 'fig:native-cases' not in text or 'not random sampling' not in text:raise ValueError('Qualitative selection scope missing')
    av=evidence['position_availability']
    for value in (f"{av['located']:,}",f"{av['displayed']:,}",f"{av['fraction']*100:.2f}%",str(av['ignored_displayed'])):
        if value not in text:raise ValueError('Written position-availability value missing')
    required=('not independent','not a calibrated','not a dense','not interchangeable','240','76','24,845','13,473')
    for phrase in required:
        if phrase not in text:raise ValueError(f'Critical qualification missing: {phrase}')
    word_count=len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b",text))
    if word_count<6500:raise ValueError('Draft is an outline rather than substantive chapters')
    result=dict(complete=True,chapters=chapters,body_and_appendix_word_count=word_count,cited_references=len(cited),
        checked_result_rows=len(checked),checked_qualitative_panels=6,rows=checked,
        manuscript_sha256=hashlib.sha256(text.encode()).hexdigest(),
        evidence_sha256=hashlib.sha256((ROOT/'thesis/evidence_snapshot_v6.json').read_bytes()).hexdigest(),
        scope='Exact result-table arithmetic/source hashes/citation keys/chapter coverage only; no School, physical identity or originality certificate')
    if args.out.exists():raise FileExistsError('Append-only review')
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'thesis/review_content_v1.json')
    run(ap.parse_args())
