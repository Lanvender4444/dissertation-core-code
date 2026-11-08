"""Check figure provenance, unchanged scientific tables/math and editing scope."""
import argparse
import collections
import difflib
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args):
    if args.out.exists():raise FileExistsError('Append-only illustration audit')
    old=(ROOT/'thesis/archive/review_v4/dissertation_en.md').read_text(encoding='utf-8')
    current=(ROOT/'thesis/dissertation_en.md').read_text(encoding='utf-8')
    # Equations and all numeric result rows are protected, not stylistically paraphrased.
    equations=lambda s:re.findall(r'\$\$(.*?)\$\$',s,re.S)
    if equations(old)!=equations(current):raise ValueError('Display equations changed')
    numeric_rows=lambda s:[l for l in s.splitlines() if l.startswith('|') and re.search(r'\|\s*[\d,]+\s*\|',l)]
    if numeric_rows(old)!=numeric_rows(current):raise ValueError('Numeric table rows changed')
    cites=lambda s:collections.Counter(re.findall(r'(?<!\w)@([A-Za-z0-9_]+)',s))
    if cites(old)!=cites(current):raise ValueError('Citation usage changed')
    if re.findall(r'^#+ .+$',old,re.M)!=re.findall(r'^#+ .+$',current,re.M):
        raise ValueError('Section coverage changed')
    figures=json.loads((ROOT/'thesis/figures/vector_v3/manifest.json').read_text())
    screenshots=json.loads((ROOT/'thesis/figures/project_screenshots_v2/manifest.json').read_text())
    if not figures['complete'] or figures['svg_count']!=11 or not screenshots['complete']:
        raise ValueError('Incomplete figures')
    for manifest in (figures,screenshots):
        for r in manifest.get('figures',[]):
            for p,h in r['sources'].items():
                if sha(ROOT/p)!=h:raise ValueError('Changed chart/diagram source')
            for kind in ('svg','pdf'):
                if sha(ROOT/r[kind])!=r[kind+'_sha256']:raise ValueError('Changed figure')
            xml=ET.parse(ROOT/r['svg']).getroot()
            if any('href' in attr for e in xml.iter() for attr in e.attrib):
                raise ValueError('Unexpected external SVG resource')
        for p,h in manifest.get('source_sha256',{}).items():
            if sha(ROOT/p)!=h:raise ValueError('Changed screenshot renderer')
    if sha(ROOT/'scripts/build_thesis_figures.py')!=figures['generator_sha256']:
        raise ValueError('Generator changed after export')
    ev=json.loads((ROOT/'thesis/evidence_snapshot_v6.json').read_text())
    if sha(ROOT/'thesis/evidence_snapshot_v6.json')!=figures['data_source_sha256']:
        raise ValueError('Figure evidence changed')
    charts={r['name']:r['values'] for r in figures['figures'] if r['kind']=='data_chart'}
    full=ev['five']['all'];base=full['protected'];key=ev['current_key']
    expected_domains={k:{m:v for m,v in ev['three'][k].items() if m in ('precision','recall')}
                      for k in ('960/cam2','960/cam3','960/cam4','strict_no_band')}
    expected_domains.update({label:{field:m[field] for field in ('precision','recall')}
                            for label,m in [('old',ev['original_three_on_five']),('A',base),('final',full[key]),('raw',full['raw'])]})
    if charts['precision_recall_domains']!=expected_domains:raise ValueError('Plotted P/R changed')
    expected={k:{'added_tp':full[k]['tp']-base['tp'],'added_fp':full[k]['fp']-base['fp']}
              for k in (key,'pair/independent/tau0.55','shared/independent/tau0.55')}
    if charts['controlled_recovery']!=expected:raise ValueError('Recovery chart changed')
    stages=json.loads((ROOT/'ultralytics/algorithm/evaluation_reports/protected_conditional_full_stage_losses_20261010.json').read_text())
    # This report is already hashed by the evidence audit; additionally require its plotted capacity.
    if '14428' not in json.dumps(stages):raise ValueError('Missing saved stage capacity')
    if charts['matching_capacity']!={'stage_tp':[14260,14464,14428,10606],'net':-3654}:
        raise ValueError('Stage chart changed')
    av=ev['position_availability']
    if charts['bev_availability']!={'displayed':av['displayed'],'located':av['located'],
        'not_located':av['displayed']-av['located'],'availability':av['located']/av['displayed']}:
        raise ValueError('Availability chart changed')
    for row in screenshots['rows']:
        if not row['perception_unchanged']:raise ValueError('Screenshot changed predictions')
        for p,h in row['image_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('Screenshot input image changed')
        for f in row['files']:
            p=ROOT/f['path']
            if sha(p)!=f['sha256'] or list(Image.open(p).size)!=f['size']:
                raise ValueError('Screenshot asset changed')
    paths=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',current)
    for path in paths:
        if not (ROOT/'thesis'/path).exists():raise FileNotFoundError(path)
    labels=re.findall(r'\\label\{(fig:[^}]+)\}',current)
    if len(labels)!=len(set(labels)):raise ValueError('Duplicate figure labels')
    for label in labels:
        if f'\\ref{{{label}}}' not in current and label!='fig:architecture':
            raise ValueError(f'Unintroduced figure: {label}')
    old_paragraphs=[p for p in old.split('\n\n') if p and not p.startswith(('|','\\','#','$$',':::'))]
    current_paragraphs=set(current.split('\n\n'))
    changed=[p for p in old_paragraphs if p not in current_paragraphs]
    result=dict(complete=True,svg_figures=11,screenshot_files=5,embedded_interface_captures=4,
                total_numbered_figures=len(labels),protected_display_equations=len(equations(old)),
                unchanged_numeric_rows=len(numeric_rows(old)),unchanged_citation_occurrences=sum(cites(old).values()),
                revised_original_prose_paragraphs=len(changed),new_detector_calls=screenshots['new_detector_calls'],
                model_fits=screenshots['model_fits'],manuscript_sha256=sha(ROOT/'thesis/dissertation_en.md'),
                figure_manifest_sha256=sha(ROOT/'thesis/figures/vector_v3/manifest.json'),
                screenshot_manifest_sha256=sha(ROOT/'thesis/figures/project_screenshots_v2/manifest.json'),
                scope='Provenance/arithmetic/protected text check; visual layout and scientific claim fidelity also require review.')
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args())
