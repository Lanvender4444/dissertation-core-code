"""Generate LaTeX table fragments and a numbers file directly from supplementary results JSON."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'ultralytics/experiments/supp_20261101'
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
j = lambda p: json.loads((R/p).read_text())
pc = lambda x: f'{100*x:.2f}'
n = lambda x: f'{int(x):,}'


def write(name, s):
    (OUT/f'{name}.tex').write_text(s)


def tab(caption, label, cols, header, rows, size='\\small'):
    return ('\\begin{table}[htbp]\n\\centering' + size + '\\singlespacing\n\\caption{' + caption + '}\\label{' + label + '}\n'
            '\\begin{tabular}{@{}' + cols + '@{}}\n\\toprule\n' + header + '\\\\\n\\midrule\n' + '\\\\\n'.join(rows) +
            '\\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


# ---------- E1 ----------
e1 = j('e1_e6_error_analysis/run01/results.json')
F, F05 = e1['full']['tau0.25'], e1['full']['tau0.05']
rows = []
for c in (1, 2, 3, 4, 5):
    a = next(x for x in F['camera'] if x['camera'] == c); b = next(x for x in F05['camera'] if x['camera'] == c)
    sz = {x['bin']: x for x in F['size'] if x['camera'] == c}
    rows.append(f"Cam{c} ({'fisheye' if c in (1,5) else 'pinhole'}) & {n(a['tp']+a['fn'])} & {pc(a['precision'])} & {pc(a['recall'])} & {pc(b['precision'])} & {pc(b['recall'])} & "
                + ' & '.join(pc(sz[h]['recall']) for h in ('25-50', '50-100', '100-200', '>=200')))
write('tab_e1_camera', tab('Own-view behaviour of the fixed detector on the five-camera crop input before any fusion (1,655 frames). '
    'Each camera is matched only to its own annotations of height at least 25 px. Height columns give recall at confidence 0.25.',
    'tab:e1-camera', 'lrrrrrrrrr',
    'Camera & GT & P$_{.25}$ & R$_{.25}$ & P$_{.05}$ & R$_{.05}$ & 25--50 & 50--100 & 100--200 & $\\ge$200', rows, '\\footnotesize'))

cls = {(x['camera'], x['group']): x for x in F['klass']}
rows = [f"Cam{c} & " + ' & '.join(f"{pc(cls[(c,g)]['recall'])} ({n(cls[(c,g)]['n'])})" for g in ('person', 'two-wheeler', 'vehicle')) for c in (1, 2, 3, 4, 5)]
write('tab_e1_class', tab('Own-view recall by class group at confidence 0.25; annotation counts in parentheses.', 'tab:e1-class',
    'lrrr', 'Camera & Person & Two-wheeler & Vehicle', rows))

AD = {(x['annotated'], x['detected']): x['n'] for x in F['annotated_vs_detected']}
rows = []
for a in (1, 2, 3, 4):
    tot = sum(v for (aa, d), v in AD.items() if aa == a)
    rows.append(f'{a} & {n(tot)} & ' + ' & '.join(f"{100*AD.get((a,d),0)/tot:.1f}" if d <= a else '--' for d in range(5)))
write('tab_e1_crossview', tab('Cross-view consistency of independent detections (confidence 0.25). Rows: number of cameras in which an '
    'object-frame instance is annotated with height $\\ge$25 px. Columns: percentage of those instances detected in $d$ of their annotated cameras.',
    'tab:e1-crossview', 'rrrrrrr', 'Annotated & Instances & $d{=}0$ & $d{=}1$ & $d{=}2$ & $d{=}3$ & $d{=}4$', rows))

cmiss = {(x['i'], x['j']): x for x in F['conditional_miss']}
nums = {}
def agg(pairs):
    m = sum(cmiss[p]['miss'] for p in pairs); t = sum(cmiss[p]['n'] for p in pairs); return m/t, t
nums['miss_fwd'], nums['miss_fwd_n'] = agg([(i, k) for i in (2, 3, 4) for k in (2, 3, 4) if i != k])
nums['miss_pin_to_fish'], _ = agg([(i, k) for i in (2, 3, 4) for k in (1, 5) if cmiss[(i, k)]['n'] >= 100])
nums['miss_fish_to_pin'], _ = agg([(i, k) for i in (1, 5) for k in (2, 3, 4) if cmiss[(i, k)]['n'] >= 100])
crop = F['crop']; nums['crop_ceiling'] = crop['ceiling_recall']; nums['crop_out'] = crop['union_all_views_outside']
nums['fish'] = F['fisheye']; nums['union_n'] = crop['union_n']
single = sum(v for (a, d), v in AD.items() if a >= 2 and d == 1); multi = sum(v for (a, d), v in AD.items() if a >= 2)
nums['multi_annot_single_det'] = single/multi; nums['multi_annot_none'] = sum(v for (a, d), v in AD.items() if a >= 2 and d == 0)/multi
nums['multi_annot'] = multi

# ---------- E6 ----------
e6 = e1['full']['e6']
def r6(cfg, kind, b): return next(x for x in e6 if x['config'] == cfg and x['kind'] == kind and x['bin'] == b)
rows = []
for kind, bins in (('size', ['25-50', '50-100', '100-200', '>=200']), ('class', ['person', 'two-wheeler', 'vehicle']),
                   ('visible', ['forward only', 'forward + lateral', 'lateral only'])):
    for b in bins:
        lab = {'25-50': 'Height 25--50 px', '50-100': 'Height 50--100 px', '100-200': 'Height 100--200 px', '>=200': 'Height $\\ge$200 px'}.get(b, b.capitalize())
        rows.append(f"{lab} & {n(r6('raw',kind,b)['n'])} & " + ' & '.join(pc(r6(c, kind, b)['recall']) for c in ('raw', 'A', 'F5P', 'F5R'))
                    + f" & {100*r6('F5R',kind,b)['recall']/max(r6('raw',kind,b)['recall'],1e-9):.0f}")
    rows[-1] += '\\\\\\midrule' if kind != 'visible' else ''
rows = [r.replace('\\\\\\midrule', '\\\\ \\midrule') for r in rows]
s = tab('Recall by object height, class group and annotated-camera set on the five-view union (full sequence). The last column is the '
    'share of the box-corrected raw-pool recall retained by F5-R.', 'tab:e6-breakdown', 'lrrrrrr',
    'Subset & GT & Raw & A & F5-P & F5-R & Kept (\\%)', rows)
write('tab_e6_breakdown', s.replace('\\\\ \\midrule\\\\', '\\\\ \\midrule'))
pr6 = {(x['config'], x['bin']): x for x in e6 if x['kind'] == 'precision_class'}
nums['prec_class'] = {k[1]: v['precision'] for k, v in pr6.items() if k[0] == 'F5R'}

# ---------- E2 ----------
e2 = j('e2_simple_baselines/run01/results.json')
lock = e2['lock']
def row2(seg, m, p):
    return next(r for r in e2['segments'][seg]['rows'] if r['method'] == m and r['params'] == p)
spec = [('Raw pool, every detection an object', 'raw', []),
        ('B1 confidence gate, best F1 ($\\tau{=}0.30$)', 'B1', lock['B1']['best_f1']),
        ('B2 ray NMS, best F1 ($\\theta{=}5^\\circ,\\tau{=}0.20$)', 'B2', lock['B2']['best_f1']),
        ('B3 $\\ge$2-view voting, dev-P90 ($\\theta{=}5^\\circ,\\tau{=}0.40$)', 'B3', lock['B3']['p90']),
        ("B3$'$ voting or single conf.\\,$\\ge$0.9, dev-P90", 'B3p', lock['B3p']['p90']),
        ('Reference A', 'A', []), ('F5-P protected scene', 'F5P', []), ('F5-R repair + recovery (final)', 'F5R', [])]
rows = []
for lab, m, p in spec:
    a = row2('full', m, list(p)); b = row2('guard_late', m, list(p))
    rows.append(f"{lab} & {n(a['tp'])} & {n(a['fp'])} & {pc(a['precision'])} & {pc(a['recall'])} & {pc(a['f1'])} & {pc(b['precision'])} & {pc(b['recall'])}")
write('tab_e2_main', tab('Main comparison on identical box-corrected five-camera detections and the five-view union (24,845 instances). '
    'Baseline hyper-parameters were chosen on the development segment only. The B1/B2 rows use their best-F1 point because neither reaches '
    '90\\% development precision at non-trivial recall.', 'tab:main-results', '>{\\raggedright\\arraybackslash}p{.40\\linewidth}rrrrrrr',
    '\\multirow{2}{*}{Method} & \\multicolumn{5}{c}{Full sequence (1,655 frames)} & \\multicolumn{2}{c}{Guard+late}\\\\\\cmidrule(lr){2-6}\\cmidrule(l){7-8} & TP & FP & P & R & F1 & P & R', rows, '\\footnotesize'))
nums['e2_lock'] = lock
nums['e2_boot'] = {k: v['F5R_minus_this'] for k, v in e2['segments']['full']['bootstrap'].items()}

# ---------- E3 ----------
arms = [('none', 'Full residual head (575 features)'), ('no_appearance', 'Without appearance: ROI descriptor + HSV (432)'),
        ('no_geometry', 'Without own observation-ray geometry (5)'), ('no_crossview', 'Without cross-view support/competition (59)'),
        ('no_camera_id', 'Without camera identity (10)'), ('detector_only', 'Detector, box and class evidence only')]
rows = []; e3 = {}
for a, lab in arms:
    p = R/f'e3_feature_ablation/{a}/run01/results.json'
    if not p.exists():
        continue
    r = json.loads(p.read_text()); e3[a] = r
    rows.append(f"{lab} & {r['tau']:.3f} & {pc(r['full']['precision'])} & {pc(r['full']['recall'])} & {pc(r['full']['f1'])} & "
                f"{pc(r['guard_late']['precision'])} & {pc(r['guard_late']['recall'])}")
if 'none' in e3:
    base = json.loads((R/'e3_feature_ablation/none/run01/per_frame.json').read_text())
    for a in e3:
        if a == 'none': continue
        pf = json.loads((R/f'e3_feature_ablation/{a}/run01/per_frame.json').read_text())
        from importlib import import_module
        sys.path.insert(0, str(Path(__file__).parent)); L = import_module('supp_lib')
        e3[a]['boot_full'] = L.block_bootstrap(base['full'], pf['full'])
        e3[a]['boot_gl'] = L.block_bootstrap(base['guard_late'], pf['guard_late'])
    for k, (a, lab) in enumerate([x for x in arms if x[0] in e3]):
        if a != 'none':
            bf = e3[a]['boot_full']
            rows[k] += f" & {100*bf['point'][1]:+.2f} [{100*bf['lo'][1]:+.2f}, {100*bf['hi'][1]:+.2f}]"
        else:
            rows[k] += ' & --'
    write('tab_e3_ablation', tab('Evidence-type ablation of the residual recovery head. Each arm re-fits the head on the identical 601-frame samples '
        '(130,355 examples, 12,335 positives) with one feature group removed, locks $\\tau$ on development at $P\\ge0.91$, and replays the '
        'frozen repaired scene. The last column is the full-sequence recall change against the full head with a paired block-bootstrap 95\\% interval.',
        'tab:e3-ablation', '>{\\raggedright\\arraybackslash}p{.36\\linewidth}rrrrrrl',
        'Arm & $\\tau$ & P & R & F1 & P$_{gl}$ & R$_{gl}$ & $\\Delta$R (pp)', rows, '\\footnotesize'))
nums['e3'] = {a: dict(tau=r['tau'], full=r['full'], gl=r['guard_late'], boot=r.get('boot_full')) for a, r in e3.items()}

e3b = j('e3b_fisheye_output/run01/results.json')
rows = []
for cfg in ('F5P', 'F5R'):
    for seg in ('full', 'guard_late'):
        r = e3b[seg][cfg]
        rows.append(f"{'F5-P' if cfg=='F5P' else 'F5-R'}, {'full' if seg=='full' else 'guard+late'} & {pc(r['five_all']['precision'])}/{pc(r['five_all']['recall'])} & "
                    f"{pc(r['five_nofish']['precision'])}/{pc(r['five_nofish']['recall'])} & {pc(r['fwd_nofish']['precision'])}/{pc(r['fwd_nofish']['recall'])} & "
                    f"{n(r['fisheye_only_objects']['tp'])}/{n(r['fisheye_only_objects']['counted'])} ({pc(r['fisheye_only_objects']['precision'])})")
write('tab_e3b_fisheye', tab('Output-level contribution of the lateral fisheye cameras. Columns 2--3: P/R on the five-view union with and without '
    'Cam1/Cam5 members; column 4: forward members only, evaluated on the Cam234 union; column 5: TP/counted objects supported only by fisheye '
    'members (precision).', 'tab:e3b-fisheye', 'lrrrr', 'Output & Five-view, all & Five-view, no fisheye & Cam234, no fisheye & Fisheye-only objects', rows, '\\footnotesize'))
nums['e3b'] = e3b

# ---------- E4 ----------
e4 = j('e4_association/run01/results.json')
rows = []
for m, lab in (('B2', 'B2 ray NMS (geometry only)'), ('B3p', "B3$'$ voting"), ('A', 'Reference A'), ('F5P', 'F5-P'), ('F5R', 'F5-R (final)')):
    a = e4['full'][m]; b = e4['guard_late'][m]
    rows.append(f"{lab} & {n(a['groups']['multiview'])} & {pc(a['precision'])} & {pc(a['recall'])} & {pc(a['f1'])} & {pc(a['evaluable_fraction'])} & {pc(a['groups']['purity'])} & {pc(b['precision'])} & {pc(b['recall'])}")
write('tab_e4_association', tab('Cross-view association accuracy against annotation identities. Pair precision: fraction of identity-labelled '
    'same-object member pairs whose IDs agree; pair recall: fraction of same-ID member pairs placed in one object; evaluable: share of '
    'predicted pairs with both members labelled; purity: fully labelled multi-view groups with a single ID.', 'tab:e4-association',
    'lrrrrrrrr', 'Method & MV groups & P$_{pair}$ & R$_{pair}$ & F1 & Eval. & Purity & P$_{gl}$ & R$_{gl}$', rows, '\\footnotesize'))
rows = []
for k, v in e4['full']['F5R']['by_camera_pair'].items():
    if v['tp']+v['fp']+v['fn'] >= 5:
        rows.append(f"Cam{k.replace('-', '--Cam')} & {n(v['tp'])} & {n(v['fp'])} & {n(v['fn'])} & {pc(v['precision'])} & {pc(v['recall'])}")
write('tab_e4_pairs', tab('F5-R association by camera pair (full sequence).', 'tab:e4-pairs', 'lrrrrr',
    'Camera pair & Correct & Wrong & Split & P$_{pair}$ & R$_{pair}$', rows))
nums['e4'] = {m: {k: e4[s][m][k] for k in ('precision', 'recall', 'f1', 'evaluable_fraction')} | {'purity': e4[s][m]['groups']['purity'],
              'pf': e4[s][m]['pinhole_fisheye_pairs'], 'groups': e4[s][m]['groups']} for s in ('full',) for m in e4[s]}
au = j('e4_identity_audit/run01/results.json')
rows = []
for st, lab in (('S3_random', 'S3 random multi-view groups'), ('S1_pure', 'S1 annotation-consistent'), ('S2_impure', 'S2 annotation-inconsistent'), ('all', 'All cases')):
    a = au[st]; w = a['wilson95']
    rows.append(f"{lab} & {a['n']} & {a['same']} & {a['different']} & {a['uncertain']} & {pc(a['precision'])} [{pc(w[0])}, {pc(w[1])}]")
write('tab_e4_audit', tab('Blind visual identity check of F5-R multi-view groups sampled from the guard and late segments. Precision excludes '
    'uncertain cases; Wilson 95\\% interval in brackets. Labels were assigned by an AI assistant from crop sheets without IDs or strata and '
    'await confirmation by the candidate (Appendix~\\ref{app:supp}).', 'tab:e4-audit', 'lrrrrl', 'Stratum & $n$ & Same & Different & Uncertain & Precision (\\%)', rows))
nums['audit'] = au

# ---------- E5 ----------
sc = j('e5_ap/run01/single_camera.json'); ga = j('e5_ap/run01/group_ap.json')
rows = [f"Cam{c} & {sc['strict'][str(c)]['AP50']*100:.1f} & {sc['strict'][str(c)]['AP75']*100:.1f} & {sc['strict'][str(c)]['AP50_95']*100:.1f} & "
        f"{sc['merged'][str(c)]['AP50']*100:.1f} & {sc['merged'][str(c)]['AP50_95']*100:.1f}" for c in (1, 2, 3, 4, 5)]
write('tab_e5_single', tab('COCO-style average precision of the fixed YOLO11n detector on each camera\'s own annotations (crop input, '
    'confidence floor 0.05, 101-point interpolation, boxes under 25 px treated as ignore regions). Strict: six COCO classes; merged: '
    'person / two-wheeler / vehicle.', 'tab:e5-single', 'lrrrrr', 'Camera & AP$_{50}$ & AP$_{75}$ & AP$_{50:95}$ & AP$_{50}$ merged & AP$_{50:95}$ merged', rows))
curves = {m: ga[m]['full']['curve50'] for m in ga}
def rec_at(m, p):
    return max((q['recall'] for q in curves[m] if q['precision'] >= p), default=0.)
def prec_at(m, r):
    return max((q['precision'] for q in curves[m] if q['recall'] >= r), default=float('nan'))
rows = []
for m, lab in (('B1', 'B1 all detections'), ('B2', 'B2 ray NMS ($\\theta{=}5^\\circ$)'), ('A', 'Reference A'), ('F5P', 'F5-P'), ('F5R', 'F5-R (final)')):
    g = ga[m]['full']; gl = ga[m]['guard_late']
    rows.append(f"{lab} & {g['AP50']*100:.1f} & {g['AP75']*100:.1f} & {g['AP50_95']*100:.1f} & {g['mAP50_class_groups']*100:.1f} & {gl['AP50']*100:.1f} & "
                f"{100*rec_at(m, .90):.1f} & " + (f"{100*prec_at(m, .4485):.1f}" if prec_at(m, .4485) == prec_at(m, .4485) else '--'))
write('tab_e5_group', tab('Group-level average precision under the dissertation protocol (objects ranked by maximum member confidence). '
    'mAP$_{50}^{cls}$ averages person, two-wheeler and vehicle AP$_{50}$. The last two columns read the full-sequence PR curve: '
    'maximum recall at precision $\\ge$90\\%, and precision at the F5-R recall of 44.85\\%.', 'tab:e5-group', 'lrrrrrrr',
    'Method & AP$_{50}$ & AP$_{75}$ & AP$_{50:95}$ & mAP$_{50}^{cls}$ & AP$_{50}^{gl}$ & R@P90 & P@R44.85', rows, '\\footnotesize'))
nums['e5'] = {m: dict(AP50=ga[m]['full']['AP50'], AP50_95=ga[m]['full']['AP50_95'], R_at_P90=rec_at(m, .9), P_at_R4485=prec_at(m, .4485),
                      per_class=ga[m]['full'].get('per_class_AP50')) for m in ga}
nums['e5_single'] = {k: {c: dict(AP50=v[c]['AP50'], AP50_95=v[c]['AP50_95']) for c in v} for k, v in sc.items()}
(OUT/'numbers.json').write_text(json.dumps(nums, indent=1, default=str))
print('tables ->', OUT)
