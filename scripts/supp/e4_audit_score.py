"""Score the visual identity audit (AI-assisted visual inspection, pending candidate confirmation)."""
import csv, math
from supp_lib import *
D = OUT_ROOT/'e4_identity_audit/run01'
cases = json.loads((D/'cases.json').read_text())
DIFFERENT = {'c031': 'Cam4 member is a white-top person, Cam3 member a red-top person'}
UNCERTAIN = {'c019': 'small, overexposed', 'c041': 'Cam2 crop too dark to compare', 'c047': 'small, blurred',
             'c058': 'blurred, clothing colour inconsistent', 'c065': 'small, colour inconsistent', 'c069': 'small, blurred',
             'c094': 'very dark', 'c102': 'small, blurred', 'c106': 'pose differs; possibly two people', 'c115': 'very dark',
             'c120': 'occluded by pole'}
rows = []
for c in cases:
    lab = 'different' if c['case'] in DIFFERENT else ('uncertain' if c['case'] in UNCERTAIN else 'same')
    ids = [m['gt_id'] for m in c['members']]
    auto = 'unlabelled' if any(i is None for i in ids) else ('same' if len(set(ids)) == 1 else 'different')
    rows.append(dict(case=c['case'], stratum=c['stratum'], frame=c['frame'], cameras='|'.join(str(m['src']) for m in c['members']),
                     annotation_label=auto, visual_label=lab, note=DIFFERENT.get(c['case'], UNCERTAIN.get(c['case'], '')),
                     reviewer='AI-assisted visual check (Claude); candidate confirmation pending', candidate_label=''))
with (D/'labels.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h, c+h]
res = {}
for st in ('S1_pure', 'S2_impure', 'S3_random', 'all'):
    R = [r for r in rows if st == 'all' or r['stratum'] == st]
    same = sum(r['visual_label'] == 'same' for r in R); diff = sum(r['visual_label'] == 'different' for r in R); unc = len(R)-same-diff
    res[st] = dict(n=len(R), same=same, different=diff, uncertain=unc, precision=same/max(same+diff, 1), wilson95=wilson(same, same+diff),
                   precision_uncertain_as_wrong=same/max(len(R), 1), wilson95_conservative=wilson(same, len(R)))
conf = {}
for r in rows:
    conf.setdefault(r['annotation_label'], {}).setdefault(r['visual_label'], 0); conf[r['annotation_label']][r['visual_label']] += 1
res['annotation_vs_visual'] = conf
res['note'] = ('Visual labels were assigned by the AI assistant from blind crop sheets (no IDs or strata shown); '
               'they are not a human audit and must be confirmed by the candidate (column candidate_label).')
dump(D/'results.json', res); print(json.dumps(res, indent=1))
