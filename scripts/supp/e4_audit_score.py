"""Score the visual identity audit.

Each case carries two labels in labels.csv: visual_label, assigned by the AI assistant from the blind crop
sheets, and candidate_label, the author's confirmed label. Re-running this script keeps the author's column,
and the scores use the confirmed label wherever it is filled (otherwise the assistant's label)."""
import csv, math
from supp_lib import *
D = OUT_ROOT/'e4_identity_audit/run01'
cases = json.loads((D/'cases.json').read_text())
DIFFERENT = {'c031': 'Cam4 member is a white-top person, Cam3 member a red-top person'}
UNCERTAIN = {'c019': 'small, overexposed', 'c041': 'Cam2 crop too dark to compare', 'c047': 'small, blurred',
             'c058': 'blurred, clothing colour inconsistent', 'c065': 'small, colour inconsistent', 'c069': 'small, blurred',
             'c094': 'very dark', 'c102': 'small, blurred', 'c106': 'pose differs; possibly two people', 'c115': 'very dark',
             'c120': 'occluded by pole'}
LABELS = ('same', 'different', 'uncertain')
PENDING = 'AI-assisted visual check (Claude); candidate confirmation pending'
CONFIRMED = 'AI-assisted visual check (Claude); confirmed by the candidate'

# the author's confirmations already recorded in labels.csv are kept
confirmed = {}
if (D/'labels.csv').exists():
    with (D/'labels.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            lab = (r.get('candidate_label') or '').strip().lower()
            if lab:
                assert lab in LABELS, f"{r['case']}: candidate_label must be one of {LABELS}, got {lab!r}"
                confirmed[r['case']] = lab

rows = []
for c in cases:
    lab = 'different' if c['case'] in DIFFERENT else ('uncertain' if c['case'] in UNCERTAIN else 'same')
    ids = [m['gt_id'] for m in c['members']]
    auto = 'unlabelled' if any(i is None for i in ids) else ('same' if len(set(ids)) == 1 else 'different')
    cand = confirmed.get(c['case'], '')
    rows.append(dict(case=c['case'], stratum=c['stratum'], frame=c['frame'], cameras='|'.join(str(m['src']) for m in c['members']),
                     annotation_label=auto, visual_label=lab, note=DIFFERENT.get(c['case'], UNCERTAIN.get(c['case'], '')),
                     reviewer=CONFIRMED if cand else PENDING, candidate_label=cand))
with (D/'labels.csv').open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def final(r):
    return r['candidate_label'] or r['visual_label']
def wilson(k, n, z=1.96):
    if n == 0: return None
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h, c+h]
res = {}
for st in ('S1_pure', 'S2_impure', 'S3_random', 'all'):
    R = [r for r in rows if st == 'all' or r['stratum'] == st]
    same = sum(final(r) == 'same' for r in R); diff = sum(final(r) == 'different' for r in R); unc = len(R)-same-diff
    res[st] = dict(n=len(R), same=same, different=diff, uncertain=unc, precision=same/max(same+diff, 1), wilson95=wilson(same, same+diff),
                   precision_uncertain_as_wrong=same/max(len(R), 1), wilson95_conservative=wilson(same, len(R)))
conf = {}
for r in rows:
    conf.setdefault(r['annotation_label'], {}).setdefault(final(r), 0); conf[r['annotation_label']][final(r)] += 1
res['annotation_vs_visual'] = conf
groups = {(c['frame'], tuple(sorted(m['uid'] for m in c['members']))) for c in cases}
n_conf = sum(bool(r['candidate_label']) for r in rows)
n_changed = sum(bool(r['candidate_label']) and r['candidate_label'] != r['visual_label'] for r in rows)
res['labels'] = dict(cases=len(rows), distinct_groups=len(groups), confirmed_by_candidate=n_conf,
                     changed_by_candidate=n_changed, pending=len(rows)-n_conf)
res['note'] = ('visual_label: assigned by the AI assistant from blind crop sheets (no IDs or strata shown); '
               'candidate_label: the author\'s confirmed label. Scores use the confirmed label where present. '
               'Strata are sampled at different rates, so "all" pools them and is not a population estimate; '
               'the random stratum S3 estimates the share of correct groups. '
               'Cases c049 (S3) and c071 (S1) are the same group, so the cases cover distinct_groups groups.')
dump(D/'results.json', res); print(json.dumps(res, indent=1))
