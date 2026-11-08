"""Append-only full-domain loss audit and selective residual recovery study.

Fixed YOLO=.05,960,no BAND. Reuses verified full low-confidence outputs. No new
depth or detector inference; fitting only post-hoc residual scoring on early GT.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'ultralytics'
sys.path[:0] = [str(BASE/'research'), str(BASE/'core_algorithms'), str(ROOT/'scripts')]
import shared_geometry as S
import residual_recovery as R
import cam15_scoring as Q
from experiment_cam15_shared import FrameEvaluator
from cache_low_conf_no_band import sha
from validate_multiview import truth, metrics, evaluate
from multiview_geometry import load_cameras
from report_low_conf_no_band import paired_bootstrap

PARENT = BASE/'experiments/low_conf_no_band_20261009/full'
CACHE = BASE/'experiments/low_conf_no_band_20261009/cache/detections'
TAUS = (.25, .4, .5, .6, .7, .8, .9, .95)
SCOPES = ('safe_side', 'safe_all', 'open_all')


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def objects(groups, by_uid):
    return [dict(src=sorted({by_uid[u]['src'] for u in g}), cls=by_uid[g[0]]['cls'],
                 dets=[by_uid[u] for u in g]) for g in groups]


def audit_stage(evaluator, objs):
    score = np.array([np.maximum.reduce([evaluator.scores[d['uid']] for d in o['dets']])
                      for o in objs]) if objs else np.zeros((0, len(evaluator.keys)))
    rr, cc = linear_sum_assignment(-((score >= .5)*1000+score*(score >= .5))) if score.size else ([], [])
    assigned = {evaluator.keys[c] for r, c in zip(rr, cc) if score[r, c] >= .5}
    hit = {evaluator.keys[j] for j in range(len(evaluator.keys)) if len(score) and np.any(score[:, j] >= .5)}
    return dict(counts=evaluator.counts(objs), hit_gt=sorted(hit), assigned_gt=sorted(assigned),
                capacity_deficit=len(hit)-len(assigned))


def load_frame(path, cameras):
    parent = json.loads(path.read_text())
    stem = path.stem
    raw, checksums = [], {}
    for c in (2, 3, 4, 1, 5):
        cp = CACHE/f'{c}_{stem}.json'
        digest = sha(cp)
        if parent['input_sha256'][str(cp.relative_to(ROOT))] != digest:
            raise ValueError('Detector input changed')
        checksums[str(cp.relative_to(ROOT))] = digest
        raw.extend(json.loads(cp.read_text())['dets'])
    # UID ordering exactly matches frozen parent. Keep .01 universe, filter later.
    pool = [S.prepare_detection(dict(d, uid=i), cameras) for i, d in enumerate(raw)]
    by_uid = {d['uid']:d for d in pool}
    grouped = parent['object_uids']
    baseline = objects(grouped['conf0.05/fish_full075'], by_uid)
    append = objects(grouped['conf0.05/five_append_control'], by_uid)
    front = objects(grouped['conf0.05/fusion_frozen'], by_uid)
    singles = [o for c in (2, 3, 4) for o in objects(grouped[f'conf0.05/cam{c}'], by_uid)]
    side = [o for o in append if o['src'][0] in (1, 5)]
    unknown = {d['uid'] for d in parent['diagnostics']['0.05']['diagnostic_unknown']}
    preprocessed = singles+side
    allowed = {d['uid'] for o in preprocessed for d in o['dets']}
    proposals = R.candidates(pool, baseline, allowed, unknown, cameras)
    stages = dict(raw=[dict(src=[d['src']], dets=[d]) for d in pool if d['conf'] >= .05],
                  native_preprocessed=preprocessed, front_filtered_plus_side=append,
                  no_deferred=[o for o in append if not (len(o['dets']) == 1 and o['dets'][0]['uid'] in unknown)],
                  final=baseline)
    return parent, pool, baseline, proposals, stages, checksums


def extract(args):
    cv2.setNumThreads(1)
    cfg = json.loads((PARENT/'protocol.json').read_text())
    names = cfg['frames'][:args.limit] if args.limit else cfg['frames']
    sources = [Path(__file__), BASE/'research/residual_recovery.py', BASE/'research/cam15_scoring.py',
               BASE/'core_algorithms/validate_multiview.py', ROOT/'scripts/experiment_cam15_shared.py']
    protected = [ROOT/'README.md', ROOT/'THESIS_REFERENCES.md',
                 *sorted((BASE/'algorithm').rglob('*.md')), *sorted((BASE/'algorithm/evaluation_reports').glob('*.json'))]
    protocol = dict(frames=names, yolo_conf=.05, imgsz=960, band=False,
        baseline='conf0.05/fish_full075', parent_protocol_sha256=sha(PARENT/'protocol.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        protected_sha256={str(p.relative_to(ROOT)):sha(p) for p in protected},
        calibration_sha256={str(p):sha(ROOT/p) for p in ('Intrinsics_cam.json', 'T_imu_cam(1).json')},
        train=[s for s in names if int(s) <= 600], development=[s for s in names if 600 < int(s) <= 800],
        guard=[s for s in names if 800 < int(s) < 1100], late=[s for s in names if int(s) >= 1100],
        taus=TAUS, scopes=SCOPES, selection='On development only: max Recall subject to P >= baseline P - 0.5pp; else no change.',
        training_target='Eligible any-member new-GT hit not covered by baseline; duplicate real-object observations are negative. Ambiguous multi-GT hits ignored.',
        limitations='Previously explored single route, NOT blind. New post-hoc scorer, NOT detector retraining. '
            'Native same-camera overlap guards only, no new metric depth. Cross-view duplicates remain possible. '
            'Safe scopes exclude already geometrically deferred side observations. Baseline never removed. '
            'A bearing-only observation is not a guaranteed new identity or a BEV point. '
            'Stage audit uses frozen-front-plus-unmerged-side control, NOT exact five-score-stage causal attribution.')
    pp = args.out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Changed protocol: new output directory required')
    if not pp.exists():
        write_new(pp, protocol)
    if args.prepare:
        print(f'Prepared {len(names)} frames', flush=True)
        return
    index = {p.stem:p for p in PARENT.glob('shard*/frames/*.json')}
    cameras = load_cameras()
    frames = names[args.shard_index::args.shard_count]
    mp = args.out/f'shard{args.shard_index}/manifest.json'
    manifest = dict(frames=frames, protocol_sha256=sha(pp), shard_index=args.shard_index, shard_count=args.shard_count)
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Changed partition')
    if not mp.exists():
        write_new(mp, manifest)
    start = time.perf_counter()
    for n, stem in enumerate(frames, 1):
        rp = mp.parent/'frames'/f'{stem}.json'
        if rp.exists():
            continue
        parent, pool, baseline, proposals, stages, checksums = load_frame(index[stem], cameras)
        # No GT above: all inference inputs/features have been frozen first.
        gt = truth(stem, (1, 2, 3, 4, 5))
        for c in (1, 2, 3, 4, 5):
            xp = BASE/f'mycar/cam_{c}/{stem}.xml'
            key = str(xp.relative_to(ROOT))
            digest = sha(xp)
            if parent['input_sha256'][key] != digest:
                raise ValueError('GT changed')
            checksums[key] = digest
        evaluator = FrameEvaluator(gt, pool)
        stage_audit = {k:audit_stage(evaluator, v) for k, v in stages.items()}
        if stage_audit['final']['counts'] != parent['counts']['conf0.05/fish_full075']:
            raise ValueError('Baseline replay mismatch')
        if stage_audit['raw']['counts'] != parent['raw_coverage']['0.05']:
            raise ValueError('Raw replay mismatch')
        covered = set(stage_audit['final']['hit_gt'])
        for p in proposals:
            d = p['detection']
            hits = {evaluator.keys[j] for j in np.flatnonzero(evaluator.scores[d['uid']] >= .5)}
            label = (-1 if len(hits) > 1 or (not hits and d['uid'] in evaluator.small_hits)
                     else int(bool(hits-covered)))
            p['label'] = label  # Training/evaluation record ONLY, not inference input.
            p['hit_gt'] = sorted(hits)
            p['x'] = p['x'].tolist()
            # Only fields needed for downstream inference; finite JSON, no rays/NaNs.
            p['detection'] = {k:d[k] for k in ('uid', 'src', 'cls', 'box', 'conf')}
        checksums[str(index[stem].relative_to(ROOT))] = sha(index[stem])
        write_new(rp, dict(frame=stem, proposals=proposals, stages=stage_audit,
            baseline_uids=[[d['uid'] for d in o['dets']] for o in baseline], input_sha256=checksums,
            parent_counts=parent['counts'], gt=len(evaluator.keys)))
        if n == 1 or n % 100 == 0 or n == len(frames):
            print(f'{n}/{len(frames)} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    write_new(mp.parent/'completion.json', dict(done=len(frames), planned=len(frames)))


def fit_eval(args):
    pp = args.out/'protocol.json'
    cfg = json.loads(pp.read_text())
    for field in ('source_sha256', 'protected_sha256', 'calibration_sha256'):
        for p, digest in cfg[field].items():
            if sha(ROOT/p) != digest:
                raise ValueError(f'Changed {field}: {p}')
    if sha(PARENT/'protocol.json') != cfg['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    from report_low_conf_no_band import verify_shards
    verify_shards(args.out, cfg['frames'], sha(pp))
    paths = sorted(args.out.glob('shard*/frames/*.json'), key=lambda p:p.stem)
    if [p.stem for p in paths] != cfg['frames']:
        raise ValueError('Incomplete extraction')
    records = [json.loads(p.read_text()) for p in paths]
    # Training is ONLY <=600. Development selection ONLY 601..800.
    train = [p for r in records if r['frame'] in cfg['train'] for p in r['proposals'] if p['label'] >= 0]
    model = Q.fit([p['x'] for p in train], [p['label'] for p in train])
    model['target'] = cfg['training_target']
    model_path = args.out/'residual_model.json'
    write_new(model_path, dict(model=model, protocol_sha256=sha(pp), train_frames=cfg['train']))
    rows = []
    start = time.perf_counter()
    for n, r in enumerate(records, 1):
        stem = r['frame']
        for p, digest in r['input_sha256'].items():
            if sha(ROOT/p) != digest:
                raise ValueError('Input changed during evaluation')
        by_uid = {p['detection']['uid']:p['detection'] for p in r['proposals']}
        raw = []
        for c in (2, 3, 4, 1, 5):
            raw.extend(json.loads((CACHE/f'{c}_{stem}.json').read_text())['dets'])
        full = {i:dict(d, uid=i) for i, d in enumerate(raw)}
        baseline = objects(r['baseline_uids'], full)
        scores = Q.predict([p['x'] for p in r['proposals']], model)
        # Deliberately strip training labels before inference.
        proposals = [{k:p[k] for k in ('detection', 'deferred', 'x')} for p in r['proposals']]
        outputs = {f'{scope}/tau{t}':R.recover(baseline, proposals, scores, t, scope)
                   for scope in SCOPES for t in TAUS}
        outputs['baseline'] = baseline
        # Models/outputs complete before GT load; no fit or selection on late.
        gt = truth(stem, (1, 2, 3, 4, 5))
        evaluator = FrameEvaluator(gt, list(full.values()))
        counts = {k:evaluator.counts(v) for k, v in outputs.items()}
        counts.update(front=r['parent_counts']['conf0.05/fusion_frozen'],
                      append=r['parent_counts']['conf0.05/five_append_control'])
        rows.append(dict(frame=stem, counts=counts, identity={k:evaluator.identity(v) for k, v in outputs.items()}))
        if n == 1 or n % 250 == 0:
            print(f'Evaluation {n}/{len(records)} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    def totals(rs):
        return {k:metrics(np.sum([r['counts'][k] for r in rs], axis=0)) for k in rows[0]['counts']}
    splits = {name:totals([r for r in rows if name == 'all' or r['frame'] in cfg[name]])
              for name in ('all', 'train', 'development', 'guard', 'late')}
    dev = splits['development']
    eligible = [k for k in dev if k.startswith(('safe_all/', 'safe_side/')) and
                dev[k]['precision'] >= dev['baseline']['precision']-.005 and
                dev[k]['recall'] > dev['baseline']['recall']]
    selected = max(eligible, key=lambda k:(dev[k]['recall'], dev[k]['precision'])) if eligible else 'baseline'
    identity = {name:{k:dict(sum((Counter(r['identity'][k]) for r in rows
                                if name == 'all' or r['frame'] in cfg[name]), Counter())) for k in rows[0]['identity']}
                for name in ('all', 'development', 'late')}
    audit_totals = {k:metrics(np.sum([r['stages'][k]['counts'] for r in records], axis=0)) for k in records[0]['stages']}
    losses = Counter()
    examples = []
    for r in records:
        s = r['stages']
        raw, pre, front, no_deferred, final = [set(s[k]['hit_gt']) for k in
            ('raw', 'native_preprocessed', 'front_filtered_plus_side', 'no_deferred', 'final')]
        classes = dict(native_preprocessing=raw-pre, front_filter_or_group=pre-front,
                       side_deferred=front-no_deferred, later_score_or_group=no_deferred-final)
        for reason, ids in classes.items():
            losses[reason] += len(ids)
            if len(examples) < 100:
                examples.extend(dict(frame=r['frame'], gt_id=i, reason=reason) for i in sorted(ids)[:3])
        losses['raw_capacity_deficit'] += s['raw']['capacity_deficit']
        losses['final_capacity_deficit'] += s['final']['capacity_deficit']
        losses['hit_set_loss'] += len(raw-final)
        losses['final_new_hit_not_raw'] += len(final-raw)
    # Exact algebra: TP difference = hit-set loss + final capacity deficit - raw deficit.
    expected = losses['hit_set_loss']-losses['final_new_hit_not_raw']+losses['final_capacity_deficit']-losses['raw_capacity_deficit']
    if expected != audit_totals['raw']['tp']-audit_totals['final']['tp']:
        raise ValueError('Loss decomposition failed')
    result = dict(complete=True, frames=len(rows), gt=sum(r['gt'] for r in records), protocol=cfg,
        model_sha256=sha(model_path), selection=dict(method=selected, used_partition='development', eligible=eligible),
        splits=splits, identity=identity, stage_totals=audit_totals, loss_hit_sets=dict(losses),
        loss_examples=examples, selected_vs_baseline={name:dict(
            delta_pp={k:100*(splits[name][selected][k]-splits[name]['baseline'][k]) for k in ('precision', 'recall', 'f1')},
            ci95=paired_bootstrap([r for r in rows if name == 'all' or r['frame'] in cfg[name]],
                                  'counts', selected, 'baseline')) for name in ('all', 'late')},
        evaluation_rows=rows, extraction_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    write_new(args.report, result)
    print(json.dumps({k:result[k] for k in ('frames', 'gt', 'selection', 'stage_totals', 'loss_hit_sets', 'selected_vs_baseline')}, ensure_ascii=False), flush=True)
    print(json.dumps(splits['all'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('extract', 'fit_eval'))
    ap.add_argument('--out', type=Path, default=BASE/'experiments/recall_recovery_20261009/run01')
    ap.add_argument('--report', type=Path, default=BASE/'algorithm/evaluation_reports/recall_recovery_results_20261009.json')
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=4)
    ap.add_argument('--shard-index', type=int, default=0)
    args = ap.parse_args()
    if args.limit < 0 or not 0 <= args.shard_index < args.shard_count:
        ap.error('Invalid domain')
    (extract if args.stage == 'extract' else fit_eval)(args)
