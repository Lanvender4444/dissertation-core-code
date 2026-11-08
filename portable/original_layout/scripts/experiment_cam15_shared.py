"""Predeclared five-camera ablations; inference is GT-free, evaluation separate.

All variants share fixed detections, side NMS/ego rules and front parent outputs.
Never selects a configuration from the late evaluation partition.
"""
import argparse
import copy
import hashlib
import itertools
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
REPORTS = BASE/'algorithm/evaluation_reports'
sys.path[:0] = [str(BASE/'research'), str(BASE/'core_algorithms'), str(BASE/'decided'), str(ROOT/'scripts')]
import shared_geometry as S
from multiview_geometry import load_cameras
from validate_multiview import truth, ev, evaluate, metrics
from yolo_hr_detect import merge_nms
from validation_inputs import load_stage1_verified
from experiment_global_fusion import EXPECTED_MODEL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def variants():
    configs = {'front_frozen': dict(mode='front'),
               'cam1_only': dict(mode='single', camera=1, threshold=.15),
               'cam5_only': dict(mode='single', camera=5, threshold=.2)}
    for threshold in (.2, .5, .75, .9):
        configs[f'append_t{threshold}'] = dict(mode='append', threshold=threshold)
    for mode, gate, threshold in itertools.product(
            ('best_pair', 'centre', 'expanded', 'independent_members', 'object_joint',
             'independent_interval', 'shared_difference', 'shared_global'), (.45, .75), (.2, .5, .75, .9)):
        name = f'{mode}_g{gate}_t{threshold}'
        configs[name] = dict(mode=mode, gate=gate, threshold=threshold, margin=.02)
    # Explicit exploratory front veto, NOT a frozen-front or default variant.
    for mode, veto in itertools.product(('object_joint', 'shared_global'), (.8, 1.5)):
        configs[f'{mode}_g0.75_t0.5_veto{veto}'] = dict(mode=mode, gate=.75, threshold=.5,
                                                     margin=.02, front_veto=veto)
    return configs


class FrameEvaluator:
    """Same historical any-member-IoU object criterion; separate identity proxy."""
    def __init__(self, gt, detections):
        self.gt = gt
        self.union = ev.build_union(gt, 25)
        self.small = ev.build_union(gt, 0)
        self.keys = list(self.union)
        self.scores = {}
        self.small_hits = set()
        self.assigned = {}
        self.ambiguous = set()
        self.dets = {d['uid']:d for d in detections}
        for uid, d in self.dets.items():
            self.scores[uid] = np.array([ev.iou(d['box'], g['boxes'][d['src']])
                if d['src'] in g['boxes'] and ev.cls_match(g['label'], d['cls']) else 0.
                for g in self.union.values()])
            if any(iid not in self.union and d['src'] in g['boxes'] and ev.cls_match(g['label'], d['cls'])
                   and ev.iou(d['box'], g['boxes'][d['src']]) >= .5 for iid, g in self.small.items()):
                self.small_hits.add(uid)
        for cid in gt:
            ds = [d for d in self.dets.values() if d['src'] == cid]
            gs = [(iid, lab, box) for iid, lab, box in gt[cid] if iid in self.union]
            score = np.zeros((len(ds), len(gs)))
            for i, d in enumerate(ds):
                for j, (_, label, box) in enumerate(gs):
                    if ev.cls_match(label, d['cls']):
                        score[i, j] = ev.iou(d['box'], box)
                if np.sum(score[i] >= .5) > 1:
                    self.ambiguous.add(d['uid'])
            if score.size:
                rr, cc = linear_sum_assignment(-((score >= .5)*1000+score*(score >= .5)))
                self.assigned.update({ds[i]['uid']:gs[j][0] for i, j in zip(rr, cc) if score[i, j] >= .5})

    def counts(self, objects):
        scores = np.zeros((len(objects), len(self.keys)))
        ignored = wrong = multi = 0
        for i, o in enumerate(objects):
            if o['dets']:
                scores[i] = np.maximum.reduce([self.scores[d['uid']] for d in o['dets']])
            hits = int(np.sum(scores[i] >= .5))
            if len(o['src']) > 1:
                multi += 1
                wrong += hits > 1
            if not hits and any(d['uid'] in self.small_hits for d in o['dets']):
                ignored += 1
        rr, cc = linear_sum_assignment(-((scores >= .5)*1000+scores*(scores >= .5))) if scores.size else ([], [])
        tp = sum(scores[i, j] >= .5 for i, j in zip(rr, cc))
        return [int(tp), len(objects)-ignored-int(tp), len(self.keys)-int(tp), wrong, multi]

    def identity(self, objects):
        out = Counter()
        matched_groups = Counter()
        for o in objects:
            known = {self.assigned[d['uid']] for d in o['dets'] if d['uid'] in self.assigned}
            out['mixed_known_groups'] += len(known) > 1
            for iid in known:
                matched_groups[iid] += 1
            for a, b in itertools.combinations(o['dets'], 2):
                if a['src'] == b['src']:
                    continue
                out['links'] += 1
                side = a['src'] in (1, 5) or b['src'] in (1, 5)
                out['side_links'] += side
                ia, ib = self.assigned.get(a['uid']), self.assigned.get(b['uid'])
                if a['uid'] in self.ambiguous or b['uid'] in self.ambiguous:
                    state = 'ambiguous'
                elif ia is None or ib is None:
                    state = 'unresolved'
                else:
                    state = 'correct' if ia == ib else 'incorrect'
                out[state+'_links'] += 1
                if side:
                    out[state+'_side_links'] += 1
        out['gt_split_across_groups'] = sum(n > 1 for n in matched_groups.values())
        out['extra_groups_for_known_gt'] = sum(max(0, n-1) for n in matched_groups.values())
        return dict(out)


def make_protocol(args, names, cfg, parent_hashes):
    return dict(date='2026-10-08', configs=variants(), frames=names,
                development_frames=[s for s in names if int(s) <= 800],
                guard_frames=[s for s in names if 800 < int(s) < 1100],
                evaluation_frames=[s for s in names if int(s) >= 1100],
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                    (Path(__file__), BASE/'research/shared_geometry.py', ROOT/'scripts/cache_cam15_research.py')},
                calibration_sha256={p:sha(ROOT/p) for p in ('Intrinsics_cam.json', 'T_imu_cam(1).json')},
                front_parent_manifests={str(p.relative_to(ROOT)):sha(p) for p in
                    sorted((BASE/'experiments/no_band_20261007_shards').glob('shard*/manifest.json'))},
                verified_stage1_parent_hashes=parent_hashes, front_params=cfg['params'],
                depth_samples=12, side_ego='fixed two corner triangles and bottom-centre y>.97; vehicles only',
                side_preprocessing='same existing vehicle-compatible merge_nms for ALL variants',
                success_pp=dict(precision=2., recall=2.),
                primary_comparisons=['matching-threshold five-camera append', 'frozen front on FIVE-camera GT'],
                limitations='Same previously inspected route, NOT blind or independent-scene evaluation. '
                    'Global XML identity semantics unconfirmed. Identity measures are annotation proxies, '
                    'ambiguous overlaps excluded from known-link precision. Certificates apply to sampled states only. '
                    'Front positions reconstructed from box geometry/size hypotheses, not claimed reused SGBM points. '
                    'Detection/parent inputs frozen; no parameter selection from evaluation/all-frame results.',
                selection='Development only: highest F1; separately highest F1 satisfying +2pp P and +2pp R '
                    'versus both its append baseline and frozen front; veto variant distinguished explicitly.')


def run(args):
    cv2.setNumThreads(1)
    cfg, _, cache, parent_hashes = load_stage1_verified()
    available = cfg['available_frames']
    names = available[:args.limit] if args.limit else available
    protocol = make_protocol(args, names, cfg, parent_hashes)
    pp = args.out/'protocol.json'
    if pp.exists() and json.loads(pp.read_text()) != protocol:
        raise ValueError('Changed study: use another output directory')
    if not pp.exists():
        dump(pp, protocol)
    if args.prepare:
        print(f'Prepared {len(names)} frames / {len(protocol["configs"])} fixed configurations', flush=True)
        return
    configs = protocol['configs']
    cameras = load_cameras()
    parent_index = {}
    signed_inputs = {}
    manifests = []
    for path in sorted((BASE/'experiments/no_band_20261007_shards').glob('shard*/manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest['limit'] or manifest['source_sha256'] != sha(ROOT/'scripts/validate_no_band.py'):
            raise ValueError('Unverified no-band parent')
        if manifest['available_frames'] != available:
            raise ValueError('Different front domain')
        parent_index.update({s:path.parent/'frames'/f'{s}.json' for s in manifest['frames']})
        signed_inputs.update(manifest['input_sha256'])
        manifests.append(manifest)
    frames = names[args.shard_index::args.shard_count]
    manifest = dict(shard_index=args.shard_index, shard_count=args.shard_count, frames=frames,
                    limit=args.limit, protocol_sha256=sha(pp))
    mp = args.out/f'shard{args.shard_index}/manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Changed shard')
    dump(mp, manifest)
    started = time.perf_counter()
    for n, stem in enumerate(frames, 1):
        output = mp.parent/'frames'/f'{stem}.json'
        if output.exists():
            continue
        front_record = json.loads(parent_index[stem].read_text())
        shard, _ = cache.frames[stem]
        raw_front = []
        front_input_hashes = {}
        for c in (2, 3, 4):
            cp = shard/f'detections/960/{c}_{stem}.json'
            key = str(cp.relative_to(ROOT))
            actual = sha(cp)
            if actual != signed_inputs[stem][key]:
                raise ValueError(f'Signed front detections changed {c}/{stem}')
            front_input_hashes[key] = actual
            raw_front.extend(json.loads(cp.read_text())['dets'])
        raw_side = []
        input_hashes = {str(parent_index[stem].relative_to(ROOT)):sha(parent_index[stem])}
        input_hashes.update(front_input_hashes)
        for cid in (1, 5):
            cp = args.cache/'detections'/f'{cid}_{stem}.json'
            if not cp.exists():
                raise ValueError(f'Side cache incomplete {cid}/{stem}')
            data = json.loads(cp.read_text())
            stat = (BASE/f'mycar/images/{cid}/{stem}.png').stat()
            if data['protocol']['model_sha256'] != EXPECTED_MODEL:
                raise ValueError('Side model changed')
            if data['protocol']['imgsz'] != 960 or data['image_signature'] != [stat.st_size, stat.st_mtime_ns]:
                raise ValueError('Side cache settings/image changed')
            raw_side.extend(data['dets'])
            input_hashes[str(cp.relative_to(ROOT))] = sha(cp)
        pool = []
        keys = {}
        for d in raw_front+raw_side:
            key = S.observation_key(d)
            if key not in keys:
                keys[key] = len(pool)
                pool.append(S.prepare_detection(dict(d, uid=len(pool)), cameras))
        by_key = {S.observation_key(d):d for d in pool}
        front = []
        for k, o in enumerate(front_record['objects']['960/current_frozen']):
            ds = [by_key[S.observation_key(d)] for d in o['dets']]
            front.append(dict(src=list(o['src']), dets=ds, cls=max(ds, key=lambda d:d['conf'])['cls'],
                              front_anchor_id=k, front_source='frozen_960_current_score_E1_parent',
                              existence_source='frozen_parent_admission_not_rescored'))
        # NMS is per camera and identical for all variants, including append.
        side_nms = [d for c in (1, 5) for d in merge_nms([d for d in pool if d['src'] == c])]
        removed_ego = [d for d in side_nms if S.side_ego(d, cameras)]
        side = [d for d in side_nms if not S.side_ego(d, cameras)]
        t = time.perf_counter()
        evidence = S.build_evidence(front, side, cameras, depth_samples=12)
        geometry_ms = (time.perf_counter()-t)*1000
        link_cache = {}
        predictions = {}
        runtime = {}
        diagnostics = {}
        for name, config in configs.items():
            t = time.perf_counter()
            mode = config['mode']
            if mode == 'front':
                predictions[name] = front
            elif mode == 'single':
                predictions[name] = [dict(src=[d['src']], cls=d['cls'], dets=[d]) for d in side
                                     if d['src'] == config['camera'] and d['conf'] >= config['threshold']]
            elif mode == 'append':
                predictions[name] = S.fuse(front, side, [], config['threshold'])
            else:
                key = (mode, config['gate'], config['margin'])
                if key not in link_cache:
                    link_cache[key] = S.sampled_assignment(front, side, evidence, cameras, mode,
                                                            config['gate'], config['margin'])
                links, diagnostic = link_cache[key]
                predictions[name] = S.fuse(front, side, links, config['threshold'],
                                          config.get('front_veto'), evidence)
                diagnostics[name] = diagnostic
            runtime[name] = (time.perf_counter()-t)*1000
        # GT first loaded AFTER all predictions. No evaluator feedback to runtime.
        gt5 = truth(stem, (1, 2, 3, 4, 5))
        for c in (1, 2, 3, 4, 5):
            xp = BASE/f'mycar/cam_{c}/{stem}.xml'
            key = str(xp.relative_to(ROOT))
            actual = sha(xp)
            if c in (2, 3, 4) and actual != signed_inputs[stem][key]:
                raise ValueError(f'Signed front annotations changed {c}/{stem}')
            input_hashes[key] = actual
        gt3 = {c:gt5[c] for c in (2, 3, 4)}
        evaluator = FrameEvaluator(gt5, pool)
        counts = {k:evaluator.counts(v) for k, v in predictions.items()}
        if evaluate(front, gt3).tolist() != front_record['counts']['960/current_frozen']:
            raise ValueError(f'Frozen front mismatch {stem}')
        if counts['front_frozen'] != evaluate(front, gt5).tolist():
            raise ValueError(f'Fast evaluator mismatch {stem}')
        conflicts = []
        labels = {}
        for c, gs in gt5.items():
            for iid, lab, box in gs:
                labels.setdefault(iid, set()).add(lab)
        conflicts = [{'instance_id':i, 'labels':sorted(labs)} for i, labs in labels.items() if len(labs) > 1]
        native = {}
        for cid, threshold in ((1, .15), (5, .2)):
            single = predictions[f'cam{cid}_only']
            native[f'cam{cid}'] = evaluate(single, {cid:gt5[cid]}).tolist()
        row = dict(frame=stem, counts=counts, native_counts=native,
                   identity={k:evaluator.identity(v) for k, v in predictions.items()},
                   geometry_ms=geometry_ms, downstream_ms=runtime, diagnostics=diagnostics,
                   input_sha256=input_hashes, invalid_geometry_side=sum(not d['geometry_valid'] for d in side),
                   removed_side_ego=[dict(src=d['src'], box=d['box'], cls=d['cls'], conf=d['conf']) for d in removed_ego],
                   removed_side_ego_gt_hits=sum(bool(evaluator.scores[d['uid']].size and
                       evaluator.scores[d['uid']].max() >= .5) for d in removed_ego),
                   annotation_label_conflicts=conflicts)
        # Persist compact object signatures for selected-method reproduction.
        row['object_uids'] = {k:[[d['uid'] for d in o['dets']] for o in v] for k, v in predictions.items()}
        dump(output, row)
        if n == 1 or n % 25 == 0 or n == len(frames):
            progress = dict(done=n, planned=len(frames), last_frame=stem,
                            elapsed_s=time.perf_counter()-started, variants=len(configs))
            dump(mp.parent/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    dump(mp.parent/'completion.json', dict(done=len(frames), planned=len(frames)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=BASE/'experiments/cam15_shared_20261008/full')
    ap.add_argument('--cache', type=Path, default=BASE/'experiments/cam15_shared_20261008/cache')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    args = ap.parse_args()
    if args.limit < 0 or not 0 <= args.shard_index < args.shard_count:
        ap.error('Invalid domain')
    run(args)
