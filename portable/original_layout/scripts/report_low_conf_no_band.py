"""Audit and aggregate the complete low-YOLO-confidence ablation, append-only.

No parameter fitting/selection happens here. Report every predeclared arm on
both GT scopes; a confidence curve maximum is exploratory, not a blind result.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'ultralytics'
sys.path[:0] = [str(BASE/'core_algorithms'), str(BASE/'research'), str(ROOT/'scripts')]
from cache_low_conf_no_band import sha
from validate_multiview import metrics


def summary(rows, field, keys):
    return {k:metrics(np.sum([r[field][k] for r in rows], axis=0)) for k in keys}


def paired_bootstrap(rows, field, a, b, iterations=2000):
    blocks = [rows[i:i+25] for i in range(0, len(rows), 25)]
    ca = np.array([np.sum([r[field][a][:3] for r in block], axis=0) for block in blocks])
    cb = np.array([np.sum([r[field][b][:3] for r in block], axis=0) for block in blocks])
    rng = np.random.default_rng(20261009)
    indices = rng.integers(0, len(blocks), (iterations, len(blocks)))
    x, y = ca[indices].sum(axis=1), cb[indices].sum(axis=1)
    def m(v):
        p = v[:, 0]/np.maximum(v[:, 0]+v[:, 1], 1)
        r = v[:, 0]/np.maximum(v[:, 0]+v[:, 2], 1)
        return np.column_stack([p, r, 2*p*r/np.maximum(p+r, 1e-9)])
    delta = (m(x)-m(y))*100
    return {k:dict(low_pp=float(np.percentile(delta[:, i], 2.5)),
                   high_pp=float(np.percentile(delta[:, i], 97.5)))
            for i, k in enumerate(('precision', 'recall', 'f1'))}


def comparisons(rows, field, totals, floors, variants):
    pairs = [(f'conf{f}/{v}', f'conf0.1/{v}') for f in floors if f != .1 for v in variants]
    pairs += [(f'conf{f}/{v}', f'conf{f}/fusion_frozen') for f in floors
              for v in ('fish_aux_only075', 'fish_full075', 'fish_full085', 'fusion_score030_control') if v in variants]
    if 'fish_full075' in variants:
        pairs += [(f'conf{f}/fish_full075', f'conf{f}/five_append_control') for f in floors]
    return {f'{a} vs {b}':dict(
        delta_pp={k:100*(totals[a][k]-totals[b][k]) for k in ('precision', 'recall', 'f1')},
        paired_25frame_block_bootstrap_95=paired_bootstrap(rows, field, a, b),
        both_2pp=totals[a]['precision']-totals[b]['precision'] >= .02 and
                  totals[a]['recall']-totals[b]['recall'] >= .02)
        for a, b in pairs}


def verify_shards(folder, names, protocol_checksum=None):
    manifests = sorted(folder.glob('shard*/manifest.json'))
    planned = []
    for path in manifests:
        m = json.loads(path.read_text())
        done = json.loads((path.parent/'completion.json').read_text())
        if done != dict(done=len(m['frames']), planned=len(m['frames'])):
            raise ValueError('Incomplete shard')
        if protocol_checksum is not None and m['protocol_sha256'] != protocol_checksum:
            raise ValueError('Mixed protocol')
        if m['frames'] != names[m['shard_index']::m['shard_count']]:
            raise ValueError('Wrong partition')
        planned += m['frames']
    if sorted(planned) != names:
        raise ValueError('Missing or duplicated domain')


def run(args):
    if args.destination.exists():
        raise FileExistsError('Report already exists; use another --destination, never overwrite')
    pp = args.out/'protocol.json'
    cfg = json.loads(pp.read_text())
    source_checksums = {k.replace('\\', '/'):v for k, v in cfg['source_sha256'].items()}
    for path, expected in cfg['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'Study source changed: {path}')
    for path, expected in cfg['historical_documents_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'Historical document/result changed: {path}')
    model_path = BASE/'experiments/cam15_shared_20261008/scoring/frozen_models.json'
    if sha(model_path) != cfg['five_scoring_model_sha256'] or sha(BASE/'yolo11n.pt') != cfg['model_sha256']:
        raise ValueError('Frozen model changed')
    reference = BASE/'experiments/cam15_shared_20261008/full/protocol.json'
    if sha(reference) != cfg['previous_five_protocol_sha256']:
        raise ValueError('Reference protocol changed')
    verify_shards(args.out, cfg['frames'], sha(pp))
    verify_shards(args.cache, cfg['frames'])
    files = sorted(args.out.glob('shard*/frames/*.json'), key=lambda p:p.stem)
    if [p.stem for p in files] != cfg['frames']:
        raise ValueError('Frame outputs are incomplete or duplicated')
    rows, drift, cap = [], Counter(), Counter()
    front_drift_examples = []
    native_counts, low_members = {}, {}
    inspected_caches = set()
    for n, path in enumerate(files, 1):
        row = json.loads(path.read_text())
        if row['frame'] != path.stem or any(v[0]+v[2] != row['gt5'] for v in row['counts'].values()) or any(v[0]+v[2] != row['gt3'] for v in row['counts3'].values()):
            raise ValueError('Inconsistent GT denominator')
        for key, expected in row['input_sha256'].items():
            if sha(ROOT/key) != expected:
                raise ValueError(f'Input changed: {key}')
        pool = []
        for c in (2, 3, 4, 1, 5):
            cp = args.cache/'detections'/f'{c}_{row["frame"]}.json'
            data = json.loads(cp.read_text())
            signature = data['protocol']
            if signature['model_sha256'] != cfg['model_sha256'] or signature['conf'] != .01 or signature['imgsz'] != 960 or signature['iou'] != .7 or signature['max_det'] != 300:
                raise ValueError('Mixed detector config')
            if signature['source_sha256'] != source_checksums['scripts/cache_low_conf_no_band.py']:
                raise ValueError('Changed producer')
            image = BASE/f'mycar/images/{c}/{row["frame"]}.png'
            st = image.stat()
            if data['image_signature'] != [st.st_size, st.st_mtime_ns] or sha(image) != data['image_sha256']:
                raise ValueError('Image bytes changed')
            if any(d['src'] != c or d['conf'] < .01 for d in data['dets']):
                raise ValueError('Wrong camera/floor')
            inspected_caches.add(cp.name)
            pool.extend(data['dets'])
            cap[f'cam{c}_images_at_300_cap'] += bool(data['max_det_cap_hit'])
            native_counts[f'cam{c}'] = native_counts.get(f'cam{c}', Counter())
            for f in cfg['floors']:
                native_counts[f'cam{c}'][str(f)] += sum(d['conf'] >= f for d in data['dets'])
            d = row['detection_drift'][str(c)]
            drift[f'cam{c}_unequal_frames'] += not d['exact_box_conf_set_equal']
            drift[f'cam{c}_old_box_count'] += d['old_count']
            drift[f'cam{c}_fresh_conf010_box_count'] += d['fresh_count']
        for key, groups in row['object_uids'].items():
            stat = low_members.setdefault(key, Counter())
            stat['output_objects'] += len(groups)
            for group in groups:
                if not group:
                    raise ValueError('Empty output')
                confidences = [pool[u]['conf'] for u in group]
                stat['contains_conf_below010'] += min(confidences) < .1
                stat['all_members_conf_below010'] += max(confidences) < .1
                stat['member_count'] += len(group)
                stat['members_conf_below010'] += sum(v < .1 for v in confidences)
        front_count_diff = row['counts']['conf0.1/fusion_frozen'] != row['counts']['original_fusion']
        drift['front_conf010_count_drift_frames'] += front_count_diff
        parent_paths = [ROOT/k for k in row['input_sha256']
                        if 'no_band_20261007_shards/' in k.replace('\\', '/') and '/frames/' in k.replace('\\', '/')]
        if len(parent_paths) != 1:
            raise ValueError('Original prediction reference missing')
        original_objects = json.loads(parent_paths[0].read_text())['objects']['960/current_frozen']
        def observation(d):
            return (d['src'], d['cls'], tuple(d['box']), d['conf'])
        old_signature = Counter(frozenset(observation(d) for d in o['dets']) for o in original_objects)
        fresh_signature = Counter(frozenset(observation(pool[u]) for u in group)
                                  for group in row['object_uids']['conf0.1/fusion_frozen'])
        front_member_diff = old_signature != fresh_signature
        drift['front_conf010_membership_drift_frames'] += front_member_diff
        if (front_count_diff or front_member_diff) and len(front_drift_examples) < 10:
            front_drift_examples.append(row['frame'])
        rows.append(row)
        if n == 1 or n % 250 == 0 or n == len(files):
            print(f'Audited {n}/{len(files)}: caches, images, XML and parents', flush=True)
    if len(inspected_caches) != len(rows)*5:
        raise ValueError('Detector cache domain incomplete')
    independent = json.loads(args.candidate_audit.read_text())
    if not independent['complete'] or independent['frames'] != len(rows):
        raise ValueError('Independent candidate audit is incomplete')
    independent_index = {r['frame']:r for r in independent['frame_counts']}
    for row in rows:
        other = independent_index[row['frame']]
        for key in other['counts5']:
            if row['counts'][key] != other['counts5'][key] or row['counts3'][key] != other['counts3'][key] or row['native_counts'][key] != other['native_counts'][key]:
                raise ValueError('Independent single-camera replay mismatch')
        if row['raw_coverage'] != other['raw_coverage']:
            raise ValueError('Independent raw candidate replay mismatch')
        for path, checksum in row['input_sha256'].items():
            if path in independent['detector_inputs_sha256'] and checksum != independent['detector_inputs_sha256'][path]:
                raise ValueError('Candidate audit used different detection inputs')
    nms_parity = json.loads(args.nms_parity.read_text())
    if not nms_parity['complete'] or not nms_parity['all_exact_equal'] or nms_parity['direct_inferences'] != 120:
        raise ValueError('Direct-threshold inference parity incomplete')
    for row in nms_parity['checks']:
        if not row['exact_box_conf_class_equal'] or sha(args.cache/'detections'/f'{row["camera"]}_{row["frame"]}.json') != row['cache_sha256']:
            raise ValueError('Direct-threshold check input changed')
    keys = cfg['configs']+['original_fusion', 'original_cam2', 'original_cam3', 'original_cam4']
    front_variants = ('cam2', 'cam3', 'cam4', 'fusion_frozen', 'fusion_score030_control')
    front_keys = [k for k in keys if k.startswith('original_') or k.rsplit('/', 1)[-1] in front_variants]
    result = dict(complete=True, frames=len(rows), configs=len(cfg['configs']),
        protocol=cfg, protocol_sha256=sha(pp), source_sha256=sha(__file__),
        frozen_five_scoring_models=json.loads(model_path.read_text()),
        source_snapshot={path:(ROOT/path).read_text(encoding='utf-8') for path in cfg['source_sha256']},
        splits={}, detector_audit=dict(images_verified=len(inspected_caches), image_caps=dict(cap),
            historical_conf010_drift=dict(drift), front_drift_example_frames=front_drift_examples,
            candidate_counts=native_counts),
        historical_documents_preserved=len(cfg['historical_documents_sha256']),
        independent_candidate_replay=dict(passed=True, frames=len(rows),
            report_sha256=sha(args.candidate_audit), source_sha256=independent['source_sha256']),
        direct_threshold_inference_parity=dict(report_sha256=sha(args.nms_parity), results=nms_parity),
        low_confidence_admission=low_members,
        inference_timing_note='Concurrent offline workers; NOT end-to-end latency or power evidence.',
        bootstrap_note='Paired 25-consecutive-frame blocks, 2000 draws. Exploratory multiple comparisons '
                       'without multiplicity correction; sequential splits are NOT independent scenes.',
        frame_counts=[{k:r[k] for k in ('frame', 'counts', 'counts3', 'native_counts', 'gt3', 'gt5')} for r in rows])
    subsets = dict(all_frames=rows,
        development=[r for r in rows if r['frame'] in set(cfg['development_frames'])],
        guard=[r for r in rows if r['frame'] in set(cfg['guard_frames'])],
        late_evaluation=[r for r in rows if r['frame'] in set(cfg['evaluation_frames'])])
    for split, subset in subsets.items():
        if not subset:
            continue
        methods5, methods3 = summary(subset, 'counts', keys), summary(subset, 'counts3', front_keys)
        identity = {k:dict(sum((Counter(r['identity'][k]) for r in subset), Counter())) for k in cfg['configs']}
        identity_rates = {}
        for key, values in identity.items():
            known = values.get('correct_links', 0)+values.get('incorrect_links', 0)
            side_known = values.get('correct_side_links', 0)+values.get('incorrect_side_links', 0)
            identity_rates[key] = dict(
                known_link_error=values.get('incorrect_links', 0)/known if known else None,
                known_side_link_error=values.get('incorrect_side_links', 0)/side_known if side_known else None,
                known_link_fraction=known/max(values.get('links', 0), 1),
                known_side_link_fraction=side_known/max(values.get('side_links', 0), 1))
        info = dict(frames=len(subset), gt3=sum(r['gt3'] for r in subset), gt5=sum(r['gt5'] for r in subset),
            on_gt5=methods5, on_gt3=methods3, identity_gt5_proxy=identity, identity_proxy_rates=identity_rates,
            native_singles=summary(subset, 'native_counts', [k for k in cfg['configs'] if k.endswith(('/cam2', '/cam3', '/cam4'))]),
            raw_pool_coverage=summary(subset, 'raw_coverage', [str(f) for f in cfg['floors']]))
        info['stage_counters'] = {}
        for f in cfg['floors']:
            ds = [r['diagnostics'][str(f)] for r in subset]
            association = Counter()
            for d in ds:
                association.update({k:v for k, v in d['association'].items() if k != 'component_sizes'})
            info['stage_counters'][str(f)] = dict(
                front_candidates=sum(d['front_candidates'] for d in ds), front_accepted=sum(d['front_accepted'] for d in ds),
                side_candidates=sum(d['side_candidates'] for d in ds), before_five_score=sum(d['before_five_score'] for d in ds),
                score_removed075=sum(d['score_removed075'] for d in ds), score_removed085=sum(d['score_removed085'] for d in ds),
                side_ego_removed=sum(len(d['removed_side_ego']) for d in ds),
                deferred_ambiguous_side=sum(len(d['diagnostic_unknown']) for d in ds), association=dict(association))
        if split in ('all_frames', 'late_evaluation'):
            info['comparisons_gt5'] = comparisons(subset, 'counts', methods5, cfg['floors'], cfg['variants'])
            info['comparisons_gt3'] = comparisons(subset, 'counts3', methods3, cfg['floors'], front_variants)
        result['splits'][split] = info
    if len(rows) == 1655:
        all_result = result['splits']['all_frames']
        if (all_result['gt3'], all_result['gt5']) != (13473, 24845):
            raise ValueError('Historical GT scope mismatch')
        baseline = all_result['on_gt5']['original_fusion']
        if (baseline['tp'], baseline['fp'], baseline['fn']) != (4963, 360, 19882):
            raise ValueError('Historical original fusion mismatch')
    for path, expected in cfg['historical_documents_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Historical document changed during aggregation')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(report=str(args.destination), frames=len(rows), audit=result['detector_audit'],
                          all_gt5=result['splits']['all_frames']['on_gt5']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=BASE/'experiments/low_conf_no_band_20261009/full')
    parser.add_argument('--cache', type=Path, default=BASE/'experiments/low_conf_no_band_20261009/cache')
    parser.add_argument('--destination', type=Path, default=BASE/'algorithm/evaluation_reports/low_conf_no_band_results_20261009.json')
    parser.add_argument('--candidate-audit', type=Path, default=BASE/'experiments/low_conf_no_band_20261009/candidate_audit.json')
    parser.add_argument('--nms-parity', type=Path, default=BASE/'experiments/low_conf_no_band_20261009/nms_parity.json')
    run(parser.parse_args())
