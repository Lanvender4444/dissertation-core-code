"""Same-input controlled ablation: ROI versus ROI + physical road support.

Freeze all detections, groups, corrected output boxes and early-only labels.
Selection is development-only with the user's explicit joint P90/raw-R gate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'ultralytics'
sys.path[:0] = [str(BASE/'research'),str(BASE/'core_algorithms'),str(ROOT/'scripts')]
import road_support_admission as A
import nonlinear_admission as N
import residual_admission as R
from experiment_detector_roi_admission import expand,CAPACITY,DESCRIPTORS,LABELS
from experiment_capacity_admission import checked
from experiment_residual_admission import verify,corrected,SOURCE,BOX,POLICY
from experiment_recall_recovery import objects,write_new
from experiment_cam15_shared import FrameEvaluator
from validate_multiview import truth,metrics
from multiview_geometry import load_cameras
from cache_low_conf_no_band import sha

REFERENCE = BASE/'algorithm/evaluation_reports/detector_roi_admission_results_20261009.json'
OLD = BASE/'experiments/budget_fovea_20261009/detector_roi_admission01'
ROAD = BASE/'experiments/road_support_20261010/full01'
TAUS = (0.,.02,.05,.15,.4,.7,.9,.98,.999)


def select_development(dev):
    """No all/guard/late data may be passed to this selector."""
    result = {}
    for arm in ('roi','road'):
        keys = sorted(k for k in dev if k.startswith(arm+'/'))
        p90 = [k for k in keys if dev[k]['precision']>=.9]
        joint = [k for k in p90 if dev[k]['recall']>=dev['raw']['recall']]
        raw_r = [k for k in keys if dev[k]['recall']>=dev['raw']['recall']]
        by_r = lambda xs:max(xs,key=lambda k:(dev[k]['recall'],dev[k]['precision'])) if xs else None
        by_p = lambda xs:max(xs,key=lambda k:(dev[k]['precision'],dev[k]['recall'])) if xs else None
        result[arm] = dict(joint_candidates=joint,joint_selected=by_r(joint),
            precision90_selected=by_r(p90),recall_raw_diagnostic=by_p(raw_r),
            best_f1_diagnostic=max(keys,key=lambda k:(dev[k]['f1'],dev[k]['precision'])) if keys else None)
    return result


def extend(s,parent,index,geometry,cameras,road_index,road_digest):
    r,pool = expand(s,parent,index,geometry,cameras)
    p = road_index[s]
    road = json.loads(p.read_text())
    if road['frame']!=s or road['protocol_sha256']!=road_digest:
        raise ValueError('Road cache protocol/frame mismatch')
    if set(road['features'])!={str(u) for u,d in pool.items() if d['conf']>=.05}:
        raise ValueError('Incomplete physical descriptors')
    for ip,d in road['input_sha256'].items():
        if sha(ROOT/ip)!=d:
            raise ValueError('Changed physical source image/detection')
    for g in r['groups']:
        g['x'] = A.augment(g['x'],[pool[u] for u in g['uids']],road['features']).tolist()
    for d in r['residuals']:
        d['x'] = A.augment(d['x'],[pool[d['uid']]],road['features'],False).tolist()
    return r,pool


def verify_hashes(proof):
    for p,d in proof.items():
        if sha(ROOT/p)!=d:
            raise ValueError(f'Changed frozen dependency: {p}')


def run(args):
    ref,parent = [json.loads(p.read_text()) for p in (REFERENCE,CAPACITY)]
    if not ref['complete'] or ref['frames']!=1655 or sha(CAPACITY)!=ref['protocol']['capacity_report_sha256']:
        raise ValueError('Incomplete/changed frozen baseline')
    verify(ref['protocol']['parent_protocol'])
    verify_hashes(ref['protocol']['source_sha256'])
    dp = json.loads((DESCRIPTORS/'protocol.json').read_text())
    rp = ROAD/'protocol.json'
    road = json.loads(rp.read_text())
    verify_hashes(dp['source_sha256'])
    verify_hashes({**road['source_sha256'],**road['calibration_sha256']})
    if sha(ROOT/'Inputs and Outputs Format.pdf')!=road['format_document_sha256']:
        raise ValueError('Changed camera format document')
    if road['frames']!=ref['protocol']['frames'] or road['floor']!=.05 or road['dimension']!=20:
        raise ValueError('Different physical feature domain')
    index = {p.stem:p for p in DESCRIPTORS.glob('shard*/frames/*.json')}
    physical = {p.stem:p for p in ROAD.glob('shard*/frames/*.json')}
    required = ref['protocol']['train'] if args.stage=='fit' else road['frames']
    if not set(required)<=set(index)&set(physical):
        raise ValueError('Required training/full feature cache not complete')
    if args.stage!='fit':
        for directory,n in ((ROAD,road['shards']),(DESCRIPTORS,dp['shard_count'])):
            for i in range(n):
                if json.loads((directory/f'shard{i}/completion.json').read_text())!=dict(protocol_sha256=sha(directory/'protocol.json')):
                    raise ValueError('Full feature cache not authoritatively complete')
    if args.report.exists() or (args.stage!='eval' and args.out.exists()):
        raise FileExistsError('Append-only experiment')
    dependencies = (Path(__file__),BASE/'research/road_support_admission.py',BASE/'research/nonlinear_admission.py',
                    ROOT/'scripts/experiment_detector_roi_admission.py',ROOT/'scripts/experiment_capacity_admission.py')
    cfg = dict(frames=road['frames'],**{p:ref['protocol'][p] for p in ('train','development','guard','late','models','parent_protocol')},
        reference_sha256=sha(REFERENCE),capacity_sha256=sha(CAPACITY),old_model_sha256=sha(OLD/'models.json'),
        road_protocol_sha256=sha(rp),roi_protocol_sha256=sha(DESCRIPTORS/'protocol.json'),
        road_source_sha256=road['source_sha256'],roi_source_sha256=dp['source_sha256'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in dependencies},
        group_taus=TAUS,residual_taus=TAUS,dimensions=dict(group=1136,residual=690),
        selection='Development-only highest Recall subject Precision>=.90; separately require Recall>=same-input raw. '
                  'No deployment candidate if joint infeasible. Low-Precision maximum-Recall and F1 are diagnostics only.',
        interpretation='ROI versus ROI + fitted road-plane/LR-stereo/ground-ray evidence. Both arms use identical extended '
            'threshold grid. Road arm refits BOTH post-hoc heads on existing early train/labels; ROI arm exactly reuses frozen '
            'existing heads/scores. No detector training/rerun, grouping, active bbox change, GT inference, or unknown hiding. '
            'New foveated input, NOT original native noBAND. Same explored route, not independent scene.')
    cfg = json.loads(json.dumps(cfg))
    if args.stage=='eval':
        if json.loads((args.out/'protocol.json').read_text())!=cfg:
            raise ValueError('Changed fitted experiment protocol')
    else:
        write_new(args.out/'protocol.json',cfg)
    geometry = {p.stem:p for p in SOURCE.glob('shard*/frames/*.json')}
    cameras,offsets = load_cameras(),json.loads(POLICY.read_text())['y0']
    mp = args.out/'models.json'
    if args.stage!='eval':
        train = {head:dict(x=[],y=[]) for head in ('group','capacity')}
        proofs = {}
        for i,s in enumerate(cfg['train'],1):
            r,_ = extend(s,parent,index,geometry,cameras,physical,sha(rp))
            lp = LABELS/f'labels/{s}.json'
            labels = checked(lp,parent['label_sha256'][str(lp.relative_to(ROOT))])
            if labels['uids']!=[d['uid'] for d in r['residuals']]:
                raise ValueError('Changed early target order')
            for d in r['groups']:
                if d['label']>=0:
                    train['group']['x'].append(d['x'])
                    train['group']['y'].append(d['label'])
            for d,y in zip(r['residuals'],labels['labels']):
                if y>=0:
                    train['capacity']['x'].append(d['x'])
                    train['capacity']['y'].append(y)
            for p in (physical[s],index[s]):
                proofs[str(p.relative_to(ROOT))] = sha(p)
            if i==1 or i%200==0:
                print(f'Road support early feature assembly {i}/{len(cfg["train"])}',flush=True)
        models = {}
        for family,params in cfg['models'].items():
            for head,data in train.items():
                models[f'{family}/{head}'] = N.fit(data['x'],data['y'],params)
                print(f'Road support model {family}/{head}: n{len(data["y"])} positive{sum(data["y"])}',flush=True)
        del train
        verify_hashes(proofs)
        write_new(mp,dict(models=models,protocol_sha256=sha(args.out/'protocol.json'),training_feature_sha256=proofs))
        if args.stage=='fit':
            print('Road support early-only fit complete; full evaluation not run',flush=True)
            return
    else:
        model_doc = json.loads(mp.read_text())
        if model_doc['protocol_sha256']!=sha(args.out/'protocol.json'):
            raise ValueError('Changed fitted model protocol')
        verify_hashes(model_doc['training_feature_sha256'])
        models = model_doc['models']
    compiled = {k:N.compile_model(v) for k,v in models.items()}
    box = json.loads(BOX.read_text())['models']['foveated']
    old_rows = {r['frame']:r for r in ref['rows']}
    rows = []
    for i,s in enumerate(cfg['frames'],1):
        r,pool = extend(s,parent,index,geometry,cameras,physical,sha(rp))
        changed = corrected(pool,box,cameras,offsets)
        seed,orphans = [d['uids'] for d in r['groups']],[d['uid'] for d in r['residuals']]
        op = OLD/f'scores/{s}.json'
        old_scores = checked(op,ref['score_sha256'][str(op.relative_to(ROOT))])['scores']
        scores = {}
        for family in cfg['models']:
            scores[f'{family}/group'] = N.predict([d['x'] for d in r['groups']],compiled[f'{family}/group']).tolist()
            scores[f'{family}/capacity'] = N.predict([d['x'] for d in r['residuals']],compiled[f'{family}/capacity']).tolist()
        e = FrameEvaluator(truth(s,(1,2,3,4,5)),list(changed.values()))
        counts = dict(r['controls'])
        for arm,sc in (('roi',old_scores),('road',scores)):
            for family in cfg['models']:
                for g in cfg['group_taus']:
                    for t in cfg['residual_taus']:
                        k = f'{family}/group{g}/capacity{t}'
                        output = objects(R.select(seed,sc[f'{family}/group'],orphans,sc[f'{family}/capacity'],g,t),changed)
                        counts[f'{arm}/{k}'] = e.counts(output)
                if counts[f'{arm}/{family}/group0.0/capacity0.0']!=old_rows[s]['counts'][f'{family}/group0.0/capacity0.0']:
                    raise ValueError('All-observation control changed')
                if arm=='roi' and counts[f'roi/{family}/group0.02/capacity0.02']!=old_rows[s]['counts'][f'{family}/group0.02/capacity0.02']:
                    raise ValueError('ROI fixed operating point changed')
        if any(counts[k]!=old_rows[s]['counts'][k] for k in r['controls']):
            raise ValueError('Frozen raw/corrected control changed')
        write_new(args.out/f'scores/{s}.json',dict(frame=s,scores=scores))
        rows.append(dict(frame=s,counts=counts))
        if i==1 or i%100==0:
            print(f'Road support full paired evaluation {i}/{len(cfg["frames"])}',flush=True)
    parts = {p:set(cfg[p]) for p in ('train','development','guard','late')}
    parts['guard_late'] = parts['guard']|parts['late']
    splits = {p:{k:metrics(np.sum([r['counts'][k] for r in rows if p=='all' or r['frame'] in parts[p]],axis=0)) for k in rows[0]['counts']}
              for p in ('all',*parts)}
    selection = select_development(splits['development'])
    picked = sorted({v for arm in selection.values() for k,v in arm.items() if k!='joint_candidates' and v is not None})
    checks = {k:{p:dict(precision90=splits[p][k]['precision']>=.9,
                recall_raw=splits[p][k]['recall']>=splits[p]['raw']['recall'],
                tp_difference=splits[p][k]['tp']-splits[p]['raw']['tp']) for p in ('all','development','guard_late','late')} for k in picked}
    verify(cfg['parent_protocol'])
    verify_hashes({**cfg['source_sha256'],**cfg['road_source_sha256'],**cfg['roi_source_sha256']})
    result = dict(complete=len(rows)==1655,frames=len(rows),protocol=cfg,model_sha256=sha(mp),selection=selection,
        selected_gate_checks=checks,splits=splits,rows=rows,
        physical_sha256={str(physical[s].relative_to(ROOT)):sha(physical[s]) for s in cfg['frames']},
        score_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((args.out/'scores').glob('*.json'))},
        limitations='Fitted road is not measured GT. Road extrapolation/covariance omit terrain, calibration and box-edge model errors. '
          'Foot-contact and rough class height can fail under occlusion/truncation; only features, never hard proof/veto. '
          'Strong LR stereo only Cam2/4, same stereo pair correlated. Early post-hoc learners not algorithmic novelty themselves. '
          'Five-view GT/identity proxy unverified, same explored route. Speed not certified; default entry unchanged.')
    write_new(args.report,result)
    print(json.dumps(dict(selection=selection,selected={k:splits['all'][k] for k in picked},checks=checks)),flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('fit','eval','fit_eval'),default='fit_eval')
    ap.add_argument('--out',type=Path,default=BASE/'experiments/road_support_20261010/admission01')
    ap.add_argument('--report',type=Path,default=BASE/'algorithm/evaluation_reports/road_support_admission_results_20261010.json')
    args = ap.parse_args()
    args.out,args.report = args.out.resolve(),args.report.resolve()
    run(args)
