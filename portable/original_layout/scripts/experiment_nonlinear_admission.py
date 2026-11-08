"""Same frozen residual features/labels; replace ONLY the post-hoc score model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'ultralytics'
sys.path[:0] = [str(BASE/'research'),str(BASE/'core_algorithms'),str(ROOT/'scripts')]
import nonlinear_admission as N
import residual_admission as R
from experiment_residual_admission import verify,load_pool,corrected,SOURCE,BOX,POLICY
from experiment_recall_recovery import objects,write_new
from experiment_cam15_shared import FrameEvaluator
from validate_multiview import truth,metrics
from multiview_geometry import load_cameras
from cache_low_conf_no_band import sha

SOURCE_FEATURES = BASE/'experiments/budget_fovea_20261009/residual01'
LINEAR = BASE/'algorithm/evaluation_reports/residual_admission_results_20261009.json'
GROUP_TAUS = (0.,.005,.02,.1)
RESIDUAL_TAUS = (0.,.001,.005,.02,.1,.3)


def configurations():
    common = dict(learning_rate=.05,min_samples_leaf=100,l2_regularization=10.,max_bins=127,
        early_stopping=False,categorical_features=None,random_state=20261009)
    return dict(shallow=dict(common,max_iter=120,max_depth=3,max_leaf_nodes=8),
        medium=dict(common,max_iter=180,max_depth=5,max_leaf_nodes=24))


def run(args):
    ref = json.loads(LINEAR.read_text())
    if not ref['complete'] or ref['frames']!=1655:
        raise ValueError('Require complete linear reference')
    verify(ref['protocol'])
    if args.out.exists() or args.report.exists():
        raise FileExistsError('Append-only experiment')
    names = ref['protocol']['frames'][:args.limit] if args.limit else ref['protocol']['frames']
    cfg = dict(frames=names,models=configurations(),group_taus=GROUP_TAUS,residual_taus=RESIDUAL_TAUS,
        linear_report_sha256=sha(LINEAR),parent_protocol=ref['protocol'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),BASE/'research/nonlinear_admission.py')},
        interpretation='Same frozen detector, geometry groups, original-coordinate bbox correction, feature vectors and early labels. '
            'Only calibration learner differs. Standard histogram-boosted trees, not detector retraining or a novel geometric method. '
            'No class weights, no test-frame fitting; small predefined shallow/medium families. No early stopping on mixed/late data. '
            'Portable finite-feature JSON inference parity checked against scikit-learn==1.7.2. All weak predictions count.',
        selection=ref['protocol']['selection'])
    write_new(args.out/'protocol.json',cfg)
    paths = [SOURCE_FEATURES/f'frames/{s}.json' for s in names]
    rs = []
    for p in paths:
        if sha(p)!=ref['extraction_sha256'][str(p.relative_to(ROOT))]:
            raise ValueError('Changed frozen feature')
        r = json.loads(p.read_text())
        for ip,d in r['input_sha256'].items():
            if sha(ROOT/ip)!=d:
                raise ValueError('Changed feature source')
        rs.append(r)
    models = {}
    for family,params in cfg['models'].items():
        for head,key in [('group','groups'),('existence','residuals'),('novelty','residuals')]:
            label = 'label' if head=='group' else head
            data = [v for r in rs if r['frame'] in ref['protocol']['train'] for v in r[key] if v[label]>=0]
            models[f'{family}/{head}'] = N.fit([v['x'] for v in data],[v[label] for v in data],params)
            print(f'Nonlinear model {family}/{head}, samples={len(data)}',flush=True)
    mp = args.out/'models.json'
    write_new(mp,dict(models=models,protocol_sha256=sha(args.out/'protocol.json')))
    compiled = {k:N.compile_model(v) for k,v in models.items()}
    geometry = {p.stem:p for p in SOURCE.glob('shard*/frames/*.json')}
    offsets,cameras = json.loads(POLICY.read_text())['y0'],load_cameras()
    box_model = json.loads(BOX.read_text())['models']['foveated']
    linear_rows = {r['frame']:r for r in ref['rows']}
    rows = []
    for n,r in enumerate(rs,1):
        pool,_ = load_pool(json.loads(geometry[r['frame']].read_text()),cameras,prepare=False)
        changed = corrected(pool,box_model,cameras,offsets)
        seed = [v['uids'] for v in r['groups']]
        orphans = [v['uid'] for v in r['residuals']]
        outputs = {}
        scores = {}
        for family in cfg['models']:
            gs = N.predict([v['x'] for v in r['groups']],compiled[f'{family}/group'])
            scores[f'{family}/group'] = gs.tolist()
            heads = {h:N.predict([v['x'] for v in r['residuals']],compiled[f'{family}/{h}']) for h in ('existence','novelty')}
            scores.update({f'{family}/{h}':p.tolist() for h,p in heads.items()})
            for g in cfg['group_taus']:
                outputs[f'{family}/group{g}/no_residual'] = objects(R.select(seed,gs,[],[],g,1.),changed)
                for h,p in heads.items():
                    for t in cfg['residual_taus']:
                        outputs[f'{family}/group{g}/{h}{t}'] = objects(R.select(seed,gs,orphans,p,g,t),changed)
        # All outputs above are GT-free; current-frame labels evaluate only.
        evaluator = FrameEvaluator(truth(r['frame'],(1,2,3,4,5)),list(changed.values()))
        counts = {k:evaluator.counts(v) for k,v in outputs.items()}
        counts.update(r['controls'])
        chosen = ref['selection']['recall_raw']
        if chosen is not None:
            counts[f'linear/{chosen}'] = linear_rows[r['frame']]['counts'][chosen]
        write_new(args.out/f'scores/{r["frame"]}.json',dict(frame=r['frame'],scores=scores))
        rows.append(dict(frame=r['frame'],counts=counts))
        if n==1 or n%250==0:
            print(f'Nonlinear admission eval {n}/{len(rs)}',flush=True)
    splits = {part:{k:metrics(np.sum([r['counts'][k] for r in rows if part=='all' or r['frame'] in ref['protocol'][part]],axis=0)) for k in rows[0]['counts']}
        for part in ('all','train','development','guard','late') if part=='all' or any(r['frame'] in ref['protocol'][part] for r in rows)}
    dev = splits.get('development',{})
    candidates = [k for k in dev if k.startswith(('shallow/','medium/'))]
    feasible = [k for k in candidates if dev[k]['recall']>=dev['raw']['recall']]
    choose = lambda ks:max(ks,key=lambda k:(dev[k]['precision'],dev[k]['recall'])) if ks else None
    selection = dict(recall_raw=choose(feasible),precision_floors={str(p):choose([k for k in feasible if dev[k]['precision']>=p]) for p in (.4,.6,.8,.9)})
    verify(ref['protocol'])
    for p,d in cfg['source_sha256'].items():
        if sha(ROOT/p)!=d:
            raise ValueError('Changed nonlinear source')
    result = dict(complete=len(rows)==1655,frames=len(rows),protocol=cfg,model_sha256=sha(mp),splits=splits,selection=selection,rows=rows,
        feature_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        score_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((args.out/'scores').glob('*.json'))},
        limitations='Same explored route. Corrected GT-hit/novelty proxies, not physical identity/existence proof. '
            'No single deployed branch latency measured; full evaluation covers two families and many thresholds.')
    write_new(args.report,result)
    picked = sorted({selection['recall_raw'],*selection['precision_floors'].values()}-{None})
    print(json.dumps(dict(selection=selection,selected={k:splits['all'][k] for k in picked},raw=splits['all']['raw'])),flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=BASE/'experiments/budget_fovea_20261009/nonlinear01')
    ap.add_argument('--report',type=Path,default=BASE/'algorithm/evaluation_reports/nonlinear_admission_results_20261009.json')
    ap.add_argument('--limit',type=int,default=0)
    args = ap.parse_args()
    if args.limit<0:
        ap.error('Invalid scope')
    args.out,args.report = args.out.resolve(),args.report.resolve()
    run(args)
