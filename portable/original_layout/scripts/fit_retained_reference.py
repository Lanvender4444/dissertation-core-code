"""Early-only cross-fitted groups -> conditional residual targets and models.

Changes the emitted-reference conditioning, not YOLO or test annotations.
No claim of full-domain performance until a separate evaluation is completed.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import experiment_road_support_admission as E
import retained_reference as C
import marginal_capacity as M
import shared_geometry as S

OUT = E.BASE/'experiments/retained_reference_20261010/run01'


def run(args):
    ref,parent = [json.loads(p.read_text()) for p in (E.REFERENCE,E.CAPACITY)]
    E.verify(ref['protocol']['parent_protocol'])
    E.verify_hashes(ref['protocol']['source_sha256'])
    if not ref['complete'] or ref['frames']!=1655 or E.sha(E.CAPACITY)!=ref['protocol']['capacity_report_sha256']:
        raise ValueError('Changed frozen ROI baseline')
    folds = C.training_folds(ref['protocol']['train'])
    cfg = dict(frames=ref['protocol']['frames'],**{k:ref['protocol'][k] for k in ('train','development','guard','late','models','parent_protocol')},
        reference_sha256=E.sha(E.REFERENCE),capacity_sha256=E.sha(E.CAPACITY),
        group_tau=.7,group_family='medium',crossfit=folds,dimension=672,arms=['context_only','recycle'],
        residual_taus=[0.,.02,.05,.15,.4,.7,.9,.98,.999],
        source_sha256={str(p.relative_to(E.ROOT)):E.sha(p) for p in (Path(__file__),E.BASE/'research/retained_reference.py',
            E.BASE/'research/marginal_capacity.py',E.BASE/'research/nonlinear_admission.py',E.ROOT/'scripts/experiment_detector_roi_admission.py')},
        interpretation='Context and capacity conditioned on emitted groups at frozen g=.7, NOT all unscored seed. '
            'Contiguous five-fold early-only group cross-fitting with five-frame gap avoids own-label group fit in residual targets. '
            'Existing output box calibration NOT cross-fitted. Same explored enhanced-input route; no blind validation. '
            'Context-only arm original orphans; recycle arm also rejected-group observations, never reuses emitted UID. '
            'Candidate is unverified singleton, no inherited rejected-group identity. No GT in inference.',
        selection='Later development-only P>=.90 max Recall and joint same-input raw-R gate, no full-domain selection.')
    cfg = json.loads(json.dumps(cfg))
    pp = args.out/'protocol.json'
    if pp.exists():
        if json.loads(pp.read_text())!=cfg:
            raise ValueError('Changed conditional-fit protocol')
    else:
        E.write_new(pp,cfg)
    E.verify_hashes(cfg['source_sha256'])
    index = {p.stem:p for p in E.DESCRIPTORS.glob('shard*/frames/*.json')}
    geometry = {p.stem:p for p in E.SOURCE.glob('shard*/frames/*.json')}
    cameras,offsets = E.load_cameras(),json.loads(E.POLICY.read_text())['y0']
    box = json.loads(E.BOX.read_text())['models']['foveated']
    frames,proofs = {},{}
    for i,s in enumerate(cfg['train'],1):
        frames[s] = E.expand(s,parent,index,geometry,cameras)
        proofs[str(index[s].relative_to(E.ROOT))] = E.sha(index[s])
        if i==1 or i%200==0:
            print(f'Retained reference early assembly {i}/601',flush=True)
    cf = args.out/'crossfit_completion.json'
    if args.stage in ('crossfit','all'):
        if cf.exists():
            raise FileExistsError('Cross-fit already complete')
        totals = Counter()
        for j,fold in enumerate(folds):
            xs,ys = [],[]
            for s in fold['train']:
                for g in frames[s][0]['groups']:
                    if g['label']>=0:
                        xs.append(g['x']);ys.append(g['label'])
            m = E.N.fit(xs,ys,cfg['models']['medium'])
            E.write_new(args.out/f'fold_models/{j}.json',dict(model=m,train=fold['train'],held=fold['held']))
            cm = E.N.compile_model(m)
            for s in fold['held']:
                scores = E.N.predict([g['x'] for g in frames[s][0]['groups']],cm).tolist()
                E.write_new(args.out/f'crossfit_scores/{s}.json',dict(frame=s,fold=j,scores=scores,protocol_sha256=E.sha(pp)))
                totals.update(frames=1,groups=len(scores),selected=sum(p>=cfg['group_tau'] for p in scores))
            print(f'Retained reference group OOF fold {j+1}/5: train{len(fold["train"])} held{len(fold["held"])}',flush=True)
        E.verify_hashes(proofs)
        E.write_new(cf,dict(protocol_sha256=E.sha(pp),totals=dict(totals),feature_sha256=proofs,
            fold_model_sha256={str(p.relative_to(E.ROOT)):E.sha(p) for p in (args.out/'fold_models').glob('*.json')}))
        if args.stage=='crossfit':
            return
    done = json.loads(cf.read_text())
    if done['protocol_sha256']!=E.sha(pp) or done['totals']['frames']!=601:
        raise ValueError('Incomplete early cross-fit')
    E.verify_hashes(done['feature_sha256']);E.verify_hashes(done['fold_model_sha256'])
    lp = args.out/'label_completion.json'
    if args.stage in ('labels','all'):
        if lp.exists():
            raise FileExistsError('Conditional targets already complete')
        totals = {arm:Counter() for arm in cfg['arms']}
        label_proof = {}
        for i,s in enumerate(cfg['train'],1):
            r,raw = frames[s]
            pool = {u:S.prepare_detection(d,cameras) for u,d in raw.items() if d['conf']>=.05}
            output_pool = E.corrected(pool,box,cameras,offsets)
            sp = args.out/f'crossfit_scores/{s}.json'
            score = json.loads(sp.read_text())
            fold = folds[score['fold']]
            if score['protocol_sha256']!=E.sha(pp) or s not in fold['held'] or s in fold['train']:
                raise ValueError('Own-frame group training contamination')
            descriptor = json.loads(index[s].read_text())['features']
            record = json.loads(geometry[s].read_text())
            unknown = {v['uid'] for v in record['diagnostics']['front0.0']['unknown']}
            seed = [g['uids'] for g in r['groups']]
            arms = {}
            # Every feature is built before current early GT supervision.
            for arm in cfg['arms']:
                kept,uids,roles = C.partition(seed,score['scores'],pool,cfg['group_tau'],recycle=arm=='recycle')
                refs = E.objects(kept,pool)
                xs = [C.features(pool[u],refs,unknown,cameras,offsets,descriptor,cfg['group_tau'],roles[u]).tolist() for u in uids]
                arms[arm] = dict(kept=kept,uids=uids,roles=[roles[u] for u in uids],x=xs)
            evaluator = E.FrameEvaluator(E.truth(s,(1,2,3,4,5)),list(output_pool.values()))
            def edges(groups):
                return np.array([np.maximum.reduce([evaluator.scores[u] for u in g])>=.5 for g in groups],bool).reshape(len(groups),len(evaluator.keys))
            for arm,a in arms.items():
                candidate_edges = edges([[u] for u in a['uids']])
                old = M.marginal_labels(edges(seed),candidate_edges)['labels']
                labels = M.marginal_labels(edges(a['kept']),candidate_edges)['labels']
                if np.any(labels<old):
                    raise ValueError('Reference-removal marginal monotonicity failed')
                labels = labels.astype(int)
                labels[[not candidate_edges[j].any() and u in evaluator.small_hits for j,u in enumerate(a['uids'])]] = -1
                a['labels'] = labels.tolist()
                totals[arm].update(frames=1,candidates=len(labels),eligible=int((labels>=0).sum()),positive=int((labels>0).sum()),
                    old_all_seed_positive=int(old.sum()),zero_to_positive=int(((old==0)&(labels>0)).sum()),
                    recycled=sum(role=='recycled_rejected_group' for role in a['roles']))
            p = args.out/f'training/{s}.json'
            E.write_new(p,dict(frame=s,arms=arms,oof_score_sha256=E.sha(sp),protocol_sha256=E.sha(pp)))
            label_proof[str(p.relative_to(E.ROOT))] = E.sha(p)
            if i==1 or i%100==0:
                print(f'Retained reference conditional early labels {i}/601',flush=True)
        E.verify_hashes(cfg['source_sha256'])
        E.write_new(lp,dict(protocol_sha256=E.sha(pp),totals={k:dict(v) for k,v in totals.items()},training_sha256=label_proof))
        print(json.dumps({k:dict(v) for k,v in totals.items()}),flush=True)
        if args.stage=='labels':
            return
    labelled = json.loads(lp.read_text())
    if labelled['protocol_sha256']!=E.sha(pp) or any(v['frames']!=601 for v in labelled['totals'].values()):
        raise ValueError('Incomplete early conditional labels')
    E.verify_hashes(labelled['training_sha256'])
    mp = args.out/'models.json'
    if mp.exists():
        raise FileExistsError('Append-only conditional models')
    training = {arm:dict(x=[],y=[]) for arm in cfg['arms']}
    for s in cfg['train']:
        doc = json.loads((args.out/f'training/{s}.json').read_text())
        for arm,a in doc['arms'].items():
            for x,y in zip(a['x'],a['labels']):
                if y>=0:
                    training[arm]['x'].append(x);training[arm]['y'].append(y)
    models = {}
    for arm,data in training.items():
        for family,params in cfg['models'].items():
            models[f'{arm}/{family}'] = E.N.fit(data['x'],data['y'],params)
            print(f'Retained reference fit {arm}/{family}: n{len(data["y"])} positive{sum(data["y"])}',flush=True)
    E.verify_hashes(cfg['source_sha256']);E.verify_hashes(labelled['training_sha256'])
    E.write_new(mp,dict(protocol_sha256=E.sha(pp),models=models,label_completion_sha256=E.sha(lp)))
    print('Retained reference early-only fitting complete; NO full-domain performance claim',flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('crossfit','labels','fit','all'),default='all')
    ap.add_argument('--out',type=Path,default=OUT)
    run(ap.parse_args())
