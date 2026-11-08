"""Nested early-only scoring references for protected conditional recovery.

Frozen globally early-trained bbox correction is NOT cross-fitted. Only the
group/residual scoring DAG excludes held-frame labels; no blind-scene claim.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import experiment_road_support_admission as E
import retained_reference as R
import protected_reference as P
import marginal_capacity as M
from joint_identity_inputs import Inputs
from fit_retained_reference import OUT as PARENT

OUT=E.BASE/'experiments/protected_conditional_20261010/run01/reference01'


def save_arrays(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)


def prepare():
    if OUT.exists():raise FileExistsError('Append-only protected reference preparation')
    inputs=Inputs();parent=json.loads((PARENT/'protocol.json').read_text())
    done=json.loads((PARENT/'crossfit_completion.json').read_text())
    E.verify_hashes(parent['source_sha256']);E.verify_hashes(done['fold_model_sha256'])
    folds=R.training_folds(parent['train'])
    if folds!=parent['crossfit'] or done['totals']['frames']!=601:raise ValueError('Changed group outer cross-fitting')
    cfg=dict(frames=parent['frames'],**{k:parent[k] for k in ('train','development','guard','late')},
        folds=folds,inner_folds={str(i):R.training_folds(f['train']) for i,f in enumerate(folds)},
        params=parent['models']['medium'],floor=.05,group_tau=.7,residual_tau=.55,reference_sha256=E.sha(E.REFERENCE),
        parent_protocol_sha256=E.sha(PARENT/'protocol.json'),parent_crossfit_sha256=E.sha(PARENT/'crossfit_completion.json'),
        original_group_fold_sha256=done['fold_model_sha256'],output_box_sha256=E.sha(E.BOX),policy_sha256=E.sha(E.POLICY),
        source_sha256={str(p.relative_to(E.ROOT)):E.sha(p) for p in (Path(__file__),E.BASE/'research/protected_reference.py',
            E.BASE/'research/retained_reference.py',E.BASE/'research/residual_recovery.py',E.BASE/'research/nonlinear_admission.py',
            E.ROOT/'scripts/joint_identity_inputs.py')},
        interpretation='Five contiguous outer scoring folds with five-frame gaps; each residual outer head '
            'gets training labels/features from five INNER group folds confined to its outer train set. '
            'Existing outer group models reused exactly. No old all601 group-OOF training labels are reused '
            'for the new outer residual head, avoiding indirect held labels through those parent group fits. '
            'Only601early GT for preparation/fitting; no development/guard/late labels or fitting. '
            'FROZEN globally601 early bbox calibration is NOT cross-fitted: this is scoring-head DAG '
            'cross-fitting, not whole-pipeline independence from held early GT. Pair head is not used here. '
            'Final deployed baseline remains exact old group .7 / novelty .55. No new P/R, speed or physical identity claim.')
    E.write_new(OUT/'protocol.json',cfg)
    cap=json.loads(E.CAPACITY.read_text());proofs={};totals=Counter();max_control_error=0.
    parentlabels=json.loads((PARENT/'label_completion.json').read_text())
    for i,s in enumerate(cfg['train'],1):
        pool,changed,descriptor,flags,crops,source=inputs.load(s)
        record=json.loads(inputs.geometry[s].read_text());unknown={d['uid'] for d in record['diagnostics']['front0.0']['unknown']}
        frame,_=E.expand(s,cap,inputs.descriptors,inputs.geometry,inputs.cameras)
        seed=[g['uids'] for g in frame['groups']];P.validate(seed,pool)
        all_seed={u for g in seed for u in g};uids=sorted(pool)
        node=np.asarray([R.features(pool[u],[],unknown,inputs.cameras,inputs.offsets,descriptor,.7,
            'recycled_rejected_group' if u in all_seed else 'original_orphan') for u in uids],float).reshape(len(uids),P.DIM)
        # Replay the old cross-fitted feature context without reading its labels.
        sp=PARENT/f'crossfit_scores/{s}.json';oldscore=json.loads(sp.read_text())
        kept,candidates,x=P.candidate_features(seed,oldscore['scores'],uids,node,pool,unknown,inputs.cameras)
        tp=PARENT/f'training/{s}.json';old=E.checked(tp,parentlabels['training_sha256'][str(tp.relative_to(E.ROOT))])['arms']['recycle']
        if old['kept']!=kept or old['uids']!=candidates:raise ValueError('Original feature control source mismatch')
        error=float(np.max(np.abs(x-np.asarray(old['x'])))) if len(x) else 0.
        max_control_error=max(max_control_error,error)
        if error>1e-12:raise ValueError('Context-only replacement differs from original 672D')
        group_x=np.asarray([g['x'] for g in frame['groups']],float).reshape(len(seed),1056)
        # Features, raw groups and output measurements constructed before GT.
        ev=E.FrameEvaluator(E.truth(s,(1,2,3,4,5)),list(changed.values()))
        edges=np.asarray([ev.scores[u]>=.5 for u in uids],bool).reshape(len(uids),len(ev.keys));rows={u:j for j,u in enumerate(uids)}
        labels=[1 if any(edges[rows[u]].any() for u in g) else -1 if any(u in ev.small_hits for u in g) else 0 for g in seed]
        if labels!=[g['label'] for g in frame['groups']]:raise ValueError('Frozen group label arithmetic mismatch')
        ap=OUT/f'prepared/{s}.npz';save_arrays(ap,node_x=node,group_x=group_x,group_y=np.asarray(labels,int),node_edges=edges)
        compact={str(u):{k:pool[u][k] for k in ('uid','src','cls','conf','box','feature','geometry_valid','angular_h')} for u in uids}
        for d in compact.values():
            d['feature']=np.asarray(d['feature']).tolist();d['angular_h']=float(d['angular_h']) if d['geometry_valid'] else 0.
        jp=OUT/f'prepared/{s}.json'
        E.write_new(jp,dict(frame=s,seed=seed,uids=uids,pool=compact,unknown=sorted(unknown),small_hits=sorted(ev.small_hits),
            gt_keys=ev.keys,array_sha256=E.sha(ap),source_sha256=source,protocol_sha256=E.sha(OUT/'protocol.json')))
        for p in (ap,jp):proofs[str(p.relative_to(E.ROOT))]=E.sha(p)
        totals.update(frames=1,observations=len(uids),groups=len(seed))
        if i==1 or i%100==0:print(f'Protected reference early preparation {i}/601',flush=True)
    E.verify_hashes(cfg['source_sha256'])
    E.write_new(OUT/'preparation_completion.json',dict(complete=True,totals=dict(totals),
        prepared_sha256=proofs,protocol_sha256=E.sha(OUT/'protocol.json'),original_672D_max_error=max_control_error,
        early_group_labels_recomputed=True))
    print('Protected reference preparation complete; no scoring/full-domain result yet',flush=True)


def load_prepared(s,done):
    jp=OUT/f'prepared/{s}.json';ap=OUT/f'prepared/{s}.npz'
    frame=E.checked(jp,done['prepared_sha256'][str(jp.relative_to(E.ROOT))])
    if E.sha(ap)!=frame['array_sha256'] or E.sha(ap)!=done['prepared_sha256'][str(ap.relative_to(E.ROOT))]:raise ValueError('Changed prepared array')
    with np.load(ap,allow_pickle=False) as a:frame.update({k:a[k].copy() for k in a.files})
    frame['pool']={int(u):d for u,d in frame['pool'].items()}
    return frame


def matrices(groups,frame):
    rows={u:i for i,u in enumerate(frame['uids'])};e=frame['node_edges']
    return np.asarray([np.maximum.reduce([e[rows[u]] for u in g]) for g in groups],bool).reshape(len(groups),len(frame['gt_keys']))


def residual_training(frame,scores,cameras):
    kept,uids,x=P.candidate_features(frame['seed'],scores,frame['uids'],frame['node_x'],frame['pool'],set(frame['unknown']),cameras)
    edges=matrices([[u] for u in uids],frame);y=M.marginal_labels(matrices(kept,frame),edges)['labels'].astype(int)
    y[[not edges[i].any() and u in frame['small_hits'] for i,u in enumerate(uids)]]=-1
    keep=y>=0
    return x[keep],y[keep]


def fit_fold(index):
    cfg=json.loads((OUT/'protocol.json').read_text());done=json.loads((OUT/'preparation_completion.json').read_text())
    if not done['complete'] or done['totals']['frames']!=601 or done['protocol_sha256']!=E.sha(OUT/'protocol.json'):
        raise ValueError('Incomplete preparation')
    E.verify_hashes(cfg['source_sha256'])
    fold=cfg['folds'][index];target=OUT/f'folds/{index}/completion.json'
    if target.exists() or (OUT/f'folds/{index}').exists():raise FileExistsError('Append-only scoring fold')
    gp=PARENT/f'fold_models/{index}.json';gd=E.checked(gp,cfg['original_group_fold_sha256'][str(gp.relative_to(E.ROOT))])
    if gd['train']!=fold['train'] or gd['held']!=fold['held']:raise ValueError('Outer group model has wrong dependency DAG')
    frames={s:load_prepared(s,done) for s in fold['train']+fold['held']};inner_scores={};proofs={};cameras=E.load_cameras()
    for j,inner in enumerate(cfg['inner_folds'][str(index)]):
        if set(inner['train']+inner['held'])-set(fold['train']) or set(inner['train'])&set(inner['held']) or set(inner['train'])&set(fold['held']):
            raise ValueError('Held-frame indirect group-label leakage')
        x=np.vstack([frames[s]['group_x'][frames[s]['group_y']>=0] for s in inner['train']])
        y=np.concatenate([frames[s]['group_y'][frames[s]['group_y']>=0] for s in inner['train']])
        model=E.N.fit(x,y,cfg['params']);mp=OUT/f'folds/{index}/inner_group_{j}.json'
        E.write_new(mp,dict(model=model,train=inner['train'],held=inner['held'],outer_held=fold['held'],protocol_sha256=E.sha(OUT/'protocol.json')))
        proofs[str(mp.relative_to(E.ROOT))]=E.sha(mp);compiled=E.N.compile_model(model)
        for s in inner['held']:inner_scores[s]=E.N.predict(frames[s]['group_x'],compiled).tolist()
        print(f'Protected scoring outer{index} inner-group{j+1}/5 complete',flush=True)
    if set(inner_scores)!=set(fold['train']):raise ValueError('Incomplete nested training references')
    xs,ys=[],[]
    for s in fold['train']:
        x,y=residual_training(frames[s],inner_scores[s],cameras);xs.extend(x);ys.extend(y.tolist())
    residual=E.N.fit(xs,ys,cfg['params']);mp=OUT/f'folds/{index}/residual.json'
    E.write_new(mp,dict(model=residual,train=fold['train'],held=fold['held'],inner_group_sha256=proofs.copy(),
        params=cfg['params'],protocol_sha256=E.sha(OUT/'protocol.json')))
    proofs[str(mp.relative_to(E.ROOT))]=E.sha(mp);cm=E.N.compile_model(residual);gm=E.N.compile_model(gd['model'])
    for s in fold['held']:
        f=frames[s];gs=E.N.predict(f['group_x'],gm).tolist()
        state=P.assemble(f['seed'],gs,f['uids'],f['node_x'],f['pool'],set(f['unknown']),cameras,cm)
        op=OUT/f'oof_frames/{s}.json'
        E.write_new(op,dict(frame=s,fold=index,**state,group_scores=gs,protocol_sha256=E.sha(OUT/'protocol.json'),
            outer_group_model_sha256=E.sha(gp),outer_residual_model_sha256=E.sha(mp),prepared_sha256=E.sha(OUT/f'prepared/{s}.json')))
        proofs[str(op.relative_to(E.ROOT))]=E.sha(op)
    E.verify_hashes(cfg['source_sha256']);E.verify_hashes(proofs)
    E.write_new(target,dict(complete=True,fold=index,held_frames=len(fold['held']),train=fold['train'],held=fold['held'],
        proof_sha256=proofs,protocol_sha256=E.sha(OUT/'protocol.json'),new_group_models=5,new_residual_models=1,
        note='Only scoring DAG excludes held labels; frozen601 bbox correction NOT cross-fitted.'))
    print(f'Protected scoring fold{index} complete held{len(fold["held"])}',flush=True)


def complete():
    cfg=json.loads((OUT/'protocol.json').read_text());target=OUT/'crossfit_completion.json'
    if target.exists():raise FileExistsError('Append-only nested scoring completion')
    proofs={};held=[]
    for i in range(5):
        p=OUT/f'folds/{i}/completion.json';d=json.loads(p.read_text())
        if not d['complete'] or d['protocol_sha256']!=E.sha(OUT/'protocol.json') or d['held']!=cfg['folds'][i]['held']:
            raise ValueError('Incomplete/changed scoring fold')
        E.verify_hashes(d['proof_sha256']);proofs.update(d['proof_sha256']);proofs[str(p.relative_to(E.ROOT))]=E.sha(p);held+=d['held']
    if sorted(held)!=cfg['train'] or len(held)!=len(set(held)):raise ValueError('Incomplete/duplicate early prediction')
    E.verify_hashes(cfg['source_sha256'])
    E.write_new(target,dict(complete=True,frames=601,protocol_sha256=E.sha(OUT/'protocol.json'),proof_sha256=proofs,
        nested_new_group_models=25,new_outer_residual_models=5,existing_outer_group_models=5,
        limitations=cfg['interpretation']))
    print('Protected scoring nested cross-fitting complete; no new full P/R claim',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','fold','completion'),required=True)
    ap.add_argument('--fold',type=int,choices=range(5));args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='fold':
        if args.fold is None:ap.error('--fold required')
        fit_fold(args.fold)
    else:complete()
