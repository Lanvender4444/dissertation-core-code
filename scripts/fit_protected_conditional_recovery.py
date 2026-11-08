"""Early-only conditional novel-object recovery; frozen reliable output stays."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import experiment_road_support_admission as E
import marginal_capacity as M
import protected_conditional_recovery as C
from protected_conditional_inputs import ProtectedInputs,REFERENCES,PARENT,PAIR

OUT=E.BASE/'experiments/protected_conditional_20261010/run01/recovery01'


def labels(groups,reference,ev):
    def edges(gs):return np.asarray([np.maximum.reduce([ev.scores[u]>=.5 for u in g]) for g in gs],bool).reshape(len(gs),len(ev.keys))
    e=edges(groups);y=M.marginal_labels(edges(reference),e)['labels'].astype(int)
    for i,g in enumerate(groups):
        hits=[set(np.flatnonzero(ev.scores[u]>=.5)) for u in g]
        known=set().union(*(h for h in hits if len(h)==1))
        if len(known)>1:y[i]=0
        if not e[i].any() and any(u in ev.small_hits for u in g):y[i]=-1
    return y.tolist()


def run():
    if OUT.exists():raise FileExistsError('Append-only protected recovery')
    inputs=ProtectedInputs();ref=json.loads((REFERENCES/'protocol.json').read_text())
    done=json.loads((REFERENCES/'crossfit_completion.json').read_text())
    if not done['complete'] or done['frames']!=601:raise ValueError('Incomplete nested scoring references')
    E.verify_hashes(done['proof_sha256']);E.verify_hashes(ref['source_sha256'])
    sources=(Path(__file__),E.ROOT/'scripts/evaluate_protected_conditional_recovery.py',
        E.ROOT/'scripts/protected_conditional_inputs.py',E.BASE/'research/protected_conditional_recovery.py',
        E.BASE/'research/protected_reference.py',E.BASE/'research/joint_identity_admission.py',
        E.BASE/'research/identity_measurement_policy.py',E.BASE/'research/marginal_capacity.py',
        E.BASE/'research/nonlinear_admission.py')
    cfg=dict(frames=ref['frames'],**{p:ref[p] for p in ('train','development','guard','late')},
        references_protocol_sha256=E.sha(REFERENCES/'protocol.json'),references_completion_sha256=E.sha(REFERENCES/'crossfit_completion.json'),
        baseline_group_model_sha256=E.sha(E.OLD/'models.json'),baseline_residual_model_sha256=E.sha(PARENT/'models.json'),
        pair_model_sha256=E.sha(PAIR/'models.json'),output_box_sha256=E.sha(E.BOX),policy_sha256=E.sha(E.POLICY),
        source_sha256={str(p.relative_to(E.ROOT)):E.sha(p) for p in sources},
        partition=dict(duplicate_iou=.7,tau=.75,margin=.1,duplicate_tau=.9),params=ref['params'],
        dimensions=dict(single=C.BASE_DIM,pair=C.BASE_DIM,shared=C.DIM),
        taus=[0.,.02,.05,.1,.15,.25,.4,.55,.7,.8,.9,.95,.98,.999],batch_policies=['independent','conservative'],
        interpretation='Immutable old baseline group .7 / residual novelty .55. Only unoccupied sources '
            'enter new singleton or complete-link additional groups; one active bbox/view, all outputs counted. '
            'Scoring reference nested cross-fitted within601early; global early bbox calibration and pair '
            'head NOT cross-fitted, so NOT whole-pipeline OOF or blind scene. New target marginal maximum '
            'matching gain relative to A, known contradictory annotation-proxy groups negative. Unknown '
            'not forced-negative identity and not automatically rejected. Same fixed detector/cache/.05 '
            'and same output correction. Single/pair same1087D, shared only extra8 common-state features. '
            'Both batch policies tested; strict pair duplicate hypothesis not physical guarantee. No GT '
            'used for reference/group/feature inference. No new detector training/default/speed claim.')
    E.write_new(OUT/'protocol.json',cfg);xs={'single':[],'pair':[]};ys={'single':[],'pair':[]};proofs={};totals=Counter()
    for i,s in enumerate(cfg['train'],1):
        pool,changed,descriptor,flags,crops,source,unknown,a=inputs.frame(s,early=True)
        probabilities,pairs,pvalues=inputs.probabilities(pool,descriptor,flags)
        groups,trace=C.partition(a,pool,probabilities,**cfg['partition']);used={u for g in a for u in g}
        single=[[u] for u in sorted(pool) if u not in used]
        # Reference, features and groups precede all current-frame supervision.
        features={k:C.features(g,a,pool,inputs.cameras,inputs.offsets,descriptor,flags,probabilities,unknown)
            for k,g in (('single',single),('pair',groups))}
        ev=E.FrameEvaluator(E.truth(s,(1,2,3,4,5)),list(changed.values()))
        targets={k:labels(g,a,ev) for k,g in (('single',single),('pair',groups))}
        for k in xs:
            y=np.asarray(targets[k]);keep=y>=0;xs[k].extend(features[k][keep]);ys[k].extend(y[keep].tolist())
            totals[k+'_samples']+=int(keep.sum());totals[k+'_positive']+=int(np.sum(y>0))
        fp=OUT/f'training/{s}.json';E.write_new(fp,dict(frame=s,reference=a,groups=dict(single=single,pair=groups),
            labels=targets,trace=trace,source_sha256=source,protocol_sha256=E.sha(OUT/'protocol.json')))
        proofs[str(fp.relative_to(E.ROOT))]=E.sha(fp);totals.update(frames=1,protected_objects=len(a))
        if i==1 or i%100==0:print(f'Protected conditional assembly {i}/601',flush=True)
    models={}
    for arm in ('single','pair','shared'):
        parent='pair' if arm=='shared' else arm
        models[arm]=E.N.fit(np.asarray(xs[parent],float)[:,:cfg['dimensions'][arm]],ys[parent],cfg['params'])
        print(f'Protected conditional {arm} fit complete',flush=True)
    E.verify_hashes(cfg['source_sha256']);E.verify_hashes(proofs)
    E.write_new(OUT/'models.json',dict(models=models,totals=dict(totals),early_training_sha256=proofs,
        protocol_sha256=E.sha(OUT/'protocol.json')))
    print(json.dumps(dict(complete=True,totals=dict(totals),note='No new scene P/R before development/full evaluation')),flush=True)


if __name__=='__main__':run()
