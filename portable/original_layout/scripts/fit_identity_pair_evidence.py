"""Early-only partial-observation identity proxy, not physical ID truth.

Known unique full-corrected or raw clipped-GT matches are training labels ONLY.
Unknown/ambiguous endpoints never become negative physical identities.
Official P/R stays full GT; no clipped GT output or denominator substitution.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np

import experiment_road_support_admission as E
import identity_pair_features as P
import crop_measurement as C
import shared_geometry as S
from experiment_residual_admission import load_pool
from validate_multiview import ev

OUT=E.BASE/'experiments/identity_pair_20261010/early01'


def run():
    if OUT.exists():raise FileExistsError('Append-only identity evidence fit')
    ref=json.loads(E.REFERENCE.read_text());E.verify(ref['protocol']['parent_protocol'])
    E.verify_hashes(ref['protocol']['source_sha256'])
    cfg=dict(train=ref['protocol']['train'],frames=ref['protocol']['frames'],
        **{p:ref['protocol'][p] for p in ('development','guard','late')},floor=.05,dimension=P.DIM,
        reference_sha256=E.sha(E.REFERENCE),source_sha256={str(p.relative_to(E.ROOT)):E.sha(p) for p in (Path(__file__),
            E.BASE/'research/identity_pair_features.py',E.BASE/'research/crop_measurement.py',E.BASE/'research/nonlinear_admission.py')},
        params=dict(ref['protocol']['models']['medium'],max_depth=4,max_leaf_nodes=16,min_samples_leaf=100,max_iter=180),
        interpretation='Early pair identity label only if each endpoint has ONE unambiguous GT-ID witness: '
            'full corrected bbox IoU>=.5 OR raw bbox versus its input-clipped GT IoU>=.5. Combined witnesses '
            'must agree; ambiguous/no witness skipped, never called different physical object. Features built '
            'before early GT, source windows/ROI/rays only. Clipped GT is supervised identity proxy ONLY, '
            'official evaluation still full GT. No detector retraining, output correction or scene decision. '
            'Centre-ray ranges are not reliable stereo points. No speed/physical-ID/new-algorithm success claim.')
    E.write_new(OUT/'protocol.json',cfg)
    descriptors={p.stem:p for p in E.DESCRIPTORS.glob('shard*/frames/*.json')}
    geometry={p.stem:p for p in E.SOURCE.glob('shard*/frames/*.json')}
    cameras,offsets=E.load_cameras(),json.loads(E.POLICY.read_text())['y0'];box=json.loads(E.BOX.read_text())['models']['foveated']
    xs,ys,totals,proofs=[],[],Counter(),{}
    for i,s in enumerate(cfg['train'],1):
        dp=descriptors[s];descriptor=E.checked(dp,ref['descriptor_sha256'][str(dp.relative_to(E.ROOT))])['features']
        record=json.loads(geometry[s].read_text());raw,_=load_pool(record,cameras,prepare=False)
        pool={u:S.prepare_detection(d,cameras) for u,d in raw.items() if d['conf']>=.05}
        crops={u:C.window(d,descriptor[str(u)],offsets)[0] for u,d in pool.items()}
        flags={u:C.edge_flags(d['box'],crops[u]) for u,d in pool.items()}
        pairs,x=P.extract(pool,cameras,descriptor,flags)
        changed=E.corrected(pool,box,cameras,offsets)
        # Pair evidence constructed BEFORE early GT identity supervision.
        evaluator=E.FrameEvaluator(E.truth(s,(1,2,3,4,5)),list(changed.values()))
        witness={};partial_added=0;ambiguous=0
        for u,d in pool.items():
            full={evaluator.keys[j] for j in np.flatnonzero(evaluator.scores[u]>=.5)}
            clipped=set()
            for key,g in evaluator.union.items():
                if d['src'] not in g['boxes'] or not ev.cls_match(g['label'],d['cls']):continue
                b=C.clip(g['boxes'][d['src']],crops[u])
                if b is not None and ev.iou(d['box'],b)>=.5:clipped.add(key)
            ids=full|clipped
            if len(ids)==1:
                witness[u]=next(iter(ids));partial_added+=not full
            else:ambiguous+=len(ids)>1
        labels=[int(witness[a]==witness[b]) if a in witness and b in witness else -1 for a,b in pairs]
        eligible=np.asarray(labels)>=0
        xs.extend(x[eligible]);ys.extend(np.asarray(labels)[eligible].tolist())
        fp=OUT/f'training/{s}.json'
        E.write_new(fp,dict(frame=s,pairs=pairs.tolist(),labels=labels,known_unique_endpoints=len(witness),
            partial_only_endpoints=partial_added,ambiguous_endpoints=ambiguous,
            input_sha256={str(p.relative_to(E.ROOT)):E.sha(p) for p in (dp,geometry[s])},protocol_sha256=E.sha(OUT/'protocol.json')))
        proofs[str(fp.relative_to(E.ROOT))]=E.sha(fp)
        totals.update(frames=1,pairs=len(pairs),eligible_pairs=int(eligible.sum()),positive_pairs=int(np.sum(np.asarray(labels)>0)),
            unique_endpoints=len(witness),partial_only_endpoints=partial_added,ambiguous_endpoints=ambiguous)
        if i==1 or i%200==0:print(f'Early identity pair evidence {i}/601',flush=True)
    model=E.N.fit(xs,ys,cfg['params']);model['target']='Unambiguous same GT-ID witness proxy from early full/clipped observations, not physical identity probability'
    E.verify_hashes(cfg['source_sha256']);E.verify_hashes(proofs)
    E.write_new(OUT/'models.json',dict(model=model,protocol_sha256=E.sha(OUT/'protocol.json'),training_sha256=proofs,totals=dict(totals)))
    print(json.dumps(dict(complete=True,training_only=True,totals=dict(totals),note='No scene P/R or identity truth claim')),flush=True)


if __name__=='__main__':run()
