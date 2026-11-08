"""Exploratory paired temporal-block intervals from frozen prediction counts.

No inference, fitting, threshold selection or ground-truth edits. Intervals do
not correct repeated model selection or prove independent-scene generalisation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'ultralytics/algorithm/evaluation_reports/protected_conditional_full_results_20261010.json'
PROTOCOL=ROOT/'ultralytics/experiments/protected_conditional_20261010/run01/recovery01/protocol.json'
KEY='single/independent/tau0.55'


def metrics(counts):
    t,p,n=np.moveaxis(np.asarray(counts,dtype=float),-1,0)
    return np.stack((t/np.maximum(t+p,1),t/np.maximum(t+n,1),2*t/np.maximum(2*t+p+n,1)),axis=-1)


def run(args):
    if args.out.exists():raise FileExistsError('Append-only interval record')
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    if not source['complete'] or source['frames']!=1655:raise ValueError('Incomplete frozen source')
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()!=source['protocol_sha256']:raise ValueError('Protocol differs from frozen evaluation')
    rows=sorted(source['rows'],key=lambda r:int(r['frame']))
    if [r['frame'] for r in rows]!=sorted(protocol['frames'],key=int) or len({r['frame'] for r in rows})!=1655:
        raise ValueError('Wrong recorded sequence')
    later=set(protocol['guard'])|set(protocol['late'])
    result={}
    for name,selected in (('all',rows),('guard_late',[r for r in rows if r['frame'] in later])):
        a=np.asarray([r['counts']['protected'][:3] for r in selected],int)
        b=np.asarray([r['counts'][KEY][:3] for r in selected],int)
        if np.any(a[:,0]>b[:,0]) or np.any(a[:,0]+a[:,2]!=b[:,0]+b[:,2]):raise ValueError('Changed protected graph/domain')
        for key,counts in (('protected',a),(KEY,b)):
            if counts.sum(axis=0).tolist()!=[source['splits'][name][key][f] for f in ('tp','fp','fn')]:
                raise ValueError('Domain totals do not close')
        point=100*(metrics(b.sum(axis=0))-metrics(a.sum(axis=0)))
        by_size={}
        for size in (10,25,50):
            blocks_a=np.asarray([a[i:i+size].sum(axis=0) for i in range(0,len(a),size)])
            blocks_b=np.asarray([b[i:i+size].sum(axis=0) for i in range(0,len(b),size)])
            rng=np.random.default_rng(args.seed+size)
            indices=rng.integers(0,len(blocks_a),size=(args.replicates,len(blocks_a)))
            delta=100*(metrics(blocks_b[indices].sum(axis=1))-metrics(blocks_a[indices].sum(axis=1)))
            interval=np.percentile(delta,[2.5,97.5],axis=0)
            by_size[str(size)]={field:dict(point_pp=float(point[i]),low_pp=float(interval[0,i]),high_pp=float(interval[1,i]))
                               for i,field in enumerate(('precision','recall','f1'))}
        result[name]=dict(frames=len(selected),reference_counts=a.sum(axis=0).tolist(),
                         final_counts=b.sum(axis=0).tolist(),block_sensitivity=by_size)
    record=dict(complete=True,date='2026-10-13',source=str(SOURCE.relative_to(ROOT)),
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                configuration=KEY,replicates=args.replicates,seed=args.seed,
                method='Paired non-overlapping blocks of retained chronological observations, percentile 95% intervals; '
                       'sample the same block indices for both methods. A final shorter block is retained. '
                       'Recorded index gaps are not assigned synthetic counts. Resampled length can vary; '
                       'metrics are computed from summed TP/FP/FN, not averaged frame ratios.',
                limitations='Exploratory within one repeatedly studied sequence; full includes fitting data. '
                            'Block lengths 10/25/50 are sensitivity checks, not an estimated decorrelation time. '
                            'No correction for repeated selection, physical-ID ambiguity or new-scene generalisation.',
                results=result)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.out),results=result)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--replicates',type=int,default=2000);ap.add_argument('--seed',type=int,default=20261013)
    args=ap.parse_args()
    if args.replicates<100:raise ValueError('Too few replicates')
    run(args)
