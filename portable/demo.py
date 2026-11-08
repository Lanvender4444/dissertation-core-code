"""Standalone score/selection replay using bundled prepared features; no GT admission."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import fusion_core as F

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(frame):
    record=json.loads((ROOT/f'examples/{frame}.json').read_text())
    npz=ROOT/f'examples/{frame}.npz'
    if digest(npz)!=record['npz_sha256']:raise ValueError('Fixture data changed')
    with np.load(npz,allow_pickle=False) as data:
        x=data['features'];expected_scores=data['expected_scores']
        probabilities={tuple(map(int,k)):float(v) for k,v in zip(data['pairs'],data['pair_scores'])}
        model=F.compile_model(json.loads((ROOT/'models/recovery_single.json').read_text()))
        scores=F.predict(x,model)
        error=float(np.max(np.abs(scores-expected_scores))) if len(scores) else 0.
        if error>1e-12:raise ValueError('Portable scores differ')
        pool={int(k):v for k,v in record['pool'].items()}
        final,trace=F.select(record['reference'],record['candidates'],scores,.55,pool,probabilities,False)
        if final!=record['expected_final']:raise ValueError('Selected group membership differs')
        if final[:len(record['reference'])]!=record['reference']:raise ValueError('Protected reference changed')
        # Offline evaluator begins only after admission. Its IoU graph never enters selection.
        graph=SimpleNamespace(keys=record['evaluation_keys'],small_hits=set(record['small_hits']),
              scores={int(u):row for u,row in zip(data['uids'],data['evaluation_iou'])})
        corrected={int(k):v for k,v in record['corrected'].items()}
        result=F.counts(graph,F.objects(final,corrected))
        base=F.counts(graph,F.objects(record['reference'],corrected))
    if result!=record['expected_counts'] or base!=record['reference_counts']:
        raise ValueError('Offline counts differ')
    if result[0]<base[0]:raise ValueError('Protected capacity decreased')
    tp,fp,fn=result[:3];p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return dict(frame=frame,counts=result,reference_counts=base,objects=len(final),added=len(trace['added']),
                precision=p,recall=r,f1=2*p*r/max(p+r,1e-12),score_max_abs_error=error,
                interpretation='These are bundled-frame checks, not the full-sequence thesis result.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--frame',choices=('00000641','00000650','00001200'))
    args=ap.parse_args()
    for frame in ([args.frame] if args.frame else ['00000641','00000650','00001200']):
        result=replay(frame);print(json.dumps(result),flush=True)
        print(f"{frame}: TP={result['counts'][0]} FP={result['counts'][1]} FN={result['counts'][2]} | "
              f"P={result['precision']:.2%} R={result['recall']:.2%} F1={result['f1']:.2%}",flush=True)
