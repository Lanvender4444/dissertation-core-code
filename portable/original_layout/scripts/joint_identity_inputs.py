"""Frozen same-input loaders for joint identity experiments; no GT access."""
import json
import numpy as np
import experiment_road_support_admission as E
import identity_pair_features as P
import joint_identity_admission as J
import crop_measurement as C
import shared_geometry as S
from experiment_residual_admission import load_pool
from fit_identity_pair_evidence import OUT as PAIR


class Inputs:
    def __init__(self):
        self.ref=json.loads(E.REFERENCE.read_text())
        self.pair_cfg=json.loads((PAIR/'protocol.json').read_text())
        self.pair_doc=json.loads((PAIR/'models.json').read_text())
        if self.pair_doc['protocol_sha256']!=E.sha(PAIR/'protocol.json'):raise ValueError('Changed pair protocol')
        E.verify_hashes(self.pair_cfg['source_sha256']);E.verify(self.ref['protocol']['parent_protocol'])
        self.pair_model=E.N.compile_model(self.pair_doc['model'])
        self.descriptors={p.stem:p for p in E.DESCRIPTORS.glob('shard*/frames/*.json')}
        self.geometry={p.stem:p for p in E.SOURCE.glob('shard*/frames/*.json')}
        self.cameras=E.load_cameras();self.offsets=json.loads(E.POLICY.read_text())['y0']
        self.box=json.loads(E.BOX.read_text())['models']['foveated']

    def load(self,s):
        dp=self.descriptors[s];gp=self.geometry[s]
        descriptor=E.checked(dp,self.ref['descriptor_sha256'][str(dp.relative_to(E.ROOT))])['features']
        raw,_=load_pool(json.loads(gp.read_text()),self.cameras,prepare=False)
        pool={u:S.prepare_detection(d,self.cameras) for u,d in raw.items() if d['conf']>=.05}
        crops={u:C.window(d,descriptor[str(u)],self.offsets)[0] for u,d in pool.items()}
        flags={u:C.edge_flags(d['box'],crops[u]) for u,d in pool.items()}
        changed=E.corrected(pool,self.box,self.cameras,self.offsets)
        proof={str(p.relative_to(E.ROOT)):E.sha(p) for p in (dp,gp)}
        return pool,changed,descriptor,flags,crops,proof

    def infer(self,pool,descriptor,flags,cfg):
        pairs,x=P.extract(pool,self.cameras,descriptor,flags)
        values=E.N.predict(x,self.pair_model);probabilities=J.pair_map(pairs,values)
        groups,trace=J.partition(pool,probabilities,**cfg['partition'])
        return groups,trace,pairs,values,probabilities

    def features(self,groups,pool,descriptor,flags,probabilities):
        return np.asarray([J.features(g,pool,self.cameras,self.offsets,descriptor,flags,probabilities)
            for g in groups],float).reshape(len(groups),J.DIM)
