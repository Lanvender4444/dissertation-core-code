"""Frozen baseline replay / early scoring-OOF references, no target decisions."""
import json
import numpy as np
import experiment_road_support_admission as E
import retained_reference as R
import protected_reference as P
from joint_identity_inputs import Inputs,PAIR
from crossfit_protected_reference import OUT as REFERENCES,PARENT


class ProtectedInputs(Inputs):
    def __init__(self):
        super().__init__()
        self.cap=json.loads(E.CAPACITY.read_text())
        gd=json.loads((E.OLD/'models.json').read_text())
        if E.sha(E.OLD/'models.json')!=self.ref['model_sha256']:raise ValueError('Changed frozen baseline group model')
        self.group_model=E.N.compile_model(gd['models']['medium/group'])
        self.residual_doc=json.loads((PARENT/'models.json').read_text())
        self.residual_model=E.N.compile_model(self.residual_doc['models']['recycle/medium'])
        self.old=json.loads((E.BASE/'algorithm/evaluation_reports/validity_admission_full_results_20261010.json').read_text())
        self.old_rows={r['frame']:r['counts'] for r in self.old['rows']}

    def frame(self,s,early=False):
        pool,changed,descriptor,flags,crops,source=self.load(s)
        geometry=json.loads(self.geometry[s].read_text())
        unknown={d['uid'] for d in geometry['diagnostics']['front0.0']['unknown']}
        if early:
            done=json.loads((REFERENCES/'crossfit_completion.json').read_text())
            if not done['complete'] or done['frames']!=601:raise ValueError('Incomplete scoring OOF reference')
            op=REFERENCES/f'oof_frames/{s}.json';o=E.checked(op,done['proof_sha256'][str(op.relative_to(E.ROOT))])
            reference=o['reference'];source[str(op.relative_to(E.ROOT))]=E.sha(op)
        else:
            frame,_=E.expand(s,self.cap,self.descriptors,self.geometry,self.cameras)
            seed=[g['uids'] for g in frame['groups']];all_seed={u for g in seed for u in g};uids=sorted(pool)
            group_x=np.asarray([g['x'] for g in frame['groups']],float).reshape(len(seed),1056)
            scores=E.N.predict(group_x,self.group_model)
            node=np.asarray([R.features(pool[u],[],unknown,self.cameras,self.offsets,descriptor,.7,
                'recycled_rejected_group' if u in all_seed else 'original_orphan') for u in uids],float).reshape(len(uids),672)
            state=P.assemble(seed,scores,uids,node,pool,unknown,self.cameras,self.residual_model)
            reference=state['reference']
            oldp=E.BASE/f'experiments/validity_admission_20261010/run01/evaluation_full/frames/{s}.json'
            old=E.checked(oldp,self.old['frame_sha256'][str(oldp.relative_to(E.ROOT))])
            expected=old['kept']+[[u] for u,p in zip(old['uids'],old['scores']['novelty']) if p>=.55]
            if reference!=expected:raise ValueError('Frozen reliable baseline changed')
            source[str(oldp.relative_to(E.ROOT))]=E.sha(oldp)
        P.validate(reference,pool)
        return pool,changed,descriptor,flags,crops,source,unknown,reference

    def probabilities(self,pool,descriptor,flags):
        import identity_pair_features as F
        import joint_identity_admission as J
        pairs,x=F.extract(pool,self.cameras,descriptor,flags)
        values=E.N.predict(x,self.pair_model)
        return J.pair_map(pairs,values),pairs,values
