import unittest
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linear_sum_assignment
import experiment_road_support_admission
import protected_conditional_recovery as C


class Contracts(unittest.TestCase):
    def data(self):
        cams=(2,3,2,4)
        pool={u:dict(uid=u,src=c,cls=2,conf=.8,box=[10.,10.,30.,40.],geometry_valid=False) for u,c in enumerate(cams)}
        cameras={c:SimpleNamespace(width=100.,height=100.) for c in cams}
        return pool,cameras

    def test_partition_excludes_protected_sources(self):
        pool,_=self.data();groups,trace=C.partition([[0,1]],pool,{(0,2):1.,(2,3):1.})
        self.assertEqual(groups,[[2,3]]);self.assertEqual(trace['retired_same_camera'],{})

    def test_context_without_self_evidence(self):
        pool,cameras=self.data();x=C.context([2,3],[[0,1]],pool,{3},cameras,{(0,2):.99,(1,3):.8})
        self.assertEqual(x.shape,(24,));self.assertTrue(np.isfinite(x).all())
        with self.assertRaises(ValueError):C.context([0],[[0,1]],pool,set(),cameras,{})

    def test_unknown_location_is_not_rejected(self):
        pool,_=self.data();a=[[0,1]];actual,trace=C.select(a,[[2],[3]],[.8,.7],.5,pool,{},True)
        self.assertEqual(actual,[[0,1],[2],[3]]);self.assertEqual(a,[[0,1]])
        self.assertEqual(trace['batch_retired'],[])

    def test_batch_duplicate_still_counted_as_decision(self):
        pool,_=self.data();actual,trace=C.select([[1]],[[0],[2]],[.8,.9],.5,pool,{(0,2):.99},True)
        self.assertEqual(actual,[[1],[2]]);self.assertEqual(trace['batch_retired'],[0])
        allout,_=C.select([[1]],[[0],[2]],[.8,.9],.5,pool,{(0,2):.99},False)
        self.assertEqual(len(allout),3)

    def test_separated_boxes_cannot_be_retired_by_pair_score_alone(self):
        pool,_=self.data();pool[2]['box']=[60.,60.,80.,90.]
        self.assertFalse(C.conflicts([0],[2],pool,{(0,2):1.}))

    def test_alternative_bbox_bag_and_reuse_forbidden(self):
        pool,_=self.data()
        with self.assertRaises(ValueError):C.select([[1]],[[0,2]],[.8],.5,pool,{})
        with self.assertRaises(ValueError):C.select([[1]],[[1]],[.8],.5,pool,{})

    def test_immutable_reference_matching_theorem(self):
        def capacity(edges):
            if not edges.size:return 0
            r,c=linear_sum_assignment(-edges.astype(int));return int(edges[r,c].sum())
        rng=np.random.default_rng(20261010)
        for _ in range(300):
            a=rng.random((int(rng.integers(0,10)),12))<.2;b=rng.random((int(rng.integers(0,10)),12))<.2
            self.assertGreaterEqual(capacity(np.vstack((a,b))),capacity(a))


if __name__=='__main__':unittest.main()
