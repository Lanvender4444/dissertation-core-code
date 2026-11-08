import unittest
from types import SimpleNamespace
import numpy as np
import experiment_road_support_admission
import protected_reference as P
import residual_recovery as C


class Contracts(unittest.TestCase):
    def data(self):
        pool={u:dict(uid=u,src=c,cls=2,conf=.8,box=[10.+u,10.,30.+u,40.],geometry_valid=False) for u,c in enumerate((2,3,2))}
        cameras={c:SimpleNamespace(width=100.,height=100.) for c in (2,3)}
        return pool,cameras

    def test_only_context_changes_and_matches_legacy(self):
        pool,cameras=self.data();x=np.arange(3*672,dtype=float).reshape(3,672);old=x.copy()
        actual=P.contextualize([0,1,2],x,[2],[[0,1]],pool,set(),cameras)
        np.testing.assert_array_equal(actual[0,:278],x[2,:278]);np.testing.assert_array_equal(actual[0,285:],x[2,285:])
        np.testing.assert_array_equal(actual[0,278:285],C.features(pool[2],P.objects([[0,1]],pool),set(),cameras)[41:])
        np.testing.assert_array_equal(x,old)

    def test_never_reuses_protected_uid(self):
        pool,cameras=self.data()
        with self.assertRaises(ValueError):P.contextualize([0,1,2],np.zeros((3,672)),[0],[[0,1]],pool,set(),cameras)
        with self.assertRaises(ValueError):P.append([[0,1]],[[1]],pool)

    def test_append_preserves_original_groups(self):
        pool,_=self.data();a=[[0,1]];result=P.append(a,[[2]],pool)
        self.assertEqual(result,[[0,1],[2]]);self.assertEqual(a,[[0,1]])

    def test_alternative_same_view_bbox_forbidden(self):
        pool,_=self.data()
        with self.assertRaises(ValueError):P.validate([[0,2]],pool)

    def test_empty_candidates(self):
        pool,cameras=self.data();x=P.contextualize([0,1,2],np.zeros((3,672)),[],[[0,1]],pool,set(),cameras)
        self.assertEqual(x.shape,(0,672))


if __name__=='__main__':unittest.main()
