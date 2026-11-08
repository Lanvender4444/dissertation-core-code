"""Synthetic contracts and frozen prepared-feature replay, no detector inference."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fusion_core as F
from demo import replay


class RecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.pool={u:dict(uid=u,src=c,cls=2,conf=.9,box=[10,10,30,40],geometry_valid=False)
                   for u,c in enumerate((2,3,4,2))}
    def test_reference_is_unchanged(self):
        a=[[0,1]];final,_=F.select(a,[[2]],[.8],.55,self.pool,{},False)
        self.assertEqual(a,[[0,1]]);self.assertEqual(final,[[0,1],[2]])
        final[0].append(9);self.assertEqual(a,[[0,1]])
    def test_unknown_location_can_be_accepted(self):
        self.assertEqual(F.select([[0]],[[2]],[.9],.55,self.pool,{},False)[0],[[0],[2]])
    def test_protected_uid_cannot_be_reused(self):
        with self.assertRaises(ValueError):F.select([[0]],[[0]],[.8],.55,self.pool,{},False)
    def test_same_camera_members_are_rejected(self):
        with self.assertRaises(ValueError):F.validate([[0,3]],self.pool)
    def test_duplicate_uid_is_rejected(self):
        with self.assertRaises(ValueError):F.validate([[0],[0]],self.pool)
    def test_unknown_uid_is_rejected(self):
        with self.assertRaises(ValueError):F.validate([[123]],self.pool)
    def test_empty_group_is_rejected(self):
        with self.assertRaises(ValueError):F.validate([[]],self.pool)
    def test_score_threshold_is_inclusive(self):
        self.assertEqual(F.select([],[[0]],[.55],.55,self.pool,{},False)[0],[[0]])
    def test_invalid_scores_are_rejected(self):
        for scores in ([float('nan')],[1.1],[-.1],[]):
            with self.assertRaises(ValueError):F.select([],[[0]],scores,.55,self.pool,{},False)
    def test_independent_policy_retains_competitors(self):
        final,trace=F.select([],[[0],[3]],[.9,.8],.55,self.pool,{(0,3):1.},False)
        self.assertEqual(final,[[0],[3]]);self.assertEqual(trace['batch_retired'],[])
    def test_candidate_positive_gains_are_not_additive(self):
        graph=SimpleNamespace(keys=['one'],scores={0:np.array([.9]),3:np.array([.8])},small_hits=set())
        self.assertEqual(F.counts(graph,F.objects([[0]],self.pool))[:3],[1,0,0])
        self.assertEqual(F.counts(graph,F.objects([[3]],self.pool))[:3],[1,0,0])
        self.assertEqual(F.counts(graph,F.objects([[0],[3]],self.pool))[:3],[1,1,0])
    def test_small_only_ignore_rule(self):
        graph=SimpleNamespace(keys=['one'],scores={0:np.array([0.])},small_hits={0})
        self.assertEqual(F.counts(graph,F.objects([[0]],self.pool))[:3],[0,0,1])
    def test_exported_predictor_checks_dimension(self):
        m=F.compile_model(dict(dim=2,baseline=0.,trees=[]))
        np.testing.assert_array_equal(F.predict([[0.,0.]],m),[.5])
        with self.assertRaises(ValueError):F.predict([[0.]],m)
    def test_three_frozen_examples(self):
        for frame in ('00000641','00000650','00001200'):
            with self.subTest(frame=frame):
                result=replay(frame)
                self.assertLessEqual(result['score_max_abs_error'],1e-12)
                self.assertGreaterEqual(result['counts'][0],result['reference_counts'][0])


if __name__=='__main__':unittest.main()
