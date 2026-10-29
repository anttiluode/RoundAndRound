"""Gate 3 invariants and causal negative controls; no model-training at import."""
import unittest
import numpy as np
from structural_memory import (Protocol,StructuralBank,DirectTransitionBank,
    SharedOperator,OracleIndexedOperators,FIFOPairs,sample_world,trace,unit,
    sihti_differences,base_rotation)
from structural_benchmark import run_seed

class StructuralTests(unittest.TestCase):
    def test_ringdown_residue_identity_and_stability(self):
        rng=np.random.default_rng(11)
        _,base,W=sample_world(Protocol(),rng)
        t=trace(base+W[0],rng.normal(size=8),5)
        self.assertTrue(np.allclose(t[:-1]-sihti_differences(t),t[1:]))
        self.assertTrue(np.allclose(t[0]-np.sum(sihti_differences(t),axis=0),t[-1]))
        self.assertLess(np.linalg.norm(t[-1]),np.linalg.norm(t[0]))

    def test_teach_silence_probe_and_novel_direction(self):
        p=Protocol(train_episodes=15)
        rng=np.random.default_rng(12)
        centers,base,teachers=sample_world(p,rng)
        mem=StructuralBank(p,base)
        for _ in range(15):
            x=unit(rng.normal(size=p.dim))
            mem.write(centers[0],trace(base+teachers[0],x,p.trace_steps))
        self.assertEqual(mem.active,1)
        # No fast activity survives in the memory object. Only operator+key.
        x=unit(rng.normal(size=p.dim))
        new=mem.awaken(centers[0],x,p.trace_steps)
        original=trace(base+teachers[0],x,p.trace_steps)
        blank=trace(base,x,p.trace_steps)
        self.assertTrue(np.array_equal(new[0],blank[0]))
        self.assertLess(np.linalg.norm(new[1:]-original[1:]),np.linalg.norm(blank[1:]-original[1:]))
        self.assertIsNone(mem.awaken(centers[2],x,p.trace_steps))

    def test_bound_contexts_cannot_silently_replace_one_another(self):
        p=Protocol(train_episodes=15)
        rng=np.random.default_rng(13)
        centers,base,teachers=sample_world(Protocol(task_count=4),rng)
        bank=StructuralBank(p,base)
        for k in range(3):
            bank.write(centers[k],trace(base+teachers[k],rng.normal(size=8),p.trace_steps))
        prior=bank.weights.copy()
        for _ in range(10):
            self.assertIsNone(bank.write(centers[3],trace(base+teachers[3],rng.normal(size=8),p.trace_steps)))
        self.assertEqual(bank.denied,10)
        self.assertTrue(np.array_equal(prior,bank.weights))

    def test_matched_direct_transition_learner_is_same_algorithm(self):
        p=Protocol(train_episodes=5)
        rng=np.random.default_rng(14)
        centers,base,teachers=sample_world(p,rng)
        a,b=StructuralBank(p,base),DirectTransitionBank(p,base)
        for k in range(3):
            for _ in range(5):
                c=unit(centers[k]+rng.normal(scale=.12,size=p.context_dim))
                t=trace(base+teachers[k],rng.normal(size=p.dim),p.trace_steps)
                a.write(c,t);b.write(c,t)
        self.assertEqual(a.scalar_budget,b.scalar_budget)
        self.assertLess(np.max(np.abs(a.weights-b.weights)),1e-12)
        self.assertLess(np.max(np.abs(a.centers-b.centers)),1e-12)
        self.assertEqual(a.scalar_budget,219)

    def test_shared_interference_and_context_collision(self):
        p=Protocol(train_episodes=14,test_probes=16)
        normal=run_seed(15,p)
        collision=run_seed(15,p,collision=True)
        self.assertAlmostEqual(normal['by_stage']['after_C']['structural']['A']['gain_over_base'],
                               normal['by_stage']['after_A']['structural']['A']['gain_over_base'],places=12)
        self.assertGreater(normal['by_stage']['after_C']['structural']['A']['gain_over_base'],
                           normal['by_stage']['after_C']['shared']['A']['gain_over_base'])
        self.assertLess(collision['by_stage']['after_C']['structural']['A']['gain_over_base'],
                        normal['by_stage']['after_C']['structural']['A']['gain_over_base'])

if __name__ == '__main__':unittest.main()
