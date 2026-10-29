"""Gate 2 invariants, fixed budgets, and the matched-rate control."""
import unittest
import numpy as np
from consolidation import Protocol, MultiTimescaleLoop, OnlineAssoc, KeyValueFIFO, sihti_encode, sihti_decode
from consolidation_benchmark import run_seed

class ConsolidationTests(unittest.TestCase):
    def test_elapsed_time_and_inverse(self):
        loop=MultiTimescaleLoop()
        x=np.random.default_rng(2).normal(size=loop.p.dim)
        self.assertTrue(np.allclose(loop.autonomous(x,1.5),
                        loop.autonomous(loop.autonomous(x,.25),1.25)))
        self.assertTrue(np.allclose(loop.inverse_slow(loop.autonomous(x,16),16),x,
                                        rtol=1e-8,atol=1e-8))

    def test_telescope_and_approximation(self):
        loop=MultiTimescaleLoop()
        loop.advance(.35,np.ones(6))
        x=loop.state.copy()
        tr=loop.ringdown()
        self.assertTrue(np.array_equal(loop.state,x))
        self.assertTrue(np.allclose((tr[:-1]-tr[1:]).sum(axis=0)+tr[-1],x))
        code=sihti_encode(tr,loop)
        self.assertEqual(code.shape,(16,))
        self.assertEqual(int(loop.slow_mask.sum()),8)
        self.assertLess(np.linalg.norm(sihti_decode(code,loop)-x),.003)

    def test_equal_charged_budget(self):
        loop=MultiTimescaleLoop()
        rates=np.where(loop.slow_mask,.12,.38)
        a,b=OnlineAssoc(12,16,rates),OnlineAssoc(12,16,rates)
        self.assertEqual(a.n_scalars,192)
        q=np.arange(12,dtype=float)+1.
        v=np.arange(16,dtype=float)-4.
        a.write(q,v);b.write(q,v)
        self.assertTrue(np.array_equal(a.W,b.W))
        fifo=KeyValueFIFO(12,16,192)
        self.assertEqual(fifo.capacity,6)
        for i in range(15):fifo.write(q,v)
        self.assertLessEqual(fifo.n_scalars,192)

    def test_sequential_training_and_identity(self):
        res=run_seed(401,Protocol(seeds=(401,)))
        self.assertEqual(list(res['by_stage']),['after_A','after_B','after_C'])
        self.assertEqual(set(res['by_stage']['after_A']['direct_fast']),{'A'})
        self.assertEqual(set(res['by_stage']['after_C']['direct_fast']),{'A','B','C'})
        self.assertLess(res['sihti_trace_identity_max_error'],1e-12)
        self.assertLess(res['sihti_coding_error'],.005)

    def test_matched_direct_rate_control(self):
        res=run_seed(402,Protocol(seeds=(402,)))
        after=res['by_stage']['after_C']
        for task in 'ABC':
            for metric in ('current_nrmse','late_nrmse'):
                self.assertLess(abs(after['sihti_selective'][task][metric]-
                       after['direct_matched_rates'][task][metric]),1e-4)

if __name__=='__main__':
    unittest.main()
