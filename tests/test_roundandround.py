import unittest
import numpy as np
from roundandround import FastLoop, Config, SlowResidueMemory, reconstruct, residues
from benchmark import run


class RoundTests(unittest.TestCase):
    def test_autonomous_motion_and_decay(self):
        f = FastLoop()
        f.restore(np.arange(1, 17.))
        original = f.state.copy()
        future = f.advance(1.)
        self.assertFalse(np.allclose(original, future))
        self.assertAlmostEqual(np.linalg.norm(future), np.linalg.norm(original)*np.exp(-1/18), places=10)
        self.assertTrue(np.allclose(f.advance(0.), future))

    def test_ringdown_is_nonmutating(self):
        f = FastLoop()
        f.advance(.1, np.ones(6))
        before = f.state.copy()
        trace = f.ringdown()
        self.assertEqual(trace.shape, (6, 16))
        self.assertTrue(np.array_equal(f.state, before))

    def test_sihti_telescoping(self):
        rng = np.random.default_rng(5)
        t = rng.normal(size=(8, 16))
        bands = residues(t)
        for i in range(len(t)):
            self.assertTrue(np.allclose(reconstruct(bands,i), t[i]))

    def test_hard_erase_and_query(self):
        m = SlowResidueMemory(1e-4)
        rng = np.random.default_rng(9)
        cues = np.eye(5)
        traces = rng.normal(size=(5, 6, 16))
        m.fit(cues, traces)
        f = FastLoop()
        f.advance(.1, np.ones(6))
        f.reset()
        recovered = m.awaken(cues[2], f)
        self.assertTrue(np.allclose(recovered, f.state))
        self.assertGreater(np.linalg.norm(recovered), .01)
        self.assertLess(np.linalg.norm(recovered-traces[2,0]), .01)

    def test_no_hidden_residue_advantage(self):
        results = run()
        self.assertLess(results['reconstruction_identity_max_error'], 1e-12)
        self.assertAlmostEqual(results['reactivated_rmse'],results['direct_state_ridge_rmse'],places=10)
        self.assertLess(results['reactivated_rmse'], results['wrong_cue_rmse'])
        self.assertLess(results['reactivated_rmse'], results['zero_fast_state_rmse'])
        self.assertLess(results['future_after_reactivation_rmse'], results['zero_fast_state_rmse'])


if __name__ == '__main__': unittest.main()
