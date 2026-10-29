"""Post-Gate-4 multi-loop recall checks (not prior preregistration)."""
import unittest
import numpy as np
from active_reentry import ActiveProtocol,unit,query
from active_reentry_benchmark import learn
from mixture_followup import run_seed
from structural_memory import trace

class MixtureFollowupTests(unittest.TestCase):
    def test_soft_readout_is_weighted_trajectories_not_weighted_operator(self):
        rng=np.random.default_rng(77)
        ops=[np.eye(2),np.array([[0.,-1.],[1.,0.]])]
        q=unit(np.array([1.,2.]));post=np.array([.3,.7])
        y=sum(w*trace(A,q,4) for A,w in zip(ops,post))
        self.assertTrue(np.allclose(y,sum(post[i]*trace(ops[i],q,4) for i in range(2))))
        interpolated=sum(w*A for A,w in zip(ops,post))
        self.assertGreater(np.linalg.norm(y-trace(interpolated,q,4)),.01)

    def test_active_and_fixed_use_same_sensor_budget_and_do_not_write(self):
        p=ActiveProtocol(episodes_per_task=7)
        rng,base,teachers,context,receiver,bank,_,_=learn(p,733)
        before=bank.weights.copy()
        for policy in ('adaptive','fixed','random'):
            ret=query(bank,teachers[0],receiver,context,np.random.default_rng(2),policy=policy)
            self.assertEqual(ret['observations'],2)
            self.assertAlmostEqual(np.sum(ret['posterior']),1.)
        self.assertTrue(np.array_equal(before,bank.weights))

    def test_seedwise_followup_reports_all_controls(self):
        p=ActiveProtocol(episodes_per_task=7)
        r=run_seed(737,p,n=8)
        self.assertIn('soft_adaptive',r['mean_gain'])
        self.assertIn('prior_mixture',r['mean_gain'])
        self.assertIn('soft_fixed',r['mean_gain'])
        self.assertTrue(0<=r['abstain_fraction']<=1)

if __name__=='__main__':unittest.main()
