"""Changing Lens Gate 5: real learned query policy, continual present and controls."""
import unittest
import numpy as np
from structural_memory import base_rotation, unit
from changing_lens import (ContinuingPresent,LensPolicy,LensProtocol,sensing_axes,
    predictions,observe_fragment,bayes_update,fixed_query)
from changing_lens_benchmark import world,train_policy,evaluate


class ChangingLensTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed=931
        cls.base,cls.teachers,cls.ops=world(cls.seed)
        cls.p=LensProtocol(train_trials=300,test_trials=45)
        cls.axes=sensing_axes(cls.p.dim,cls.p.channels,cls.seed+3000)

    def test_only_an_observed_fragment_can_break_identical_priors(self):
        rng=np.random.default_rng(4)
        trigger=unit(rng.normal(size=self.p.dim))
        mu=predictions(self.ops,trigger,self.axes)
        observed=observe_fragment(self.teachers[2],trigger,self.axes,.001,rng)
        none=fixed_query(mu,observed,.001,3,method='none')
        self.assertTrue(np.allclose(none[0],np.ones(3)/3))
        best=fixed_query(mu,observed,.001,3,method='best')
        self.assertGreater(best[0][2],.80)
        self.assertEqual(best[1],3)

    def test_probabilities_are_normalized_and_no_hidden_label_at_query(self):
        rng=np.random.default_rng(5)
        p=LensPolicy(self.p.channels,9)
        mu=rng.normal(size=(3,self.p.channels));obs=rng.normal(size=self.p.channels)
        old=p.theta.copy(),p.bias.copy()
        post,used,actions=p.query(mu,obs,.15,3,mode='greedy')
        self.assertAlmostEqual(float(post.sum()),1.)
        self.assertLessEqual(used,3)
        self.assertEqual(p.training_steps,0)
        self.assertTrue(np.array_equal(p.theta,old[0]))
        self.assertTrue(np.array_equal(p.bias,old[1]))
        with self.assertRaises(ValueError):
            p.query(mu,obs,.15,3,train=True,source=None)

    def test_old_present_never_overwritten_by_a_saved_loop(self):
        o=ContinuingPresent(self.base,.4,seed=5)
        before=o.state.copy()
        no_input=o.advance(1.7)
        self.assertFalse(np.allclose(before,no_input))
        self.assertAlmostEqual(float(np.linalg.norm(no_input)),float(np.linalg.norm(before))*np.exp(-1.7/50),places=10)
        with self.assertRaises(ValueError):o.advance(-1.)
        for _ in range(60):
            p=np.array([.9,.1,0.])
            next_state,baseline=o.step(np.ones(8),self.ops,p)
            self.assertLessEqual(float(np.linalg.norm(next_state-baseline))/.001 if np.linalg.norm(baseline)<.001
                                 else float(np.linalg.norm(next_state-baseline))/np.linalg.norm(baseline),.15000001)
        self.assertEqual(o.resets,0)
        self.assertEqual(o.steps,60)
        self.assertEqual(o.autonomous_updates,1)
        self.assertFalse(np.array_equal(before,o.state))
        self.assertLessEqual(o.max_relative_graze,.15000001)

    def test_training_updates_policy_but_not_memories(self):
        originals=self.ops.copy()
        policy=train_policy(self.ops,self.teachers,self.axes,self.p,self.seed+10,cost=.02,train_trials=350)
        self.assertEqual(policy.training_steps,350)
        self.assertGreater(np.linalg.norm(policy.theta)+np.linalg.norm(policy.bias),0.01)
        self.assertTrue(np.array_equal(originals,self.ops))
        self.assertEqual(policy.scalar_budget,18)

    def test_policy_swap_vs_content_swap(self):
        a=train_policy(self.ops,self.teachers,self.axes,self.p,self.seed+10,cost=.02,train_trials=450)
        b=train_policy(self.ops,self.teachers,self.axes,self.p,self.seed+10,cost=.70,train_trials=450)
        origin=evaluate(self.ops,self.teachers,self.base,self.axes,self.p,1133,{'a':a,'b':b},N=35)
        reversed=evaluate(self.ops,self.teachers,self.base,self.axes,self.p,1133,{'a_got_b':b,'b_got_a':a},N=35)
        self.assertEqual(origin['a']['first_actions'],reversed['b_got_a']['first_actions'])
        self.assertEqual(origin['b']['mean_probes'],reversed['a_got_b']['mean_probes'])
        self.assertTrue(all(v['steps']==35 and v['autonomous_updates']==35 and v['resets']==0 for v in reversed.values()))

if __name__=='__main__':unittest.main()

class AdditionalGuards(unittest.TestCase):
    def test_with_zero_graze_memory_content_cannot_change_present(self):
        base=base_rotation(8)
        rng=np.random.default_rng(10)
        ops=np.stack([base+.15*rng.normal(size=(8,8)) for _ in range(3)])
        z1=ContinuingPresent(base,0.,seed=10)
        z2=ContinuingPresent(base,0.,seed=10)
        for _ in range(30):
            x=rng.normal(size=8)
            a,_=z1.step(x,ops,np.array([1.,0.,0.]))
            b,_=z2.step(x,ops,np.array([0.,0.,1.]))
            self.assertTrue(np.array_equal(a,b))

    def test_self_reading_does_not_modify_slow_operators(self):
        base,teachers,ops=world(939)
        old=ops.copy()
        policy=LensPolicy(6,seed=33)
        axes=sensing_axes(8,6,123)
        _=evaluate(ops,teachers,base,axes,LensProtocol(test_trials=30),93999,{'read_only':policy},N=30)
        self.assertTrue(np.array_equal(ops,old))
