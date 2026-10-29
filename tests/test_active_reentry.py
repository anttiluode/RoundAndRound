"""Gate 4: cue collision, teacher-free routing, active experiments and write guards."""
import unittest
import numpy as np
from active_reentry import (ActiveProtocol, TransitionMemory, world, observations,
                            query, unit, optimized_probe, receiver_response)
from active_reentry_benchmark import learn, run_seed


class ActiveReentryTests(unittest.TestCase):
    def test_discover_operators_despite_identical_context_centers(self):
        p=ActiveProtocol(episodes_per_task=7)
        rng,base,teachers,context,receiver,bank,direct,routes=learn(p,709)
        self.assertEqual(bank.active,3)
        self.assertTrue(all(a==b for a,b in routes))
        self.assertEqual(bank.scalar_budget,219)
        self.assertLess(np.max(np.abs(bank.weights-direct.weights)),1e-12)
        self.assertLess(max(np.linalg.norm(bank.candidates()[i]-teachers[i]) for i in range(3)),.04)

    def test_response_channel_needed_to_resolve_identical_cues(self):
        p=ActiveProtocol(episodes_per_task=8)
        rng,base,teachers,context,receiver,bank,_,_=learn(p,711)
        self.assertTrue(np.allclose(bank.cue_prior(context),[1/3]*3))
        for t in range(3):
            self.assertEqual(query(bank,teachers[t],receiver,context,rng,'cue_only')['slot'],0)
        # A physical response (one scalar) may distinguish the loops.
        a=query(bank,teachers[1],receiver,context,np.random.default_rng(300),policy='adaptive',budget=2,sensor_noise=0.0+1e-5)
        self.assertEqual(a['slot'],1)

    def test_adaptive_second_probe_uses_first_response(self):
        p=ActiveProtocol(episodes_per_task=8)
        rng,base,teachers,context,receiver,bank,_,_=learn(p,713)
        outcomes=[]
        for task in range(3):
            r=query(bank,teachers[task],receiver,context,np.random.default_rng(99),policy='adaptive',sensor_noise=.045)
            outcomes.append(r)
        first=[r['records'][0][0] for r in outcomes]
        second=[r['records'][1][0] for r in outcomes]
        self.assertTrue(all(np.allclose(first[0],f) for f in first[1:]))
        # The posterior-weighted second action is conditional, up to sign.
        self.assertGreater(max(np.linalg.norm(np.outer(second[0],second[0])-np.outer(s,s)) for s in second[1:]),.05)

    def test_optimized_probe_maximizes_predicted_variance(self):
        rng=np.random.default_rng(7)
        operators=rng.normal(size=(3,8,8));receiver=unit(rng.normal(size=8))
        post=np.array([.2,.5,.3]);a=optimized_probe(operators,receiver,post)
        def variance(x):
            z=np.einsum('i,nij,j->n',receiver,operators,x)
            return float(np.dot(post,(z-np.dot(post,z))**2))
        val=variance(a)
        self.assertTrue(np.isclose(np.linalg.norm(a),1))
        self.assertGreaterEqual(val,max(variance(unit(rng.normal(size=8))) for _ in range(200))-1e-8)

    def test_unobservable_direction_cannot_be_interrogated(self):
        rng=np.random.default_rng(11)
        receiver=unit(rng.normal(size=8))
        orth=unit(rng.normal(size=8)-receiver*np.dot(receiver,rng.normal(size=8)))
        # Recompute exactly orthogonal to receiver (no causal sensor channel).
        orth=unit(orth-receiver*np.dot(receiver,orth))
        a=np.eye(8);b=a+.1*np.outer(orth,unit(rng.normal(size=8)))
        self.assertLess(np.linalg.norm(receiver@(a-b)),1e-12)
        for i in range(30):
            x=unit(rng.normal(size=8))
            self.assertAlmostEqual(receiver_response(a,x,receiver),receiver_response(b,x,receiver),places=11)

    def test_replay_provenance_and_uncertain_write_protect(self):
        p=ActiveProtocol(episodes_per_task=5)
        rng,base,teachers,context,receiver,bank,_,_=learn(p,715)
        copy=bank.weights.copy();x=unit(rng.normal(size=8))
        self.assertFalse(bank.update_from_external_probe(0,x,receiver,999.,1.,'internal_replay'))
        self.assertFalse(bank.update_from_external_probe(0,x,receiver,999.,.10,'external'))
        self.assertTrue(np.array_equal(bank.weights,copy))
        self.assertEqual(bank.self_replay_rejections,1)
        self.assertEqual(bank.uncertain_rejections,1)
        # A trusted, high-confidence observation does permit a local write.
        self.assertTrue(bank.update_from_external_probe(0,x,receiver,999.,1.,'external'))
        self.assertGreater(np.linalg.norm(bank.weights-copy),0)
        self.assertTrue(np.array_equal(bank.weights[1:],copy[1:]))

    def test_full_bank_refuses_new_dynamics_without_corrupting_old_slots(self):
        p=ActiveProtocol(episodes_per_task=8)
        rng,base,teachers,context,receiver,bank,_,_=learn(p,725)
        frozen=bank.weights.copy()
        unknown=base+.24*np.outer(unit(rng.normal(size=p.dim)),unit(rng.normal(size=p.dim)))
        result=bank.fit_episode(context,observations(unknown,p,rng))
        self.assertIsNone(result)
        self.assertEqual(bank.rejections,1)
        self.assertTrue(np.array_equal(bank.weights,frozen))

    def test_interleaved_rehearsal_uses_measured_dynamics_not_task_label(self):
        p=ActiveProtocol(episodes_per_task=8)
        rng,base,teachers,context,receiver,bank,_,_=learn(p,727)
        for task in [2,0,1,1,0,2,0,2,1]:
            cue=unit(context+rng.normal(scale=p.cue_noise,size=p.context_dim))
            slot=bank.fit_episode(cue,observations(teachers[task],p,rng))
            self.assertEqual(slot,task)
        self.assertEqual(bank.active,3)

    def test_small_end_to_end(self):
        p=ActiveProtocol(episodes_per_task=9)
        out=run_seed(719,p,queries_per_task=20)
        self.assertEqual(out['slots'],3)
        self.assertEqual(out['memory_scalars'],219)
        self.assertEqual(out['retrieval']['cue_only']['accuracy'],1/3)
        self.assertGreater(out['retrieval']['adaptive']['accuracy'],out['retrieval']['random']['accuracy'])
        self.assertLess(out['max_matched_direct_weight_difference'],1e-12)

if __name__=='__main__':unittest.main()
