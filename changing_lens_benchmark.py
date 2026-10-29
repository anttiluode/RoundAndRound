"""Gate 5: learned attention, ongoing present, swapping policies and memories.

Development seeds and held-out confirmation are disjoint; no label enters query.
The reward environment is deliberately changed between policies (probe cost).
This experiment identifies computational retrieval styles, NOT human personality.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from active_reentry import ActiveProtocol
from active_reentry_benchmark import learn
from structural_memory import unit
from changing_lens import (LensProtocol, LensPolicy, ContinuingPresent,
    sensing_axes, predictions, observe_fragment, fixed_query)


def gain(error,blank):
    return float(1-error/(blank+1e-12))


def world(seed:int):
    ap=ActiveProtocol(episodes_per_task=18)
    rng,base,teachers,_,_,bank,_,_=learn(ap,seed)
    if bank.active != ap.slots:
        raise RuntimeError('did not discover all three operator memories')
    return base,teachers,bank.candidates()


def trial(rng,ops,teachers,axes,noise,source=None,z=None):
    if source is None:source=int(rng.integers(len(teachers)))
    if z is None:z=unit(rng.normal(size=ops.shape[1]))
    trigger=unit(z+.35*rng.normal(size=len(z)))
    mus=predictions(ops,trigger,axes)
    observed=observe_fragment(teachers[source],trigger,axes,noise,rng)
    return source,trigger,mus,observed


def train_policy(ops,teachers,axes,p,seed,cost,train_trials=None):
    rng=np.random.default_rng(seed)
    pol=LensPolicy(p.channels,seed+71000)
    N=p.train_trials if train_trials is None else train_trials
    # All training cues depend on the continuously evolving present, not a reset state.
    # The learned bank stores only operators; the current state stays private.
    from structural_memory import base_rotation
    present=ContinuingPresent(base_rotation(p.dim),p.rho,seed+7000)
    for i in range(N):
        present.advance(float(rng.uniform(.1,1.0)))
        k=int(rng.integers(len(teachers)))
        external=unit(rng.normal(size=p.dim))
        trigger=unit(.70*external+.30*unit(present.state))
        mu=predictions(ops,trigger,axes)
        obs=observe_fragment(teachers[k],trigger,axes,p.noise,rng)
        post,_,_=pol.query(mu,obs,p.noise,p.probe_budget,train=True,
                  cost=cost,source=k,lr=.030 if i<N*.75 else .010)
        present.step(external,ops,post)
    return pol


def evaluate(ops,teachers,base,axes,p,seed,policies,N=None):
    """Every policy gets exactly the same stream and fresh continuous present.

    To hold cue evidence independent of diverging z, trigger is generated from
    shared exogenous events, not the policy's private state. Feedback uses the
    current z (which DOES diverge when different memories graze it).
    """
    rng=np.random.default_rng(seed)
    N=p.test_trials if N is None else N
    streams={name:ContinuingPresent(base,p.rho,seed+9000) for name in policies}
    metric={name:{'used':[],'correct':[],'logp':[], 'gain':[], 'graze':[],
                  'actions':[],'present_error':[], 'base_error':[], 'final_state':None}
            for name in policies}
    for t in range(N):
        gap=float(rng.uniform(.1,1.0))
        for present in streams.values():present.advance(gap)
        k=int(rng.integers(len(teachers)))
        ext=unit(rng.normal(size=p.dim))
        shared_noise=rng.normal(scale=p.noise,size=p.channels)
        for name,policy in policies.items():
            z=streams[name].state.copy()
            trigger=unit(.70*ext+.30*unit(z))
            mu=predictions(ops,trigger,axes)
            obs=axes@(teachers[k]@trigger)+shared_noise
            if isinstance(policy,LensPolicy):
                post,used,acts=policy.query(mu,obs,p.noise,p.probe_budget,mode='greedy')
            else:
                post,used,acts=fixed_query(mu,obs,p.noise,p.probe_budget,method=policy)
            prior=np.ones(len(ops))/len(ops)
            # Current-world target uses unchanged exogenous input but actual
            # hidden operator. The agent only gets its estimated mixture.
            reference=streams[name]
            predicted,blank=reference.step(ext,ops,post,couple=True)
            # Evaluate the same pre-state under the true operator.
            oracle=blank+p.rho*((teachers[k]-base)@z)
            metric[name]['present_error'].append(float(np.sum((predicted-oracle)**2)))
            metric[name]['base_error'].append(float(np.sum((blank-oracle)**2)))
            metric[name]['used'].append(used)
            metric[name]['correct'].append(int(post.argmax()==k))
            metric[name]['logp'].append(float(np.log(max(post[k],1e-12))))
            metric[name]['graze'].append(float(np.linalg.norm(predicted-blank)/max(np.linalg.norm(blank),1e-12)))
            metric[name]['actions'].append(tuple(acts))
    result={}
    for name,v in metric.items():
        result[name]={
            'mean_probes':float(np.mean(v['used'])),
            'accuracy':float(np.mean(v['correct'])),
            'log_score':float(np.mean(v['logp'])),
            'relative_recall_gain':gain(np.mean(v['present_error']),np.mean(v['base_error'])),
            'graze_strength':float(np.mean(v['graze'])),
            'steps':streams[name].steps,'autonomous_updates':streams[name].autonomous_updates,'resets':streams[name].resets,
            'max_relative_graze':streams[name].max_relative_graze,
            'first_actions':[list(x) for x in v['actions'][:12]],
            'final_state':streams[name].state.tolist(),
        }
    return result


def do_world(seed:int,p:LensProtocol=LensProtocol(),N=None,train=None):
    base,teachers,ops=world(seed)
    axes=sensing_axes(p.dim,p.channels,seed+3000)
    # Both policies have the identical bank and training experience stream,
    # but different penalties for probing: an intentional design intervention.
    discover=train_policy(ops,teachers,axes,p,seed+10,cost=.02,train_trials=train)
    economical=train_policy(ops,teachers,axes,p,seed+10,cost=.70,train_trials=train)
    evals=evaluate(ops,teachers,base,axes,p,seed+50000,{
        'discover':discover,'economical':economical,
        'best_discrete': 'best','first_discrete':'first','cue_only':'none'},N=N)
    # Evaluate all policies on the SAME stream and memory. Swapping their
    # learned policy objects between hosts swaps behaviour without modifying M.
    swap=evaluate(ops,teachers,base,axes,p,seed+50000,{
        'host_A_with_B_policy':economical,'host_B_with_A_policy':discover},N=N)
    # Exchange a bank, keeping the policy fixed: only memory content changes.
    foreign_base,foreign_teachers,foreign=world(seed+131)
    foreign_eval=evaluate(foreign,teachers,base,axes,p,seed+50000,{
        'discover_foreign_memory':discover,'economical_foreign_memory':economical},N=N)
    # Exploratory *policy* transfer: remembered operators and sensory responses
    # now both come from a new world; the learned policy is frozen.
    transfer=evaluate(foreign,foreign_teachers,foreign_base,axes,p,seed+50000,{
        'discover_unseen_world':discover,'economical_unseen_world':economical},N=N)
    return {'seed':seed,'retrieval':evals,'policy_swap':swap,'memory_swap':foreign_eval,
            'unseen_world_policy_transfer':transfer,
            'scalars_per_policy':discover.scalar_budget,
            'memory_matrix_scalars':int(ops.size),
            'separate_source_labels_during_query':False,
            'style_gap':evals['discover']['mean_probes']-evals['economical']['mean_probes'],
            'weights_changed_by_recall':False}


def run(seeds=tuple(range(901,909)),p:LensProtocol=LensProtocol(),N=None,train=None):
    worlds=[do_world(seed,p,N=N,train=train) for seed in seeds]
    names=list(worlds[0]['retrieval'])
    summaries={name:{key:float(np.mean([w['retrieval'][name][key] for w in worlds]))
                     for key in ('mean_probes','accuracy','log_score','relative_recall_gain','graze_strength')}
               for name in names}
    swap_exact=all(w['policy_swap']['host_A_with_B_policy']['mean_probes']==w['retrieval']['economical']['mean_probes']
                   and w['policy_swap']['host_B_with_A_policy']['mean_probes']==w['retrieval']['discover']['mean_probes']
                   for w in worlds)
    foreign={name:float(np.mean([w['memory_swap'][name]['relative_recall_gain'] for w in worlds]))
            for name in worlds[0]['memory_swap']}
    transfer={name:{metric:float(np.mean([w['unseen_world_policy_transfer'][name][metric] for w in worlds]))
                    for metric in ('mean_probes','accuracy','relative_recall_gain')}
              for name in worlds[0]['unseen_world_policy_transfer']}
    return {'status':'Gate5 exploratory; training costs differ by construction; no self/personality claim','unseen_world_policy_transfer_exploratory':transfer,
        'seed_list':list(seeds),'protocol':asdict(p),'n_worlds':len(worlds),
        'summary':summaries,'memory_swap_relative_gains':foreign,
        'policy_swap_exact_probe_count':swap_exact,
        'mean_style_gap':float(np.mean([w['style_gap'] for w in worlds])),
        'style_gap_positive_seeds':sum(w['style_gap']>.10 for w in worlds),
        'learned_vs_best_accuracy_delta':summaries['discover']['accuracy']-summaries['best_discrete']['accuracy'],
        'worlds':worlds}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',default='results/changing_lens_development.json')
    a.add_argument('--confirm',action='store_true')
    a.add_argument('--quick',action='store_true')
    args=a.parse_args()
    seeds=range(1001,1009) if args.confirm else range(901,909)
    res=run(tuple(seeds),N=80 if args.quick else None,train=400 if args.quick else None)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:res[k] for k in ('summary','memory_swap_relative_gains','unseen_world_policy_transfer_exploratory','mean_style_gap','style_gap_positive_seeds','learned_vs_best_accuracy_delta')},indent=2))

if __name__=='__main__':main()
