"""Gate 4 — active interrogation of ambiguous learned dynamical memories.

The information source is an external world responding to interventions.
Memory-only introspection with the identical context contains no task ID.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from active_reentry import (ActiveProtocol, TransitionMemory, world, observations,
    query, unit, receiver_response)
from structural_memory import trace

POLICIES=('cue_only','basis','random','fixed','adaptive')


def learn(p,seed):
    rng=np.random.default_rng(seed)
    base,operators,center,receiver=world(p,rng)
    memory=TransitionMemory(p,base)
    control=TransitionMemory(p,base,direct=True)
    routes=[]
    for task in range(p.slots):
        for _ in range(p.episodes_per_task):
            cue=unit(center+rng.normal(scale=p.cue_noise,size=p.context_dim))
            frames=observations(operators[task],p,rng)
            slot=memory.fit_episode(cue,frames)
            directslot=control.fit_episode(cue,frames)
            assert slot == directslot
            routes.append((task,slot))
    return rng,base,operators,center,receiver,memory,control,routes


def trajectory_gain(estimated,true,base,probe_set):
    targets=np.stack([trace(true,x,5)[1:] for x in probe_set])
    guess=np.stack([trace(estimated,x,5)[1:] for x in probe_set])
    blank=np.stack([trace(base,x,5)[1:] for x in probe_set])
    return float(1-np.sum((guess-targets)**2)/(np.sum((blank-targets)**2)+1e-12))


def run_seed(seed:int,p:ActiveProtocol=ActiveProtocol(),queries_per_task:int=120):
    rng,base,teachers,context,receiver,bank,direct,routes=learn(p,seed)
    errors=np.array([np.linalg.norm(bank.candidates()[i]-teachers[i]) for i in range(p.slots)])
    paired=float(np.max(np.abs(bank.weights-direct.weights)))
    route_success=float(np.mean([int(a==b) for a,b in routes]))
    metrics={name:{'correct':0,'total':0,'gains':[],'observations':0} for name in POLICIES}
    # Every policy receives same latent task, noisy cue and observation-noise draw;
    # only action changes which noiseless response is observed.
    for task in range(p.slots):
        for trial in range(queries_per_task):
            cue=unit(context+rng.normal(scale=p.cue_noise,size=p.context_dim))
            jointseed=int(rng.integers(0,2**31-1))
            probes=np.array([unit(rng.normal(size=p.dim)) for _ in range(7)])
            for policy in POLICIES:
                local=np.random.default_rng(jointseed)
                ans=query(bank,teachers[task],receiver,cue,local,policy=policy)
                got=ans['slot']
                metrics[policy]['correct']+=int(got==task)
                metrics[policy]['total']+=1
                metrics[policy]['observations']+=ans['observations']
                metrics[policy]['gains'].append(trajectory_gain(bank.candidates()[got],teachers[task],base,probes))
    summary={key:{'accuracy':float(v['correct']/v['total']),
                  'response_gain':float(np.mean(v['gains'])),
                  'observed_scalars':v['observations'],
                  'observations_per_query':v['observations']/v['total']}
             for key,v in metrics.items()}
    # An exact ID-indexed oracle only assesses the upper bound, does not
    # compete as an implementable system with the same information.
    oracle_gains=[]
    for task in range(p.slots):
        x=np.eye(p.dim)
        oracle_gains.append(trajectory_gain(bank.candidates()[task],teachers[task],base,x))
    # If the observed response is drawn from a different regime, identification
    # will follow that regime, not infer the true hidden index magically.
    scrambled_correct=0
    for task in range(p.slots):
        for trial in range(queries_per_task):
            swapped=(task+1)%p.slots
            got=query(bank,teachers[swapped],receiver,context,np.random.default_rng(seed*100000+task*1000+trial),policy='adaptive')['slot']
            scrambled_correct+=int(got==task)
    return {'seed':seed,'learn_route_accuracy':route_success,'slots':bank.active,
            'memory_scalars':bank.scalar_budget,'operator_rmse':float(np.mean(errors)),
            'max_matched_direct_weight_difference':paired,
            'retrieval':summary,'oracle_indexed_response_gain':float(np.mean(oracle_gains)),
            'mismatched_world_feedback_accuracy':scrambled_correct/(p.slots*queries_per_task)}


def reconsolidation_attack(seed:int,p:ActiveProtocol=ActiveProtocol(),trials=300):
    """Compare stationary read-only, trusted external plasticity and false replay.

    This test never gives original task ID to the query controller; it uses
    queries on a selected known world (A) and measures all slot changes.
    """
    rng,base,teachers,context,receiver,bank,_,_=learn(p,seed)
    initial=bank.weights.copy()
    # Change hidden A dynamics after training, without telling the memory.
    changed=teachers[0]+.025*np.outer(unit(rng.normal(size=p.dim)),unit(rng.normal(size=p.dim)))
    read_only=TransitionMemory(p,base)
    protected=TransitionMemory(p,base)
    self_replay=TransitionMemory(p,base)
    for obj in (read_only,protected,self_replay):
        obj.weights[:]=bank.weights;obj.keys[:]=bank.keys;obj.counts[:]=bank.counts;obj.active=bank.active
    probe_eval=np.eye(p.dim)
    before=trajectory_gain(bank.candidates()[0],changed,base,probe_eval)
    # For this causal test, use a known A cue? No: all contexts are identical,
    # so a query still needs external observations to identify a slot.
    correct=0
    for i in range(trials):
        cue=unit(context+rng.normal(scale=p.cue_noise,size=p.context_dim))
        shared_seed=int(rng.integers(0,2**31-1))
        a=query(read_only,changed,receiver,cue,np.random.default_rng(shared_seed),policy='adaptive')
        b=query(protected,changed,receiver,cue,np.random.default_rng(shared_seed),policy='adaptive',allow_reconsolidation=True,provenance='external')
        c=query(self_replay,changed,receiver,cue,np.random.default_rng(shared_seed),policy='adaptive',allow_reconsolidation=True,provenance='internal_replay')
        correct+=int(b['slot']==0)
    return {'seed':seed,'original_gain_on_changed_A':before,
        'A_correct_query_fraction':correct/trials,
        'read_only_after':trajectory_gain(read_only.candidates()[0],changed,base,probe_eval),
        'external_plastic_after':trajectory_gain(protected.candidates()[0],changed,base,probe_eval),
        'self_replay_after':trajectory_gain(self_replay.candidates()[0],changed,base,probe_eval),
        'external_updates':protected.external_updates,
        'self_replay_rejections':self_replay.self_replay_rejections,
        'other_slot_drift':float(np.max(np.abs(protected.weights[1:]-initial[1:])))}


def run(p:ActiveProtocol=ActiveProtocol(),queries_per_task=120,drift_trials=300):
    records=[run_seed(seed,p,queries_per_task) for seed in p.seeds]
    summary={}
    for policy in POLICIES:
        values=np.array([v['retrieval'][policy]['accuracy'] for v in records])
        gains=np.array([v['retrieval'][policy]['response_gain'] for v in records])
        summary[policy]={'accuracy_mean':float(values.mean()),'accuracy_sd':float(values.std(ddof=1)),
            'response_gain_mean':float(gains.mean()),'accuracy_by_seed':values.tolist(),
            'gain_by_seed':gains.tolist(), 'scalars_per_query':records[0]['retrieval'][policy]['observations_per_query']}
    recon=[reconsolidation_attack(seed,p,trials=drift_trials) for seed in p.seeds]
    recon_sum={key:float(np.mean([r[key] for r in recon])) for key in
       ('original_gain_on_changed_A','read_only_after','external_plastic_after','self_replay_after',
        'A_correct_query_fraction','external_updates','self_replay_rejections','other_slot_drift')}
    diff=np.array(summary['adaptive']['accuracy_by_seed'])-np.array(summary['fixed']['accuracy_by_seed'])
    diff_random=np.array(summary['adaptive']['accuracy_by_seed'])-np.array(summary['random']['accuracy_by_seed'])
    gate={
     'auto_discovered_distinct_operators':bool(min(r['learn_route_accuracy'] for r in records)>.95),
     'matched_direct_learning_ties':bool(max(r['max_matched_direct_weight_difference'] for r in records)<1e-10),
     'active_beats_random_by_0_10':bool(float(diff_random.mean())>.10),
     'active_beats_best_fixed_by_0_05':bool(float(diff.mean())>.05),
     'reactivation_gain_above_0_80':bool(summary['adaptive']['response_gain_mean']>.80),
     'external_reconsolidation_improves':bool(recon_sum['external_plastic_after']>recon_sum['read_only_after']+.02),
     'internal_replay_does_not_write':bool(all(r['self_replay_rejections']==2*drift_trials for r in recon)),
     'all_advantage_gates':False}
    gate['all_advantage_gates']=all(v for k,v in gate.items() if k!='all_advantage_gates')
    return {'protocol':{**asdict(p),'seeds':list(p.seeds)},'queries_per_task':queries_per_task,
       'summary':summary,'oracle_gain_mean':float(np.mean([x['oracle_indexed_response_gain'] for x in records])),
       'learn_route_accuracy_mean':float(np.mean([x['learn_route_accuracy'] for x in records])),
       'feedback_scramble_accuracy_mean':float(np.mean([x['mismatched_world_feedback_accuracy'] for x in records])),
       'reconsolidation':recon_sum,'gate':gate,'seedwise':records,'drift_seedwise':recon}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/active_reentry_v1.json')
    parser.add_argument('--confirm',action='store_true',help='independent seeds 701–708')
    parser.add_argument('--queries',type=int,default=120)
    parser.add_argument('--drift-trials',type=int,default=300)
    args=parser.parse_args()
    p=ActiveProtocol(seeds=tuple(range(701,709))) if args.confirm else ActiveProtocol()
    report=run(p,queries_per_task=args.queries,drift_trials=args.drift_trials)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('summary','oracle_gain_mean','learn_route_accuracy_mean','feedback_scramble_accuracy_mean','reconsolidation','gate')},indent=2))

if __name__=='__main__':main()
