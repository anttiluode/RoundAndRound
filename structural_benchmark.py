"""Gate 3: online structural coexistence from ringdown residues.

Each teacher possesses an unknown dynamical operator. Memory learns only from
observed short sequences and noisy context. Never sees task ID (except oracle
comparator). After learning A/B/C in order, erase all transient state and
interrogate with fresh, previously unseen probes. Report all seed receipts.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from structural_memory import (Protocol,StructuralBank,DirectTransitionBank,SharedOperator,
    OracleIndexedOperators,FIFOPairs,sample_world,trace,unit,base_rotation)


def probe_set(dim:int,n:int,rng)->np.ndarray:
    # The same unseen probe directions are applied to every learned context.
    q=rng.normal(size=(n,dim))
    q=np.array([unit(x) for x in q])
    q[0:dim]=np.eye(dim)
    return q


def quality(pred,target,base):
    # Task-normalized ability to recover *operator-specific* response, not
    # a flattering score against the large generic baseline trajectory.
    num=np.sum((np.asarray(pred)[:,1:]-target[:,1:])**2)
    den=np.sum((base[:,1:]-target[:,1:])**2)
    return float(1-num/(den+1e-12))


def run_seed(seed:int,p:Protocol=Protocol(), collision=False, capacity=False, test_context_noise=None)->dict:
    if capacity:
        p=replace(p,task_count=4)
    rng=np.random.default_rng(seed)
    centers,base,Wtrue=sample_world(p,rng)
    if collision:
        centers[1]=unit(centers[0]+.035*rng.normal(size=p.context_dim))
    structural=StructuralBank(p,base)
    direct_routed=DirectTransitionBank(p,base)
    shared=SharedOperator(p,base)
    privileged=OracleIndexedOperators(p,base)
    fifo=FIFOPairs(p,structural.scalar_budget,base)
    models={'structural':structural,'direct_routed':direct_routed,'shared':shared,'oracle_index':privileged,'fifo':fifo}
    test_x=probe_set(p.dim,p.test_probes,rng)
    # Explicitly evaluate all contexts at the identical probe state.
    tests={k:np.asarray([trace(base+Wtrue[k],x,p.trace_steps) for x in test_x])
           for k in range(p.task_count)}
    generic=np.asarray([trace(base,x,p.trace_steps) for x in test_x])
    by_stage={}
    train_route=[]
    after_first=None
    for task in range(p.task_count):
        for episode in range(p.train_episodes):
            cue=unit(centers[task]+rng.normal(scale=p.context_noise,size=p.context_dim))
            x=unit(rng.normal(size=p.dim))
            frames=trace(base+Wtrue[task],x,p.trace_steps)
            noisy=frames+rng.normal(scale=p.observed_noise,size=frames.shape)
            inferred=structural.write(cue,noisy)
            direct_routed.write(cue,noisy)
            train_route.append({'task':task,'slot':inferred})
            shared.write(cue,noisy)
            privileged.write(task,noisy)
            fifo.write(cue,noisy)
        if task==0:
            after_first=structural.weights[0].copy()
        stage=f'after_{"ABCD"[task]}'
        scores={name:{} for name in ['structural','direct_routed','shared','oracle_index','fifo','base_no_memory']}
        routing_hits=0
        total_queries=0
        for old in range(task+1):
            test_cues=[unit(centers[old]+rng.normal(scale=(p.context_noise if test_context_noise is None else test_context_noise),size=p.context_dim))
                       for _ in test_x]
            truth=tests[old]
            for name in scores:
                replies=[]
                for c,x in zip(test_cues,test_x):
                    if name=='base_no_memory':t=trace(base,x,p.trace_steps)
                    elif name=='oracle_index':t=privileged.awaken(old,x,p.trace_steps)
                    else:
                        t=models[name].awaken(c,x,p.trace_steps)
                        if t is None:t=trace(base,x,p.trace_steps)
                    replies.append(t)
                scores[name]["ABCD"[old]]={
                    'gain_over_base':quality(np.array(replies),truth,generic),
                    'trajectory_rmse':float(np.sqrt(np.mean((np.asarray(replies)-truth)**2))),
                    'initial_probe_error':float(np.max(np.abs(np.asarray(replies)[:,0,:]-test_x)))
                }
            for c in test_cues:
                idx=structural._route(c,False)
                if idx==old:routing_hits+=1
                total_queries+=1
        by_stage[stage]=scores
        by_stage[stage]['context_route_accuracy']=routing_hits/total_queries
    # A must remain unchanged when B and C are learned, modulo routing.
    a_drift=0. if after_first is None else float(np.linalg.norm(structural.weights[0]-after_first))
    valid_route=sum(1 for e in train_route if e['slot']==e['task'])/len(train_route)
    # Slow operator and centroid memory only; no saved fast episode trace.
    return {'seed':seed,'by_stage':by_stage,'storage':{
        'slow_scalars':structural.scalar_budget,'matrix_scalars':structural.weights.size,
        'context_scalars':structural.centers.size,'counts':structural.counts.size,
        'fifo_records':fifo.capacity,'fifo_scalars':fifo.n_scalar_records,
        'fast_state_persisted':0,'stored_training_traces':0},
        'slots_created':structural.active,'denied_writes':structural.denied,
        'fraction_training_correctly_routed':float(valid_route),
        'old_A_operator_drift_after_new_tasks':a_drift,
        'full_slot_count':structural.allocated,'collision':collision,'capacity':capacity,
        'test_context_noise':test_context_noise,
        'matched_direct_max_weight_difference':float(np.max(np.abs(structural.weights-direct_routed.weights)))}


def run(p:Protocol=Protocol())->dict:
    seeds=[run_seed(s,p) for s in p.seeds]
    stages=['after_A','after_B','after_C']
    summary={}
    for stage in stages:
        summary[stage]={}
        for name in ['structural','direct_routed','shared','oracle_index','fifo','base_no_memory']:
            summary[stage][name]={}
            for task in 'ABC'[:stages.index(stage)+1]:
                vals=np.array([v['by_stage'][stage][name][task]['gain_over_base'] for v in seeds])
                summary[stage][name][task]={
                    'gain_mean':float(np.mean(vals)),
                    'gain_sd':float(np.std(vals,ddof=1)),
                    'per_seed':[float(x) for x in vals]}
        summary[stage]['route_accuracy']=float(np.mean([v['by_stage'][stage]['context_route_accuracy'] for v in seeds]))
    collisions=[run_seed(s,replace(p,train_episodes=p.train_episodes),collision=True) for s in p.seeds]
    capacities=[run_seed(s,p,capacity=True) for s in p.seeds]
    noisy_queries=[run_seed(s,p,test_context_noise=.38) for s in p.seeds]
    collision_oldA=float(np.mean([v['by_stage']['after_C']['structural']['A']['gain_over_base'] for v in collisions]))
    capacity_D=float(np.mean([v['by_stage']['after_D']['structural']['D']['gain_over_base'] for v in capacities]))
    oldvals=np.array(summary['after_C']['structural']['A']['per_seed'])
    sharedvals=np.array(summary['after_C']['shared']['A']['per_seed'])
    oracvals=np.array(summary['after_C']['oracle_index']['A']['per_seed'])
    newvals=np.array(summary['after_C']['structural']['C']['per_seed'])
    # Confirmatory gate set before full 8-seed run; trials 301-308 only.
    gates={
       'old_A_gain_above_0_70':bool(float(oldvals.mean())>.70),
       'new_C_gain_above_0_70':bool(float(newvals.mean())>.70),
       'A_gain_advantage_over_shared_above_0_40':bool(float((oldvals-sharedvals).mean())>.40),
       'A_within_0_10_of_privileged_oracle':bool(float((oracvals-oldvals).mean())<.10),
       'at_least_7_seeds_old_A_better_shared':bool(int(np.sum(oldvals>sharedvals))>=7),
       'no_old_A_matrix_rewriting':bool(max(v['old_A_operator_drift_after_new_tasks'] for v in seeds)<1e-10),
       'residue_advantage_over_matched_direct':bool(max(v['matched_direct_max_weight_difference'] for v in seeds)>1e-5),
       'all_gates_pass':False,
    }
    gates['all_gates_pass']=all(v for k,v in gates.items() if k not in ('all_gates_pass','residue_advantage_over_matched_direct'))
    return {'protocol':{**asdict(p),'seeds':list(p.seeds)},'summary':summary,
            'gates':gates,'additional_attackers':{
            'near_collision_A_after_C_gain':collision_oldA,
            'full_three_slots_fourth_task_D_gain':capacity_D,
            'collision_seeds':collisions,'capacity_seeds':capacities,
            'unseen_high_noise_cues_A_gain_after_C':float(np.mean([v['by_stage']['after_C']['structural']['A']['gain_over_base'] for v in noisy_queries])),
            'unseen_high_noise_route_accuracy':float(np.mean([v['by_stage']['after_C']['context_route_accuracy'] for v in noisy_queries])),
            'high_noise_seeds':noisy_queries},
            'seedwise':seeds}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',default='results/structural_v1.json')
    a.add_argument('--seeds',default='301,302,303,304,305,306,307,308',help='Comma-separated integer seeds; independent confirmation: 501,...,508')
    args=a.parse_args()
    seeds=tuple(int(x) for x in args.seeds.split(',') if x.strip())
    if len(seeds)<2:raise ValueError('Need at least two seeds to estimate variability')
    outcome=run(replace(Protocol(),seeds=seeds))
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(outcome,indent=2)+'\n')
    print(json.dumps({'gates':outcome['gates'],
     'summary':{s:{k:v for k,v in vals.items() if k in ('structural','shared','oracle_index','fifo','route_accuracy')}
                for s,vals in outcome['summary'].items()},
     'collision_A':outcome['additional_attackers']['near_collision_A_after_C_gain'],
     'full_bank_D':outcome['additional_attackers']['full_three_slots_fourth_task_D_gain']},indent=2))

if __name__=='__main__':main()
