"""Sequential A/B/C interference under the fixed-scalar Gate 2 budget.

Run: python consolidation_benchmark.py --output results/consolidation_v1.json
Only standard NumPy. Retains full seedwise receipts. No hyperparameter search.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from consolidation import Protocol, MultiTimescaleLoop, OnlineAssoc, KeyValueFIFO, sihti_encode, sihti_decode

def rms(a:np.ndarray)->float:
    return float(np.sqrt(np.mean(np.asarray(a)**2)))

def score(pred:np.ndarray,target:np.ndarray)->float:
    return rms(np.asarray(pred)-target)/(rms(target)+1e-12)

def generate(p:Protocol,seed:int):
    rng=np.random.default_rng(seed)
    loop=MultiTimescaleLoop(p,seed=91)
    n=p.ids_per_task*3
    family=rng.normal(size=(3,p.cue_dim))
    family/=np.linalg.norm(family,axis=1,keepdims=True)
    local=rng.normal(size=(n,p.cue_dim))
    local/=np.linalg.norm(local,axis=1,keepdims=True)
    cues=.55*np.repeat(family,p.ids_per_task,axis=0)+.45*local
    cues/=np.linalg.norm(cues,axis=1,keepdims=True)
    events=rng.choice([-1.,1.],size=(n,8,p.input_dim))
    targets=[]; traces=[]
    for ident in range(n):
        loop.reset()
        for u in events[ident]:
            loop.advance(.35,u)
        targets.append(loop.state.copy())
        traces.append(loop.ringdown())
    return rng,loop,cues,np.asarray(events),np.asarray(targets),np.asarray(traces)

def run_seed(seed:int,p:Protocol=Protocol())->dict:
    rng,loop,cues,events,targets,traces=generate(p,seed)
    mask=loop.slow_mask
    rates=np.where(mask,p.lr_slow,p.lr_fast)
    wrong=np.where(mask,p.lr_fast,p.lr_slow)
    budget=p.cue_dim*p.dim
    models={
        'sihti_selective':OnlineAssoc(p.cue_dim,p.dim,rates),
        'direct_matched_rates':OnlineAssoc(p.cue_dim,p.dim,rates),
        'direct_fast':OnlineAssoc(p.cue_dim,p.dim,p.lr_fast),
        'direct_slow':OnlineAssoc(p.cue_dim,p.dim,p.lr_slow),
        'sihti_wrong_band_rates':OnlineAssoc(p.cue_dim,p.dim,wrong),
        'fifo_kv':KeyValueFIFO(p.cue_dim,p.dim,budget)
    }
    stages={}; n=p.ids_per_task
    for task in range(3):
        order=np.repeat(np.arange(task*n,(task+1)*n),p.train_repeats)
        rng.shuffle(order)
        for ident in order:
            loop.reset()
            for u in events[ident]:
                loop.advance(.35,u+rng.normal(scale=p.event_noise_train,size=p.input_dim))
            trace=loop.ringdown()
            encoded=sihti_encode(trace,loop)
            x=trace[0]
            noisy=cues[ident]+rng.normal(scale=p.cue_noise_train,size=p.cue_dim)
            models['sihti_selective'].write(noisy,encoded)
            models['direct_matched_rates'].write(noisy,x)
            models['direct_fast'].write(noisy,x)
            models['direct_slow'].write(noisy,x)
            models['sihti_wrong_band_rates'].write(noisy,encoded)
            models['fifo_kv'].write(noisy,x)
        stage=f'after_{"ABC"[task]}'
        stages[stage]={}
        for query_task in range(task+1):
            ids=np.arange(query_task*n,(query_task+1)*n)
            queries=np.repeat(cues[ids],3,axis=0)+rng.normal(scale=p.cue_noise_test,size=(3*len(ids),p.cue_dim))
            truth=np.repeat(targets[ids],3,axis=0)
            late_truth=np.stack([loop.autonomous(x,p.times[-1]) for x in truth])
            for key,model in models.items():
                values=np.stack([model.recall(q) for q in queries])
                if key.startswith('sihti_'):
                    values=np.stack([sihti_decode(v,loop) for v in values])
                late_pred=np.stack([loop.autonomous(x,p.times[-1]) for x in values])
                stages[stage].setdefault(key,{})["ABC"[query_task]]={
                    'current_nrmse':score(values,truth),
                    'late_nrmse':score(late_pred,late_truth)
                }
    # Offline comparator has privileged access to all task identities.
    X=np.concatenate([np.repeat(cues[i:i+1],p.train_repeats,axis=0) for i in range(3*n)])
    Y=np.concatenate([np.repeat(targets[i:i+1],p.train_repeats,axis=0) for i in range(3*n)])
    W=np.linalg.solve(X.T@X+.1*np.eye(p.cue_dim),X.T@Y)
    offline={}
    for query_task in range(3):
        ids=np.arange(query_task*n,(query_task+1)*n)
        pred=cues[ids]@W
        late=np.stack([loop.autonomous(x,p.times[-1]) for x in pred])
        late_target=np.stack([loop.autonomous(x,p.times[-1]) for x in targets[ids]])
        offline["ABC"[query_task]]={
            'current_nrmse':score(pred,targets[ids]),
            'late_nrmse':score(late,late_target)
        }
    telescope=np.stack([(tr[:-1]-tr[1:]).sum(axis=0)+tr[-1] for tr in traces])
    coded=np.stack([sihti_decode(sihti_encode(tr,loop),loop) for tr in traces])
    return {'seed':seed,'by_stage':stages,'offline_joint_ridge':offline,
            'scalar_budget':budget,'fifo_capacity':models['fifo_kv'].capacity,
            'mask_slow_dims':int(mask.sum()),
            'sihti_trace_identity_max_error':float(np.max(np.abs(telescope-targets))),
            'sihti_coding_error':rms(coded-targets)}

def run(p:Protocol=Protocol())->dict:
    results=[run_seed(seed,p) for seed in p.seeds]
    models=['sihti_selective','direct_matched_rates','direct_fast','direct_slow',
            'sihti_wrong_band_rates','fifo_kv']
    summary={}
    for stage,last in [('after_A','A'),('after_B','B'),('after_C','C')]:
        summary[stage]={}
        seen="ABC"[:"ABC".index(last)+1]
        for name in models:
            summary[stage][name]={}
            for task in seen:
                summary[stage][name][task]={}
                for metric in ('current_nrmse','late_nrmse'):
                    z=np.array([r['by_stage'][stage][name][task][metric] for r in results])
                    summary[stage][name][task][metric]=float(z.mean())
                    summary[stage][name][task][metric+'_sd']=float(z.std(ddof=1))
    offline={}
    for task in 'ABC':
        offline[task]={metric:float(np.mean([r['offline_joint_ridge'][task][metric] for r in results]))
                       for metric in ('current_nrmse','late_nrmse')}
    diffs=[r['by_stage']['after_C']['sihti_selective']['A']['late_nrmse']
           -r['by_stage']['after_C']['direct_fast']['A']['late_nrmse'] for r in results]
    matched=[r['by_stage']['after_C']['sihti_selective']['A']['late_nrmse']
             -r['by_stage']['after_C']['direct_matched_rates']['A']['late_nrmse'] for r in results]
    gap=float(np.mean(diffs))
    return {
      'protocol':{**p.__dict__,'seeds':list(p.seeds),'times':list(p.times)},
      'allocation':{'online_weight_scalars':p.cue_dim*p.dim,
                    'key_value_slots':results[0]['fifo_capacity'],
                    'key_value_used_scalars':results[0]['fifo_capacity']*(p.cue_dim+p.dim),
                    'extra_memory':0,'mask_slow_dims':results[0]['mask_slow_dims']},
      'summary':summary,'offline_joint_ridge':offline,
      'identity_max_error':max(r['sihti_trace_identity_max_error'] for r in results),
      'coding_error_mean':float(np.mean([r['sihti_coding_error'] for r in results])),
      'gate':{
       'after_C_old_A_late_improvement_over_direct_fast':-gap,
       'old_A_delta_over_direct_matched':float(np.mean(matched)),
       'old_A_seedwise_advantage_count':int(sum(d<0 for d in diffs)),
       'old_A_5pct_margin_met':bool(gap<=-.05),
       'new_C_within_5pct_of_direct_fast':bool(summary['after_C']['sihti_selective']['C']['current_nrmse']<=summary['after_C']['direct_fast']['C']['current_nrmse']+.05),
       'beats_matched_feature_rate_control':bool(np.mean(matched)<-.01),
       'status':'exploratory gate; no evidence of advantage unless all criteria and independent confirmatory seeds pass'
      },'seeds':results
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/consolidation_v1.json')
    args=parser.parse_args()
    result=run()
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'gate':result['gate'],'allocation':result['allocation'],
        'after_C':{name:{task:{m:round(v[m],4) for m in ('late_nrmse','current_nrmse')}
                         for task,v in groups.items()}
                   for name,groups in result['summary']['after_C'].items()}},indent=2))

if __name__=='__main__':
    main()
