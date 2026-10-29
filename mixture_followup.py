"""Post-confirmation exploratory study: read several stored loops at once.

A posterior mixture of predicted *trajectories* is not the same as updating
one interpolated operator. It uses the same measured scalars and the same
stored matrices as hard selection, with extra compute for parallel rollouts.
This follow-up was chosen after inspecting Gate-4 confirm outcomes; it is
NOT a preregistered Gate-4 claim or independent confirmation.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
from active_reentry import ActiveProtocol, unit, query
from active_reentry_benchmark import learn
from structural_memory import trace


def gain(pred,target,base):
    return float(1-np.sum((pred-target)**2)/(np.sum((base-target)**2)+1e-12))


def run_seed(seed,p:ActiveProtocol=ActiveProtocol(),n=120):
    rng,base,teachers,context,receiver,bank,direct,_=learn(p,seed)
    out={kind:[] for kind in ('prior_mixture','hard_adaptive','soft_adaptive','hard_fixed','soft_fixed','abstain_adaptive')}
    abstain_count=0
    for k in range(p.slots):
      for j in range(n):
        cue=unit(context+rng.normal(scale=p.cue_noise,size=p.context_dim))
        shared=int(rng.integers(0,2**31-1))
        vectors=np.array([unit(rng.normal(size=p.dim)) for _ in range(7)])
        truth=np.stack([trace(teachers[k],v,5)[1:] for v in vectors])
        blank=np.stack([trace(base,v,5)[1:] for v in vectors])
        # All candidate predictions use the same fixed slow memory: no writes.
        bank_preds=np.stack([np.stack([trace(op,v,5)[1:] for v in vectors]) for op in bank.candidates()])
        out['prior_mixture'].append(gain(bank_preds.mean(axis=0),truth,blank))
        for policy in ('adaptive','fixed'):
            ans=query(bank,teachers[k],receiver,cue,np.random.default_rng(shared),policy=policy)
            posterior=ans['posterior'];slot=ans['slot']
            out['hard_'+policy].append(gain(bank_preds[slot],truth,blank))
            out['soft_'+policy].append(gain(np.einsum('s,sntd->ntd',posterior,bank_preds),truth,blank))
            if policy=='adaptive':
                known=posterior.max()>=.85
                abstain_count+=int(not known)
                out['abstain_adaptive'].append(gain(bank_preds[slot] if known else blank,truth,blank))
    return {'seed':seed,'mean_gain':{key:float(np.mean(v)) for key,v in out.items()},
            'abstain_fraction':abstain_count/(p.slots*n),
            'per_query_gain':{key:[float(x) for x in val] for key,val in out.items()}}


def run(p=ActiveProtocol(seeds=tuple(range(701,709))),queries_per_task=120):
    results=[run_seed(seed,p,queries_per_task) for seed in p.seeds]
    means={key:float(np.mean([r['mean_gain'][key] for r in results]))
           for key in results[0]['mean_gain']}
    effects={
        'soft_minus_hard_adaptive':means['soft_adaptive']-means['hard_adaptive'],
        'soft_minus_hard_fixed':means['soft_fixed']-means['hard_fixed'],
        'soft_adaptive_minus_soft_fixed':means['soft_adaptive']-means['soft_fixed'],
        'soft_adaptive_minus_prior':means['soft_adaptive']-means['prior_mixture'],
    }
    return {'status':'post-Gate4 mixture follow-up; 801–808 are independent of mixture development, but the hypothesis was generated after Gate4',
       'protocol':asdict(p),'means':means,'effects':effects,
       'abstain_rate_mean':float(np.mean([r['abstain_fraction'] for r in results])),
       'seedwise':[{key:value for key,value in r.items() if key!='per_query_gain'} for r in results]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results/mixture_followup.json')
    ap.add_argument('--confirm',action='store_true',help='fresh post-follow-up seeds 801–808')
    args=ap.parse_args()
    p=ActiveProtocol(seeds=tuple(range(801,809))) if args.confirm else ActiveProtocol(seeds=tuple(range(701,709)))
    r=run(p)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','means','effects','abstain_rate_mean')},indent=2))

if __name__=='__main__':main()
