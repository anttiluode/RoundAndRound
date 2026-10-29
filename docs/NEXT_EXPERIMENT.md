# After Gate 4: learning to interrogate, rather than coding the information-gain rule

Gate 4's eigenvector policy is a specified active experiment design rule; the memory operators are learned, **but the probing intelligence is not**. The next falsifiable experiment should replace this engineered rule with a small trainable controller that receives (i) its current probabilistic hypothesis over long-term loops, (ii) the observable state/query/available action repertoire, and (iii) the actual previous sensor prediction errors. Train it only from **action cost and subsequent prediction accuracy**, with held-out unseen operator families and variable bank sizes.

A suitable sequence of tests:

1. Three known operators with indistinguishable context and matched 1/2/3 probe budgets. Compare learned controller against *analytic optimal-variance*, fixed optimal, random and passive context readout. Include policy compute and memory in the budget.
2. New fourth operator when all slots are full. The correct response could be **uncertain/unknown**, not forcibly interpreting it as A/B/C. Compare a calibrated unknown detector against posterior-only selection and a key/value baseline.
3. Consolidation from a stream of replayed and new evidence: false self-generated echo should not become new historical fact. Restrict writes to experimentally grounded evidence, audit contamination and compare forgetting on genuinely shifted tasks.
4. Transfer learned policy to a qualitatively different teacher family (non-normal matrices, temporal drives, nonlinear and partially observed dynamics). Showing gains in one hand-designed linear family is not broad continual intelligence.

A positive result would be a learned intervention policy **without being given the answer or an explicit information-gain oracle** that preserves earlier task utility with competitive total memory/compute. A negative result—especially if conventional recurrent controls or Bayesian model averaging tie—would still locate the architecture's real constraints.
