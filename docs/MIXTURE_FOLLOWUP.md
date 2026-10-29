# Post-Gate-4 follow-up — awaken several loops, not necessarily one

**Exploratory mechanism chosen after observing Gate 4 confirmation seeds 701–708.** The high-confidence hard winner was sometimes wrong, producing negative response gain. The natural Bayes least-squares comparator is to keep the **full posterior**, roll out each candidate operator in parallel, and average its predicted trajectory, not to interpolate matrices and pretend that result is one ordinary loop.

No additional slow state or external observations: three learned 8×8 matrices plus prototypes/counters (219 scalars), two scalar active probes, and extra compute for three trajectories. Memory weights are frozen during these queries. The response is a probability-weighted *output*, not a claim that simultaneously active oscillators are biological consciousness.

## Outcomes and comparators

- **hard_adaptive:** two posterior-dependent probes, then choose one operator (Gate 4).
- **soft_adaptive:** same two probes, run every recalled candidate's trajectory, average outputs with the posterior (new post-hoc method).
- **hard_fixed / soft_fixed:** same with strongest globally optimized fixed probe repeated twice (tests whether adaptive intervention itself matters).
- **prior_mixture:** uniform mixture of all memories with **no external observations**. An essential control against merely receiving an ensemble averaging benefit.
- **abstain_adaptive:** when confidence is below 0.85, use no-memory base instead of risking an incorrect old loop (a decision-cost alternative).

Scored on seven fresh five-step probe directions per retrieval episode, never on the two informative scalar sensor measurements. The exact same hidden worlds, cues and scalar sensor noise draws are used for hard versus soft comparisons. `gain = 1 − SSE(predicted−truth)/SSE(base−truth)`; a negative score is worse than empty memory.

Initial exploratory run uses seeds **701–708**, 120 questions per hidden regime. Local non-external confirmatory split uses **801–808**, frozen after development: soft adaptive should exceed hard adaptive by **0.05** mean gain and improve in **at least 7/8 worlds**, and it should beat soft fixed by **0.03** mean gain, while producing an absolute mean gain **>0.60**. These criteria are a narrow follow-up and do not change the original Gate-4 failure decision. They are not a test of improved memory *capacity*, only of using uncertain retrieved dynamics effectively.

Limitations: mixture of trajectories is strictly more computational work than a single loop rollout; the policy still depends on an **external observed response**. Without such a response and with exactly identical contextual cues, source identity is not identifiable from the memories alone. Strong alternatives include Bayesian prediction/mixture of experts and trained state-space models; no novel theorem or biological implementation is claimed.

Run `python mixture_followup.py --confirm --output results/mixture_confirm.json`.
