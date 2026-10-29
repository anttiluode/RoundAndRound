# Gate 5 results — The Changing Lens (29 October 2026)

**Status: partial positive mechanism, no new performance edge.** Development worlds 901–908 and separate replication worlds 1001–1008. The 1001–1008 results were inspected before adding autonomous-time advancement, so the final version reuses those worlds; it is **not a blind independent preregistered confirmation**. Trained memory-access policies use retrospective correct-source feedback and **different engineered probe costs**; this is *not* autonomous human personality.

## Fresh eight-world confirmation (final autonomous-time version)

| Reader | Observed scalars / cue | Correct memory selected | Reduction of present-state squared error versus no-memory influence |
|---|---:|---:|---:|
| Learned inquisitive (probe cost 0.02) | 3.0 | **51.39%** | **42.06%** |
| Learned economical (cost 0.70) | 0.0 | 35.11% | 33.71% |
| Bayesian best of six discrete sensors (3 views) | 3.0 | **55.39%** | **45.69%** |
| Fixed first three sensors | 3.0 | 49.96% | 40.59% |
| No attention reads, uniform prior | 0.0 | 35.11% | 33.71% |

Chance memory-ID classification is 33.33%. The **no-reads control also has a positive present-state effect** because simply averaging available learned operators is better than using no memory at all. Therefore one must compare *additional* gains: inquisitive retrieval improves the present error reduction only about **8.3 percentage points** beyond uniform-prior mixing; the analytic variance control is better by another **3.6 points**. It is not evidence that learned style beats conventional Bayesian attention.

Across confirmation seeds, the trained probe-frequency gap was 3.0 scalars in all 8/8 worlds. This is not subtle: one policy used all its allowed probes, the other used none, a result strongly encouraged by the explicitly assigned reward costs. It verifies that access rules can be learned separately from memory content, **not** emergent complex personality.

Mean memory-induced effect on the ongoing next-state norm was only **0.79%** in the inquisitive condition and **0.73%** in the economical condition. The 15% per-step safety cap was respected. Every trial also advanced autonomous time before new input, and all 350 test episodes used the same continuously evolving live state without resets.

### Swaps and perturbations

- **Policy swap:** keeping the identical stored operators and initial present while exchanging the two access policies exchanges their sensor choices and their probe-count signature. This is expected from the modular implementation and confirms the policy and memory are not the same object.
- **Wrong memory bank:** with a totally unrelated bank but the original world supplying observations, the present-state error *increased* relative to no learned memory: inquisitive **-40.35%** relative gain; economical **-36.35%**. Content matters and a policy alone cannot reproduce the old experiences.
- **Post-hoc unknown-world policy transfer:** on a new bank with correctly matched new teachers, without retraining the attention policy, inquisitive identification accuracy was **47.64%** with 42.16% relative present-error reduction; economical remained near chance **35.11%** with 35.93% reduction. This test was added after reviewing earlier outcomes, so it is exploratory evidence, not an untouched confirmatory claim.
- **No-memory control:** setting the coupling to zero removes memory effects on the live present exactly, independently of posterior or retrieved content.
- **Read-only safeguard:** the stored slow operators do not change while the system queries. This avoids false reconsolidation by *prohibiting writes*, not by learning to distinguish genuine evidence from self-generated recall.

## Development check

On development worlds 901–908, the inquisitive policy selected the correct memory **51.36%**; the economical policy **34.43%**; the analytic best-discrete sensor **53.21%**. The learned inquisitive reader was below the analytic reader by about 1.86 accuracy points. The same sign repeated on the eight-world replication split (gap 4.00 points).

## What this does and does not establish

The active state is influenced by a posterior mixture over **learned structural operators** without resetting to a historical state. The policy controls *which limited fragments are observed*, learns that policy from delayed feedback, and can be swapped independently of the bank. This operationalizes the proposal that a system's access habits can differ even when its memories are identical.

However the policies use simple linear features, deliberate reward preferences, and an engineered Gaussian source and observation model. The present is an 8D numerical recurrent vector, not a model of cortical body–brain loops. Multi-area memory binding, safe reconsolidation, personal identity, phenomenology, and a competitive continual-learning solution are not demonstrated.

### Next decisive experiment

Make both agents optimize the **same** utility while experiencing different histories, and train a *continuous confidence-conditional* access rule rather than yielding the current all-or-none two extremes. Require the policy's access style to transfer across a new memory bank and beat cost-matched conventional active sensing. Only then consider allowing **trusted, provenance-checked** retrieval-linked changes to durable operators, with false internal replay as the principal attacker.
