# Gate 3 — learned dynamics that coexist, then reawaken (29 October 2026)

**Hypothesis:** It is possible for bounded slow *operators* learned from a short sequence of fading states to preserve multiple distinct dynamics while a short-term state is erased. A later neutral probe can reactivate those dynamics. The specific protection mechanism is **learned context routing into separate slow operators**, not Sihti's difference identity by itself.

## Inputs and memory

The teacher is an 8-dimensional autonomous damped rotation (`0.78` spectral radius) plus a hidden, rank-two context-specific linear residual with spectral norm `0.20`. A, B and C arrive sequentially, each supplying 80 independent five-step ringdowns from different random initial states (no full traces stored between updates). Observed states have Gaussian noise SD `0.0015`. The slow learner updates its operator from the Sihti residue `r_t=x_t-x_(t+1)`, reconstructing a transition locally with `x_(t+1)=x_t-r_t`. Its online normalized-LMS learning rate is `0.42`. The base rotation is known to all comparators and is not counted as task-specific memory.

Each regime has an eight-dimensional noisy observable context vector, sampled around a different orthogonal center (train/test context noise SD `0.12`). The learner receives **no A/B/C task labels**. It allocates a new prototype/operator slot only when cosine similarity to all existing prototypes is less than `0.70`. It can hold three slots; when full, unknown contexts are rejected instead of overwriting a remembered operator. The stored slow state is **3×8×8 operator coefficients + 3×8 learned prototypes + 3 counters = 219 scalars**. Initial fast state and teaching traces are erased after each update. A read query does not update the memory.

At checkpoints after A, B and C, give every context **the same 64 fresh probe vectors**, including all eight basis directions plus random unseen directions. Hard-reset transient state before every probe. Measure the five subsequent autonomous time steps with no additional stimuli. The target is the unknown teacher's corresponding response. Define `gain = 1 - SSE(learner-target) / SSE(base_only-target)` across all probe trajectories except their common initial state. 1.0 means faithful recovery of the missing operator-specific effect; 0.0 is the no-memory baseline; negative values are worse than no memory.

## Baselines and attackers

- **Shared**: one overwritable operator, charged the same 219 scalar allocation (unused slots); tests catastrophic interference.
- **Oracle-indexed**: same dynamics/learning rule in three separate operators, but privileged true task IDs. It is an *upper reference*, not a fair learned-routing competitor.
- **Direct-transition routed learner**: identical three operators, prototype router and 219-scalar budget, except learned directly from `(x_t,x_(t+1))` rather than from differences. **This is the key residue-specific control.** It should tie numerically for this linear transformation.
- **FIFO**: last nine context+input+output triplets, 9×24=216 scalars, with local context-weighted ridge. No old task replay; no uncharged storage.
- **Near-collision attacker**: A and B context centroids artificially brought close; there is no reliable cue to distinguish them. Measure old A after C.
- **Full-bank attacker**: introduce D after three slots are full; check refusal, old-slot preservation and D retrieval.
- **Unseen noisy-cue attack**: keep training contexts SD `0.12`, but test with SD `0.38`, to expose routing reliability.

## Decision rules

Mechanism pass requires after-C old A and new C gain each `>0.70`, old A gain greater than shared by `>0.40`, learned routing A within `0.10` of task-ID privileged oracle, old A better than shared in at least seven of eight seeds, and **exactly zero** change to the A operator when only B and C are written. A separate, more demanding residue-specific advance would require beating the matched direct-transition router. **If it ties, claim only the operator/routing mechanism.** Independent memory scale, novel topology or compression advantages are not established. Report failure cases without changing the protocol.

Development/pilot used seeds `301–308`; tests and comparator were refined following that run. **Independent confirmation uses new fixed seeds `501–508`**, not used to choose the method. The protocol was prepared before confirmation; it is not an external/public pre-registration. Report means, seedwise values and all attacker outcomes. Capacities and real-world or biological implications are limited by the synthetic task design.
