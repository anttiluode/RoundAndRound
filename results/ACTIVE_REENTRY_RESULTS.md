# Gate 4 — ambiguous memory, two external probes, optional reconsolidation

**Status 29 October 2026:** ***partial support for adaptive access; full gate FAILED.*** Independent development 601–608 and confirmation 701–708. [Frozen-before-confirmation local protocol](../docs/ACTIVE_REENTRY_GATE.md); complete [seedwise numerical receipts](active_reentry_confirm.json). The companion [posterior-mixture follow-up](../docs/MIXTURE_FOLLOWUP.md) has a separate independent split and must not be conflated with the originally frozen gate.

## What is actually learned?

Three 8D damped rotating systems have different rank-two transition perturbations but **identical contextual cue centers**. A transient training episode supplies 12 random start states and three observed transitions each. Unsupervised-to-label **dynamics clustering** assigns these episodes to three independent slow operators in sequence A→B→C, without knowing those labels. The bank retains **219 scalars**: 3×8×8 matrices, 3×8 context prototypes, 3 counts. All transient episodes are erased. This is ordinary local transition regression with a hand-coded novelty threshold; **the query policy is also engineered using predictive variance, not spontaneously discovered by reinforcement learning**.

At retrieval no past fast state or correct memory label is supplied. Each query can set two unit-norm probe vectors and receive one scalar fixed-receiver measurement for each. Identification requires **real new evidence**: with no observations the identical context cue is chance level. A later seven-vector, five-step trajectory test is separate from the two scalar interrogation probes.

## Independent Gate-4 confirmation (seeds 701–708)

| Policy | Hidden regime accuracy (↑) | Reactivated-trajectory gain (↑) | Scalars measured per query |
|---|---:|---:|---:|
| Cue alone (no observations) | 0.3333 | −0.5791 | 0 |
| Two pre-chosen basis probes | 0.5312 | 0.0000 | 2 |
| Two random probes | 0.5462 | 0.0047 | 2 |
| **Optimal fixed probe, repeated twice** | **0.7226** | **0.4026** | **2** |
| **Posterior-dependent active probes** | **0.7729** | **0.5025** | **2** |
| Privileged operator ID (not a deployable competitor) | 1.0000 | 0.99970 | 0 |

Gain is `1 − SSE(predicted trajectory−true)/SSE(base trajectory−true)`, not classification accuracy. Can be negative. The mean adaptive minus optimal-fixed accuracy is **+0.05035** (positive in 8/8 seeds); its mean advantage over random is **+0.22674**. Two-probe active selection of a memory is useful on this constructed problem, but many incorrect selections generate misleading state evolutions. The predeclared `response gain > 0.80` requirement FAILS.

### Accuracy by world

| Seed | Random | Optimized fixed | Adaptive |
|---|---:|---:|---:|
| 701 | .6750 | .8833 | .8972 |
| 702 | .5000 | .6722 | .7667 |
| 703 | .4472 | .5528 | .5750 |
| 704 | .4833 | .6250 | .6389 |
| 705 | .6028 | .7361 | .8889 |
| 706 | .5000 | .6250 | .6306 |
| 707 | .5611 | .8194 | .8639 |
| 708 | .6000 | .8667 | .9222 |

All three operators were discovered from teaching transitions: mean assignment accuracy **1.000** for the engineered separated dynamics. Sihti-residue learned matrices tie a direct transition-regression control to **5.6e-17** maximum coefficient difference. The performance reflects learning distinct operators and choosing informative experiments, **not special advantages of difference-based Sihti storage**. When response evidence is swapped to a different hidden regime, accuracy compared with the original target drops to **0.1156**: probing faithfully reports the *other* environment, not any magical direct knowledge of the original identity. A receiver-null perturbation is unobservable under the one-step scalar measurement by construction.

## Retrieval can damage memory

A separate controlled test changes operator A slightly after training and allows high-confidence (posterior ≥ .90) writes from observed external scalar probes. Averaged over 8 confirmation worlds:

| Mechanism | Changed-A response gain |
|---|---:|
| Read-only stored memory | **0.99168** |
| Confidence-gated writes from actual noisy retrieval probes | **0.96776** |
| Replay explicitly marked *internal*, forbidden to write | **0.99168** |

The plastic writer writes an average **186** scalar updates per world and unintentionally modifies other memories because some wrong operator selections still achieve high posterior confidence (non-target maximum element drift **0.00234**). All 600 internal-replay update attempts per world were refused, preventing such corruption. Thus **confidence plus provenance controls are not sufficient for safe reconsolidation**. The particular small true change is already poorly identifiable under the sensor noise, making frequent writing counterproductive. Read-only retrieval is the safer mechanism under these conditions. No general claim about human recall or courtroom evidence follows.

## Gate-4 verdict

- PASS: separate dynamics discoverable despite indistinguishable context cues using sufficient *teaching responses*.
- PASS: active two-probe interrogation outperforms random and an optimized fixed two-probe policy on the 8-world confirmation.
- FAIL: hard-selected loop useful-trajectory gain does not reach the frozen 0.80 threshold.
- FAIL: retrieval-driven slow writes do not improve changed memory; they damage it.
- PASS: flagged internally generated replay causes no slow-memory write.
- FAIL: any Sihti-specific computational advantage, general continual learning, unlimited capacity, or autonomous emergence of a self.

## Post-confirmation follow-up: several loops in the present

After the Gate-4 failure, we considered **posterior-weighted prediction across all three stored dynamical memories**, not winner-takes-all retrieval. This uses no additional slow-state scalars or external measurements but evaluates three candidate trajectories per question instead of one. A development evaluation used 701–708; fresh follow-up seed split 801–808 tested [locally frozen follow-up criteria](../docs/MIXTURE_FOLLOWUP.md).

| Same two-measurement budget, follow-up seeds 801–808 | Response gain (↑) |
|---|---:|
| Prior mixture with **zero** external evidence | 0.2885 |
| One hard-selected loop, adaptive queries | 0.7059 |
| **Posterior mixture, adaptive queries** | **0.8045** |
| One hard-selected loop, globally optimal fixed queries | 0.5207 |
| Posterior mixture, globally optimal fixed queries | 0.7124 |
| Hard-selected loop with confidence-based abstention | 0.6377 |

Soft-minus-hard gain **+0.09855**, positive in **7/8** worlds; soft-adaptive-minus-soft-fixed **+0.09204**, positive in **8/8** worlds. The locally frozen post-Gate-4 follow-up criteria pass, but this is *not* a rescue or retest of the originally failed Gate-4 preregistration. The mixture works because under squared error loss the conditional mean is generally a safer prediction than an uncertain hard choice. This is standard Bayesian model averaging, not a novel form of physically entangled loops or proof of emergent cognition.

The important architectural possibility is **multiple learned slow dynamics concurrently contributing to a temporary predictive present**, weighted by new measured consequences. The bottleneck is still learning good operators, choosing meaningful interventions, dealing with unknown dynamics, and deciding when retrieved activity may safely change the store.
