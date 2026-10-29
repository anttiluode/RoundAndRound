# Gate 4 — Can a learned loop ask which past operator applies?

**Date:** 2026-10-29. Prepared and revised after development seeds **601–608**, before the independent evaluation on **701–708**. This is a local frozen-confirmation protocol, *not* an externally preregistered trial. Source code deliberately exposes all choices.

## Hypothesis and necessary information boundary

When two or more prior dynamics share **the same context cue**, a stored bank cannot determine which hidden world is currently active merely by examining its own contents. An action can resolve the ambiguity **only if it produces a new observation** that depends on the hidden world. Gate 4 studies this active identification: choose a probe, observe a **single scalar** response from a fixed receiver, update belief about which learned operator matches it, and choose a second probe conditionally. The system never reads the true hidden regime label at train or test time; a simulation oracle uses it *only to generate physical responses and score outcomes*.

This is active system identification, not a claim that introspection alone creates external evidence, not a neural model and not evidence for consciousness or a novel learning theorem.

## Teacher and memory

- Three 8D independent damped rotation + rank-two teacher residual operators, each residual spectral norm **0.24**; background 8D rotation and random scalar receiver known to all models.
- **Exactly identical mean contextual cue** for all three regimes. Per-episode cue noise SD **0.20**. Context-only decisions should score ~1/3.
- Sequential A then B then C, each with 24 episodes, each episode 12 random 8D starting states and three observed transitions, state measurement noise SD **0.0015**.
- A local episode operator is inferred from these observations, either as Sihti differences `r=x_now-x_next` or directly from observed `(x_now, x_next)`. Both are expected to be algebraically identical. A new slow operator slot is created when the nearest learned operator has Frobenius distance **>0.075**; otherwise that slow slot is updated with EMA rate **0.45**. Existing slots can be revisited in arbitrary order from measured transitions. No actual A/B/C labels are supplied to the learner.
- Slow memory has exactly **3×8×8 + 3×8 + 3 = 219 scalars** (operators, cue prototypes and access counts). Old transient episode traces are not stored.

## Retrieval and active experimental design

- All fast activity is erased before each query. Same noisy cue in every hidden regime. Only the returned scalar `y=receiver · operator · probe + independent Gaussian noise SD 0.045` can distinguish regimes. Each intervention tests one freely selected unit-norm 8D probe. Budget: **two scalar observations per query**.
- Candidate operators predict their responses. Initially uniform posterior (cue gives no task information). At each step, the *adaptive controller* selects the eigenvector maximizing posterior-weighted variance of predicted scalar responses, then updates posterior by Gaussian likelihood. It has no privileged world labels.
- Compare 0-observation cue-only, two coordinate-basis probes, two random probes, **globally optimized fixed probe repeated twice** (important strong alternative), adaptive first + conditioned second, and privileged known-ID upper reference. The adaptive and fixed policies use the same models and exactly two scalar observations; the adaptive policy incurs additional internal computation.
- Outcome A: hidden-regime identification accuracy, balanced three tasks. Outcome B: independent five-step **operator-specific trajectory error reduction** against known no-memory base on seven fresh vectors per query. This score may be negative and is **not classification accuracy**. Training starts are different from retrieval probe directions.
- Negative controls: shuffle the world that produces feedback while leaving the scored true regime unchanged (should destroy identification); identical-cue zero-observation chance; receiver-null changes are not observable; a wholly new fourth teacher after all three slots filled should not silently rewrite existing slots.

## Reconsolidation control

Probe observations may also adjust stored response operators via local normalized scalar gradient when the estimated slot posterior is at least **0.90** and the observation is declared **external**, update rate **0.03**. **Internal replay is forbidden to write.** A separate posttraining change of A by rank-one amplitude 0.025 challenges this writer. Over 300 queries per seed, compare read-only A, adaptive external writes, and simulated self-generated replay flagged correctly as internal. This is intended to expose corruption as well as potential adaptation; confidence is not a guarantee of correctness.

## Confirmatory rules frozen before seeds 701–708

1. Data-dependent training discovers 3 independent operators with >95% correct sample assignment without labels.
2. Direct transition regression ties Sihti residual regression to 1e-10 max weight difference, as a *negative* claim about uniqueness.
3. Adaptive identification improves over random two-probe accuracy by **>10 percentage points** and over the *optimal fixed two-probe* accuracy by **>5 percentage points**, both mean differences across eight independent worlds.
4. Adaptive operator-specific trajectory error reduction after selecting a loop exceeds **0.80**. This high requirement deliberately tests useful reactivation, not just above-chance slot decisions.
5. Real external cue-dependent plasticity improves changed-A response gain over read-only by **>0.02**, while all 600 replay probe attempts per seed are rejected from writing.

**Full gate passes only if all conditions hold.** Individual sub-results can still be informative. Do not alter thresholds after the confirmatory run. Report all seedwise accuracies and numerical failure modes. Any improvement beyond the strong fixed-probe comparison is an *active experimental design* gain, not proof that the memory representation is new. A matched direct-transition bank is expected to tie the Sihti-coded bank.

Reproduction: `python -m unittest discover -s tests -v` and `python active_reentry_benchmark.py --output results/active_reentry_confirm.json` with `ActiveProtocol(seeds=(701,702,703,704,705,706,707,708))` selected in the benchmark runner; the CLI includes an `--confirm` switch to avoid changing defaults or development records.
