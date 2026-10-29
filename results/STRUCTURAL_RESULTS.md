# Gate 3: teach a slow operator; erase; probe; reawaken

**Decision (29 October 2026): bounded coexistence PASS, Sihti-specific advantage FAIL.** The *eight development* seeds were 301–308; after the mechanism, strong direct-transition control, tests and thresholds were finalized locally, a separate *eight confirmation* seeds 501–508 reproduced the result. Protocol: [STRUCTURAL_GATE.md](../docs/STRUCTURAL_GATE.md). Complete seedwise results can be produced with `python structural_benchmark.py`; committed [compact confirmation JSON](structural_confirm_summary.json) includes every main seedwise checkpoint and all attacker means.

Three distinct rank-two response perturbations are learned from noisy five-step ringdowns, each under a distinct observed contextual cue, in order A → B → C. A finite prototype router recognizes contexts, assigns 3 operators, and refuses to silently overwrite unfamiliar contexts when full. **The fast state is erased after every learning episode**; retrieval sees only a new noisy context and a fresh, never-before-used probe state. The continuation is autonomous: no task-specific input after the probe.

## Eight independent confirmation worlds (seeds 501–508)

The score is **operator-specific response gain**: `1-SSE(memory_response-teacher_response)/SSE(base_response-teacher_response)` across steps 1–5 and the same 64 novel probe vectors in all tasks. `1` is perfect, `0` is the untrained base, and negative values are worse than no memory. It is *not classification accuracy*.

| After learning C | A response gain | B response gain | C response gain |
|---|---:|---:|---:|
| **3 context-routed slow operators** | **0.997089** | **0.996825** | **0.997703** |
| Context-routed **direct transition** model, same 219 scalars | **0.997089** | **0.996825** | **0.997703** |
| Privileged true-ID three-operator reference | 0.997089 | 0.996825 | 0.997703 |
| One shared overwritable operator (allocated 219 scalars) | **−1.095388** | **−1.211567** | 0.997703 |
| Last-nine transition FIFO (216 scalars) | 0.004771 | 0.007747 | 0.562644 |

The learned context router has **100%** accuracy on the constructed, well-separated noisy context clusters at all three checkpoints. The stored A operator has exactly **zero numerical drift** after B and C in all eight confirmation seeds, as expected for isolated slots. Its response remains readable after the same fast reset used for every memory. This meets the stated **bounded-coexistence mechanism gate** on all eight seeds.

The **strongest matched control ties to numerical precision**: the maximum learned-operator weight difference is below `5e-17`. The identity `x_next = x - (x-x_next)` means that LMS trained on a Sihti difference is algebraically just LMS trained on the directly observed transition. This is emphatically **not** evidence of a special memory compression, new learning rule or hippocampal physics.

## Attacks: where it breaks

| Attack (confirmation means) | Outcome |
|---|---|
| A/B context identities nearly collide | Old A response gain becomes **−1.1094** after C; contexts alias and writing damages A |
| Present unknown D after all three slots filled | D response gain **~0**; writes are refused, and existing slots are protected |
| Independent noisy query context SD 0.38 instead of 0.12 | Context routing accuracy **52.93%**, old A response gain **0.5308** |

The success therefore depends on distinguishable contexts and enough allocated expert capacity. It is a tractable architecture for **three known operating regimes**, not a cure for limitless continual learning. Neither the dummy shared operator nor the tiny FIFO is the strongest possible alternative. Ordinary context-gated transitions are the decisive comparison—and they tie.

## Next unfalsified question

A genuinely new contribution would require learned **context splitting/merging and capacity allocation without experimenter-designed separated keys**, plus selective consolidation when the third slot is exhausted. A performance benefit over context-routed direct transition learning would require a stricter information/computation advantage or a genuinely different mechanism (e.g. distributed retained geometry or noncommuting active read paths). Keep these as hypotheses, not results.
