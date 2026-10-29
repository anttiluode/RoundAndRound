# RoundAndRound — train the residues, reawaken the loop

**First gate, 29 October 2026.** An offline NumPy instrument for a narrowly testable idea: **can a fast autonomous loop leave a *learned* slow memory of its ringdown, then be erased and reactivated from a partial cue?**

This is an engineering analogy to [SighImageSuper](https://github.com/anttiluode/SighImageSuper), [Sihti](https://github.com/anttiluode/Sihti), [MoebiusLoopMemory](https://github.com/anttiluode/MoebiusLoopMemory), [PresentMoment](https://github.com/anttiluode/PresentMoment), and [LoopAImemory](https://github.com/anttiluode/LoopAImemory). It is **not** a hippocampal model, a consciousness claim, a wave simulator or a solution to catastrophic forgetting.

## Mechanism

1. **Fast activity:** independent two-coordinate phase planes rotate continuously with real elapsed time, even without inputs, while decaying exponentially. Input events write through a fixed contextual multiplicative gate.
2. **Ringdown:** freeze the current state and inspect the hypothetical autonomous states at times `[0,1,2,4,8,16]` without changing the live state.
3. **Sihti-style residues:** save the differences `r_i=x(t_i)-x(t_(i+1))`, plus the final core `x(t_last)`. Their sum telescopes to `x(0)` exactly, regardless of the update operator.
4. **Slow training:** fit a regularized cue → residue-band association from episodes. It stores learned weights, **not original traces or individual fast states**.
5. **Reawakening:** erase the fast state completely; feed a noisy learned cue into the slow reader; reconstruct `x(0)` from predicted residues and put that state back into the autonomous loop. Observe its evolution after eight time units.

### Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python benchmark.py
```

## First measured local test (seed 7)

12 synthetic histories; 8 noisy training repetitions each; 24-dimensional independent cue keys; 16-dimensional fast state; held-out noisy cue. The cue identities are **trained and already known**: no new-key generalization is claimed.

| Measured quantity | Result |
|---|---:|
| Maximum Sihti telescoping reconstruction error | 3.55e-15 |
| Fast state erased, then cue-trained residue reactivation RMSE | 0.22174 |
| Equal-cost direct cue → state ridge RMSE | **0.22174 (tie)** |
| Wrong cue RMSE | 3.67203 |
| Empty fast state RMSE | 2.95856 |
| Reawakened state evolved 8 units: RMSE vs original future | 0.14218 |

The direct ridge model **must tie in this linear setup**: summing ridge predictions of telescoping differences equals ridge prediction of the initial state. A Sihti decomposition reorganizes the target but cannot create information or independently establish an advantage. The positive result is a reproducible **mechanism check**: activity can be erased, a cue can recall a trained representation of its past ringdown, and activity can resume.


## Gate 2 — sequential interference: learned residues do not yet solve forgetting

**October 29, 2026, eight seeds (401–408).** New: [continuous-timescale loop and selective memory code](consolidation.py), [A→B→C sequential benchmark](consolidation_benchmark.py), [fixed local protocol](docs/CONSOLIDATION_GATE.md), [source-linked result ledger with eight seedwise primary receipts](results/CONSOLIDATION_RESULTS.md), and [CPU validation with full-JSON artifact](.github/workflows/validate.yml).

This tests a **16D fast state with four short-lived and four long-lived 2D planes**. The readout stores only 16 learned numbers per cue: Sihti's *summed fast residues* and *surviving slow core*, not the entire six-stage trace. An online 12×16 weight matrix (192 scalars) learns three sequential task sets, without old-task rehearsal. Baselines use the same 192-scalar memory or six FIFO key/state slots (168 scalars).

| After learning A, B and C | Old A late-state NRMSE ↓ | New C current-state NRMSE ↓ |
|---|---:|---:|
| Selective Sihti residues/core | 0.9885 | 0.7565 |
| **Direct state with matched slow/fast learning rates** | **0.9885** | **0.7565** |
| Direct state, all-fast learning rate | 1.2700 | **0.6622** |
| Direct state, all-slow learning rate | 0.9885 | 0.7933 |
| Sihti, learning-rate bands reversed | 1.2700 | 0.7046 |
| Six-item FIFO key/value memory | 1.4816 | 1.0044 |

**Decision: failed the compound success criterion.** Selective gating improved A relative to fast-only writing in all 8 seeds (0.2815 lower NRMSE), but sacrificed too much new C learning and tied the direct matched-timescale memory. Even old A NRMSE ≈0.99 is little better than predicting zero (NRMSE=1.0). The old A late-state error degraded from 0.7694 immediately after A to 0.9885 after C. This is not protected old memory. Sihti re-encoding does not create information or an extra capacity advantage.

### Run it

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python consolidation_benchmark.py --output results/consolidation_v1.json
```

The benchmark writes every seed and intermediate checkpoint, including the noncausal all-task ridge reference, to local JSON. The GitHub Actions workflow also uploads the full result. The protocol was drafted locally before the eight-seed run, **but not publicly preregistered**.

**Next scientific direction:** after this negative result, move beyond linearly invertible residue coordinates. Test *learned operator/transition memory* that changes how a neutral later probe travels after the fast loop is erased; keep direct-state and time-scale-matched controls. The open question is structural consolidation and active reactivation, not more memory filtering.

## Gate 3 — structural memories that learn and coexist

**October 29, 2026.** This follows the negative Gate 2: instead of remembering the fast vector, a slow memory learns how earlier activity changes a **response operator**. Each short five-step ringdown is encoded as Sihti-style successive differences; an online local update learns a context-specific perturbation of a stable autonomous rotation. The transient loop is discarded. After independently learning A, B, C, a fresh neutral probe starts from the **same initial fast state** in every condition, and the stored slow operator determines the response.

**Implemented:** [three-slot learned operator bank](structural_memory.py) · [A→B→C benchmark, collisions, full-bank and noisy-cue attacks](structural_benchmark.py) · [five unit tests](tests/test_structural_memory.py) · [fixed local protocol](docs/STRUCTURAL_GATE.md) · [full result interpretation](results/STRUCTURAL_RESULTS.md) · [compact independent confirmation receipts](results/structural_confirm_summary.json).

An independent eight-seed confirmation (501–508; development 301–308) reports **operator-specific response gain** = 1 − (model trajectory squared error)/(no-memory baseline squared error), on 64 *new probe directions* per regime. It is *not* classification accuracy.

| After learning A, B, C | Old A gain ↑ | Old B gain ↑ | New C gain ↑ |
|---|---:|---:|---:|
| **Context-gated learned operators (219 scalar storage)** | **0.9971** | **0.9968** | **0.9977** |
| **Matched direct-transition gated memory (219 scalars)** | **0.9971** | **0.9968** | **0.9977** |
| Privileged true-label independent operators | 0.9971 | 0.9968 | 0.9977 |
| One overwritable shared operator | −1.0954 | −1.2116 | 0.9977 |
| Recent transition FIFO (216 scalar storage) | 0.0048 | 0.0077 | 0.5626 |

**Bounded coexistence gate PASS:** A and B are retained as independently learned response dynamics while C is acquired, no old fast activity or original training traces are stored, and A's operator has **zero drift** during later writes. **Residue-specific gain FAIL:** an ordinary *equally budgeted context-gated direct transition learner* produces the same operator matrices within 5×10⁻¹⁷. The benefit comes from **separate writing and learned contextual routing**, not from a novel Sihti learning rule.

**Failure modes are essential:** nearly colliding A/B contexts yield old A gain **−1.1094**; an unrecognized fourth task D gets gain ≈0 when the three slots are full because it is deliberately refused; on substantially noisier *test* context cues (SD 0.38 versus training 0.12), routing accuracy falls to **52.93%**. The separate context clusters are designed by the experimenter, not discovered autonomously. This is **not yet open-ended continual learning, hippocampal biology, a self-model or general visual memory**.

Run:
```bash
python -m unittest discover -s tests -v
python structural_benchmark.py --seeds 501,502,503,504,505,506,507,508 --output results/structural_confirm.json
```

**Next gate:** learning context boundaries (split/merge), replacing full operators without unacceptable forgetting, and avoiding contamination by self-generated replay. Keep the direct routed learner as the decisive comparison.

## What would earn a genuine advantage?

- A storage budget that is smaller than a direct state/key-value store, charged honestly for *every* slow weight.
- Band-selective retention: allow fast residues to be replaced/forgotten while older structural or task-relevant bands survive, compared with matched direct-state compression.
- Delayed cues and independently corrupted contexts; test unknown and conflicting cues and old-item forgetting after many new writes.
- Learned active re-entry: a controller must choose a probe/phase schedule and beat passive or equally expressive readouts at matched cost.
- Separate self-generated replays from *new evidence* before allowing them to rewrite slow memory (see SighImageSuper's self-query contamination controls).
- Compare ordinary cyclic rotation, linear reservoirs, GRU/SSM and explicit episodic memory. No Möbius-specific advantage is assumed.

**Research status:** Gates 1–3 are now in this repository. The next unsolved experiment is context discovery and controlled memory allocation when slots are full, with strong matched controls.
