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

## What would earn a genuine advantage?

- A storage budget that is smaller than a direct state/key-value store, charged honestly for *every* slow weight.
- Band-selective retention: allow fast residues to be replaced/forgotten while older structural or task-relevant bands survive, compared with matched direct-state compression.
- Delayed cues and independently corrupted contexts; test unknown and conflicting cues and old-item forgetting after many new writes.
- Learned active re-entry: a controller must choose a probe/phase schedule and beat passive or equally expressive readouts at matched cost.
- Separate self-generated replays from *new evidence* before allowing them to rewrite slow memory (see SighImageSuper's self-query contamination controls).
- Compare ordinary cyclic rotation, linear reservoirs, GRU/SSM and explicit episodic memory. No Möbius-specific advantage is assumed.

**Next pre-registered gate to build:** online band-specific consolidation under a fixed scalar budget and a distribution shift. Require a real improvement over equal-parameter direct state memory, and report how performance changes when old cues are displaced by newly learned cues.
