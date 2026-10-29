# Gate 2 results — October 29, 2026

**Decision: FAIL for a residue-specific benefit.** Local eight-seed run, seeds 401–408, no parameter search. Full seedwise checkpoint metrics are reproducible from the committed `consolidation_benchmark.py` and stored by the CI job as a JSON artifact.

The system learns A, then B, then C without revisiting earlier episodes. Every example begins with an erased fast state. All weight-based online memories have **192 trained scalar weights**, compared with six FIFO key/value slots charging **168 scalars**. The model uses 16-dimensional state, 12-dimensional noisy identity cues and a declared four-fast/four-slow set of rotation planes.

NRMSE is normalized by the RMS of the true state. **A zero prediction scores 1.0**. Lower is better.

## Primary eight-seed means (after learning C)

| Model | Old A, 16-time-unit state NRMSE | New C, current state NRMSE |
|---|---:|---:|
| Sihti selective core/residue | **0.9885** | 0.7565 |
| Direct state, matched per-feature learning rates | **0.9885** | 0.7565 |
| Direct state, uniform fast writes | 1.2700 | **0.6622** |
| Direct state, uniform slow writes | **0.9885** | 0.7933 |
| Sihti, reversed fast/slow rates | 1.2700 | 0.7046 |
| FIFO, six last key/value items | 1.4816 | 1.0044 |

A slow-mode gating policy beats uniform-fast old memory by 0.2815 NRMSE (in all 8 seeds), but **fails to stay within +0.05 on new C** (difference +0.0943) and **fails to beat direct matched-rate memory** (difference <4×10⁻⁵ in every examined seed/task). Uniform slow writing ties the old late-state NRMSE, confirming that slower writes rather than special residues explain this metric. Even 0.9885 is barely better than an empty-state prediction.

## Per-seed primary receipts

| Seed | Selective A late | Direct-fast A late | Matched direct A late | Selective C now | Direct-fast C now |
|---|---:|---:|---:|---:|---:|
| 401 | .9044 | 1.1641 | .9044 | .7240 | .5888 |
| 402 | 1.0247 | 1.1526 | 1.0247 | .7219 | .6272 |
| 403 | 1.0214 | 1.3787 | 1.0214 | .7947 | .6927 |
| 404 | .8862 | 1.1730 | .8862 | .7874 | .7272 |
| 405 | 1.1964 | 1.6486 | 1.1964 | .6740 | .5485 |
| 406 | .9041 | 1.1556 | .9041 | .7628 | .6555 |
| 407 | 1.0081 | 1.1985 | 1.0081 | .8363 | .7907 |
| 408 | .9627 | 1.2891 | .9627 | .7507 | .6668 |

## Interference and negative controls

Old A selective late NRMSE deteriorated from **0.7694 after A** to **0.8883 after B** and **0.9885 after C**. The system has **not** shown durable old-episode recall under interference. Sihti code is 16 scalars per cue, not a hidden 96-scalar residue bank. Telescoping identity error is effectively numerical zero; discarding the fast core produces small error (~0.000399 RMS).

The direct matched-rate memory ties because the coded slow core is simply a known invertible transform of those slow coordinates; the fast differences nearly reproduce current fast coordinates. Hence selective update rates, not residue storage, account for performance against a uniform-rate baseline. No special neural dynamics, memory-compression superiority or catastrophic-forgetting solution follows.

## Next experiment implied by this failure

A useful successor must change the **memory mechanism**, not just the coordinate basis: e.g. protect a sparse learned slow *operator* or conditional transition, rather than fit a cue directly to the same initial state. Run teach → erase fast state → replay from learned structural memory → test untrained delayed queries, contrasting exact-state and equal-parameter operator controls. Keep the direct-time-scale rate control in place.
