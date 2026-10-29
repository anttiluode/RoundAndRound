"""RoundAndRound: autonomous fast loop and trained Sihti-style residue memory.

This is a synthetic dynamical system, not a physical wave or neuronal model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    dim: int = 16
    input_dim: int = 6
    decay_time: float = 18.0
    gate: float = 0.4
    seed: int = 29


class FastLoop:
    """Continuous-time block rotations, exponential decay, and contextual event writes."""

    def __init__(self, cfg: Config = Config()):
        if cfg.dim < 2 or cfg.dim % 2 or cfg.input_dim < 1 or cfg.decay_time <= 0:
            raise ValueError("dim must be positive even; input_dim and decay_time must be positive")
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.B = rng.normal(size=(cfg.dim, cfg.input_dim)) / np.sqrt(cfg.input_dim)
        self.A = rng.normal(size=(cfg.dim, cfg.dim)) / np.sqrt(cfg.dim)
        self.freq = np.linspace(0.2, 1.4, cfg.dim // 2)
        self.state = np.zeros(cfg.dim, dtype=np.float64)

    def reset(self):
        self.state[:] = 0.0

    def advance(self, dt: float, event: np.ndarray | None = None) -> np.ndarray:
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be a finite nonnegative duration")
        phase = self.freq * dt
        x = self.state.reshape(-1, 2).copy()
        c, s = np.cos(phase), np.sin(phase)
        rotated = np.column_stack((c * x[:, 0] - s * x[:, 1],
                                   s * x[:, 0] + c * x[:, 1])).reshape(-1)
        self.state = np.exp(-dt / self.cfg.decay_time) * rotated
        if event is not None:
            event = np.asarray(event, dtype=np.float64)
            if event.shape != (self.cfg.input_dim,):
                raise ValueError("wrong event dimensions")
            basal = self.B @ event
            self.state += basal * (1 + self.cfg.gate * np.tanh(self.A @ self.state))
        return self.state.copy()

    def restore(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.cfg.dim,):
            raise ValueError("wrong state shape")
        self.state[:] = state

    def ringdown(self, delays: tuple[float, ...] = (0., 1., 2., 4., 8., 16.)) -> np.ndarray:
        """Observe an autonomous trajectory without modifying the live state."""
        if len(delays) < 2 or delays[0] != 0 or any(b <= a for a, b in zip(delays, delays[1:])):
            raise ValueError("delays must start at 0 and be strictly increasing")
        old = self.state.copy()
        result = [old.copy()]
        last = 0.
        for t in delays[1:]:
            result.append(self.advance(t-last))
            last = t
        self.state = old
        return np.stack(result)


def residues(trace: np.ndarray) -> np.ndarray:
    """Sihti telescope: differences between successive states, plus final core."""
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 2 or len(trace) < 2:
        raise ValueError("trace should have shape (time>=2, dim)")
    return np.concatenate((trace[:-1] - trace[1:], trace[-1:]), axis=0)


def reconstruct(bands: np.ndarray, start: int = 0) -> np.ndarray:
    """Reconstruct a trace stage by summing its remaining residues and core."""
    bands = np.asarray(bands, dtype=np.float64)
    if bands.ndim != 2 or not (0 <= start < len(bands)):
        raise ValueError("invalid bands or stage")
    return bands[start:].sum(axis=0)


class SlowResidueMemory:
    """Ridge-trained cue -> residue bands. Only weights persist, not fast traces.

    fit() is supervised associative storage. No claims of unsupervised memory,
    automatic consolidation, continual learning, or special compression.
    """
    def __init__(self, ridge: float = 1e-2):
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        self.ridge = ridge
        self.weights: np.ndarray | None = None
        self.band_shape: tuple[int, int] | None = None

    def fit(self, cues: np.ndarray, traces: np.ndarray) -> None:
        cues, traces = np.asarray(cues, dtype=float), np.asarray(traces, dtype=float)
        if cues.ndim != 2 or traces.ndim != 3 or len(cues) != len(traces):
            raise ValueError("expected (episodes,cue_dim) and (episodes,times,dim)")
        self.band_shape = traces.shape[1:]
        targets = np.stack([residues(t) for t in traces]).reshape(len(cues), -1)
        design = np.column_stack([cues, np.ones(len(cues))])
        reg = np.eye(design.shape[1]) * self.ridge
        reg[-1, -1] = 0  # don't regularize bias
        self.weights = np.linalg.solve(design.T @ design + reg, design.T @ targets)

    def recall_bands(self, cue: np.ndarray) -> np.ndarray:
        if self.weights is None or self.band_shape is None:
            raise RuntimeError("must fit before recalling")
        cue = np.asarray(cue, dtype=float)
        if cue.shape != (self.weights.shape[0] - 1,):
            raise ValueError("wrong cue dimensions")
        return (np.r_[cue, 1.] @ self.weights).reshape(self.band_shape)

    def awaken(self, cue: np.ndarray, fast: FastLoop, stage: int = 0) -> np.ndarray:
        """Restore reconstructed state into an erased fast loop; return state."""
        value = reconstruct(self.recall_bands(cue), start=stage)
        fast.restore(value)
        return value.copy()
