"""Gate 4 — infer a memory from real consequences, then probe it deliberately.

No task indices are accessible to the bank or query policy. During learning,
multiple observed transitions help discover contexts with identical cue vectors.
At read time the query policy may *act on the external world*, measuring only
one fixed sensor per intervention. Probing the bank alone cannot reveal which
of several identical-cue memories the external world currently follows.

All operators and routines are ordinary numerical least-squares, clustering,
active experimental design and Bayesian likelihood, not new neuronal physics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from structural_memory import base_rotation, trace, unit


@dataclass(frozen=True)
class ActiveProtocol:
    dim: int = 8
    slots: int = 3
    context_dim: int = 8
    trace_starts: int = 12
    trace_steps: int = 3
    episodes_per_task: int = 24
    observation_noise: float = .0015
    cue_noise: float = .20
    sensor_noise: float = .045
    split_threshold: float = .075
    learning_rate: float = .45
    probe_budget: int = 2
    seeds: tuple[int, ...] = (601, 602, 603, 604, 605, 606, 607, 608)


def world(p: ActiveProtocol, rng: np.random.Generator):
    """Distinct rank-two teacher operators, indistinguishable cue centers."""
    base = base_rotation(p.dim)
    operators = []
    for _ in range(p.slots):
        a = rng.normal(size=(p.dim, 2)) @ rng.normal(size=(2, p.dim))
        a *= .24 / np.linalg.norm(a, 2)
        operators.append(base + a)
    cue_center = unit(rng.normal(size=p.context_dim))
    receiver = unit(rng.normal(size=p.dim))
    return base, np.asarray(operators), cue_center, receiver


def observations(matrix: np.ndarray, p: ActiveProtocol, rng: np.random.Generator):
    """An episode contains unbiased observed transitions from new start states."""
    initial = rng.normal(size=(p.trace_starts, p.dim))
    clean = np.stack([trace(matrix, x, p.trace_steps) for x in initial])
    return clean + rng.normal(scale=p.observation_noise, size=clean.shape)


def infer_local_transition(frames: np.ndarray, base: np.ndarray, ridge: float = .002):
    """Solve transition residual using Sihti differences; no saved traces."""
    if frames.ndim != 3 or frames.shape[-1] != base.shape[0]:
        raise ValueError("invalid episode dimensions")
    x = frames[:, :-1, :].reshape(-1, base.shape[0])
    diffs = (frames[:, :-1, :] - frames[:, 1:, :]).reshape(-1, base.shape[0])
    next_states = x - diffs
    target = next_states - x @ base.T
    return np.linalg.solve(x.T @ x + ridge*np.eye(base.shape[0]), x.T @ target).T


class TransitionMemory:
    """Learned, independent slow operators; novelty derives from dynamics.

    Storage: S*D*D operator coefficients + S*C context prototypes + S counters.
    The observed transition episode is transient. Inference itself is not a
    magical hippocampal operation: it is multi-probe ridge regression.
    """
    def __init__(self, p: ActiveProtocol, base: np.ndarray, direct: bool = False):
        self.p, self.base, self.direct = p, base.copy(), direct
        self.weights = np.zeros((p.slots, p.dim, p.dim))
        self.keys = np.zeros((p.slots, p.context_dim))
        self.counts = np.zeros(p.slots, dtype=int)
        self.active = 0
        self.rejections = 0
        self.external_updates = 0
        self.self_replay_rejections = 0
        self.uncertain_rejections = 0

    @property
    def scalar_budget(self):
        return int(self.weights.size + self.keys.size + self.counts.size)

    def fit_episode(self, cue: np.ndarray, frames: np.ndarray):
        sample = infer_local_transition(frames, self.base) if not self.direct else self._direct(frames)
        if self.active == 0:
            slot = 0
            self.active = 1
        else:
            distances = np.linalg.norm((self.weights[:self.active] - sample).reshape(self.active, -1), axis=1)
            slot = int(np.argmin(distances))
            if distances[slot] > self.p.split_threshold:
                if self.active == self.p.slots:
                    self.rejections += 1
                    return None
                slot = self.active
                self.active += 1
        n = self.counts[slot]
        self.weights[slot] = sample if n == 0 else (1-self.p.learning_rate)*self.weights[slot] + self.p.learning_rate*sample
        self.keys[slot] = unit(unit(cue) if n == 0 else (1-1/min(n+1,20))*self.keys[slot]+unit(cue)/min(n+1,20))
        self.counts[slot] += 1
        return slot

    def _direct(self, frames):
        """Same exact regression from directly observed next states."""
        x=frames[:, :-1, :].reshape(-1,self.p.dim)
        y=frames[:, 1:, :].reshape(-1,self.p.dim) - x@self.base.T
        return np.linalg.solve(x.T@x+.002*np.eye(self.p.dim),x.T@y).T

    def candidates(self):
        return self.base[None] + self.weights[:self.active]

    def cue_prior(self, cue, strength: float = 0.):
        # For the collision experiment, no privileged task-ID signal may
        # enter via context: uniform prior is the honest default.
        if not self.active:
            raise RuntimeError('bank is empty')
        if strength == 0:
            return np.ones(self.active)/self.active
        logits = strength*(self.keys[:self.active]@unit(cue))
        logits -= np.max(logits)
        probs = np.exp(logits)
        return probs/probs.sum()

    def update_from_external_probe(self, slot: int, probe: np.ndarray, receiver: np.ndarray,
                                   observation: float, posterior: float, provenance: str,
                                   rate: float = .03, confidence_threshold: float = .90):
        """Optional reconsolidation. Only authentic external sensor results can write.

        A replay is intentionally forbidden even if its numeric value looks
        convincing; overconfidence is not proof of trustworthy provenance.
        """
        if provenance != 'external':
            self.self_replay_rejections += 1
            return False
        if posterior < confidence_threshold:
            self.uncertain_rejections += 1
            return False
        if not 0 <= slot < self.active:
            raise ValueError('unallocated slot')
        pred = float(receiver@(self.base+self.weights[slot])@probe)
        error = observation-pred
        self.weights[slot] += rate*error*np.outer(receiver,probe)/(
            float(receiver@receiver)*float(probe@probe)+1e-12)
        self.external_updates += 1
        return True


def receiver_response(matrix: np.ndarray, probe: np.ndarray, receiver: np.ndarray):
    return float(receiver @ matrix @ probe)


def optimized_probe(operators: np.ndarray, receiver: np.ndarray, posterior: np.ndarray):
    """Unit probe maximizing predicted between-hypothesis scalar variance.

    Unlike a memorized answer, this choice is made only from already learned
    operators. Actual unknown-world response must still be observed.
    """
    rows = np.einsum('i,nij->nj', receiver, operators)
    center = np.sum(posterior[:,None]*rows,axis=0)
    diff = rows-center
    cov = (diff.T*posterior)@diff
    w,v = np.linalg.eigh(cov)
    return unit(v[:,-1])


def fixed_probe(operators: np.ndarray, receiver: np.ndarray):
    weights = np.full(len(operators),1./len(operators))
    return optimized_probe(operators,receiver,weights)


def query(bank: TransitionMemory, true_operator: np.ndarray, receiver: np.ndarray,
          cue: np.ndarray, rng: np.random.Generator, policy='adaptive',
          budget: int | None = None, sensor_noise: float | None = None,
          allow_reconsolidation: bool = False, provenance: str = 'external'):
    """Choose experiments, read actual external consequences, infer memory slot.

    No true-operator parameter enters the policy: it is used *only* in the
    measurement channel, analogous to a plant/system being investigated.
    Querying the stored bank without this observation cannot resolve
    identical-context hypotheses. Two probes = two measured scalars.
    """
    ops = bank.candidates()
    if len(ops) == 0:
        raise RuntimeError('empty memory bank')
    if budget is None: budget = bank.p.probe_budget
    if sensor_noise is None: sensor_noise = bank.p.sensor_noise
    if not 0 <= budget <= 20: raise ValueError('invalid probe budget')
    posterior = bank.cue_prior(cue)
    records=[]
    static = fixed_probe(ops,receiver)
    first_guess = int(np.argmax(posterior))
    for k in range(budget):
        if policy == 'adaptive': x=optimized_probe(ops,receiver,posterior)
        elif policy == 'fixed': x=static
        elif policy == 'random': x=unit(rng.normal(size=bank.p.dim))
        elif policy == 'basis': x=np.eye(bank.p.dim)[k % bank.p.dim]
        elif policy == 'cue_only': break
        else: raise ValueError('unknown policy')
        predicted=np.array([receiver_response(op,x,receiver) for op in ops])
        observation=receiver_response(true_operator,x,receiver)+rng.normal(scale=sensor_noise)
        # Independent Gaussian sensor noise; external observation costs one.
        logp=np.log(np.maximum(posterior,1e-200))-.5*((predicted-observation)/sensor_noise)**2
        logp-=np.max(logp)
        posterior=np.exp(logp);posterior/=posterior.sum()
        records.append((x,observation,predicted,posterior.copy()))
    identified=int(np.argmax(posterior)) if records else first_guess
    if allow_reconsolidation:
        for x,y,_,post in records:
            bank.update_from_external_probe(identified,x,receiver,y,float(post[identified]),provenance)
    return {'slot':identified,'posterior':posterior,'records':records,
            'observations':len(records),'cue_only_slot':first_guess}
