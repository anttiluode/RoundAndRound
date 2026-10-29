"""Gate 3: learned *operators*, not remembered states.

A ringdown of an unknown system supplies transitions; the slow memory learns
its transition residual relative to a known base rotation. After destroying
all transient states, a neutral probe excites the *learned* dynamics.

This is a synthetic linear system with prototype-routed operators. Gating,
ordinary LMS and the residual calculation are established methods, not a
neuroscience or unique wave/memory claim.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Protocol:
    dim: int = 8
    context_dim: int = 8
    slots: int = 3
    task_count: int = 3
    train_episodes: int = 80
    trace_steps: int = 5
    test_probes: int = 64
    observed_noise: float = 0.0015
    context_noise: float = 0.12
    learning_rate: float = 0.42
    allocation_cosine: float = 0.70
    residual_spectral_norm: float = 0.20
    seeds: tuple[int, ...] = (301, 302, 303, 304, 305, 306, 307, 308)


def unit(v):
    a = np.asarray(v, dtype=float)
    return a / max(float(np.linalg.norm(a)), 1e-12)


def base_rotation(dim: int) -> np.ndarray:
    """Stable damped rotation; same base dynamics for every history."""
    if dim < 2 or dim % 2:
        raise ValueError("dimension must be positive and even")
    a = np.linspace(0.25, 0.90, dim // 2)
    m = np.zeros((dim, dim))
    for i, theta in enumerate(a):
        c, s = np.cos(theta), np.sin(theta)
        m[2*i:2*i+2, 2*i:2*i+2] = 0.78*np.array([[c, -s],[s,c]])
    return m


def sample_world(p: Protocol, rng: np.random.Generator):
    """Fixed teacher operators and independent contextual vectors.

    The context is observable; the true task index is NOT supplied to the
    learned router. Teachers have distinct rank-two residuals.
    """
    q, _ = np.linalg.qr(rng.normal(size=(p.context_dim,p.context_dim)))
    centers = q.T[:p.task_count].copy()
    op = base_rotation(p.dim)
    residuals=[]
    for _ in range(p.task_count):
        u = rng.normal(size=(p.dim,2))
        v = rng.normal(size=(2,p.dim))
        w = u@v
        w *= p.residual_spectral_norm / np.linalg.norm(w,2)
        residuals.append(w)
    return centers,op,np.asarray(residuals)


def trace(matrix:np.ndarray, x0:np.ndarray, steps:int)->np.ndarray:
    x=unit(x0)
    frames=[x]
    for _ in range(steps):
        x=matrix@x
        frames.append(x.copy())
    return np.asarray(frames)


def sihti_differences(frames:np.ndarray):
    frames=np.asarray(frames)
    return frames[:-1]-frames[1:]


def learn_ringdown(operator:np.ndarray, base:np.ndarray,
                   frames:np.ndarray, rate:float) -> None:
    """Online local LMS using temporal differences (Sihti residues).

    x_next = x_now - residue. No stored raw episode after each update.
    Operator contains only the learned change relative to the fixed base.
    """
    diffs = sihti_differences(frames)
    for x, residue in zip(frames[:-1],diffs):
        x_next = x-residue
        missing = x_next-base@x-operator@x
        operator += rate*np.outer(missing,x)/(float(x@x)+1e-5)


class StructuralBank:
    """Finite set of independently writable slow operators and learned keys.

    New context allocates a slot by novelty against prior normalized keys;
    existing slots are never rewritten by a *different* recognized context.
    When full, unknown contexts are explicitly rejected, never silently
    treated as a familiar experience. All routing uses observed noisy cues.
    """
    def __init__(self, p: Protocol, base:np.ndarray):
        self.p=p
        self.base=base.copy()
        self.weights=np.zeros((p.slots,p.dim,p.dim))
        self.centers=np.zeros((p.slots,p.context_dim))
        self.counts=np.zeros(p.slots,dtype=int)
        self.active=0
        self.allocated=0
        self.denied=0
        self.routing=[]

    @property
    def scalar_budget(self):
        # Charged: learned operators, learned prototypes, and access counters.
        return self.weights.size+self.centers.size+self.counts.size

    def _route(self, context, write:bool):
        c=unit(context)
        sims=self.centers[:self.active]@c
        if len(sims):
            idx=int(np.argmax(sims))
            if sims[idx]>=self.p.allocation_cosine:
                return idx
        if write and self.active<self.p.slots:
            idx=self.active
            self.active+=1
            self.allocated+=1
            self.centers[idx]=c
            return idx
        if write:self.denied+=1
        return None

    def write(self, context, frames):
        idx=self._route(context,True)
        self.routing.append(idx)
        if idx is None:return None
        learn_ringdown(self.weights[idx],self.base,frames,self.p.learning_rate)
        self.counts[idx]+=1
        if self.counts[idx]>1:
            # A slow running average prevents one noisy context from moving a
            # learned prototype too far; no task IDs or explicit class labels.
            alpha=1/min(self.counts[idx],20)
            self.centers[idx]=unit((1-alpha)*self.centers[idx]+alpha*unit(context))
        return idx

    def retrieve(self,context):
        idx=self._route(context,False)
        if idx is None:return None
        return self.base+self.weights[idx]

    def awaken(self,context,probe,steps):
        """Hard reset; only a fresh probe and remembered operator persist."""
        matrix=self.retrieve(context)
        if matrix is None:return None
        return trace(matrix,probe,steps)


class DirectTransitionBank(StructuralBank):
    """Exact-budget ordinary context-routed direct transition regression.

    Matches every routing, learning-rate, matrix and prototype operation in
    StructuralBank, but uses observed next states directly instead of Sihti
    ringdown differences. This is the *decisive* residue-specific control.
    """
    def write(self, context, frames):
        idx=self._route(context,True)
        self.routing.append(idx)
        if idx is None:return None
        for x, x_next in zip(frames[:-1],frames[1:]):
            residual=x_next-self.base@x-self.weights[idx]@x
            self.weights[idx]+=self.p.learning_rate*np.outer(residual,x)/(float(x@x)+1e-5)
        self.counts[idx]+=1
        if self.counts[idx]>1:
            alpha=1/min(self.counts[idx],20)
            self.centers[idx]=unit((1-alpha)*self.centers[idx]+alpha*unit(context))
        return idx


class SharedOperator:
    """Ungated current-task operator, allocated equal scalar capacity.

    Unused matrix rows and prototype scalars remain charged. This deliberately
    isolates the routing/protection mechanism; it is not the strongest possible
    context-conditional alternative.
    """
    def __init__(self,p:Protocol,base:np.ndarray):
        self.p=p;self.base=base.copy()
        self.W=np.zeros((p.slots,p.dim,p.dim))
        self.prototypes=np.zeros((p.slots,p.context_dim))
        self.counters=np.zeros(p.slots)

    @property
    def scalar_budget(self):
        return self.W.size+self.prototypes.size+self.counters.size

    def write(self,context,frames):
        # Identical one shared operator updated for every experience.
        learn_ringdown(self.W[0],self.base,frames,self.p.learning_rate)

    def awaken(self,context,probe,steps):
        return trace(self.base+self.W[0],probe,steps)


class OracleIndexedOperators:
    """Privileged comparator: same matrices, but externally provided true slot."""
    def __init__(self,p:Protocol,base:np.ndarray):
        self.p=p;self.base=base.copy()
        self.W=np.zeros((p.task_count,p.dim,p.dim))
    def write(self,label,frames):
        learn_ringdown(self.W[label],self.base,frames,self.p.learning_rate)
    def awaken(self,label,probe,steps):
        return trace(self.base+self.W[label],probe,steps)


class FIFOPairs:
    """Explicit recently observed input/output pairs with matched budget."""
    def __init__(self,p:Protocol,budget:int,base:np.ndarray):
        self.p=p;self.base=base.copy()
        # Each record stores context, x and its next state, plus no metadata.
        self.capacity=budget//(p.context_dim+2*p.dim)
        self.pairs=deque(maxlen=self.capacity)
        self.n_scalar_records=self.capacity*(p.context_dim+2*p.dim)
    def write(self,context,frames):
        # A stream of local transitions, no uncharged full trace retention.
        for x,y in zip(frames[:-1],frames[1:]):
            self.pairs.append((unit(context).copy(),x.copy(),y.copy()))
    def awaken(self,context,probe,steps):
        # Context-weighted local ridge, with baseline as prior, estimated
        # from only the records that remain in the finite FIFO.
        if not self.pairs:return trace(self.base,probe,steps)
        c=unit(context)
        X=[];Y=[];weights=[]
        for key,x,y in self.pairs:
            X.append(x);Y.append(y-self.base@x)
            weights.append(float(np.exp(5*(key@c-1))))
        X=np.array(X);Y=np.array(Y);weights=np.asarray(weights)
        z=(X.T*weights)@X+.1*np.eye(self.p.dim)
        W=((Y.T*weights)@X)@np.linalg.inv(z)
        return trace(self.base+W,probe,steps)
