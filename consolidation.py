"""Gate 2: heterogeneous autonomous loops, Sihti residues, and equal-budget online memory.

Synthetic systems, not biological neurons. All online weight-based methods store
exactly cue_dim * state_dim trainable scalars.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

TIMES = (0., 1., 2., 4., 8., 16.)

@dataclass(frozen=True)
class Protocol:
    dim: int = 16
    input_dim: int = 6
    cue_dim: int = 12
    ids_per_task: int = 10
    train_repeats: int = 6
    fast_tau: float = 2.
    slow_tau: float = 35.
    lr_fast: float = 0.38
    lr_slow: float = 0.12
    cue_noise_train: float = 0.035
    cue_noise_test: float = 0.04
    event_noise_train: float = 0.018
    seeds: tuple[int, ...] = (401, 402, 403, 404, 405, 406, 407, 408)
    times: tuple[float, ...] = TIMES

class MultiTimescaleLoop:
    """Continuous-time block-rotation FastLoop variant with multiple damping rates."""
    def __init__(self, p: Protocol = Protocol(), seed: int = 71):
        if p.dim % 4 or p.dim < 4:
            raise ValueError("dim must be a multiple of four")
        self.p = p
        rng = np.random.default_rng(seed)
        self.B = rng.normal(size=(p.dim,p.input_dim))/np.sqrt(p.input_dim)
        self.A = rng.normal(size=(p.dim,p.dim))/np.sqrt(p.dim)
        self.omega = np.linspace(0.2,1.4,p.dim//2)
        self.tau = np.array([p.fast_tau]*(p.dim//4)+[p.slow_tau]*(p.dim//4))
        self.state = np.zeros(p.dim)

    def reset(self):
        self.state[:] = 0.

    def autonomous(self, x: np.ndarray, dt: float) -> np.ndarray:
        if dt < 0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and nonnegative")
        x = np.asarray(x,dtype=float).reshape(-1,2)
        a = self.omega*dt
        c,s = np.cos(a),np.sin(a)
        gain = np.exp(-dt/self.tau)
        return (np.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),axis=-1)*gain[:,None]).reshape(-1)

    def inverse_slow(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x,dtype=float).reshape(-1,2)
        a = self.omega*dt
        c,s = np.cos(a),np.sin(a)
        gain = np.exp(dt/self.tau)
        return (np.stack((c*x[:,0]+s*x[:,1],-s*x[:,0]+c*x[:,1]),axis=-1)*gain[:,None]).reshape(-1)

    @property
    def slow_mask(self) -> np.ndarray:
        survival = np.exp(-self.p.times[-1]/self.tau)
        return np.repeat(survival>.25,2)

    def advance(self, dt: float, event: np.ndarray | None = None) -> np.ndarray:
        self.state = self.autonomous(self.state,dt)
        if event is not None:
            event = np.asarray(event)
            if event.shape != (self.p.input_dim,):
                raise ValueError("bad event shape")
            basal = self.B@event
            self.state += basal*(1+0.4*np.tanh(self.A@self.state))
        return self.state.copy()

    def ringdown(self) -> np.ndarray:
        return np.stack([self.autonomous(self.state,t) for t in self.p.times])

def sihti_encode(trace: np.ndarray, loop: MultiTimescaleLoop) -> np.ndarray:
    """16-scalar code: early residues for fast planes; final core for slow planes.

    Fast core is discarded; slow core is inverse-transported at readout.
    This is not storage of all (time,dim) residues.
    """
    trace = np.asarray(trace)
    if trace.shape != (len(loop.p.times),loop.p.dim):
        raise ValueError("bad trace shape")
    diffs = trace[:-1]-trace[1:]
    y = diffs.sum(axis=0)
    y[loop.slow_mask] = trace[-1,loop.slow_mask]
    return y

def sihti_decode(code: np.ndarray, loop: MultiTimescaleLoop) -> np.ndarray:
    if np.shape(code)!=(loop.p.dim,):
        raise ValueError("bad code shape")
    x = np.asarray(code).copy()
    backwards = loop.inverse_slow(x,loop.p.times[-1])
    x[loop.slow_mask] = backwards[loop.slow_mask]
    return x

class OnlineAssoc:
    """Fixed-size normalized LMS, optionally per-coordinate learning rates."""
    def __init__(self, cue_dim: int, dim: int, rates: float | np.ndarray):
        self.W = np.zeros((cue_dim,dim))
        r = np.broadcast_to(np.asarray(rates,dtype=float),(dim,)).copy()
        if np.any((r<=0)|(r>1)):
            raise ValueError("rates must lie in (0,1]")
        self.rates=r
        self.n_writes=0

    @property
    def n_scalars(self):
        return self.W.size

    def write(self,cue:np.ndarray,value:np.ndarray):
        cue,value=np.asarray(cue),np.asarray(value)
        error=value-cue@self.W
        self.W+=np.outer(cue,self.rates*error)/(float(cue@cue)+1e-12)
        self.n_writes+=1

    def recall(self,cue:np.ndarray)->np.ndarray:
        return np.asarray(cue)@self.W

class KeyValueFIFO:
    """Recent-only episodic key/value memory with counted scalar capacity."""
    def __init__(self,cue_dim:int,dim:int,scalar_budget:int):
        self.capacity=scalar_budget//(cue_dim+dim)
        self.items:list[tuple[np.ndarray,np.ndarray]]=[]
        if self.capacity<1:
            raise ValueError("budget too small")

    @property
    def n_scalars(self):
        return self.capacity*(len(self.items[0][0])+len(self.items[0][1])) if self.items else 0

    def write(self,cue:np.ndarray,state:np.ndarray):
        self.items.append((np.asarray(cue).copy(),np.asarray(state).copy()))
        if len(self.items)>self.capacity:
            self.items.pop(0)

    def recall(self,cue:np.ndarray)->np.ndarray:
        if not self.items:
            raise ValueError("empty")
        j=int(np.argmax([float(np.asarray(cue)@k) for k,_ in self.items]))
        return self.items[j][1].copy()
