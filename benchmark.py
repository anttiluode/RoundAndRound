"""Controlled synthetic cue-to-residue retrieval; exact-state ridge is the fair control."""
import numpy as np
from roundandround import Config, FastLoop, SlowResidueMemory, residues


def run(seed: int = 7, episodes: int = 12, examples_per_episode: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    cfg = Config()
    loop = FastLoop(cfg)
    cues = rng.normal(size=(episodes, 24))
    cues /= np.linalg.norm(cues, axis=1, keepdims=True)
    patterns = rng.choice([-1., 1.], size=(episodes, 8, cfg.input_dim))
    signals = []
    keys = []
    ideal = []
    for episode in range(episodes):
        loop.reset()
        for u in patterns[episode]:
            loop.advance(0.35, u)
        ideal.append(loop.ringdown())
        for _ in range(examples_per_episode):
            loop.reset()
            for u in patterns[episode]:
                loop.advance(0.35, u + rng.normal(scale=0.025, size=cfg.input_dim))
            signals.append(loop.ringdown())
            keys.append(cues[episode] + rng.normal(scale=0.015, size=cues.shape[1]))
    keys, signals, ideal = np.array(keys), np.array(signals), np.array(ideal)
    memory = SlowResidueMemory(ridge=0.01)
    memory.fit(keys, signals)

    # Matched direct ridge: identical input/output dimensions and regularization.
    design = np.column_stack((keys, np.ones(len(keys))))
    reg = np.eye(design.shape[1]) * memory.ridge
    reg[-1, -1] = 0
    direct_weights = np.linalg.solve(design.T @ design + reg,
                                     design.T @ signals[:, 0, :])

    recalled, direct, scrambled, future, target = [], [], [], [], []
    for i in range(episodes):
        query = cues[i] + rng.normal(scale=0.015, size=cues.shape[1])
        loop.reset()  # hard erase of all fast activity
        recall = memory.awaken(query, loop)
        recalled.append(recall)
        direct.append(np.r_[query, 1] @ direct_weights)
        scrambled.append(memory.recall_bands(cues[(i+1) % episodes]).sum(axis=0))
        future.append(loop.advance(8.))
        target.append(ideal[i, 0])
    recalled, direct = np.array(recalled), np.array(direct)
    target = np.array(target)
    denom = np.mean(target ** 2)
    rmse = lambda x, t: float(np.sqrt(np.mean((np.asarray(x) - np.asarray(t))**2)))
    max_telescoping = max(float(np.max(np.abs(residues(t).sum(axis=0)-t[0]))) for t in signals)
    return {
        "reconstruction_identity_max_error": max_telescoping,
        "reactivated_rmse": rmse(recalled, target),
        "direct_state_ridge_rmse": rmse(direct, target),
        "wrong_cue_rmse": rmse(scrambled, target),
        "zero_fast_state_rmse": float(np.sqrt(denom)),
        "future_after_reactivation_rmse": rmse(future, ideal[:, 4, :]),
        "episodes": episodes,
        "examples_per_episode": examples_per_episode,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
