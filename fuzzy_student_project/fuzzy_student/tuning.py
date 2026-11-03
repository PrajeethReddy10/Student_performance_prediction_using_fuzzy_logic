"""Calibration of the membership-function centres (a light-weight alternative
to ANFIS-style learning).

Each input has one free parameter: the relative position of its "medium" peak.
The search maximises the Pearson correlation between FIS scores and the true
scores on the TRAINING split only; the test split is used afterwards to check
whether tuning actually generalises.
"""
from __future__ import annotations

import numpy as np

from .data import COLUMNS
from .evaluate import pearson
from .model import FISConfig, build_fis

BOUNDS = (0.30, 0.70)


def _objective(cfg: FISConfig, X: np.ndarray, y: np.ndarray) -> float:
    scores, _ = build_fis(cfg).predict_matrix(X, COLUMNS)
    return pearson(scores, y)


def tune_centers(X: np.ndarray, y: np.ndarray, base: FISConfig | None = None,
                 n_random: int = 40, refine_rounds: int = 3, seed: int = 0
                 ) -> tuple[FISConfig, list[float]]:
    """Random search followed by coordinate refinement. Returns (config, history)."""
    base = base or FISConfig()
    names = base.input_names()
    rng = np.random.default_rng(seed)

    best_c = {n: 0.5 for n in names}
    best = _objective(base.with_(centers=dict(best_c)), X, y)
    history = [best]

    for _ in range(n_random):                      # global exploration
        cand = {n: float(rng.uniform(*BOUNDS)) for n in names}
        val = _objective(base.with_(centers=cand), X, y)
        if val > best:
            best, best_c = val, cand
        history.append(best)

    step = 0.08
    for _ in range(refine_rounds):                 # local refinement
        for n in names:
            for delta in (-step, step):
                cand = dict(best_c)
                cand[n] = float(np.clip(cand[n] + delta, *BOUNDS))
                val = _objective(base.with_(centers=cand), X, y)
                if val > best:
                    best, best_c = val, cand
                history.append(best)
        step /= 2
    return base.with_(centers=best_c), history
