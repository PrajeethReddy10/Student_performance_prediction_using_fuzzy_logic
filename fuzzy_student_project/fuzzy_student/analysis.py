"""Model analysis: sensitivity, what-if advice, rule usage, robustness, ablation."""
from __future__ import annotations

import numpy as np

from .data import COLUMNS
from .engine import MamdaniFIS
from .evaluate import evaluate_fis
from .model import FISConfig, build_fis


# ----------------------------------------------------------- sensitivity
def sweep(fis: MamdaniFIS, base: dict[str, float], var: str, n: int = 81) -> tuple[np.ndarray, np.ndarray]:
    """Score as one input moves across its range, others held at `base`."""
    v = fis.inputs[var]
    xs = np.linspace(v.lo, v.hi, n)
    return xs, np.array([fis.infer(**{**base, var: x}).score for x in xs])


def sensitivity_table(fis: MamdaniFIS, base: dict[str, float]) -> list[dict]:
    """One-at-a-time effect: score range produced by each input alone."""
    rows = []
    for var in fis.inputs:
        _, ys = sweep(fis, base, var)
        rows.append({"input": var, "min": float(ys.min()), "max": float(ys.max()),
                     "range": float(ys.max() - ys.min())})
    return sorted(rows, key=lambda r: -r["range"])


# ----------------------------------------------------------- what-if advice
def advise(fis: MamdaniFIS, student: dict[str, float], step: float = 0.15) -> list[dict]:
    """Which single improvement helps most?

    For each input, raise it by `step` of its range (hours: same fraction) and
    report the resulting score gain, ranked. A counterfactual, not a causal claim.
    A negative gain means the model is non-monotone at that point (see
    monotonicity_audit); the advice for that input should be ignored.
    """
    base = fis.infer(**student)
    out = []
    for var, v in fis.inputs.items():
        new = min(v.hi, student[var] + step * (v.hi - v.lo))
        if new == student[var]:
            out.append({"input": var, "from": student[var], "to": new, "gain": 0.0, "label": base.label})
            continue
        r = fis.infer(**{**student, var: new})
        out.append({"input": var, "from": student[var], "to": new,
                    "gain": r.score - base.score, "label": r.label})
    return sorted(out, key=lambda r: -r["gain"])


def monotonicity_audit(fis: MamdaniFIS, n_points: int = 150, tol: float = 1.0, seed: int = 0) -> dict:
    """Is a higher input ever predicted to LOWER the score?

    Mamdani + centroid is not guaranteed monotone: when a stronger input
    activates a better category it can also add area that pulls the centroid
    down. Samples random base points, sweeps each input, counts drops > tol.
    """
    rng = np.random.default_rng(seed)
    per_var = {}
    for var, v in fis.inputs.items():
        viol, worst = 0, 0.0
        for _ in range(n_points):
            base = {n: float(rng.uniform(w.lo, w.hi)) for n, w in fis.inputs.items()}
            _, ys = sweep(fis, base, var, n=41)
            drop = float(np.max(np.maximum.accumulate(ys) - ys))
            viol += drop > tol
            worst = max(worst, drop)
        per_var[var] = {"violating_sweeps": viol / n_points, "worst_drop": worst}
    return per_var


# ----------------------------------------------------------- rule usage
def rule_usage(fis: MamdaniFIS, X: np.ndarray, columns: list[str]) -> list[dict]:
    """How often each rule fires, and how often it is the strongest rule."""
    idx = {c: i for i, c in enumerate(columns)}
    fired = np.zeros(len(fis.rules))
    top = np.zeros(len(fis.rules))
    pos = {id(r): i for i, r in enumerate(fis.rules)}
    for row in X:
        res = fis.infer(**{n: row[idx[n]] for n in fis.inputs})
        for rule, _ in res.firing:
            fired[pos[id(rule)]] += 1
        top[pos[id(res.firing[0][0])]] += 1
    return [{"rule": str(r), "source": r.source, "fired": int(f), "top": int(t)}
            for r, f, t in zip(fis.rules, fired, top)]


# ----------------------------------------------------------- robustness
def robustness(fis: MamdaniFIS, X: np.ndarray, columns: list[str],
               sigmas=(0.02, 0.05, 0.10), trials: int = 10, seed: int = 0) -> list[dict]:
    """Add Gaussian noise (sigma = fraction of each input's range) to every
    input and measure how much the score moves and how often the label flips."""
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(columns)}
    cols = [idx[n] for n in fis.inputs]
    lo = np.array([fis.inputs[n].lo for n in fis.inputs])
    hi = np.array([fis.inputs[n].hi for n in fis.inputs])
    base_s, base_l = fis.predict_matrix(X, columns)
    out = []
    for sg in sigmas:
        shifts, flips = [], []
        for _ in range(trials):
            Xn = X.copy()
            noise = rng.standard_normal((len(X), len(cols))) * sg * (hi - lo)
            Xn[:, cols] = np.clip(X[:, cols] + noise, lo, hi)
            s, l = fis.predict_matrix(Xn, columns)
            shifts.append(np.abs(s - base_s).mean())
            flips.append(np.mean([a != b for a, b in zip(l, base_l)]))
        out.append({"sigma": sg, "mean_abs_shift": float(np.mean(shifts)),
                    "label_flip_rate": float(np.mean(flips))})
    return out


# ----------------------------------------------------------- ablation
def default_ablation_grid() -> list[tuple[str, FISConfig]]:
    b = FISConfig()
    return [
        ("Baseline (tri, min, centroid, 4 inputs)", b),
        ("3 inputs (no assignments)", b.with_(use_assignments=False)),
        ("Trapezoidal MFs", b.with_(shape="trap")),
        ("Gaussian MFs", b.with_(shape="gauss")),
        ("AND = product", b.with_(and_op="prod")),
        ("Implication = product", b.with_(implication="prod")),
        ("Defuzz: bisector", b.with_(defuzz="bisector")),
        ("Defuzz: mean of maxima", b.with_(defuzz="mom")),
        ("Rules: weighted grid only", b.with_(rule_mode="weighted")),
    ]


def compare_configs(grid, X_tr, y_tr, X_te, y_te, c_te) -> list[dict]:
    rows = []
    for name, cfg in grid:
        m = evaluate_fis(build_fis(cfg), X_tr, y_tr, X_te, y_te, c_te, COLUMNS)
        rows.append({"name": name, "n_rules": len(build_fis(cfg).rules), **m})
    return rows
