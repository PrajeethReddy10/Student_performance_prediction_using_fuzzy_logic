"""Metrics and baselines (NumPy only)."""
from __future__ import annotations

import numpy as np

from .data import CATEGORIES


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_ranks(a), _ranks(b))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 4) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def quadratic_kappa(cm: np.ndarray) -> float:
    k = cm.shape[0]
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    denom = (w * expected).sum()
    return float(1.0 - (w * cm).sum() / denom) if denom else 0.0


def macro_f1(cm: np.ndarray) -> float:
    f1 = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        p = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        r = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1))


def category_index(labels: list[str]) -> np.ndarray:
    return np.array([CATEGORIES.index(l) for l in labels])


def affine_fit(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares y ~ a + b*pred (puts FIS scores on the target scale)."""
    A = np.column_stack([np.ones_like(pred), pred])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    r = pearson(pred, y)
    return {"r": r, "r2": r * r, "rho": spearman(pred, y),
            "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}


def classification_metrics(y_cat: np.ndarray, pred_cat: np.ndarray) -> dict[str, float]:
    cm = confusion(y_cat, pred_cat)
    return {"accuracy": float(np.trace(cm) / cm.sum()),
            "within_one": float((np.abs(y_cat - pred_cat) <= 1).mean()),
            "kappa_qw": quadratic_kappa(cm), "macro_f1": macro_f1(cm)}


class LinearRegression:
    """Ordinary least squares with intercept - the baseline (the article also
    reports a regression analysis)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        A = np.column_stack([np.ones(len(X)), X])
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X]) @ self.coef_


def evaluate_fis(fis, X_train, y_train, X_test, y_test, cat_test, columns) -> dict[str, float]:
    """Raw metrics + metrics after an affine fit learned on the training set.

    FIS outputs live on their own 0-100 scale (centroids never reach the
    extremes), so raw MAE/RMSE are pessimistic. The calibrated figures put the
    FIS and the regression baseline on equal footing.
    """
    s_tr, _ = fis.predict_matrix(X_train, columns)
    s_te, lab_te = fis.predict_matrix(X_test, columns)
    out = {f"raw_{k}": v for k, v in regression_metrics(y_test, s_te).items()}
    a, b = affine_fit(s_tr, y_train)
    cal = regression_metrics(y_test, a + b * s_te)
    out.update({"cal_mae": cal["mae"], "cal_rmse": cal["rmse"]})
    out.update(classification_metrics(cat_test, category_index(lab_te)))
    return out
