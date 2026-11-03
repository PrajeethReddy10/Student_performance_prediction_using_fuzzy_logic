"""Synthetic student dataset and CSV helpers.

No real student data is available for this case study, so a synthetic cohort
is generated. Two latent traits drive the observed inputs: academic ability
and diligence (moderately correlated). The "true" performance score is a
NON-LINEAR function of the inputs (diminishing returns on study hours) plus
noise, so it is not simply the weighted rule grid the FIS was built from.

Results on this data demonstrate the pipeline; they are not evidence of
accuracy on real students.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

COLUMNS = ["attendance", "marks", "assignments", "study_hours"]
CATEGORIES = ["poor", "average", "good", "excellent"]
TRUE_CUTS = (25.0, 50.0, 75.0)       # score -> category for the ground truth


def true_score(X: np.ndarray) -> np.ndarray:
    att, marks, asg, study = X.T
    study_term = 100.0 * (1 - np.exp(-study / 2.5)) / (1 - np.exp(-8 / 2.5))
    return 0.40 * marks + 0.25 * asg + 0.20 * att + 0.15 * study_term


def to_category(score: np.ndarray) -> np.ndarray:
    return np.digitize(score, TRUE_CUTS)          # 0..3 -> CATEGORIES index


def generate(n: int = 300, seed: int = 42, noise: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (n x 4, COLUMNS order), true score y, true category index."""
    rng = np.random.default_rng(seed)
    ability = rng.standard_normal(n)
    diligence = 0.4 * ability + np.sqrt(1 - 0.4 ** 2) * rng.standard_normal(n)
    marks = 50 + 22 * ability + 5 * rng.standard_normal(n)
    attendance = 64 + 20 * diligence + 5 * rng.standard_normal(n)
    assignments = 52 + 18 * (0.5 * ability + 0.5 * diligence) + 6 * rng.standard_normal(n)
    study = 3.2 + 2.0 * diligence + 0.7 * ability + 0.5 * rng.standard_normal(n)
    X = np.column_stack([
        np.clip(attendance, 0, 100), np.clip(marks, 0, 100),
        np.clip(assignments, 0, 100), np.clip(study, 0, 8)])
    y = np.clip(true_score(X) + noise * rng.standard_normal(n), 0, 100)
    return X, y, to_category(y)


def train_test_split(n: int, test_frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    k = int(round(n * (1 - test_frac)))
    return idx[:k], idx[k:]


def read_csv(path: str | Path) -> tuple[list[str], np.ndarray, list[dict]]:
    """Read student rows. Returns (columns found, matrix, raw rows)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("CSV has no data rows")
    cols = [c for c in COLUMNS if c in rows[0]]
    if not {"attendance", "marks", "study_hours"} <= set(cols):
        raise ValueError("CSV needs at least: attendance, marks, study_hours "
                         "(and optionally assignments)")
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    return cols, X, rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
