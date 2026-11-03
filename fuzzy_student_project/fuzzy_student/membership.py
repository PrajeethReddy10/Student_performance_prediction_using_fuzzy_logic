"""Membership functions and linguistic variables.

The source article uses triangular membership functions. Trapezoidal and
Gaussian shapes are also provided so the effect of the shape choice can be
studied (see analysis.compare_configs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- array forms
def trimf(x, a: float, b: float, c: float) -> np.ndarray:
    """Triangular MF. a == b or b == c gives a left / right shoulder."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        m = (x > a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    if c > b:
        m = (x > b) & (x < c)
        y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    return y


def trapmf(x, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal MF: rises a->b, equals 1 on [b, c], falls c->d."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        m = (x > a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    if d > c:
        m = (x > c) & (x < d)
        y[m] = (d - x[m]) / (d - c)
    y[(x >= b) & (x <= c)] = 1.0
    return y


def gaussmf(x, m: float, s: float, side: str = "") -> np.ndarray:
    """Gaussian MF. side='L' keeps 1 left of m, side='R' keeps 1 right of m."""
    x = np.asarray(x, dtype=float)
    y = np.exp(-0.5 * ((x - m) / s) ** 2)
    if side == "L":
        y = np.where(x <= m, 1.0, y)
    elif side == "R":
        y = np.where(x >= m, 1.0, y)
    return y


# ---------------------------------------------------------------- scalar forms
def _tri_s(x, a, b, c):
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    return 0.0


def _trap_s(x, a, b, c, d):
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0


def _gauss_s(x, m, s, side=""):
    if (side == "L" and x <= m) or (side == "R" and x >= m):
        return 1.0
    return math.exp(-0.5 * ((x - m) / s) ** 2)


@dataclass(frozen=True)
class MF:
    kind: str                 # "tri" | "trap" | "gauss"
    params: tuple
    side: str = ""            # gauss only

    def __call__(self, x) -> np.ndarray:
        if self.kind == "tri":
            return trimf(x, *self.params)
        if self.kind == "trap":
            return trapmf(x, *self.params)
        return gaussmf(x, *self.params, side=self.side)

    def scalar(self, x: float) -> float:
        if self.kind == "tri":
            return _tri_s(x, *self.params)
        if self.kind == "trap":
            return _trap_s(x, *self.params)
        return _gauss_s(x, *self.params, side=self.side)


def from_triangle(tri: tuple[float, float, float], shape: str = "tri") -> MF:
    """Build an MF of the requested shape from triangle breakpoints (a, b, c).

    The converted shapes keep the same support and peak location. Note: a
    trapezoid made by flattening the top of a triangle is a rescaled triangle
    (alpha-cuts coincide until a degree saturates at 1), so under min/clip
    inference it gives almost identical results to the triangular sets. That
    equivalence is expected, not a bug (see the ablation study).
    """
    a, b, c = tri
    if shape == "tri":
        return MF("tri", (a, b, c))
    if shape == "trap":
        dl = 0.0 if a == b else 0.25 * (b - a)
        dr = 0.0 if c == b else 0.25 * (c - b)
        return MF("trap", (a, b - dl, b + dr, c))
    if shape == "gauss":
        if a == b:                      # left shoulder
            return MF("gauss", (b, (c - b) / 2.5), "L")
        if b == c:                      # right shoulder
            return MF("gauss", (b, (b - a) / 2.5), "R")
        return MF("gauss", (b, (c - a) / 4.0))
    raise ValueError(f"Unknown MF shape: {shape}")


@dataclass
class LinguisticVariable:
    """A named variable on [lo, hi] with a set of fuzzy terms."""

    name: str
    lo: float
    hi: float
    terms: dict[str, MF]
    unit: str = ""
    resolution: int = 1001
    universe: np.ndarray = field(init=False, repr=False)
    _curves: dict[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.universe = np.linspace(self.lo, self.hi, self.resolution)
        self._curves = {t: mf(self.universe) for t, mf in self.terms.items()}

    def curve(self, term: str) -> np.ndarray:
        """Pre-computed membership of `term` over the whole universe."""
        return self._curves[term]

    def mu(self, term: str, x) -> np.ndarray:
        return self.terms[term](x)

    def fuzzify(self, x: float) -> dict[str, float]:
        x = min(max(float(x), self.lo), self.hi)
        return {t: mf.scalar(x) for t, mf in self.terms.items()}

    def best_term(self, x: float) -> str:
        d = self.fuzzify(x)
        return max(d, key=d.get)
