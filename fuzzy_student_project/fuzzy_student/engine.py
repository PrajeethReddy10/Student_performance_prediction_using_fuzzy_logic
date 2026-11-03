"""Mamdani fuzzy inference engine.

Pipeline: fuzzification -> rule evaluation (AND) -> implication
          -> aggregation (max) -> defuzzification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .membership import LinguisticVariable

DEFUZZ_METHODS = ("centroid", "bisector", "mom", "som", "lom")


def defuzzify(x: np.ndarray, mu: np.ndarray, method: str = "centroid") -> float:
    """Crisp value from an aggregated output set sampled on x."""
    area = mu.sum()
    if area <= 0.0:
        return float((x[0] + x[-1]) / 2)
    if method == "centroid":
        return float((x * mu).sum() / area)
    if method == "bisector":
        cs = np.cumsum(mu)
        return float(x[int(np.searchsorted(cs, area / 2.0))])
    if method in ("mom", "som", "lom"):
        xs = x[mu >= mu.max() - 1e-12]
        return float({"mom": xs.mean(), "som": xs.min(), "lom": xs.max()}[method])
    raise ValueError(f"Unknown defuzzification method: {method}")


@dataclass
class Rule:
    """IF v1 is t1 AND v2 is t2 ... THEN output is `consequent`."""

    antecedents: dict[str, str]
    consequent: str
    weight: float = 1.0
    source: str = "grid"      # "slide" (explicit rule of slide 4) or "grid"
    note: str = ""

    def __str__(self) -> str:
        cond = " AND ".join(f"{v} is {t}" for v, t in self.antecedents.items())
        return f"IF {cond} THEN performance is {self.consequent}"


@dataclass
class InferenceResult:
    crisp_inputs: dict[str, float]
    fuzzified: dict[str, dict[str, float]]
    firing: list[tuple[Rule, float]]       # rules with strength > 0, strongest first
    output_strength: dict[str, float]      # strongest firing per output term
    aggregated: np.ndarray
    score: float
    label: str


class MamdaniFIS:
    def __init__(self, inputs: dict[str, LinguisticVariable],
                 output: LinguisticVariable, rules: list[Rule],
                 and_op: str = "min", implication: str = "min",
                 defuzz: str = "centroid") -> None:
        if and_op not in ("min", "prod"):
            raise ValueError("and_op must be 'min' or 'prod'")
        if implication not in ("min", "prod"):
            raise ValueError("implication must be 'min' or 'prod'")
        if defuzz not in DEFUZZ_METHODS:
            raise ValueError(f"defuzz must be one of {DEFUZZ_METHODS}")
        self.inputs, self.output, self.rules = inputs, output, rules
        self.and_op, self.implication, self.defuzz = and_op, implication, defuzz
        self._validate()

    def _validate(self) -> None:
        for r in self.rules:
            for v, t in r.antecedents.items():
                if v not in self.inputs or t not in self.inputs[v].terms:
                    raise ValueError(f"Unknown input term in rule: {r}")
            if r.consequent not in self.output.terms:
                raise ValueError(f"Unknown output term in rule: {r}")

    # ------------------------------------------------------------- inference
    def infer(self, **crisp: float) -> InferenceResult:
        missing = set(self.inputs) - set(crisp)
        if missing:
            raise ValueError(f"Missing inputs: {sorted(missing)}")
        fuzz = {n: v.fuzzify(crisp[n]) for n, v in self.inputs.items()}

        firing: list[tuple[Rule, float]] = []
        strongest = {t: 0.0 for t in self.output.terms}
        use_min = self.and_op == "min"
        for rule in self.rules:
            vals = [fuzz[v][t] for v, t in rule.antecedents.items()]
            s = (min(vals) if use_min else math.prod(vals)) * rule.weight
            if s <= 0.0:
                continue
            firing.append((rule, s))
            if s > strongest[rule.consequent]:
                strongest[rule.consequent] = s

        # With max-aggregation, rules sharing a consequent only matter through
        # their strongest firing, so the aggregate is built per output term.
        agg = np.zeros_like(self.output.universe)
        for term, s in strongest.items():
            if s > 0.0:
                curve = self.output.curve(term)
                implied = np.minimum(s, curve) if self.implication == "min" else s * curve
                np.maximum(agg, implied, out=agg)

        score = defuzzify(self.output.universe, agg, self.defuzz)
        firing.sort(key=lambda rs: -rs[1])
        return InferenceResult({n: float(crisp[n]) for n in self.inputs}, fuzz,
                               firing, strongest, agg, score,
                               self.output.best_term(score))

    def predict(self, **crisp: float) -> tuple[float, str]:
        r = self.infer(**crisp)
        return r.score, r.label

    def predict_matrix(self, X: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[str]]:
        """Score every row of X. `columns` names the columns of X."""
        idx = {c: i for i, c in enumerate(columns)}
        need = list(self.inputs)
        scores = np.empty(len(X))
        labels: list[str] = []
        for i, row in enumerate(X):
            r = self.infer(**{n: row[idx[n]] for n in need})
            scores[i] = r.score
            labels.append(r.label)
        return scores, labels
