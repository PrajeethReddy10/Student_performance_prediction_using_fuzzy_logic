"""The student-performance FIS described in the slides.

Inputs  (slide 3): attendance, marks, assignment performance, study hours
Output  (slide 4): performance = poor / average / good / excellent
Rules   (slide 4): four explicit rules + a complete rule grid, in the spirit of
                   the article's 3^4 = 81 rule base.

The slides name the categories but not the numeric breakpoints or the rule
consequents of the full grid, so those are documented assumptions below.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

from .engine import MamdaniFIS, Rule
from .membership import LinguisticVariable, from_triangle

ALL_INPUTS = ("attendance", "marks", "assignments", "study_hours")

# name -> (lo, hi, unit, (term_low, term_mid, term_high))
INPUT_SPECS = {
    "attendance":  (0.0, 100.0, "%",     ("low", "medium", "high")),
    "marks":       (0.0, 100.0, "/100",  ("poor", "average", "good")),
    "assignments": (0.0, 100.0, "/100",  ("poor", "average", "good")),
    "study_hours": (0.0, 8.0,   "h/day", ("low", "medium", "high")),
}
LEVEL = {"low": 0, "medium": 1, "high": 2, "poor": 0, "average": 1, "good": 2}

# Assumption: relative importance of each factor when generating the rule grid.
IMPORTANCE = {"marks": 0.40, "assignments": 0.25, "attendance": 0.20, "study_hours": 0.15}
# Assumption: composite score (0-1) -> output category.
OUTPUT_CUTS = [(0.30, "poor"), (0.60, "average"), (0.85, "good")]

# Output variable: triangle breakpoints (a, b, c), 0-100 score.
OUTPUT_TRIANGLES = {
    "poor": (0, 0, 35), "average": (20, 40, 60),
    "good": (45, 65, 85), "excellent": (70, 100, 100),
}

# The four rules printed on slide 4 (partial antecedents are allowed).
SLIDE_RULES = [
    Rule({"attendance": "high", "marks": "good"}, "excellent", source="slide", note="slide 4 rule 1"),
    Rule({"attendance": "medium", "marks": "average"}, "average", source="slide", note="slide 4 rule 2"),
    Rule({"attendance": "low", "marks": "poor"}, "poor", source="slide", note="slide 4 rule 3"),
    Rule({"study_hours": "high", "marks": "good"}, "excellent", source="slide", note="slide 4 rule 4"),
]


@dataclass(frozen=True)
class FISConfig:
    """Everything that defines a variant of the model (used for ablations)."""

    shape: str = "tri"                  # tri | trap | gauss
    use_assignments: bool = True        # False -> the 3-input model of slides 4-5
    rule_mode: str = "slides_first"     # slides_first | weighted
    and_op: str = "min"                 # min | prod
    implication: str = "min"            # min | prod
    defuzz: str = "centroid"            # centroid | bisector | mom | som | lom
    # Relative position (0-1) of each input's "medium" peak. 0.5 is the default.
    centers: dict[str, float] = field(default_factory=dict)

    def input_names(self) -> tuple[str, ...]:
        return ALL_INPUTS if self.use_assignments else tuple(
            n for n in ALL_INPUTS if n != "assignments")

    def with_(self, **kw) -> "FISConfig":
        return replace(self, **kw)

    def label(self) -> str:
        parts = [self.shape, "4in" if self.use_assignments else "3in", self.rule_mode,
                 f"and={self.and_op}", f"impl={self.implication}", self.defuzz]
        return "/".join(parts)


def _triangles(lo: float, hi: float, c_rel: float) -> list[tuple[float, float, float]]:
    """Low / medium / high triangles from the relative position of the medium peak.

    c_rel = 0.5 on a 0-100 scale gives (0,0,50), (25,50,75), (50,100,100).
    """
    c = lo + c_rel * (hi - lo)
    return [(lo, lo, c), (lo + (c - lo) / 2, c, c + (hi - c) / 2), (c, hi, hi)]


def build_variables(cfg: FISConfig) -> tuple[dict[str, LinguisticVariable], LinguisticVariable]:
    inputs: dict[str, LinguisticVariable] = {}
    for name in cfg.input_names():
        lo, hi, unit, terms = INPUT_SPECS[name]
        tris = _triangles(lo, hi, cfg.centers.get(name, 0.5))
        inputs[name] = LinguisticVariable(
            name, lo, hi, unit=unit,
            terms={t: from_triangle(tr, cfg.shape) for t, tr in zip(terms, tris)})
    output = LinguisticVariable(
        "performance", 0, 100, unit="score",
        terms={t: from_triangle(tr, cfg.shape) for t, tr in OUTPUT_TRIANGLES.items()})
    return inputs, output


def _grid_consequent(combo: dict[str, str]) -> str:
    w = sum(IMPORTANCE[v] for v in combo)
    s = sum(IMPORTANCE[v] * LEVEL[t] / 2 for v, t in combo.items()) / w
    for cut, term in OUTPUT_CUTS:
        if s < cut:
            return term
    return "excellent"


def build_rules(inputs: dict[str, LinguisticVariable], rule_mode: str = "slides_first") -> list[Rule]:
    """Complete rule base: one rule for every combination of input terms.

    slides_first : a combination covered by one of the slide-4 rules takes that
                   rule's consequent; every other combination uses the weighted
                   composite score.
    weighted     : ignore the slide rules, use the weighted composite for all.
    """
    if rule_mode not in ("slides_first", "weighted"):
        raise ValueError("rule_mode must be 'slides_first' or 'weighted'")
    names = list(inputs)
    rules: list[Rule] = []
    for terms in itertools.product(*(inputs[n].terms for n in names)):
        combo = dict(zip(names, terms))
        slide = None
        if rule_mode == "slides_first":
            slide = next((r for r in SLIDE_RULES
                          if all(v in combo and combo[v] == t
                                 for v, t in r.antecedents.items())), None)
        if slide:
            rules.append(Rule(combo, slide.consequent, source="slide", note=slide.note))
        else:
            rules.append(Rule(combo, _grid_consequent(combo)))
    return rules


def build_fis(cfg: FISConfig | None = None) -> MamdaniFIS:
    cfg = cfg or FISConfig()
    inputs, output = build_variables(cfg)
    return MamdaniFIS(inputs, output, build_rules(inputs, cfg.rule_mode),
                      and_op=cfg.and_op, implication=cfg.implication, defuzz=cfg.defuzz)
