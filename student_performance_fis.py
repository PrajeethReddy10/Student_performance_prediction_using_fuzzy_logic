"""
Student Performance Prediction Using Fuzzy Logic
================================================

Implements the Fuzzy Inference System (FIS) described in the accompanying
slide deck, using the methodology of:

    Şenel A.A., Göksu B., Şenel E., Solmaz F.İ. "Predicting academic
    performance with fuzzy logic in prospective physical education and
    sports teachers." Scientific Reports 15, 28241 (2025).
    https://doi.org/10.1038/s41598-025-99124-3

Pipeline (slide 3 / article Fig. 1):
    crisp inputs -> fuzzification -> rule base -> inference
                 -> aggregation -> defuzzification -> crisp score + label

Method choices taken from the article
    * Triangular membership functions  f(x; a, b, c)
    * IF-THEN rules joined with AND, with the rule base covering every
      combination of linguistic terms (the article uses 3^4 = 81 rules)
    * Mamdani inference (min for AND / implication, max for aggregation)
    * Centroid defuzzification

Model content taken from the slides
    * Inputs : attendance (low/medium/high), marks (poor/average/good),
               study hours (low/medium/high)
    * Output : performance (poor/average/good/excellent)
    * The four explicit rules of slide 4
    * The illustrative case study of slide 5 (students A-D)

Only NumPy is required for the model. Matplotlib is optional (plots).

Usage
    python student_performance_fis.py                       # case study + checks
    python student_performance_fis.py --predict 85 72 6     # attendance marks hours
    python student_performance_fis.py --predict 85 72 6 --explain
    python student_performance_fis.py --plots               # save figures to ./output
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# 1. Fuzzy-set primitives
# --------------------------------------------------------------------------
def trimf(x: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function (article's definition).

        0                  x < a
        (x - a)/(b - a)    a <= x <= b
        (c - x)/(c - b)    b <= x <= c
        0                  x > c

    Degenerate triangles (a == b or b == c) give "shoulder" sets, which is
    how the open-ended low / high categories are represented.
    """
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


@dataclass
class LinguisticVariable:
    """A named variable on a universe of discourse with triangular terms."""

    name: str
    lo: float
    hi: float
    terms: dict[str, tuple[float, float, float]]
    unit: str = ""
    resolution: int = 1001
    universe: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.universe = np.linspace(self.lo, self.hi, self.resolution)

    def mu(self, term: str, x: np.ndarray | float) -> np.ndarray:
        return trimf(x, *self.terms[term])

    def fuzzify(self, x: float) -> dict[str, float]:
        """Crisp value -> degree of membership in every term."""
        x = float(np.clip(x, self.lo, self.hi))
        return {t: float(self.mu(t, x)) for t in self.terms}

    def best_term(self, x: float) -> str:
        degrees = self.fuzzify(x)
        return max(degrees, key=degrees.get)


@dataclass
class Rule:
    """IF v1 is t1 AND v2 is t2 ... THEN output is term."""

    antecedents: dict[str, str]
    consequent: str
    weight: float = 1.0
    source: str = "grid"  # "slide" for explicit rules, "grid" for generated

    def __str__(self) -> str:
        cond = " AND ".join(f"{v} is {t}" for v, t in self.antecedents.items())
        return f"IF {cond} THEN performance is {self.consequent}"


# --------------------------------------------------------------------------
# 2. Mamdani inference engine
# --------------------------------------------------------------------------
@dataclass
class InferenceResult:
    crisp_inputs: dict[str, float]
    fuzzified: dict[str, dict[str, float]]
    firing: list[tuple[Rule, float]]          # rules with strength > 0
    output_strength: dict[str, float]         # strongest firing per output term
    aggregated: np.ndarray
    score: float                              # defuzzified output
    label: str                                # category with highest membership


class MamdaniFIS:
    def __init__(
        self,
        inputs: dict[str, LinguisticVariable],
        output: LinguisticVariable,
        rules: list[Rule],
    ) -> None:
        self.inputs = inputs
        self.output = output
        self.rules = rules
        self._validate()

    def _validate(self) -> None:
        for r in self.rules:
            for v, t in r.antecedents.items():
                if v not in self.inputs or t not in self.inputs[v].terms:
                    raise ValueError(f"Unknown term in rule: {r}")
            if r.consequent not in self.output.terms:
                raise ValueError(f"Unknown output term in rule: {r}")

    def infer(self, **crisp: float) -> InferenceResult:
        missing = set(self.inputs) - set(crisp)
        if missing:
            raise ValueError(f"Missing inputs: {sorted(missing)}")

        # Step 1 - fuzzification
        fuzz = {n: v.fuzzify(crisp[n]) for n, v in self.inputs.items()}

        # Step 2 - rule evaluation (AND = min), Step 3 - implication (clip)
        out = self.output
        aggregated = np.zeros_like(out.universe)
        firing: list[tuple[Rule, float]] = []
        strongest = {t: 0.0 for t in out.terms}
        for rule in self.rules:
            strength = min(fuzz[v][t] for v, t in rule.antecedents.items())
            strength *= rule.weight
            if strength <= 0.0:
                continue
            firing.append((rule, strength))
            strongest[rule.consequent] = max(strongest[rule.consequent], strength)
            clipped = np.minimum(strength, out.mu(rule.consequent, out.universe))
            # Step 4 - aggregation (max)
            aggregated = np.maximum(aggregated, clipped)

        # Step 5 - centroid defuzzification
        area = aggregated.sum()
        score = float((out.universe * aggregated).sum() / area) if area > 0 \
            else (out.lo + out.hi) / 2
        label = out.best_term(score)

        firing.sort(key=lambda rs: -rs[1])
        return InferenceResult(
            {n: float(crisp[n]) for n in self.inputs}, fuzz, firing,
            strongest, aggregated, score, label,
        )

    def predict(self, **crisp: float) -> tuple[float, str]:
        r = self.infer(**crisp)
        return r.score, r.label


# --------------------------------------------------------------------------
# 3. The model from the slides
# --------------------------------------------------------------------------
def build_variables() -> tuple[dict[str, LinguisticVariable], LinguisticVariable]:
    """Membership functions (triangular, overlapping - slide 4).

    The slides give the category names but not the numeric breakpoints, so
    the breakpoints below are reasonable defaults. Edit them to calibrate.
    """
    attendance = LinguisticVariable(
        "attendance", 0, 100, unit="%",
        terms={"low": (0, 0, 50), "medium": (25, 50, 75), "high": (50, 100, 100)},
    )
    marks = LinguisticVariable(
        "marks", 0, 100, unit="/100",
        terms={"poor": (0, 0, 50), "average": (25, 50, 75), "good": (50, 100, 100)},
    )
    study = LinguisticVariable(
        "study_hours", 0, 8, unit="h/day",
        terms={"low": (0, 0, 4), "medium": (2, 4, 6), "high": (4, 8, 8)},
    )
    performance = LinguisticVariable(
        "performance", 0, 100, unit="score",
        terms={
            "poor": (0, 0, 35),
            "average": (20, 40, 60),
            "good": (45, 65, 85),
            "excellent": (70, 100, 100),
        },
    )
    return {"attendance": attendance, "marks": marks, "study_hours": study}, performance


# The four rules printed on slide 4. Partial antecedents are allowed: a rule
# that does not mention a variable ignores it.
SLIDE_RULES = [
    Rule({"attendance": "high", "marks": "good"}, "excellent", source="slide"),
    Rule({"attendance": "medium", "marks": "average"}, "average", source="slide"),
    Rule({"attendance": "low", "marks": "poor"}, "poor", source="slide"),
    Rule({"study_hours": "high", "marks": "good"}, "excellent", source="slide"),
]

# Level index of each term (0 = weakest, 2 = strongest) for rule generation.
LEVEL = {"low": 0, "medium": 1, "high": 2, "poor": 0, "average": 1, "good": 2}
# Marks matter most, then attendance, then study hours (assumption).
IMPORTANCE = {"marks": 0.5, "attendance": 0.3, "study_hours": 0.2}
# Composite score (0-1) -> output category.
OUTPUT_CUTS = [(0.30, "poor"), (0.60, "average"), (0.85, "good")]


def _grid_consequent(combo: dict[str, str]) -> str:
    s = sum(IMPORTANCE[v] * LEVEL[t] / 2 for v, t in combo.items())
    for cut, term in OUTPUT_CUTS:
        if s < cut:
            return term
    return "excellent"


def build_rules(inputs: dict[str, LinguisticVariable]) -> list[Rule]:
    """Slide rules + generated grid so every input combination is covered.

    Like the article (81 rules = every combination of 4 inputs x 3 terms), the
    rule base is complete. Here 3 inputs x 3 terms = 27 combinations. A
    combination already matched by an explicit slide rule is not generated
    again, so the slide's rules always take precedence.
    """
    names = list(inputs)
    rules = list(SLIDE_RULES)
    for terms in itertools.product(*(inputs[n].terms for n in names)):
        combo = dict(zip(names, terms))
        covered = any(
            all(combo[v] == t for v, t in r.antecedents.items()) for r in SLIDE_RULES
        )
        if not covered:
            rules.append(Rule(combo, _grid_consequent(combo)))
    return rules


def build_fis() -> MamdaniFIS:
    inputs, output = build_variables()
    return MamdaniFIS(inputs, output, build_rules(inputs))


# --------------------------------------------------------------------------
# 4. Case study (slide 5) - illustrative, NOT the paper's data
# --------------------------------------------------------------------------
# Crisp values chosen so each student falls in the slide's linguistic categories.
CASE_STUDY = [
    # name, attendance %, marks, study h/day, expected (slide 5)
    ("A", 92, 88, 7.0, "excellent"),   # High / Good / High
    ("B", 52, 52, 4.0, "average"),     # Medium / Average / Medium
    ("C", 25, 22, 1.0, "poor"),        # Low / Poor / Low
    ("D", 90, 55, 7.0, "good"),        # High / Average / High
]


def run_case_study(fis: MamdaniFIS) -> bool:
    print("Illustrative case study (slide 5) - not data from the paper\n")
    header = f"{'Student':<8}{'Attend.':>8}{'Marks':>7}{'Hours':>7}  {'Inputs as terms':<28}{'Score':>7}  {'Predicted':<10}{'Slide':<10}"
    print(header)
    print("-" * len(header))
    ok = True
    for name, att, mk, hrs, expected in CASE_STUDY:
        r = fis.infer(attendance=att, marks=mk, study_hours=hrs)
        terms = " / ".join(
            fis.inputs[v].best_term(x)
            for v, x in (("attendance", att), ("marks", mk), ("study_hours", hrs))
        )
        match = "OK" if r.label == expected else "MISMATCH"
        ok &= r.label == expected
        print(f"{name:<8}{att:>8}{mk:>7}{hrs:>7}  {terms:<28}{r.score:>7.1f}  "
              f"{r.label:<10}{expected:<10}{match}")
    print()
    return ok


# --------------------------------------------------------------------------
# 5. Explanation (the "explainable" selling point of slide 6)
# --------------------------------------------------------------------------
def explain(fis: MamdaniFIS, **crisp: float) -> None:
    r = fis.infer(**crisp)
    print("Crisp inputs:", ", ".join(f"{k}={v:g}" for k, v in r.crisp_inputs.items()))
    print("\nFuzzification:")
    for var, degs in r.fuzzified.items():
        txt = ", ".join(f"{t}={d:.2f}" for t, d in degs.items() if d > 0)
        print(f"  {var:<12} {txt}")
    print(f"\nRules fired: {len(r.firing)} of {len(fis.rules)}  (top 5)")
    for rule, s in r.firing[:5]:
        print(f"  [{s:.2f}] {rule}")
    print("\nStrongest support per performance category:")
    for t, s in r.output_strength.items():
        print(f"  {t:<10} {'#' * int(round(s * 30)):<30} {s:.2f}")
    print(f"\nDefuzzified score (centroid): {r.score:.1f} / 100")
    print(f"Predicted performance       : {r.label.upper()}")


# --------------------------------------------------------------------------
# 6. Plots (optional)
# --------------------------------------------------------------------------
def make_plots(fis: MamdaniFIS, outdir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Membership functions (article Fig. 2 style)
    variables = list(fis.inputs.values()) + [fis.output]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, var in zip(axes.ravel(), variables):
        for term in var.terms:
            ax.plot(var.universe, var.mu(term, var.universe), label=term, lw=2)
        ax.set_title(f"{var.name} ({var.unit})" if var.unit else var.name)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("membership")
        ax.legend(loc="center right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Triangular membership functions")
    fig.tight_layout()
    p = outdir / "membership_functions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # Control surfaces (article Figs. 4-5 style): performance vs two inputs
    def surface(xv, yv, fixed, fname):
        xs = np.linspace(fis.inputs[xv].lo, fis.inputs[xv].hi, 25)
        ys = np.linspace(fis.inputs[yv].lo, fis.inputs[yv].hi, 25)
        X, Y = np.meshgrid(xs, ys)
        Z = np.array([[fis.infer(**{xv: x, yv: y, **fixed}).score for x in xs] for y in ys])
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_xlabel(xv)
        ax.set_ylabel(yv)
        ax.set_zlabel("predicted performance")
        fixed_txt = ", ".join(f"{k}={v:g}" for k, v in fixed.items())
        ax.set_title(f"Performance surface ({fixed_txt})")
        fig.tight_layout()
        path = outdir / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    surface("attendance", "marks", {"study_hours": 4.0}, "surface_attendance_marks.png")
    surface("marks", "study_hours", {"attendance": 70.0}, "surface_marks_study.png")
    return saved


# --------------------------------------------------------------------------
# 7. CLI
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predict", nargs=3, type=float,
                    metavar=("ATTENDANCE", "MARKS", "STUDY_HOURS"),
                    help="predict for one student (attendance %%, marks /100, hours/day)")
    ap.add_argument("--explain", action="store_true", help="show fuzzification and fired rules")
    ap.add_argument("--rules", action="store_true", help="print the full rule base")
    ap.add_argument("--plots", action="store_true", help="save figures to ./output")
    args = ap.parse_args()

    fis = build_fis()
    n_slide = sum(r.source == "slide" for r in fis.rules)
    print(f"Fuzzy inference system: {len(fis.inputs)} inputs, {len(fis.rules)} rules "
          f"({n_slide} from slide 4 + {len(fis.rules) - n_slide} generated)\n")

    if args.rules:
        for i, r in enumerate(fis.rules, 1):
            print(f"{i:>3}. [{r.source}] {r}")
        print()

    if args.predict:
        att, mk, hrs = args.predict
        kw = dict(attendance=att, marks=mk, study_hours=hrs)
        if args.explain:
            explain(fis, **kw)
        else:
            score, label = fis.predict(**kw)
            print(f"Score {score:.1f}/100 -> {label}")
    else:
        ok = run_case_study(fis)
        print("Case study matches slide 5." if ok else "Case study does NOT match slide 5.")

    if args.plots:
        for p in make_plots(fis, Path("output")):
            print("saved", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
