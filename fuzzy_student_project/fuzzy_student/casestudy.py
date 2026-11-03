"""Illustrative case study of slide 5 (students A-D) - not data from the paper."""
from __future__ import annotations

from .engine import MamdaniFIS

# name, attendance %, marks, assignments, study h/day, category on slide 5.
# Slide 5 gives linguistic levels only; the crisp numbers (and the assignment
# scores, which slide 5 omits) are assumptions chosen to fall in those levels.
CASE_STUDY = [
    ("A", 92, 88, 90, 7.0, "excellent"),   # High / Good / High
    ("B", 52, 52, 50, 4.0, "average"),     # Medium / Average / Medium
    ("C", 25, 22, 20, 1.0, "poor"),        # Low / Poor / Low
    ("D", 90, 55, 60, 7.0, "good"),        # High / Average / High
]


def _inputs(fis: MamdaniFIS, att, mk, asg, hrs) -> dict[str, float]:
    full = {"attendance": att, "marks": mk, "assignments": asg, "study_hours": hrs}
    return {k: v for k, v in full.items() if k in fis.inputs}


def run_case_study(fis: MamdaniFIS, verbose: bool = True) -> list[dict]:
    rows = []
    for name, att, mk, asg, hrs, expected in CASE_STUDY:
        r = fis.infer(**_inputs(fis, att, mk, asg, hrs))
        terms = " / ".join(fis.inputs[v].best_term(x) for v, x in r.crisp_inputs.items())
        rows.append(dict(student=name, inputs=r.crisp_inputs, terms=terms,
                         score=round(r.score, 1), label=r.label, expected=expected,
                         match=r.label == expected, fired=len(r.firing)))
    if verbose:
        print("Illustrative case study (slide 5) - not data from the paper\n")
        hdr = f"{'Student':<8}{'Inputs':<26}{'Linguistic levels':<34}{'Score':>6}  {'Predicted':<10}{'Slide':<10}"
        print(hdr + "\n" + "-" * len(hdr))
        for x in rows:
            vals = "/".join(f"{v:g}" for v in x["inputs"].values())
            print(f"{x['student']:<8}{vals:<26}{x['terms']:<34}{x['score']:>6.1f}  "
                  f"{x['label']:<10}{x['expected']:<10}{'OK' if x['match'] else 'MISMATCH'}")
        print()
    return rows


def explain(fis: MamdaniFIS, **crisp: float) -> None:
    """Step-by-step trace of one inference (the 'explainable' point of slide 6)."""
    r = fis.infer(**crisp)
    print("Crisp inputs:", ", ".join(f"{k}={v:g}" for k, v in r.crisp_inputs.items()))
    print("\n1. Fuzzification")
    for var, degs in r.fuzzified.items():
        print(f"   {var:<12}", ", ".join(f"{t}={d:.2f}" for t, d in degs.items() if d > 0))
    print(f"\n2. Rule evaluation - {len(r.firing)} of {len(fis.rules)} rules fired (top 5)")
    for rule, s in r.firing[:5]:
        tag = f" [{rule.note}]" if rule.note else ""
        print(f"   [{s:.2f}] {rule}{tag}")
    print("\n3. Strongest support per output category")
    for t, s in r.output_strength.items():
        print(f"   {t:<10}{'#' * int(round(s * 30)):<30} {s:.2f}")
    print(f"\n4. Defuzzification ({fis.defuzz}): {r.score:.1f} / 100")
    print(f"   Predicted performance: {r.label.upper()}")
