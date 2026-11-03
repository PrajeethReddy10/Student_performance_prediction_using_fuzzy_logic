"""Student performance prediction using fuzzy logic - command line interface.

Examples
    python main.py casestudy
    python main.py predict --attendance 85 --marks 72 --assignments 65 --hours 6 --explain --advise
    python main.py batch students.csv predictions.csv
    python main.py rules
    python main.py evaluate
    python main.py tune
    python main.py ablation
    python main.py sensitivity
    python main.py all            # full pipeline: results.json + figures in ./output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fuzzy_student import analysis as an
from fuzzy_student import data, tuning
from fuzzy_student import evaluate as ev
from fuzzy_student.casestudy import CASE_STUDY, explain, run_case_study
from fuzzy_student.model import FISConfig, build_fis


# ------------------------------------------------------------------ helpers
def _cfg(args) -> FISConfig:
    return FISConfig(use_assignments=not args.no_assignments, shape=args.shape,
                     defuzz=args.defuzz)


def _split(args):
    X, y, c = data.generate(args.n, args.seed)
    tr, te = data.train_test_split(args.n, 0.3, args.seed)
    return X, y, c, tr, te


def _fmt(d: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


# ----------------------------------------------------------------- commands
def cmd_casestudy(args):
    fis = build_fis(_cfg(args))
    print(f"FIS: {len(fis.inputs)} inputs, {len(fis.rules)} rules "
          f"({sum(r.source == 'slide' for r in fis.rules)} derived from slide-4 rules)\n")
    rows = run_case_study(fis)
    print("All four match slide 5." if all(r["match"] for r in rows) else "Mismatch with slide 5!")


def cmd_predict(args):
    fis = build_fis(_cfg(args))
    student = {"attendance": args.attendance, "marks": args.marks,
               "assignments": args.assignments, "study_hours": args.hours}
    student = {k: v for k, v in student.items() if k in fis.inputs}
    if len(student) != len(fis.inputs) or any(v is None for v in student.values()):
        raise SystemExit("Provide --attendance --marks --hours (and --assignments for the 4-input model)")
    if args.explain:
        explain(fis, **student)
    else:
        score, label = fis.predict(**student)
        print(f"Score {score:.1f}/100 -> {label}")
    if args.advise:
        print("\nWhat-if advice (raise one input by 15% of its range):")
        for a in an.advise(fis, student):
            note = "" if a["gain"] > 0 else "  (no benefit - model flat/non-monotone here)"
            print(f"  {a['input']:<12} {a['from']:g} -> {a['to']:g}: {a['gain']:+.1f} pts{note}")


def cmd_batch(args):
    cols, X, rows = data.read_csv(args.input)
    cfg = _cfg(args).with_(use_assignments="assignments" in cols and not args.no_assignments)
    fis = build_fis(cfg)
    scores, labels = fis.predict_matrix(X, cols)
    out = []
    for row, s, l, x in zip(rows, scores, labels, X):
        res = fis.infer(**{n: x[cols.index(n)] for n in fis.inputs})
        out.append({**row, "score": round(float(s), 1), "performance": l,
                    "top_rule": str(res.firing[0][0])})
    data.write_csv(args.output, out)
    print(f"Wrote {len(out)} predictions to {args.output}")


def cmd_rules(args):
    fis = build_fis(_cfg(args))
    for i, r in enumerate(fis.rules, 1):
        print(f"{i:>3}. [{r.source:<5}] {r}")


def _evaluate(args):
    X, y, c, tr, te = _split(args)
    fis = build_fis(_cfg(args))
    fm = ev.evaluate_fis(fis, X[tr], y[tr], X[te], y[te], c[te], data.COLUMNS)
    ols = ev.LinearRegression().fit(X[tr], y[tr])
    po = ols.predict(X[te])
    om = {**{f"raw_{k}": v for k, v in ev.regression_metrics(y[te], po).items()},
          **ev.classification_metrics(c[te], np.digitize(po, data.TRUE_CUTS))}
    return X, y, c, tr, te, fis, fm, om, ols


def cmd_evaluate(args):
    *_, fm, om, ols = _evaluate(args)
    print(f"Synthetic cohort n={args.n}, seed={args.seed}, 70/30 split (NOT real student data)\n")
    print(f"{'metric':<14}{'FIS':>10}{'Linear reg.':>14}")
    for k in ("raw_r", "raw_r2", "raw_rho", "raw_mae", "raw_rmse", "accuracy", "within_one", "kappa_qw", "macro_f1"):
        print(f"{k:<14}{fm[k]:>10.3f}{om[k]:>14.3f}")
    print(f"{'cal_rmse':<14}{fm['cal_rmse']:>10.3f}   (FIS after affine calibration on train)")
    print("\nOLS coefficients [intercept, " + ", ".join(data.COLUMNS) + "]:", np.round(ols.coef_, 3))


def cmd_tune(args):
    X, y, c, tr, te = _split(args)
    base = _cfg(args)
    cfg, hist = tuning.tune_centers(X[tr], y[tr], base, seed=args.seed)
    print("Tuned medium-peak positions:", {k: round(v, 3) for k, v in cfg.centers.items()})
    print(f"train r: {hist[0]:.3f} -> {hist[-1]:.3f}")
    for name, cf in (("default", base), ("tuned", cfg)):
        m = ev.evaluate_fis(build_fis(cf), X[tr], y[tr], X[te], y[te], c[te], data.COLUMNS)
        print(f"{name:<8} test r={m['raw_r']:.3f}  cal_rmse={m['cal_rmse']:.2f}  "
              f"acc={m['accuracy']:.3f}  kappa={m['kappa_qw']:.3f}")


def cmd_ablation(args):
    X, y, c, tr, te = _split(args)
    rows = an.compare_configs(an.default_ablation_grid(), X[tr], y[tr], X[te], y[te], c[te])
    print(f"{'configuration':<42}{'rules':>6}{'r':>7}{'rho':>7}{'calRMSE':>9}{'acc':>7}{'kappa':>7}")
    for r in rows:
        print(f"{r['name']:<42}{r['n_rules']:>6}{r['raw_r']:>7.3f}{r['raw_rho']:>7.3f}"
              f"{r['cal_rmse']:>9.2f}{r['accuracy']:>7.3f}{r['kappa_qw']:>7.3f}")


def cmd_sensitivity(args):
    fis = build_fis(_cfg(args))
    base = {k: v for k, v in dict(attendance=60, marks=60, assignments=60, study_hours=4).items() if k in fis.inputs}
    print("One-at-a-time sensitivity at", base)
    for r in an.sensitivity_table(fis, base):
        print(f"  {r['input']:<12} score range {r['min']:.1f} - {r['max']:.1f}  (span {r['range']:.1f})")
    print("\nMonotonicity audit (random base points):")
    for v, r in an.monotonicity_audit(fis).items():
        print(f"  {v:<12} sweeps with a >1pt drop: {r['violating_sweeps']:.0%}   worst drop: {r['worst_drop']:.1f}")
    print("\nSame audit with the slide rules removed (rule_mode='weighted'):")
    for v, r in an.monotonicity_audit(build_fis(_cfg(args).with_(rule_mode="weighted"))).items():
        print(f"  {v:<12} sweeps with a >1pt drop: {r['violating_sweeps']:.0%}   worst drop: {r['worst_drop']:.1f}")


def cmd_all(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    from fuzzy_student import plots

    X, y, c, tr, te, fis, fm, om, ols = _evaluate(args)
    res: dict = {"config": {"n": args.n, "seed": args.seed, "n_rules": len(fis.rules)}}
    res["case_study"] = run_case_study(fis)
    res["evaluation"] = {"fis": _fmt(fm), "ols": _fmt(om)}
    res["ols_coef"] = dict(zip(["intercept"] + data.COLUMNS, map(float, ols.coef_)))
    res["class_counts_true"] = np.bincount(c, minlength=4).tolist()

    s_te, l_te = fis.predict_matrix(X[te], data.COLUMNS)
    cm = ev.confusion(c[te], ev.category_index(l_te))
    res["confusion_fis"] = cm.tolist()

    ab = an.compare_configs(an.default_ablation_grid(), X[tr], y[tr], X[te], y[te], c[te])
    res["ablation"] = [_fmt(r) for r in ab]

    cfg_t, hist = tuning.tune_centers(X[tr], y[tr], _cfg(args), seed=args.seed)
    mt = ev.evaluate_fis(build_fis(cfg_t), X[tr], y[tr], X[te], y[te], c[te], data.COLUMNS)
    res["tuning"] = {"centers": cfg_t.centers, "train_r_before": hist[0], "train_r_after": hist[-1],
                     "test_default": _fmt(fm), "test_tuned": _fmt(mt)}

    base = dict(attendance=60, marks=60, assignments=60, study_hours=4)
    res["sensitivity"] = an.sensitivity_table(fis, base)
    res["monotonicity"] = an.monotonicity_audit(fis)
    res["monotonicity_weighted_only"] = an.monotonicity_audit(build_fis(_cfg(args).with_(rule_mode="weighted")))
    res["robustness"] = an.robustness(fis, X[te], data.COLUMNS)
    usage = an.rule_usage(fis, X, data.COLUMNS)
    res["rule_usage"] = {"never_fired": sum(u["fired"] == 0 for u in usage),
                         "never_strongest": sum(u["top"] == 0 for u in usage), "total": len(usage)}
    res["rules"] = [{"source": r.source, "rule": str(r), "note": r.note} for r in fis.rules]
    res["advice_example"] = an.advise(fis, dict(attendance=60, marks=55, assignments=50, study_hours=3))

    figs = [
        plots.membership_functions(fis, out / "fig_membership.png"),
        plots.surface(fis, "attendance", "marks", dict(assignments=60, study_hours=4), out / "fig_surface_att_marks.png"),
        plots.surface(fis, "marks", "study_hours", dict(attendance=70, assignments=60), out / "fig_surface_marks_study.png"),
        plots.aggregation(fis, dict(zip(fis.inputs, [90, 55, 60, 7])), out / "fig_aggregation_D.png"),
        plots.sensitivity(fis, base, out / "fig_sensitivity.png"),
        plots.predicted_vs_actual(y[te], s_te, "FIS (raw score) vs truth", out / "fig_pred_vs_actual.png"),
        plots.confusion_matrix(cm, data.CATEGORIES, "FIS confusion matrix (test)", out / "fig_confusion.png"),
        plots.ablation_bars(ab, out / "fig_ablation.png"),
        plots.rule_usage_bars(usage, out / "fig_rule_usage.png"),
        plots.tuning_history(hist, out / "fig_tuning.png"),
    ]
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nSaved results.json and {len(figs)} figures to {out}/")
    cmd_evaluate(args)


# --------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-assignments", action="store_true", help="use the 3-input model of slides 4-5")
    ap.add_argument("--shape", default="tri", choices=["tri", "trap", "gauss"])
    ap.add_argument("--defuzz", default="centroid", choices=["centroid", "bisector", "mom", "som", "lom"])
    ap.add_argument("--n", type=int, default=300, help="synthetic cohort size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="output")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("casestudy").set_defaults(fn=cmd_casestudy)
    p = sub.add_parser("predict")
    p.add_argument("--attendance", type=float)
    p.add_argument("--marks", type=float)
    p.add_argument("--assignments", type=float)
    p.add_argument("--hours", type=float)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--advise", action="store_true")
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("batch")
    p.add_argument("input")
    p.add_argument("output")
    p.set_defaults(fn=cmd_batch)
    for name, fn in (("rules", cmd_rules), ("evaluate", cmd_evaluate), ("tune", cmd_tune),
                     ("ablation", cmd_ablation), ("sensitivity", cmd_sensitivity), ("all", cmd_all)):
        sub.add_parser(name).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
