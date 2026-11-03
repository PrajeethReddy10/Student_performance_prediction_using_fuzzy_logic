"""Run with:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuzzy_student import analysis as an
from fuzzy_student import data
from fuzzy_student import evaluate as ev
from fuzzy_student.casestudy import run_case_study
from fuzzy_student.engine import defuzzify
from fuzzy_student.membership import MF, from_triangle, gaussmf, trapmf, trimf
from fuzzy_student.model import FISConfig, build_fis


class MembershipTests(unittest.TestCase):
    def test_triangle_values(self):
        x = np.array([0, 25, 50, 75, 100.0])
        np.testing.assert_allclose(trimf(x, 25, 50, 75), [0, 0, 1, 0, 0])
        np.testing.assert_allclose(trimf(x, 50, 100, 100), [0, 0, 0, .5, 1])
        np.testing.assert_allclose(trimf(x, 0, 0, 50), [1, .5, 0, 0, 0])

    def test_scalar_matches_array(self):
        xs = np.linspace(-5, 105, 221)
        for shape in ("tri", "trap", "gauss"):
            for tri in ((0, 0, 50), (25, 50, 75), (50, 100, 100)):
                mf = from_triangle(tri, shape)
                np.testing.assert_allclose(mf(xs), [mf.scalar(float(v)) for v in xs], atol=1e-12)

    def test_degrees_within_unit_interval(self):
        xs = np.linspace(0, 100, 301)
        for mf in (MF("tri", (25, 50, 75)), MF("trap", (0, 10, 20, 30)), MF("gauss", (50, 10))):
            y = mf(xs)
            self.assertTrue(((y >= 0) & (y <= 1)).all())


class DefuzzTests(unittest.TestCase):
    def test_symmetric_set(self):
        x = np.linspace(0, 100, 1001)
        mu = trimf(x, 20, 50, 80)
        for m in ("centroid", "bisector", "mom"):
            self.assertAlmostEqual(defuzzify(x, mu, m), 50, delta=0.2)

    def test_empty_set_returns_midpoint(self):
        x = np.linspace(0, 100, 11)
        self.assertEqual(defuzzify(x, np.zeros_like(x)), 50.0)


class ModelTests(unittest.TestCase):
    def test_rule_counts(self):
        self.assertEqual(len(build_fis().rules), 81)                       # 3^4, as in the article
        self.assertEqual(len(build_fis(FISConfig(use_assignments=False)).rules), 27)

    def test_slide_rules_present(self):
        rules = build_fis().rules
        self.assertTrue(any(r.source == "slide" for r in rules))
        self.assertEqual(sum(r.source == "slide" for r in build_fis(FISConfig(rule_mode="weighted")).rules), 0)

    def test_case_study_matches_slide5(self):
        for cfg in (FISConfig(), FISConfig(use_assignments=False)):
            self.assertTrue(all(r["match"] for r in run_case_study(build_fis(cfg), verbose=False)))

    def test_full_coverage(self):
        fis = build_fis()
        rng = np.random.default_rng(1)
        for _ in range(300):
            res = fis.infer(attendance=rng.uniform(0, 100), marks=rng.uniform(0, 100),
                            assignments=rng.uniform(0, 100), study_hours=rng.uniform(0, 8))
            self.assertGreater(len(res.firing), 0)
            self.assertTrue(0 <= res.score <= 100)

    def test_out_of_range_inputs_are_clamped(self):
        fis = build_fis()
        a = fis.predict(attendance=150, marks=120, assignments=100, study_hours=20)[0]
        b = fis.predict(attendance=100, marks=100, assignments=100, study_hours=8)[0]
        self.assertAlmostEqual(a, b)

    def test_extremes(self):
        fis = build_fis()
        self.assertEqual(fis.predict(attendance=100, marks=100, assignments=100, study_hours=8)[1], "excellent")
        self.assertEqual(fis.predict(attendance=0, marks=0, assignments=0, study_hours=0)[1], "poor")

    def test_missing_input_raises(self):
        with self.assertRaises(ValueError):
            build_fis().infer(attendance=50)

    def test_every_variant_runs(self):
        pt = dict(attendance=70, marks=65, assignments=60, study_hours=5)
        for _, cfg in an.default_ablation_grid():
            fis = build_fis(cfg)
            s, _ = fis.predict(**{k: v for k, v in pt.items() if k in fis.inputs})
            self.assertTrue(0 <= s <= 100)


class EvaluationTests(unittest.TestCase):
    def test_pearson_spearman(self):
        a = np.arange(10.0)
        self.assertAlmostEqual(ev.pearson(a, 2 * a + 1), 1.0)
        self.assertAlmostEqual(ev.spearman(a, a ** 3), 1.0)

    def test_ranks_ties(self):
        np.testing.assert_allclose(ev._ranks(np.array([1.0, 2.0, 2.0, 3.0])), [1, 2.5, 2.5, 4])

    def test_kappa_perfect(self):
        cm = np.diag([5, 5, 5, 5])
        self.assertAlmostEqual(ev.quadratic_kappa(cm), 1.0)

    def test_ols_recovers_linear_law(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (100, 3))
        y = 2 + X @ np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(ev.LinearRegression().fit(X, y).coef_, [2, 1, -2, 3], atol=1e-9)

    def test_dataset_reproducible_and_in_range(self):
        X1, y1, _ = data.generate(50, 3)
        X2, y2, _ = data.generate(50, 3)
        np.testing.assert_array_equal(X1, X2)
        self.assertTrue((X1[:, :3] >= 0).all() and (X1[:, :3] <= 100).all())
        self.assertTrue((X1[:, 3] >= 0).all() and (X1[:, 3] <= 8).all())


if __name__ == "__main__":
    unittest.main()
