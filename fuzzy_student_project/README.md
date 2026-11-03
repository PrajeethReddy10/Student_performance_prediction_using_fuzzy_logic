# Student Performance Prediction Using Fuzzy Logic

Mamdani fuzzy inference system for predicting student performance, built from the
slide deck and following the methodology of Şenel et al., *Scientific Reports* 15,
28241 (2025) (https://doi.org/10.1038/s41598-025-99124-3).

## Structure
```
main.py                     CLI (casestudy, predict, batch, rules, evaluate, tune, ablation, sensitivity, all)
fuzzy_student/
  membership.py             triangular / trapezoidal / Gaussian MFs, LinguisticVariable
  engine.py                 Mamdani engine: AND (min|prod), implication (min|prod), max aggregation,
                            defuzzification (centroid, bisector, MOM, SOM, LOM)
  model.py                  variables, slide-4 rules, 81-rule base, FISConfig
  casestudy.py              slide-5 students A-D, step-by-step explanation
  data.py                   synthetic cohort generator, CSV I/O
  evaluate.py               r, R^2, Spearman, MAE, RMSE, accuracy, kappa, F1, OLS baseline
  tuning.py                 calibration of MF centres (random search + coordinate refinement)
  analysis.py               sensitivity, monotonicity audit, what-if advisor, rule usage,
                            noise robustness, ablation grid
  plots.py                  all figures
tests/test_fis.py           18 unit tests
```

## Run
```
pip install -r requirements.txt
python main.py casestudy
python main.py predict --attendance 85 --marks 72 --assignments 65 --hours 6 --explain --advise
python main.py batch students.csv predictions.csv     # columns: attendance,marks,assignments,study_hours
python main.py ablation
python main.py all                                     # results.json + 10 figures in ./output
python -m unittest discover -s tests -v
```
Use `--no-assignments` for the 3-input model shown on slides 4-5.

## Assumptions (not given in the slides or article)
* Numeric MF breakpoints, input ranges (study hours 0-8 h/day) and the output score scale.
* Consequents of the generated rules: a weighted composite of the input levels
  (marks 0.40, assignments 0.25, attendance 0.20, study hours 0.15).
  The four slide-4 rules take precedence on the combinations they cover.
* Crisp values for students A-D (and their assignment scores) are chosen to match slide 5's levels.
* The evaluation cohort is **synthetic**; results demonstrate the pipeline, not accuracy on real students.
