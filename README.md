# Who Gets Flagged? Accuracy and Fairness of Early-Warning Models for Undergraduate Dropout

Code for a study that evaluates how accurately undergraduate dropout can be predicted, and whether
the predictions work equally well for different groups of students.

**Author:** Priscilla Sarfoa Anim

## Research questions

1. **RQ1:** How accurately can dropout be predicted at enrollment, compared with after the first semester?
2. **RQ2:** Does XGBoost predict dropout better than logistic regression?
3. **RQ3:** Are the predictions equally accurate, well calibrated, and equitable in their errors across
   gender, scholarship status, and age at enrollment?

## Data

The data come from the UCI Machine Learning Repository dataset
[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697) (id 697).
It contains records for 4,424 undergraduates at a Portuguese public higher-education institution
and is licensed under CC BY 4.0. The dataset is described in:

> Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and
> academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146

The code downloads the data directly from UCI using the `ucimlrepo` package. If you are offline,
place the UCI `data.csv` file (semicolon-separated) in the project folder instead.

## Repository contents

| File | Description |
|---|---|
| `dropout_fairness_analysis.ipynb` | Full analysis in one notebook (runs in Google Colab or Jupyter) |
| `01_descriptives.py` | Descriptive statistics: outcome counts, Table 1, dropout rates by group, Figures 1–3 |
| `02_models_fairness.py` | Prediction models, fairness analysis, SHAP, and robustness checks |
| `requirements.txt` | Python packages needed |
| `output/`, `output_models/` | Tables (CSV) and figures (PNG) produced by the code |

## How to run

**Option 1: Google Colab.** Upload `dropout_fairness_analysis.ipynb` to
[Colab](https://colab.research.google.com) and choose *Runtime → Run all*. The first cell installs
the packages and downloads the data.

**Option 2: Locally.**

```bash
pip install -r requirements.txt
python 01_descriptives.py
python 02_models_fairness.py
```

Running the model script takes about 2–3 minutes. Results are saved to `output/` and
`output_models/`.

## Analysis overview

- **Outcome:** Dropout vs. Graduate. Students still enrolled (outcome unknown) are excluded from
  the main analysis, giving an analytic sample of 3,630 students.
- **Two prediction points:** *at enrollment* (background and admission information) and *after
  the first semester* (adds first-semester grades, approved units, and pass rate). Second-semester
  variables are excluded to avoid information leakage.
- **Models:** L2-penalized logistic regression and XGBoost, tuned with 5-fold cross-validation on a
  70% training set and evaluated on a 30% held-out test set (random seed 42).
- **Evaluation:** AUC, Brier score, and calibration overall; paired bootstrap comparisons of AUC.
- **Fairness:** calibration, AUC, true and false positive rates, and flag rates by gender,
  scholarship status, and age group, when the 30% highest-risk students are flagged, with 95%
  bootstrap confidence intervals.
- **Interpretation:** SHAP values for the XGBoost models.
- **Robustness checks:** adding debtor and tuition-fee status, removing gender, and an alternative
  outcome definition that includes enrolled students.

## Main results

| | At enrollment | After 1st semester |
|---|---|---|
| AUC, logistic regression | 0.819 | 0.927 |
| AUC, XGBoost | 0.822 | 0.929 |

- First-semester information improves prediction substantially; XGBoost adds almost nothing over
  logistic regression.
- The models are well calibrated within every group, but at a top-30% flagging rule they identify
  80% of men who drop out vs. 66% of women, and 36% of scholarship holders who drop out vs. 76% of
  non-holders.

