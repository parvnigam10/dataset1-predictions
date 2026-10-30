# Football Match Outcome Prediction

Predict Home win (H), Draw (D), or Away win (A) from pre-match information across the Bundesliga, Premier League, Ligue 1, Serie A and La Liga.

The current E9 benchmark uses 56 features covering Elo, recent form, historical goals and match statistics, venue history, market odds, rest days and head-to-head history. Training uses fixtures through 2022, testing uses calendar-year 2023, and holdout evaluation uses the later fixtures actually present in the input file.

## Reproduce the results

Place your exact input snapshot at `data/raw/Matches.csv`. Raw and processed datasets are excluded from Git. The committed metadata records the input SHA-256, actual dates, fixture counts, source hashes and package versions.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python src/run_all.py
python -m unittest discover -s tests -v
```

The runner fits models once, exports predictions, derives diagnostics, updates the generated results below and writes an artifact checksum manifest. It replaces generated files under `results/`; keep a copy of that directory before rerunning if you want to retain another run. `python src/run_pipeline.py` and `python src/run_smote_experiments.py` are compatibility aliases for this same complete run.

`requirements-lock.txt` pins the training/evaluation environment used here. The original `requirements.txt` also includes optional notebook tools. To rebuild the read-only results notebook, run `python src/build_notebook.py`. No dataset is needed to run the committed-results consistency tests.

## One ensemble definition

The standard ensemble combines Extra Trees, XGBoost, HistGradientBoosting, Random Forest and **ordinary logistic regression** with weights **0.35, 0.20, 0.20, 0.15, 0.10**. Balanced logistic regression has a separate model key and remains a separate benchmark. The SMOTE ensemble uses the same members and weights after training-only oversampling.

Every evaluation row has a matching CSV in `results/predictions/`, including fixture identifiers, actual and predicted classes, P(Home), P(Draw), P(Away), Elo difference and correctness. Confusion matrices, ROC curves and error-analysis summaries read these same predictions. Feature importance comes from the exact benchmark fits, without retraining.

## Interpretation and scope

- Full-population accuracy, macro F1, per-class recall and ROC-AUC measure different properties. Report them together.
- Confidence filtering reports accuracy on accepted fixtures together with coverage. It does not establish 70% accuracy across every match.
- Draw multipliers change class decisions. They do not demonstrate calibrated probabilities.
- Thresholds and intervention comparisons here are descriptive test/holdout sweeps. Select operating thresholds using separate earlier validation data before making deployment claims.
- This run does not independently certify zero leakage, statistical superiority, a universal accuracy ceiling or betting profitability.
- Earlier README/notebook/report numbers mixed saved runs and different ensemble definitions. The generated section below supersedes those summaries for this exact snapshot. Historical 2024–2026 figures from the earlier 4,502-fixture holdout have not been reproduced with the available shorter dataset.
- `RESEARCH_PAPER.md` remains a historical manuscript draft. Use the generated tables and provenance below when updating a paper or presentation.

<!-- BEGIN GENERATED RESULTS -->

## Verified run and data coverage

Input SHA-256: `eac9d0096993c8a5705f724e0ddc5513f71aca29058106182bc9a2c6340bcc0e`. The run contains **230,557 raw fixtures** and **43,708 selected five-league fixtures**.

| Partition | First fixture | Last fixture | Matches |
|---|---|---|---|
| train | 2000-07-28 | 2022-12-31 | 39,138 |
| test | 2023-01-01 | 2023-12-31 | 1,919 |
| holdout | 2024-01-01 | 2025-05-25 | 2,651 |

These dates describe the available snapshot, not a promised calendar-year endpoint. Full provenance: [run metadata](results/run_metadata.json). Artifact hashes: [manifest](results/artifact_manifest.json).

## 2023 feature progression

Model families change during the progression, so this is not a controlled feature-only ablation.

| Experiment | Columns | Accuracy | Macro F1 | Draw recall |
|---|---|---|---|---|
| E0_Baseline_logistic_regression | 5 | 46.12% | 0.2104 | 0.00% |
| E1_Elo_logistic_regression | 11 | 53.10% | 0.3842 | 0.00% |
| E2_Form_logistic_regression | 20 | 53.15% | 0.3845 | 0.00% |
| E3_Goals_xgboost | 35 | 53.52% | 0.3885 | 0.21% |
| E4_Venue_xgboost | 44 | 53.31% | 0.3870 | 0.21% |
| E5_MatchStats_xgboost | 63 | 53.52% | 0.3916 | 0.21% |
| E6_Odds_xgboost | 71 | 54.35% | 0.4108 | 1.66% |
| E7_Selected_xgboost | 28 | 53.41% | 0.4021 | 1.25% |
| E8_Paper_Advanced_xgboost | 39 | 53.31% | 0.4074 | 2.29% |
| E9_Expanded_Market_Fatigue_extra_trees | 56 | 54.19% | 0.3970 | 0.00% |

## E9 model benchmark

| Model | Accuracy | Macro F1 | Home recall | Draw recall | Away recall |
|---|---|---|---|---|---|
| Logistic Regression | 53.88% | 0.4035 | 80.00% | 0.83% | 58.23% |
| Balanced Logistic Regression | 49.50% | 0.4594 | 54.12% | 22.66% | 65.46% |
| Random Forest | 54.04% | 0.4030 | 81.24% | 0.83% | 56.78% |
| Extra Trees | 54.19% | 0.3970 | 83.16% | 0.00% | 54.97% |
| HistGradientBoosting | 53.83% | 0.4107 | 80.11% | 2.29% | 56.60% |
| XGBoost | 54.04% | 0.4119 | 79.66% | 1.87% | 58.41% |
| Multi-Layer Perceptron (MLP) | 54.19% | 0.4142 | 81.02% | 2.70% | 56.06% |

## Ensemble and SMOTE results

| Experiment | Evaluation | Accuracy | Macro F1 | Draw recall |
|---|---|---|---|---|
| Ensemble_5Models_E9 | 2023 Benchmark Test | 54.30% | 0.4038 | 0.62% |
| Ensemble_5Models_DrawCalibrated | 2023 Benchmark Test | 52.37% | 0.4171 | 5.61% |
| Holdout_Ensemble | Holdout (2024-01-01 to 2025-05-25, n=2,651) | 53.79% | 0.4077 | 0.29% |
| SMOTE_Ensemble_5Models_E9 | 2023 Benchmark Test | 52.21% | 0.4454 | 10.81% |
| SMOTE_Holdout_Ensemble | Holdout (2024-01-01 to 2025-05-25, n=2,651) | 53.90% | 0.4655 | 11.49% |

Detailed baseline/SMOTE comparisons: [CSV](results/smote_vs_baseline_comparison.csv).

## XGBoost draw multiplier sweep

| Multiplier | Accuracy | Macro F1 | Draw recall |
|---|---|---|---|
| 1.00 | 54.04% | 0.4119 | 1.87% |
| 1.15 | 52.84% | 0.4204 | 4.99% |
| 1.25 | 52.11% | 0.4363 | 10.40% |
| 1.35 | 51.85% | 0.4593 | 19.96% |
| 1.50 | 49.14% | 0.4605 | 32.22% |

## Standard ensemble confusion matrix (2023)

| Actual / predicted | H | D | A |
|---|---|---|---|
| H | 725 | 6 | 154 |
| D | 300 | 3 | 178 |
| A | 237 | 2 | 314 |

There are 1,919 fixtures and 1,042 correct predictions. Error analysis and ROC evaluation use these same prediction exports.

## Confidence filtering

| Evaluation | Confidence at least | Selected fixtures | Coverage | Selected accuracy |
|---|---|---|---|---|
| 2023 Benchmark Test | 0.40 | 1572 | 81.92% | 56.93% |
| 2023 Benchmark Test | 0.45 | 1253 | 65.29% | 59.94% |
| 2023 Benchmark Test | 0.50 | 966 | 50.34% | 62.73% |
| 2023 Benchmark Test | 0.55 | 786 | 40.96% | 65.14% |
| 2023 Benchmark Test | 0.60 | 504 | 26.26% | 70.83% |
| 2023 Benchmark Test | 0.65 | 376 | 19.59% | 74.47% |
| 2023 Benchmark Test | 0.70 | 246 | 12.82% | 80.08% |
| Holdout (2024-01-01 to 2025-05-25, n=2,651) | 0.40 | 2219 | 83.70% | 57.55% |
| Holdout (2024-01-01 to 2025-05-25, n=2,651) | 0.45 | 1757 | 66.28% | 61.18% |
| Holdout (2024-01-01 to 2025-05-25, n=2,651) | 0.50 | 1369 | 51.64% | 64.86% |
| Holdout (2024-01-01 to 2025-05-25, n=2,651) | 0.55 | 1082 | 40.81% | 66.73% |
| Holdout (2024-01-01 to 2025-05-25, n=2,651) | 0.60 | 758 | 28.59% | 71.11% |
| Holdout (2024-01-01 to 2025-05-25, n=2,651) | 0.65 | 580 | 21.88% | 75.00% |
| Holdout (2024-01-01 to 2025-05-25, n=2,651) | 0.70 | 394 | 14.86% | 76.65% |

## One-versus-rest ROC-AUC

| Model | Evaluation | Home AUC | Draw AUC | Away AUC | Macro AUC |
|---|---|---|---|---|---|
| Extra Trees | 2023 Benchmark Test | 0.7165 | 0.5635 | 0.7169 | 0.6660 |
| XGBoost | 2023 Benchmark Test | 0.7145 | 0.5674 | 0.7153 | 0.6661 |
| HistGradientBoosting | 2023 Benchmark Test | 0.7131 | 0.5645 | 0.7152 | 0.6647 |
| Random Forest | 2023 Benchmark Test | 0.7148 | 0.5598 | 0.7144 | 0.6634 |
| Logistic Regression | 2023 Benchmark Test | 0.7160 | 0.5696 | 0.7166 | 0.6678 |
| Multi-Layer Perceptron (MLP) | 2023 Benchmark Test | 0.7124 | 0.5611 | 0.7121 | 0.6623 |
| Soft-Voting Ensemble (5 Models) | 2023 Benchmark Test | 0.7160 | 0.5645 | 0.7167 | 0.6661 |
| Soft-Voting Ensemble (5 Models) | Holdout (2024-01-01 to 2025-05-25, n=2,651) | 0.7392 | 0.5730 | 0.7552 | 0.6894 |
| SMOTE Soft-Voting Ensemble | 2023 Benchmark Test | 0.7170 | 0.5629 | 0.7165 | 0.6659 |

## Top XGBoost feature importances

| Feature | XGBoost importance | Random Forest importance |
|---|---|---|
| NormProbHome | 0.2612 | 0.0805 |
| OddsRatio_H_A | 0.0892 | 0.0800 |
| NormProbAway | 0.0796 | 0.0672 |
| X2Odd | 0.0766 | 0.0791 |
| 1XOdd | 0.0366 | 0.0789 |
| NormProbDraw | 0.0302 | 0.0266 |
| NormHome_x_NormDraw | 0.0219 | 0.0376 |
| 12Odd | 0.0179 | 0.0179 |
| EloWinProbHome | 0.0119 | 0.0308 |
| MarketHandiSize | 0.0114 | 0.0300 |

Importance describes the fitted model and does not establish causation.

## Execution environment

python 3.12.14, numpy 2.5.4, pandas 2.2.3, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1, imbalanced-learn 0.14.2, matplotlib 3.10.8, seaborn 0.13.2.

See `results/figures/` for generated confusion matrices, ROC curves and feature-importance charts.

<!-- END GENERATED RESULTS -->
