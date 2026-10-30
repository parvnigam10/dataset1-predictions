# Result reconciliation — 30 October 2026

Base commit: `1c15bc0acf920d0bf7021db8de3f58cc2e041949`.

The corrected column uses the complete regenerated run recorded in `results/run_manifest.json`. It is not a transcription of the old saved CSV or of the earlier reproduction package.

| Experiment | Previous README accuracy | Previous saved CSV accuracy | Corrected run accuracy | Corrected Macro F1 |
| --- | --- | --- | --- | --- |
| E9_Logistic Regression | 53.93% | 53.88% | 53.88% | 0.4035 |
| E9_Balanced Logistic Regression | 49.82% | 49.56% | 49.50% | 0.4594 |
| E9_Random Forest | 54.30% | 54.04% | 54.04% | 0.4030 |
| E9_Extra Trees | 54.46% | 54.19% | 54.19% | 0.3970 |
| E9_HistGradientBoosting | 53.36% | 53.83% | 53.62% | 0.4072 |
| E9_XGBoost | 54.04% | 53.78% | 53.57% | 0.4061 |
| E9_Multi-Layer Perceptron (MLP) | 53.05% | 54.19% | 54.19% | 0.4142 |
| Ensemble_5Models_E9 | 54.35% | 54.30% | 54.25% | 0.4019 |
| Ensemble_5Models_DrawCalibrated | 52.37% | 52.11% | 52.89% | 0.4229 |

The holdout is deliberately not compared as an equivalent sample: the old README claimed 4,502 fixtures through 2026, while the available input supplies 2,651 five-league fixtures from 1 January 2024 to 25 May 2025. The corrected holdout accuracy is 53.90%.

## Why the numbers changed

1. Ordinary and balanced LR previously shared a dictionary key, so balanced LR accidentally entered the main ensemble. They now have separate keys, and ordinary LR is used consistently.
2. Error analysis previously trained a separate three-model ensemble. It now reads the main five-model predictions, including all H/D/A probabilities.
3. ROC and feature plots now consume outputs from the same fitted models; they do not independently retrain.
4. Exact input hashes, split dates, feature lists, fixed weights, thread counts and package versions are recorded. This run uses XGBoost 3.1.1 and scikit-learn 1.8.0; earlier environments were not locked, so individual model metrics can differ.
5. Historical notebook narratives and figures with inconsistent values are replaced. The regenerated research report contains the reconciled empirical results; the former write-up is available in Git history.
6. Claims that the observed accuracy establishes a universal ceiling or proves calibration/zero degradation have been removed.

## Validation

- Completed all 33 baseline, feature-progression, draw-multiplier and SMOTE experiments.
- Recomputed every metric from saved labels and predictions.
- Checked confidence-selected counts, coverage and accuracy; per-class AUC; and exact probability weights.
- Validated the notebook schema, executed all notebook code cells in process, and inspected regenerated plots. A separate Jupyter kernel could not start because this execution environment blocks its network interface discovery.
- Eight regression/consistency tests pass. GitHub Actions runs the same checks without requiring the raw dataset.

Separate Library PDF/PPT artifacts are outside this repository correction.
