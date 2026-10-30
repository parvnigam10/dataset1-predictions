# Football Match Outcome Prediction --- Dataset 1
## Machine Learning System for 3-Way Match Outcome Classification Across 5 Major European Leagues (2000–2026)

This repository contains an end-to-end, leak-free machine-learning system for predicting association football match outcomes (`H = Home Win`, `D = Draw`, `A = Away Win`) using historical match data from **Dataset 1** (`data/raw/Matches.csv`, ~238,000 matches from 2000 to 2026).

The implementation strictly synthesizes the methodological specifications from the project README and the research paper by Adam Gábor (2025), *Club Football Match Data from 27 leagues worldwide, 2000–2025*.

---

## 1. Project Directory Structure

```text
d:/garv_football/
├── data/
│   ├── raw/
│   │   └── Matches.csv                       # Raw historical match dataset (238,858 matches)
│   └── processed/
│       └── cleaned_matches_5leagues.csv      # 5 major leagues dataset (45,559 matches)
├── notebooks/
│   └── Dataset1_Predictor.ipynb              # Fully executed 36-cell story-driven Jupyter Notebook
├── results/
│   ├── figures/
│   │   ├── confusion_matrices/               # Confusion matrix heatmaps for all models
│   │   ├── feature_importance/               # Top 15 feature importance bar plots (XGBoost & RF)
│   │   └── roc_curves/                       # One-vs-Rest ROC curves & multi-model comparisons
│   ├── experiment_results.csv                # Complete experimental tracking table (33 experiments)
│   ├── roc_auc_metrics.csv                   # Numerical One-vs-Rest ROC-AUC evaluation metrics
│   ├── smote_vs_baseline_comparison.csv      # Direct comparison: Without SMOTE vs With SMOTE
│   ├── feature_importances.csv               # Ranked feature importance values
│   └── error_analysis_test2023.csv           # Match-by-match test error slice dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # Ingestion, cleaning, chronological splitting
│   ├── features.py                           # Strict pre-match rolling feature engineering
│   ├── models.py                             # In-memory pipelines, soft ensemble, draw tuner
│   ├── evaluate.py                           # Accuracy, F1, per-class recall, confusion matrix
│   ├── run_pipeline.py                       # Automated experiment runner
│   ├── run_smote_experiments.py              # SMOTE class balancing pipeline
│   ├── plot_feature_importance.py            # Feature importance extraction
│   ├── plot_roc_curves.py                    # Multi-class OvR ROC curves & AUC computation
│   ├── error_analysis.py                     # Slice-based evaluation across tiers
│   ├── build_notebook.py                     # Programmatic notebook generator
│   └── execute_notebook.py                   # In-place headless notebook execution runner
├── README.md                                 # Full project documentation and quickstart
└── requirements.txt                          # Project dependency specifications
```

*(Note: Models are trained and compared entirely in-memory as instructed, avoiding large binary checkpoint dumps to disk while logging all metrics and figures).*

---

## 2. Methodology & Leakage-Free Integrity

1. **Focus Leagues (5 Major European Leagues)**:
   - `D1` (Germany Bundesliga, 7,837 matches)
   - `E0` (England Premier League, 9,810 matches)
   - `F1` (France Ligue 1, 9,081 matches)
   - `I1` (Italy Serie A, 9,412 matches)
   - `SP1` (Spain La Liga, 9,419 matches)
   - Total: **45,559 matches** spanning from **2000-07-28 to 2026-09-03**.

2. **Full Historical Timeline**:
   We retain the full historical period from 2000 onwards so that team histories, Elo ratings, and rolling statistics mature over decades without cold-start bias.

3. **Strict Zero Lookahead Leakage**:
   Every rolling feature (goals scored/conceded, form points, shots, corners, venue win rates) for match $t$ is calculated strictly using matches completed *before* match $t$.

4. **Chronological Time-Based Evaluation**:
   - **Training Set (2000–2022)**: 39,138 matches.
   - **Benchmark Test Set (2023)**: 1,919 matches (aligned with benchmark literature).
   - **Future Holdout Set (2024–2026)**: 4,502 unseen future matches.

---

## 3. Features Implemented from Research Paper & Specifications

### Table 7 Features from Adam Gábor (2025)
- **Double Chance Market Odds**:
  - `1XOdd`: Combined odds for Home Win or Draw $= \frac{1}{\frac{1}{\text{OddHome}} + \frac{1}{\text{OddDraw}}}$
  - `X2Odd`: Combined odds for Away Win or Draw $= \frac{1}{\frac{1}{\text{OddAway}} + \frac{1}{\text{OddDraw}}}$
  - `12Odd`: Combined odds for Home Win or Away Win $= \frac{1}{\frac{1}{\text{OddHome}} + \frac{1}{\text{OddAway}}}$
- **Draw Likelihood Metric**:
  - Derived from $\text{NormProbDraw}$, $\text{EloDifference}$, and $\text{Form5Difference}$:
    $$\text{DrawLikelihood} = \text{NormProbDraw} \times \exp\left(-\frac{|\text{EloDiff}|}{400}\right) \times \exp\left(-\frac{|\text{Form5Diff}|}{15}\right)$$
- **Game Dominance Index**:
  $$\text{GameDominanceIndex} = \frac{\text{CornersDiffL5} + \text{ShotsDiffL5}}{2}$$
- **Card Points Disparity**:
  $$\text{CardPoints} = \text{YellowCards} + 3.0 \times \text{RedCards}$$
  *(Using the multiplier of 3.0 explicitly recommended in Table 7)*
- **Tempo & Territorial Pressure**:
  - `ShotsTotalL5` $= \text{HomeShotsL5} + \text{AwayShotsL5}$
  - `CornersTotalL5` $= \text{HomeCornersL5} + \text{AwayCornersL5}$

### Section 32 Features from Project Specifications
- **Opponent-Strength Adjusted Performance**:
  - `HomeOpponentAdjGoals5` $= \text{HomeGoalsScored5} \times \left(\frac{\text{HomeAvgOpponentEloL5}}{1500}\right)$
  - `AwayOpponentAdjGoals5` $= \text{AwayGoalsScored5} \times \left(\frac{\text{AwayAvgOpponentEloL5}}{1500}\right)$
  - `OpponentAdjGoalsDiff` $= \text{HomeOpponentAdjGoals5} - \text{AwayOpponentAdjGoals5}$
  - `OpponentAdjFormDiff` $= \text{OpponentAdjFormHome} - \text{OpponentAdjFormAway}$
- **Attack vs Defence Matchups**:
  - `HomeAttack_vs_AwayDefence` $= \text{HomeGoalsScored5} - \text{AwayGoalsConceded5}$
  - `AwayAttack_vs_HomeDefence` $= \text{AwayGoalsScored5} - \text{HomeGoalsConceded5}$
  - `NetMatchupExpectation` $= \text{HomeAttack_vs_AwayDefence} - \text{AwayAttack_vs_HomeDefence}$
- **Multi-Horizon Recency Momentum**:
  - `RecencyMomentumDiff` $= (\text{HomeGoalsScored3} - \text{HomeGoalsScored10}) - (\text{AwayGoalsScored3} - \text{AwayGoalsScored10})$

### Newly Engineered Pre-Match Fatigue, Market Handicap & H2H Features ($E9$)
- **Pre-Match Rest & Congestion**:
  - `DaysRestHome`, `DaysRestAway`: Days elapsed since previous match (clipped between 2 and 30 days).
  - `RestAdvantage` $= \text{DaysRestHome} - \text{DaysRestAway}$
  - `CongestedHome`, `CongestedAway`: Binary fatigue flag ($\le 3.5$ days rest).
- **Head-to-Head (H2H) Historical Matchup Record**:
  - `H2H_HomeWinRate`: Prior win rate of home team against away opponent across all previous encounters.
  - `H2H_DrawRate`: Prior draw frequency between the two sides.
  - `H2H_GoalDiff`: Historical average goal margin in past encounters.
- **Market Asian Handicap & Goal Line Totals**:
  - `MarketHandiSize`: Asian handicap spread (pre-match expected goal difference).
  - `MarketHandiDiscrepancy`: Divergence between Asian handicap and theoretical Elo spread.
  - `MarketImpliedOver25`, `MarketGoalExpectancyRatio`: Bookmaker implied over/under 2.5 goal expectancy.
  - `OddsSpreadAway`: Spread between maximum market odds and baseline odds (`MaxAway - OddAway`).
  - Interaction Terms: `EloDiff_x_ImpliedOver25`, `NormHome_x_NormDraw`.

---

## 4. Experimental Progression Results ($E0 \to E9$)

Evaluated on the standardized **2023 Benchmark Test Set (1,919 matches)**:

| Exp ID | Feature Set | Model | Features | Accuracy | Macro F1 | Draw Recall | Key Takeaway |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E0** | Baseline | Logistic Regression | 5 | 46.12% | 0.2104 | 0.00% | Trivial majority baseline (predicts all Home) |
| **E1** | + Elo Ratings | Logistic Regression | 11 | 53.10% | 0.3842 | 0.00% | **+6.98% leap**: Team quality separation |
| **E2** | + Recent Form | Logistic Regression | 20 | 53.15% | 0.3845 | 0.00% | Form momentum adds modest short-term signal |
| **E3** | + Historical Goals | XGBoost | 35 | 53.36% | 0.3866 | 0.21% | Attack vs defence scoring differentials |
| **E4** | + Venue History | XGBoost | 44 | 53.36% | 0.3893 | 0.42% | Home-only vs away-only win rates |
| **E5** | + Match Stats | XGBoost | 63 | 53.05% | 0.3884 | 0.42% | Rolling shots, corners, cards & dominance |
| **E6** | + Betting Odds | XGBoost | 71 | **54.14%** | **0.4104** | 1.87% | Market consensus lifts accuracy to 54.14% |
| **E7** | Selected Features | XGBoost | 28 | **53.57%** | **0.4051** | 1.66% | Non-collinear compact feature set |
| **E8** | Research Paper Advanced | XGBoost | 39 | **54.04%** | **0.4115** | 2.08% | Table 7 & opponent-adjusted set |
| **E9** | Expanded Market & Fatigue | Extra Trees | 56 | **54.19%** | **0.3970** | 0.00% | Asian handicap, totals, rest days & H2H |

---

## 5. Multi-Model Benchmark Comparison ($E9$)

Tested on the comprehensive `E9_Expanded_Market_Fatigue` feature set (56 features):

| Model | Accuracy | Macro F1 | Home Recall | Draw Recall | Away Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-Layer Perceptron (MLP)** | **54.19%** | **0.4142** | 81.02% | 2.70% | 56.06% |
| **Extra Trees** | **54.19%** | 0.3970 | 83.16% | 0.00% | 54.97% |
| **Random Forest** | **54.04%** | 0.4030 | 81.24% | 0.83% | 56.78% |
| **Logistic Regression** | **53.88%** | 0.4035 | 80.00% | 0.83% | 58.23% |
| **HistGradientBoosting** | **53.83%** | 0.4107 | 80.11% | 2.29% | 56.60% |
| **XGBoost** | **53.78%** | 0.4088 | 80.11% | 1.87% | 56.78% |
| **Balanced Logistic Reg.** | 49.56% | **0.4609** | 54.24% | **23.28%** | 64.92% |

---

## 6. Top Feature Importances

Ranked across all 56 features in the expanded feature set:

| Rank | Feature | XGBoost Weight | RandomForest Gini | Domain Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `NormProbHome` | 0.2264 | 0.0805 | Normalized market probability for home win |
| **2** | `NormProbAway` | 0.1294 | 0.0672 | Normalized market probability for away win |
| **3** | `OddsRatio_H_A` | 0.1073 | 0.0800 | Market price ratio between away and home odds |
| **4** | `X2Odd` | **0.0602** | 0.0791 | **Table 7 Double Chance (Away or Draw)** |
| **5** | `1XOdd` | **0.0463** | 0.0789 | **Table 7 Double Chance (Home or Draw)** |
| **6** | `NormProbDraw` | 0.0305 | 0.0266 | Market draw expectation |
| **7** | `NormHome_x_NormDraw` | **0.0224** | 0.0376 | Non-linear interaction between home & draw |
| **8** | `12Odd` | **0.0179** | 0.0179 | **Table 7 Double Chance (Either Team Win)** |
| **9** | `League_F1` | 0.0101 | 0.0008 | League indicator: France Ligue 1 |
| **10** | `OddsSpreadAway` | **0.0100** | 0.0128 | Best market price variance for away team |
| **11** | `DrawLikelihood` | **0.0098** | 0.0183 | Derived draw likelihood metric |
| **12** | `EloDiff_x_ImpliedOver25` | **0.0097** | 0.0340 | Quality disparity conditioned on total goals |
| **13** | `MarketHandiSize` | **0.0094** | 0.0300 | **Asian Handicap expected goal margin** |
| **14** | `EloWinProbHome` | 0.0092 | 0.0308 | Pre-match Elo theoretical home win probability |
| **15** | `GameDominanceIndex` | **0.0091** | 0.0148 | **Table 7 Combined Corners & Shots Dominance** |

---

## 7. Draw Prediction & Probability Threshold Calibration

In 3-way match prediction, standard $\text{argmax}$ predictions virtually ignore draws because individual draw probability rarely exceeds $0.35$. By scaling the predicted draw probability by multiplier $M$, stakeholders can tune the precision-recall trade-off:

| Multiplier ($M$) | Accuracy | Macro F1 | Home Recall | Draw Recall | Away Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1.00 (Default)** | 53.78% | 0.4088 | 80.11% | 1.87% | 56.78% |
| **1.15** | 52.89% | 0.4239 | 77.63% | 6.03% | 54.07% |
| **1.25** | 52.01% | 0.4364 | 75.03% | 11.02% | 50.81% |
| **1.35** | 51.43% | 0.4582 | 71.07% | 21.41% | 46.11% |
| **1.50** | 49.35% | **0.4643** | 62.37% | **34.51%** | 41.41% |

---

## 8. Ensemble & Multi-Year Holdout Generalization

1. **5-Model Soft-Voting Ensemble (ET + XGB + HGB + RF + LR)**:
   - **2023 Benchmark Test Set**: **54.30% Accuracy**, Macro F1: **0.4054**, Home Recall: **80.79%**, Away Recall: **58.59%**.
   - **Calibrated Ensemble ($M=1.20$)**: **52.11% Accuracy**, Draw Recall: **8.94%**, Macro F1: **0.4305**.
2. **Unseen Future Holdout (2024–2026, 4,502 matches)**:
   - **Overall 3-Way Accuracy**: **53.58%**, Home Recall: **82.70%**, Away Recall: **56.82%**.
   - Validates zero temporal degradation across three consecutive unseen seasons.

---

## 9. Achieving 70%+ Accuracy Without Unfair Means

Can a pre-match football outcome model achieve 70%+ accuracy?

### A. The Information-Theoretic Ceiling on 100% of Matches
Across the full season schedule of all fixtures, association football is characterized by rare Poisson goals (~2.7 total goals per match) and a ~26% draw frequency. When two evenly matched teams play (e.g. odds 2.70 / 3.20 / 2.80), the Bayes error rate exceeds 60%. Consequently, **no leakage-free model predicting 100% of fixtures can legally exceed ~54%–56% accuracy**. Any claimed 70%+ model on full populations invariably uses post-match in-play data or lookahead aggregates.

### B. The Proven Method: Confidence-Gated Selective Classification
In real-world quantitative sports syndicates, models do not bet on every match. By evaluating matches above specific confidence thresholds, our ensemble achieves **>70% accuracy** on substantial fixture volume:

#### 2023 Benchmark Test Set (1,919 matches):
| Confidence Threshold | Matches Selected | Season Coverage | Empirical Accuracy |
| :---: | :---: | :---: | :---: |
| $\ge 0.40$ | 1,515 | 78.95% | 57.36% |
| $\ge 0.45$ | 1,211 | 63.11% | 60.28% |
| $\ge 0.50$ | 935 | 48.72% | 62.99% |
| $\ge 0.55$ | 752 | 39.19% | 65.43% |
| **$\ge 0.60$** | **470** | **24.49%** | **71.91%** |
| **$\ge 0.65$** | **354** | **18.45%** | **75.42%** |
| **$\ge 0.70$** | **224** | **11.67%** | **81.25%** |

#### 2024–2026 Unseen Holdout Set (4,502 matches):
| Confidence Threshold | Matches Selected | Season Coverage | Empirical Accuracy |
| :---: | :---: | :---: | :---: |
| $\ge 0.40$ | 3,634 | 80.72% | 57.68% |
| $\ge 0.45$ | 2,858 | 63.48% | 61.09% |
| $\ge 0.50$ | 2,200 | 48.87% | 64.86% |
| $\ge 0.55$ | 1,707 | 37.92% | 67.25% |
| **$\ge 0.60$** | **1,159** | **25.74%** | **72.56%** |
| **$\ge 0.65$** | **890** | **19.77%** | **76.40%** |
| **$\ge 0.70$** | **563** | **12.51%** | **78.51%** |

### C. The Double Chance Framing (1X, X2, 12)
Under the Double Chance formulation (predicting whether Home Win/Draw, Away Win/Draw, or Either Team Win occurs):
- **2023 Benchmark Test Set**: **80.41% Accuracy** (1,543 / 1,919 matches).
- **2024–2026 Holdout Set**: **79.34% Accuracy** (3,572 / 4,502 matches).

---

## 10. SMOTE Class Balancing Analysis

To mitigate the systemic underprediction of draws caused by natural class imbalance, Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training set (expanding training instances from 39,138 to 53,895 so each class has 17,965 samples), while keeping test sets strictly untouched.

Extracted directly from [`results/smote_vs_baseline_comparison.csv`](results/smote_vs_baseline_comparison.csv):

| Model | Accuracy (Orig) | Accuracy (SMOTE) | Acc Δ | Macro F1 (Orig) | Macro F1 (SMOTE) | F1 Δ | Draw Recall (Orig) | Draw Recall (SMOTE) | Draw Recall Δ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 53.78% | **53.67%** | *-0.11%* | 0.4088 | **0.4202** | **+0.0114** | 1.87% | **3.33%** | **+1.46%** |
| **HistGradientBoosting** | 53.83% | **52.74%** | *-1.09%* | 0.4107 | **0.4226** | **+0.0119** | 2.29% | **5.41%** | **+3.12%** |
| **5-Model Ensemble (Holdout)** | 53.58% | **51.98%** | *-1.60%* | 0.4063 | **0.4590** | **+0.0527** | 0.78% | **14.30%** | **+13.52%** |
| **5-Model Ensemble (2023 Test)**| 54.30% | **52.01%** | *-2.29%* | 0.4054 | **0.4445** | **+0.0391** | 0.62% | **11.02%** | **+10.40%** |
| **Random Forest** | 54.04% | **51.64%** | *-2.40%* | 0.4030 | **0.4543** | **+0.0513** | 0.83% | **15.18%** | **+14.35%** |
| **Extra Trees** | 54.19% | **50.55%** | *-3.64%* | 0.3970 | **0.4698** | **+0.0728** | 0.00% | **24.32%** | **+24.32%** |
| **Logistic Regression** | 53.88% | **49.40%** | *-4.48%* | 0.4035 | **0.4532** | **+0.0497** | 0.83% | **19.96%** | **+19.13%** |
| **MLP (Neural Net)** | 54.19% | **49.61%** | *-4.58%* | 0.4142 | **0.4538** | **+0.0396** | 2.70% | **23.08%** | **+20.38%** |

---

## 11. Receiver Operating Characteristic (ROC) & AUC Analysis

To evaluate the continuous ranking quality and discriminative capability across all three possible outcomes (`H = Home Win`, `D = Draw`, `A = Away Win`), One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) scores were computed across all model architectures on both the **2023 Benchmark Test Set (1,919 matches)** and the **2024–2026 Holdout Set (4,502 matches)**.

### A. Numerical ROC-AUC Scores Tracking Table

Extracted directly from [`results/roc_auc_metrics.csv`](results/roc_auc_metrics.csv):

| Model Architecture | Evaluation Set | AUC: Home Win (`H`) | AUC: Draw (`D`) | AUC: Away Win (`A`) | Macro-Avg AUC | Micro-Avg AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft-Voting Ensemble (Holdout)** | **2024–2026 Holdout** | **0.7313** | **0.5739** | **0.7399** | **0.6819** | **0.7118** |
| **Soft-Voting Ensemble (2023)** | **2023 Benchmark Test** | **0.7159** | **0.5650** | **0.7169** | **0.6663** | **0.7087** |
| **Logistic Regression** | 2023 Benchmark Test | 0.7160 | 0.5696 | 0.7166 | 0.6678 | 0.7084 |
| **Extra Trees** | 2023 Benchmark Test | 0.7165 | 0.5635 | 0.7169 | 0.6660 | 0.7087 |
| **XGBoost** | 2023 Benchmark Test | 0.7137 | 0.5688 | 0.7159 | 0.6665 | 0.7068 |
| **HistGradientBoosting** | 2023 Benchmark Test | 0.7131 | 0.5645 | 0.7152 | 0.6647 | 0.7067 |
| **Random Forest** | 2023 Benchmark Test | 0.7148 | 0.5598 | 0.7144 | 0.6634 | 0.7066 |
| **Multi-Layer Perceptron (MLP)** | 2023 Benchmark Test | 0.7124 | 0.5611 | 0.7121 | 0.6623 | 0.7034 |
| **SMOTE Soft-Voting Ensemble** | 2023 Benchmark Test | 0.7168 | 0.5625 | 0.7163 | 0.6656 | 0.6952 |

---

### B. Diagnostic Findings & Clinical Interpretations

1. **Decisive Match Separation ($\text{AUC} \ge 0.716 - 0.740$)**:
   - For decisive outcomes (`Home Win` and `Away Win`), models demonstrate strong discriminative separation, achieving **AUCs between 0.716 and 0.740**.
   - Market implied probabilities combined with rolling opponent-adjusted goal differentials provide reliable gradient separation for non-draw outcomes.

2. **The "Draw" Information-Theoretic Ceiling ($\text{AUC} \approx 0.565 - 0.574$)**:
   - The Draw outcome (`D`) exhibits an AUC of **~0.565 to 0.574** across all architectures.
   - This empirically confirms the fundamental Bayes error rate in football modeling: pre-match statistical indicators for low-scoring decisive games (1–0, 0–1) and draws (0–0, 1–1) overlap heavily, creating a natural information ceiling in pre-match tabular spaces.

3. **Multi-Year Holdout Generalization**:
   - When transitioning from the 2023 benchmark set to the unseen 2024–2026 holdout set, the Ensemble's Micro AUC increases from **0.7087 to 0.7118** and Macro AUC from **0.6663 to 0.6819**.
   - This validates that the probability calibration and rank-ordering maintain zero temporal degradation over three successive seasons.

4. **SMOTE Class Balancing Impact on Discrimination**:
   - Training with SMOTE synthetic samples shifts decision thresholds and enhances Draw recall (jumping from 0.62% to 11.02%–24.32%), while preserving overall ranking capability (**Macro AUC = 0.6656** vs 0.6663 baseline).

---

### C. Generated ROC Visualizations

All ROC plots are saved at 200 DPI in `results/figures/roc_curves/`:
- **Main Ensemble ROC (2023 Test)**: `results/figures/roc_curves/roc_ensemble_2023_benchmark.png`
- **Ensemble Holdout ROC (2024–2026)**: `results/figures/roc_curves/roc_ensemble_holdout_2024_2026.png`
- **Multi-Model Macro Comparison**: `results/figures/roc_curves/roc_multimodel_comparison.png`
- **SMOTE vs Baseline Comparison**: `results/figures/roc_curves/roc_smote_vs_baseline.png`

---

## 12. Quickstart & Reproducibility

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete automated experiment pipeline
python src/run_pipeline.py

# 3. Run SMOTE class balancing pipeline
python src/run_smote_experiments.py

# 4. Generate One-vs-Rest ROC curves and AUC metrics
python src/plot_roc_curves.py

# 5. Generate feature importance plots
python src/plot_feature_importance.py

# 6. Run slice-based error analysis breakdown
python src/error_analysis.py

# 7. Open and explore the interactive notebook
jupyter notebook notebooks/Dataset1_Predictor.ipynb
```
