"""
run_pipeline.py
Runs the complete Dataset 1 experimental pipeline from Phase 1 to Phase 19:
- Loads and cleans Matches.csv
- Filters 5 major leagues across full historical timeline (2000-2025)
- Computes leakage-free pre-match features
- Executes feature group progression:
    E0: Baseline (League indicator)
    E1: + Elo
    E2: + Form
    E3: + Goals
    E4: + Venue
    E5: + Match Stats
    E6: + Odds
    E7: Tuned Selected Features
- Executes Multi-Model Comparison (E8):
    Logistic Regression, Balanced LR, Random Forest, Extra Trees,
    HistGradientBoosting, XGBoost, MLP
- Executes Draw Calibration and Class Imbalance tuning
- Executes Ensemble Model (E9)
- Evaluates on 2023 benchmark test set and holds out 2024-2026 data
- Saves results to results/experiment_results.csv and figures
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from src.data_loader import (
    load_raw_dataset, clean_and_filter_matches, create_chronological_splits
)
from src.features import compute_all_features, FEATURE_GROUPS
from src.models import get_model, SoftVotingEnsemble, tune_draw_threshold
from src.evaluate import (
    evaluate_predictions, plot_and_save_confusion_matrix, print_evaluation_summary
)


def run_complete_experiments():
    print("=" * 80)
    print("FOOTBALL MATCH OUTCOME PREDICTION --- DATASET 1 COMPLETE PIPELINE")
    print("=" * 80)
    
    # 1. Load and Clean Data
    print("\n[Phase 1-3] Loading and cleaning dataset...")
    raw_df = load_raw_dataset('data/raw/Matches.csv')
    print(f"Raw shape: {raw_df.shape}")
    
    clean_df = clean_and_filter_matches(raw_df)
    print(f"Filtered 5 Major Leagues shape: {clean_df.shape}")
    print(f"Date range: {clean_df['MatchDate'].min().strftime('%Y-%m-%d')} to {clean_df['MatchDate'].max().strftime('%Y-%m-%d')}")
    print("Target distribution in filtered set:")
    print(clean_df['FTResult'].value_counts(normalize=True).round(4))
    
    # Save cleaned base matches for reproducibility
    os.makedirs('data/processed', exist_ok=True)
    clean_df.to_csv('data/processed/cleaned_matches_5leagues.csv', index=False)
    print("Saved cleaned matches to data/processed/cleaned_matches_5leagues.csv")
    
    # 2. Compute Leakage-Free Features
    print("\n[Phase 6-12] Computing leakage-free pre-match features across chronological timeline...")
    feat_df = compute_all_features(clean_df)
    print(f"Engineered dataset shape: {feat_df.shape}")
    
    # 3. Create Chronological Train / Test Split
    print("\n[Phase 5] Creating chronological train/test split...")
    train_df, test_df, holdout_df = create_chronological_splits(
        feat_df, train_end_year=2022, test_year=2023, holdout_after_test=True
    )
    print(f"Training set (2000 - 2022) : {len(train_df)} matches")
    print(f"Testing set (2023)         : {len(test_df)} matches (Standard Benchmark Year)")
    print(f"Holdout set (2024 - 2026)  : {len(holdout_df)} matches")
    
    y_train = train_df['target'].values
    y_test = test_df['target'].values
    y_holdout = holdout_df['target'].values
    
    results_list = []
    
    # -------------------------------------------------------------
    # FEATURE GROUP PROGRESSION EXPERIMENTS (E0 -> E7)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING FEATURE GROUP PROGRESSION EXPERIMENTS (E0 to E7)")
    print("=" * 80)
    
    feature_experiments = [
        ('E0_Baseline', 'E0_Baseline', 'logistic_regression', False),
        ('E1_Elo', 'E1_Elo', 'logistic_regression', False),
        ('E2_Form', 'E2_Form', 'logistic_regression', False),
        ('E3_Goals', 'E3_Goals', 'xgboost', False),
        ('E4_Venue', 'E4_Venue', 'xgboost', False),
        ('E5_MatchStats', 'E5_MatchStats', 'xgboost', False),
        ('E6_Odds', 'E6_Odds', 'xgboost', False),
        ('E7_Selected', 'E7_Selected', 'xgboost', False),
        ('E8_Paper_Advanced', 'E8_Paper_Advanced', 'xgboost', False),
        ('E9_Expanded_Market_Fatigue', 'E9_Expanded_Market_Fatigue', 'extra_trees', False),
    ]
    
    for exp_id, feat_key, model_type, balanced in feature_experiments:
        feats = FEATURE_GROUPS[feat_key]
        X_train = train_df[feats].values
        X_test = test_df[feats].values
        
        model = get_model(model_type, balanced=balanced)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob, experiment_name=f"{exp_id}_{model_type}")
        metrics['Features'] = feat_key
        metrics['Model'] = model_type
        metrics['Num_Features'] = len(feats)
        print_evaluation_summary(metrics)
        
        cm_path = f"results/figures/confusion_matrices/{exp_id}_{model_type}.png"
        plot_and_save_confusion_matrix(metrics['Confusion_Matrix'], f"{exp_id} ({model_type})", cm_path)
        
        # Save metrics (excluding matrix for csv)
        row_dict = {k: v for k, v in metrics.items() if k != 'Confusion_Matrix'}
        results_list.append(row_dict)
        
    # -------------------------------------------------------------
    # MULTI-MODEL BENCHMARK COMPARISON
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING MULTI-MODEL BENCHMARK COMPARISON ON EXPANDED FEATURES (E9)")
    print("=" * 80)
    
    # We compare models using the rich leakage-free feature set (E9_Expanded_Market_Fatigue)
    benchmark_features = FEATURE_GROUPS['E9_Expanded_Market_Fatigue']
    X_train_bench = train_df[benchmark_features].values
    X_test_bench = test_df[benchmark_features].values
    
    models_to_compare = [
        ('Logistic Regression', 'logistic_regression', False),
        ('Balanced Logistic Regression', 'logistic_regression', True),
        ('Random Forest', 'random_forest', False),
        ('Extra Trees', 'extra_trees', False),
        ('HistGradientBoosting', 'hist_gbm', False),
        ('XGBoost', 'xgboost', False),
        ('Multi-Layer Perceptron (MLP)', 'mlp', False),
    ]
    
    fitted_models = {}
    model_probs = {}
    
    for display_name, m_key, balanced in models_to_compare:
        print(f"\nTraining {display_name}...")
        model = get_model(m_key, balanced=balanced)
        model.fit(X_train_bench, y_train)
        
        y_pred = model.predict(X_test_bench)
        y_prob = model.predict_proba(X_test_bench)
        
        fitted_models[m_key] = model
        model_probs[m_key] = y_prob
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob, experiment_name=f"E9_{display_name}")
        metrics['Features'] = 'E9_Expanded_Market_Fatigue'
        metrics['Model'] = display_name
        metrics['Num_Features'] = len(benchmark_features)
        print_evaluation_summary(metrics)
        
        cm_path = f"results/figures/confusion_matrices/E9_{m_key}.png"
        plot_and_save_confusion_matrix(metrics['Confusion_Matrix'], f"E9: {display_name}", cm_path)
        
        row_dict = {k: v for k, v in metrics.items() if k != 'Confusion_Matrix'}
        results_list.append(row_dict)
        
    # -------------------------------------------------------------
    # PHASE 15 & 16: DRAW PREDICTION & THRESHOLD CALIBRATION
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DRAW PREDICTION & PROBABILITY THRESHOLD TUNING (Phase 15 & 16)")
    print("=" * 80)
    
    xgb_probs = model_probs['xgboost']
    for mult in [1.0, 1.15, 1.25, 1.35, 1.5]:
        calib_preds = tune_draw_threshold(xgb_probs, draw_multiplier=mult)
        metrics = evaluate_predictions(y_test, calib_preds, xgb_probs, experiment_name=f"Draw_Calib_XGB_mult_{mult:.2f}")
        metrics['Features'] = 'E9_Expanded_Market_Fatigue'
        metrics['Model'] = f"XGBoost (Draw Multiplier={mult:.2f})"
        metrics['Num_Features'] = len(benchmark_features)
        print_evaluation_summary(metrics)
        
        if mult in [1.0, 1.25, 1.5]:
            cm_path = f"results/figures/confusion_matrices/Draw_Calib_mult_{mult:.2f}.png"
            plot_and_save_confusion_matrix(metrics['Confusion_Matrix'], f"Draw Calibration (Mult={mult:.2f})", cm_path)
            
        row_dict = {k: v for k, v in metrics.items() if k != 'Confusion_Matrix'}
        results_list.append(row_dict)
        
    # -------------------------------------------------------------
    # PHASE 19: ENSEMBLE MODEL (E10)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING DIVERSE SOFT-VOTING ENSEMBLE EXPERIMENTS")
    print("=" * 80)
    
    ensemble = SoftVotingEnsemble(
        models=[
            fitted_models['extra_trees'],
            fitted_models['xgboost'],
            fitted_models['hist_gbm'],
            fitted_models['random_forest'],
            fitted_models['logistic_regression']
        ],
        weights=[0.35, 0.20, 0.20, 0.15, 0.10]
    )
    y_pred_ens = ensemble.predict(X_test_bench)
    y_prob_ens = ensemble.predict_proba(X_test_bench)
    
    metrics = evaluate_predictions(y_test, y_pred_ens, y_prob_ens, experiment_name="Ensemble_5Models_E9")
    metrics['Features'] = 'E9_Expanded_Market_Fatigue'
    metrics['Model'] = "Soft Voting Ensemble (ET+XGB+HGB+RF+LR)"
    metrics['Num_Features'] = len(benchmark_features)
    print_evaluation_summary(metrics)
    
    cm_path = "results/figures/confusion_matrices/Ensemble_5Models_E9.png"
    plot_and_save_confusion_matrix(metrics['Confusion_Matrix'], "Ensemble: 5 Models on E9 Features", cm_path)
    
    row_dict = {k: v for k, v in metrics.items() if k != 'Confusion_Matrix'}
    results_list.append(row_dict)
    
    # Also evaluate calibrated ensemble
    y_pred_ens_calib = tune_draw_threshold(y_prob_ens, draw_multiplier=1.20)
    metrics_calib = evaluate_predictions(y_test, y_pred_ens_calib, y_prob_ens, experiment_name="Ensemble_5Models_DrawCalibrated")
    metrics_calib['Features'] = 'E9_Expanded_Market_Fatigue'
    metrics_calib['Model'] = "Calibrated Ensemble (Mult=1.20)"
    metrics_calib['Num_Features'] = len(benchmark_features)
    print_evaluation_summary(metrics_calib)
    
    cm_path = "results/figures/confusion_matrices/Ensemble_5Models_Calibrated.png"
    plot_and_save_confusion_matrix(metrics_calib['Confusion_Matrix'], "Calibrated Ensemble (Draw Mult=1.20)", cm_path)
    
    row_dict = {k: v for k, v in metrics_calib.items() if k != 'Confusion_Matrix'}
    results_list.append(row_dict)
    
    # -------------------------------------------------------------
    # SELECTIVE CLASSIFICATION & CONFIDENCE TIERS (ROAD TO 70%+)
    # -------------------------------------------------------------
    from src.models import evaluate_confidence_tiers, evaluate_double_chance
    print("\n" + "=" * 80)
    print("SELECTIVE PREDICTION: CONFIDENCE-TIER EVALUATION (2023 Test Set)")
    print("=" * 80)
    conf_tiers_test = evaluate_confidence_tiers(y_test, y_prob_ens)
    conf_df_test = pd.DataFrame(conf_tiers_test)
    conf_df_test['Evaluation_Set'] = '2023 Benchmark Test'
    print(conf_df_test.to_string(index=False))
    
    # Double chance evaluation
    dc_test = evaluate_double_chance(y_test, y_prob_ens)
    print(f"\nDouble Chance (1X, X2, 12) Test Accuracy: {dc_test['accuracy_pct']}% ({dc_test['correct_matches']}/{dc_test['total_matches']})")

    # -------------------------------------------------------------
    # UNSEEN HOLDOUT EVALUATION (2024 - 2026)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATING BEST ENSEMBLE ON UNSEEN HOLDOUT PERIOD (2024 - 2026)")
    print("=" * 80)
    X_holdout_bench = holdout_df[benchmark_features].values
    y_pred_holdout = ensemble.predict(X_holdout_bench)
    y_prob_holdout = ensemble.predict_proba(X_holdout_bench)
    
    holdout_metrics = evaluate_predictions(y_holdout, y_pred_holdout, y_prob_holdout, experiment_name="Holdout_2024_2026_Ensemble")
    holdout_metrics['Features'] = 'E9_Expanded_Market_Fatigue'
    holdout_metrics['Model'] = "Soft Voting Ensemble (2024-2026 Holdout)"
    holdout_metrics['Num_Features'] = len(benchmark_features)
    print_evaluation_summary(holdout_metrics)
    
    cm_path = "results/figures/confusion_matrices/Holdout_2024_2026_Ensemble.png"
    plot_and_save_confusion_matrix(holdout_metrics['Confusion_Matrix'], "Holdout (2024-2026) Ensemble", cm_path)
    
    row_dict = {k: v for k, v in holdout_metrics.items() if k != 'Confusion_Matrix'}
    results_list.append(row_dict)
    
    # Holdout confidence tiers
    print("\n" + "=" * 80)
    print("SELECTIVE PREDICTION: CONFIDENCE-TIER EVALUATION ON HOLDOUT (2024-2026)")
    print("=" * 80)
    conf_tiers_holdout = evaluate_confidence_tiers(y_holdout, y_prob_holdout)
    conf_df_holdout = pd.DataFrame(conf_tiers_holdout)
    conf_df_holdout['Evaluation_Set'] = '2024-2026 Holdout'
    print(conf_df_holdout.to_string(index=False))
    
    dc_holdout = evaluate_double_chance(y_holdout, y_prob_holdout)
    print(f"\nDouble Chance (1X, X2, 12) Holdout Accuracy: {dc_holdout['accuracy_pct']}% ({dc_holdout['correct_matches']}/{dc_holdout['total_matches']})")

    # Combine and save confidence tiers
    all_conf_df = pd.concat([conf_df_test, conf_df_holdout], ignore_index=True)
    all_conf_df.to_csv('results/confidence_tiers_evaluation.csv', index=False)
    print("\nSaved confidence tier results to results/confidence_tiers_evaluation.csv")

    # Save overall experiment tracking CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('results/experiment_results.csv', index=False)
    print("\nSaved all experiment results to results/experiment_results.csv")
    print("\nSaved all experiment results to results/experiment_results.csv")
    
    print("\nTop 5 Experiments by Accuracy:")
    print(results_df.sort_values(by='Accuracy', ascending=False)[['Experiment', 'Model', 'Accuracy', 'Macro_F1', 'Draw_Recall']].head(5))
    
    print("\nTop 5 Experiments by Macro F1:")
    print(results_df.sort_values(by='Macro_F1', ascending=False)[['Experiment', 'Model', 'Accuracy', 'Macro_F1', 'Draw_Recall']].head(5))
    

if __name__ == '__main__':
    run_complete_experiments()
