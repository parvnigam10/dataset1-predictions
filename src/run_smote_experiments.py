"""
run_smote_experiments.py
Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the training set
classes (Home Win, Draw, Away Win) while preserving all original data and past experiment records.

- Keeps raw and processed match datasets intact.
- Preserves all 26 existing non-SMOTE experiment results in results/experiment_results.csv.
- Evaluates SMOTE-trained models on:
    * Standardized 2023 Benchmark Test Set (1,919 matches)
    * Multi-Year 2024-2026 Unseen Holdout Set (4,502 matches)
- Generates side-by-side comparison table: Without SMOTE vs With SMOTE.
- Saves SMOTE confusion matrices to results/figures/confusion_matrices/SMOTE_*.png.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src.data_loader import create_chronological_splits
from src.features import compute_all_features, FEATURE_GROUPS
from src.models import (
    get_model, SoftVotingEnsemble, evaluate_confidence_tiers, evaluate_double_chance
)
from src.evaluate import (
    evaluate_predictions, plot_and_save_confusion_matrix, print_evaluation_summary
)


def run_smote_pipeline():
    print("=" * 80)
    print("FOOTBALL MATCH OUTCOME PREDICTION --- SMOTE CLASS BALANCING EXPERIMENT")
    print("=" * 80)
    
    # 1. Load existing cleaned data (keeping original files untouched)
    clean_csv_path = 'data/processed/cleaned_matches_5leagues.csv'
    if not os.path.exists(clean_csv_path):
        raise FileNotFoundError(f"Cleaned matches file not found at {clean_csv_path}")
        
    print(f"\n[Step 1] Loading existing cleaned dataset from {clean_csv_path}...")
    clean_df = pd.read_csv(clean_csv_path, low_memory=False)
    clean_df['MatchDate'] = pd.to_datetime(clean_df['MatchDate'])
    print(f"Total historical matches: {len(clean_df):,}")
    
    # 2. Compute leakage-free features
    print("\n[Step 2] Computing pre-match feature set...")
    feat_df = compute_all_features(clean_df)
    
    # 3. Create chronological train / test / holdout split
    print("\n[Step 3] Chronological splitting (preserving untouched test & holdout sets)...")
    train_df, test_df, holdout_df = create_chronological_splits(
        feat_df, train_end_year=2022, test_year=2023, holdout_after_test=True
    )
    print(f"  Training Set (2000-2022) : {len(train_df):,} matches")
    print(f"  Benchmark Test (2023)    : {len(test_df):,} matches")
    print(f"  Holdout Set (2024-2026)  : {len(holdout_df):,} matches")
    
    benchmark_features = FEATURE_GROUPS['E9_Expanded_Market_Fatigue']
    X_train_raw = train_df[benchmark_features].values
    y_train_raw = train_df['target'].values
    
    X_test = test_df[benchmark_features].values
    y_test = test_df['target'].values
    
    X_holdout = holdout_df[benchmark_features].values
    y_holdout = holdout_df['target'].values
    
    # 4. Impute training features and apply SMOTE
    print("\n[Step 4] Applying SMOTE Balancing on Training Set (2000-2022)...")
    target_names = {0: 'Home Win (0)', 1: 'Draw (1)', 2: 'Away Win (2)'}
    
    counts_before = Counter(y_train_raw)
    print("  Pre-SMOTE Training Class Counts:")
    for cls_idx in sorted(counts_before.keys()):
        print(f"    {target_names[cls_idx]}: {counts_before[cls_idx]:,} matches ({counts_before[cls_idx]/len(y_train_raw)*100:.2f}%)")
        
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_raw)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_imp, y_train_raw)
    
    counts_after = Counter(y_train_smote)
    print("\n  Post-SMOTE Training Class Counts:")
    for cls_idx in sorted(counts_after.keys()):
        print(f"    {target_names[cls_idx]}: {counts_after[cls_idx]:,} synthetic/real samples ({counts_after[cls_idx]/len(y_train_smote)*100:.2f}%)")
    print(f"  Total balanced training samples: {len(y_train_smote):,} (added {len(y_train_smote) - len(y_train_raw):,} synthetic minority samples)")

    # Optionally persist SMOTE training data to separate file (keeping older data untouched)
    smote_df = pd.DataFrame(X_train_smote, columns=benchmark_features)
    smote_df['target'] = y_train_smote
    smote_data_path = 'data/processed/train_smote_E9_features.csv'
    smote_df.to_csv(smote_data_path, index=False)
    print(f"  Saved resampled training data to {smote_data_path} (older data files preserved intact).")

    # 5. Train benchmark models on SMOTE-balanced data
    print("\n" + "=" * 80)
    print("TRAINING & EVALUATING MODELS WITH SMOTE BALANCING (2023 Benchmark Test)")
    print("=" * 80)
    
    models_to_test = [
        ('Logistic Regression', 'logistic_regression'),
        ('Random Forest', 'random_forest'),
        ('Extra Trees', 'extra_trees'),
        ('HistGradientBoosting', 'hist_gbm'),
        ('XGBoost', 'xgboost'),
        ('Multi-Layer Perceptron (MLP)', 'mlp'),
    ]
    
    os.makedirs('results/figures/confusion_matrices', exist_ok=True)
    smote_results = []
    fitted_smote_models = {}
    smote_probs_test = {}
    
    for display_name, m_key in models_to_test:
        exp_name = f"SMOTE_{m_key}"
        print(f"\nTraining {display_name} with SMOTE...")
        
        model = get_model(m_key, balanced=False)
        model.fit(X_train_smote, y_train_smote)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        fitted_smote_models[m_key] = model
        smote_probs_test[m_key] = y_prob
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob, experiment_name=exp_name)
        metrics['Features'] = 'E9_Expanded_Market_Fatigue (SMOTE)'
        metrics['Model'] = f"{display_name} (SMOTE)"
        metrics['Num_Features'] = len(benchmark_features)
        print_evaluation_summary(metrics)
        
        cm_path = f"results/figures/confusion_matrices/SMOTE_{m_key}.png"
        plot_and_save_confusion_matrix(metrics['Confusion_Matrix'], f"SMOTE: {display_name}", cm_path)
        
        row_dict = {k: v for k, v in metrics.items() if k != 'Confusion_Matrix'}
        smote_results.append(row_dict)

    # 6. Soft-Voting Ensemble on SMOTE Models
    print("\n" + "=" * 80)
    print("EVALUATING SMOTE SOFT-VOTING ENSEMBLE (2023 Benchmark Test)")
    print("=" * 80)
    
    smote_ensemble = SoftVotingEnsemble(
        models=[
            fitted_smote_models['extra_trees'],
            fitted_smote_models['xgboost'],
            fitted_smote_models['hist_gbm'],
            fitted_smote_models['random_forest'],
            fitted_smote_models['logistic_regression']
        ],
        weights=[0.35, 0.20, 0.20, 0.15, 0.10]
    )
    
    y_pred_ens = smote_ensemble.predict(X_test)
    y_prob_ens = smote_ensemble.predict_proba(X_test)
    
    ens_metrics = evaluate_predictions(y_test, y_pred_ens, y_prob_ens, experiment_name="SMOTE_Ensemble_5Models_E9")
    ens_metrics['Features'] = 'E9_Expanded_Market_Fatigue (SMOTE)'
    ens_metrics['Model'] = "Soft Voting Ensemble (SMOTE)"
    ens_metrics['Num_Features'] = len(benchmark_features)
    print_evaluation_summary(ens_metrics)
    
    cm_path = "results/figures/confusion_matrices/SMOTE_Ensemble_5Models_E9.png"
    plot_and_save_confusion_matrix(ens_metrics['Confusion_Matrix'], "SMOTE Soft-Voting Ensemble", cm_path)
    
    row_dict = {k: v for k, v in ens_metrics.items() if k != 'Confusion_Matrix'}
    smote_results.append(row_dict)

    # Confidence tiers on SMOTE test
    conf_tiers_test = evaluate_confidence_tiers(y_test, y_prob_ens)
    conf_df_test = pd.DataFrame(conf_tiers_test)
    conf_df_test['Evaluation_Set'] = '2023 Benchmark Test (SMOTE)'
    print("\nSMOTE Ensemble Selective Prediction Confidence Tiers (2023 Test):")
    print(conf_df_test.to_string(index=False))

    # Double chance on SMOTE test
    dc_test = evaluate_double_chance(y_test, y_prob_ens)
    print(f"\nDouble Chance (1X, X2, 12) Test Accuracy: {dc_test['accuracy_pct']}% ({dc_test['correct_matches']}/{dc_test['total_matches']})")

    # 7. Evaluate on Unseen Holdout Set (2024-2026)
    print("\n" + "=" * 80)
    print("EVALUATING SMOTE ENSEMBLE ON UNSEEN HOLDOUT SET (2024 - 2026)")
    print("=" * 80)
    
    y_pred_ho = smote_ensemble.predict(X_holdout)
    y_prob_ho = smote_ensemble.predict_proba(X_holdout)
    
    ho_metrics = evaluate_predictions(y_holdout, y_pred_ho, y_prob_ho, experiment_name="SMOTE_Holdout_2024_2026_Ensemble")
    ho_metrics['Features'] = 'E9_Expanded_Market_Fatigue (SMOTE)'
    ho_metrics['Model'] = "Soft Voting Ensemble (SMOTE 2024-2026 Holdout)"
    ho_metrics['Num_Features'] = len(benchmark_features)
    print_evaluation_summary(ho_metrics)
    
    cm_path = "results/figures/confusion_matrices/SMOTE_Holdout_2024_2026_Ensemble.png"
    plot_and_save_confusion_matrix(ho_metrics['Confusion_Matrix'], "SMOTE Holdout (2024-2026) Ensemble", cm_path)
    
    row_dict = {k: v for k, v in ho_metrics.items() if k != 'Confusion_Matrix'}
    smote_results.append(row_dict)

    # Confidence tiers on holdout
    conf_tiers_ho = evaluate_confidence_tiers(y_holdout, y_prob_ho)
    conf_df_ho = pd.DataFrame(conf_tiers_ho)
    conf_df_ho['Evaluation_Set'] = '2024-2026 Holdout (SMOTE)'

    # 8. Preserve Existing Results & Append SMOTE Results
    print("\n" + "=" * 80)
    print("UPDATING RESULTS & CREATING COMPARATIVE TABLES (Keeping Older Data Intact)")
    print("=" * 80)
    
    exp_results_path = 'results/experiment_results.csv'
    smote_df_results = pd.DataFrame(smote_results)
    
    if os.path.exists(exp_results_path):
        existing_df = pd.read_csv(exp_results_path)
        print(f"Existing experiment results table contains: {len(existing_df)} experiments.")
        
        # Remove any previous SMOTE runs if re-running, while keeping all older non-SMOTE runs
        older_non_smote_df = existing_df[~existing_df['Experiment'].str.startswith('SMOTE_')].copy()
        print(f"Retaining {len(older_non_smote_df)} original experiments without modification.")
        
        combined_df = pd.concat([older_non_smote_df, smote_df_results], ignore_index=True)
    else:
        combined_df = smote_df_results
        
    combined_df.to_csv(exp_results_path, index=False)
    print(f"Updated {exp_results_path} (Now contains {len(combined_df)} total experiments).")
    
    # Save standalone SMOTE results file
    smote_df_results.to_csv('results/smote_experiment_results.csv', index=False)
    print("Saved standalone SMOTE table to results/smote_experiment_results.csv")
    
    # Build Direct Comparison Table: Without SMOTE vs With SMOTE
    comparison_rows = []
    model_mapping = {
        'Logistic Regression': ('E9_Logistic Regression', 'SMOTE_logistic_regression'),
        'Random Forest': ('E9_Random Forest', 'SMOTE_random_forest'),
        'Extra Trees': ('E9_Extra Trees', 'SMOTE_extra_trees'),
        'HistGradientBoosting': ('E9_HistGradientBoosting', 'SMOTE_hist_gbm'),
        'XGBoost': ('E9_XGBoost', 'SMOTE_xgboost'),
        'Multi-Layer Perceptron (MLP)': ('E9_Multi-Layer Perceptron (MLP)', 'SMOTE_mlp'),
        '5-Model Ensemble (2023 Test)': ('Ensemble_5Models_E9', 'SMOTE_Ensemble_5Models_E9'),
        '5-Model Ensemble (Holdout)': ('Holdout_2024_2026_Ensemble', 'SMOTE_Holdout_2024_2026_Ensemble')
    }
    
    comp_records = combined_df.set_index('Experiment')
    
    for model_name, (orig_exp, smote_exp) in model_mapping.items():
        if orig_exp in comp_records.index and smote_exp in comp_records.index:
            r_orig = comp_records.loc[orig_exp]
            r_smote = comp_records.loc[smote_exp]
            
            comparison_rows.append({
                'Model': model_name,
                'Accuracy_Orig': round(float(r_orig['Accuracy']) * 100, 2),
                'Accuracy_SMOTE': round(float(r_smote['Accuracy']) * 100, 2),
                'Acc_Delta': round((float(r_smote['Accuracy']) - float(r_orig['Accuracy'])) * 100, 2),
                'MacroF1_Orig': round(float(r_orig['Macro_F1']), 4),
                'MacroF1_SMOTE': round(float(r_smote['Macro_F1']), 4),
                'MacroF1_Delta': round(float(r_smote['Macro_F1']) - float(r_orig['Macro_F1']), 4),
                'DrawRecall_Orig': round(float(r_orig['Draw_Recall']) * 100, 2),
                'DrawRecall_SMOTE': round(float(r_smote['Draw_Recall']) * 100, 2),
                'DrawRecall_Delta': round((float(r_smote['Draw_Recall']) - float(r_orig['Draw_Recall'])) * 100, 2),
                'HomeRecall_Orig': round(float(r_orig['Home_Recall']) * 100, 2),
                'HomeRecall_SMOTE': round(float(r_smote['Home_Recall']) * 100, 2),
                'AwayRecall_Orig': round(float(r_orig['Away_Recall']) * 100, 2),
                'AwayRecall_SMOTE': round(float(r_smote['Away_Recall']) * 100, 2),
            })
            
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = 'results/smote_vs_baseline_comparison.csv'
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved Direct Comparison table to {comp_csv_path}")
    print("\n" + "=" * 80)
    print("DIRECT COMPARISON: WITHOUT SMOTE vs WITH SMOTE")
    print("=" * 80)
    print(comp_df[['Model', 'Accuracy_Orig', 'Accuracy_SMOTE', 'Acc_Delta', 'MacroF1_Orig', 'MacroF1_SMOTE', 'DrawRecall_Orig', 'DrawRecall_SMOTE']].to_string(index=False))


if __name__ == '__main__':
    run_smote_pipeline()

