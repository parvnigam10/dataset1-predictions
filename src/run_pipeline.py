"""Train once and export the probabilities used by all downstream diagnostics."""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import pandas as pd
from sklearn.impute import SimpleImputer
from src.data_loader import load_raw_dataset, clean_and_filter_matches, create_chronological_splits
from src.features import compute_all_features, FEATURE_GROUPS
from src.models import (get_model, benchmark_key, build_standard_ensemble, ENSEMBLE_CONFIG,
                        tune_draw_threshold, evaluate_confidence_tiers, evaluate_double_chance)
from src.evaluate import evaluate_predictions, plot_and_save_confusion_matrix
from src.artifacts import RESULTS, save_predictions, write_metadata, holdout_label

BENCHMARKS = [('Logistic Regression', 'logistic_regression', False),
              ('Balanced Logistic Regression', 'logistic_regression', True),
              ('Random Forest', 'random_forest', False), ('Extra Trees', 'extra_trees', False),
              ('HistGradientBoosting', 'hist_gbm', False), ('XGBoost', 'xgboost', False),
              ('Multi-Layer Perceptron (MLP)', 'mlp', False)]
PROGRESSION = [('E0_Baseline', 'logistic_regression'), ('E1_Elo', 'logistic_regression'),
               ('E2_Form', 'logistic_regression'), ('E3_Goals', 'xgboost'), ('E4_Venue', 'xgboost'),
               ('E5_MatchStats', 'xgboost'), ('E6_Odds', 'xgboost'), ('E7_Selected', 'xgboost'),
               ('E8_Paper_Advanced', 'xgboost'), ('E9_Expanded_Market_Fatigue', 'extra_trees')]


def run_complete_experiments(include_smote=True):
    # Check input before removing stale generated outputs.
    raw_path = 'data/raw/Matches.csv'
    raw = load_raw_dataset(raw_path)
    clean = clean_and_filter_matches(raw)
    frame = compute_all_features(clean)
    train, test, holdout = create_chronological_splits(frame)
    if train.empty or test.empty:
        raise ValueError('The input needs training fixtures through 2022 and a nonempty 2023 benchmark')
    RESULTS.mkdir(exist_ok=True)
    for folder in ['predictions', 'figures']:
        shutil.rmtree(RESULTS / folder, ignore_errors=True)
    for old in RESULTS.glob('*.csv'):
        old.unlink()
    for old in RESULTS.glob('*.json'):
        old.unlink()
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    clean.to_csv('data/processed/cleaned_matches_5leagues.csv', index=False)
    features = FEATURE_GROUPS['E9_Expanded_Market_Fatigue']
    meta = write_metadata(raw_path, len(raw), clean, train, test, holdout, ENSEMBLE_CONFIG, features)
    holdout_name = holdout_label(meta)
    x_train, y_train = train[features].to_numpy(), train.target.to_numpy()
    fitted, rows, confidence, double_chance = {}, [], [], []

    def record(experiment, model, data, probs, feature_group='E9_Expanded_Market_Fatigue', predictions=None):
        if data.empty:
            return
        export = save_predictions(experiment, data, probs, predictions)
        metrics = evaluate_predictions(data.target.to_numpy(), export.pred_target.to_numpy(), probs, experiment)
        plot_and_save_confusion_matrix(metrics.pop('Confusion_Matrix'), experiment,
                                      f'results/figures/confusion_matrices/{experiment}.png')
        metrics.update(Features=feature_group, Model=model, Num_Features=len(FEATURE_GROUPS[feature_group]),
                       Evaluation_Set='2023 Benchmark Test' if data.Year.eq(2023).all() else holdout_name)
        rows.append(metrics)
        print(f"{experiment}: accuracy={metrics['Accuracy']:.6f} macro_F1={metrics['Macro_F1']:.6f}", flush=True)

    for feature_group, model_key in PROGRESSION:
        columns = FEATURE_GROUPS[feature_group]
        model = get_model(model_key)
        model.fit(train[columns].to_numpy(), y_train)
        record(f'{feature_group}_{model_key}', model_key, test, model.predict_proba(test[columns].to_numpy()), feature_group)
    for display, model_key, balanced in BENCHMARKS:
        model = get_model(model_key, balanced=balanced)
        model.fit(x_train, y_train)
        fitted[benchmark_key(model_key, balanced)] = model
        record(f'E9_{display}', display, test, model.predict_proba(test[features].to_numpy()))
    xgb_probs = fitted['xgboost'].predict_proba(test[features].to_numpy())
    for multiplier in [1.0, 1.15, 1.25, 1.35, 1.5]:
        record(f'Draw_Calib_XGB_mult_{multiplier:.2f}', f'XGBoost (Draw Multiplier={multiplier:.2f})', test,
               xgb_probs, predictions=tune_draw_threshold(xgb_probs, multiplier))
    ensemble = build_standard_ensemble(fitted)
    for data, name, label in [(test, 'Ensemble_5Models_E9', '2023 Benchmark Test'),
                              (holdout, 'Holdout_Ensemble', holdout_name)]:
        if data.empty:
            continue
        probs = ensemble.predict_proba(data[features].to_numpy())
        record(name, 'Standard five-model ensemble', data, probs)
        for tier in evaluate_confidence_tiers(data.target.to_numpy(), probs):
            confidence.append({**tier, 'Evaluation_Set': label})
        double_chance.append({**evaluate_double_chance(data.target.to_numpy(), probs), 'Evaluation_Set': label})
        if name == 'Ensemble_5Models_E9':
            record('Ensemble_5Models_DrawCalibrated', 'Ensemble (Draw multiplier=1.20)', data, probs,
                   predictions=tune_draw_threshold(probs, 1.20))
    # Extract importances from these exact benchmark fits, without another training run.
    pd.DataFrame({'Feature': features,
                  'XGBoost': fitted['xgboost'].named_steps['clf'].feature_importances_,
                  'RandomForest': fitted['random_forest'].named_steps['clf'].feature_importances_}).sort_values(
                      'XGBoost', ascending=False).to_csv(RESULTS / 'feature_importances.csv', index=False)
    if include_smote:
        from imblearn.over_sampling import SMOTE
        x_smote, y_smote = SMOTE(random_state=42).fit_resample(SimpleImputer(strategy='median').fit_transform(x_train), y_train)
        smote_fitted = {}
        for display, key, balanced in BENCHMARKS:
            if balanced:
                continue
            model = get_model(key)
            model.fit(x_smote, y_smote)
            smote_fitted[key] = model
            record(f'SMOTE_{key}', f'{display} (SMOTE)', test, model.predict_proba(test[features].to_numpy()))
        smote_ensemble = build_standard_ensemble(smote_fitted)
        for data, name in [(test, 'SMOTE_Ensemble_5Models_E9'), (holdout, 'SMOTE_Holdout_Ensemble')]:
            if not data.empty:
                record(name, 'Standard five-model ensemble (SMOTE)', data,
                       smote_ensemble.predict_proba(data[features].to_numpy()))
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / 'experiment_results.csv', index=False)
    results[results.Experiment.str.startswith('SMOTE_')].to_csv(RESULTS / 'smote_experiment_results.csv', index=False)
    pd.DataFrame(confidence).to_csv(RESULTS / 'confidence_tiers_evaluation.csv', index=False)
    pd.DataFrame(double_chance).to_csv(RESULTS / 'double_chance_evaluation.csv', index=False)
    pairs = [(display, f'E9_{display}', f'SMOTE_{key}') for display, key, balanced in BENCHMARKS if not balanced]
    pairs += [('Ensemble (2023)', 'Ensemble_5Models_E9', 'SMOTE_Ensemble_5Models_E9'),
              (holdout_name, 'Holdout_Ensemble', 'SMOTE_Holdout_Ensemble')]
    comparisons = []
    for display, original, smote in pairs:
        a, b = results[results.Experiment == original], results[results.Experiment == smote]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        comparisons.append({'Model': display, 'Accuracy_Orig': round(a.Accuracy * 100, 2),
                            'Accuracy_SMOTE': round(b.Accuracy * 100, 2),
                            'Acc_Delta': round((b.Accuracy - a.Accuracy) * 100, 2),
                            'MacroF1_Orig': round(a.Macro_F1, 4), 'MacroF1_SMOTE': round(b.Macro_F1, 4),
                            'DrawRecall_Orig': round(a.Draw_Recall * 100, 2), 'DrawRecall_SMOTE': round(b.Draw_Recall * 100, 2)})
    pd.DataFrame(comparisons).to_csv(RESULTS / 'smote_vs_baseline_comparison.csv', index=False)
    return meta


if __name__ == '__main__':
    from src.run_all import main
    main()
