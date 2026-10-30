"""Train once and export every result from the same fitted models.

Run from any directory: python src/run_pipeline.py
The raw CSV remains local. Results include its SHA-256 and exact split dates.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Bound numerical parallelism to keep runs practical on laptops and CI.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '4')

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from src.data_loader import load_raw_dataset, clean_and_filter_matches, create_chronological_splits
from src.features import compute_all_features, FEATURE_GROUPS
from src.models import get_model, make_ensemble, tune_draw_threshold, ENSEMBLE_KEYS, ENSEMBLE_WEIGHTS
from src.evaluate import evaluate_predictions, plot_and_save_confusion_matrix

MODELS = [
    ('Logistic Regression', 'logistic_regression', False),
    ('Balanced Logistic Regression', 'logistic_regression', True),
    ('Random Forest', 'random_forest', False),
    ('Extra Trees', 'extra_trees', False),
    ('HistGradientBoosting', 'hist_gbm', False),
    ('XGBoost', 'xgboost', False),
    ('Multi-Layer Perceptron (MLP)', 'mlp', False),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_description(df):
    return {'matches': len(df), 'date_min': df.MatchDate.min().strftime('%Y-%m-%d'),
            'date_max': df.MatchDate.max().strftime('%Y-%m-%d'),
            'class_counts': {k: int(v) for k, v in df.FTResult.value_counts().items()}}


def export_predictions(frame, probs, path):
    out = frame[['MatchDate', 'Division', 'HomeTeam', 'AwayTeam', 'FTResult', 'target', 'EloDiff']].copy()
    out['pred_target'] = probs.argmax(axis=1)
    for index, name in enumerate(('H', 'D', 'A')):
        out[f'pred_prob_{name}'] = probs[:, index]
    out['is_correct'] = (out.target == out.pred_target).astype(int)
    out['MaxProb'] = probs.max(axis=1)
    out.to_csv(path, index=False)


def run_complete_experiments():
    os.chdir(ROOT)
    out = Path('results')
    out.mkdir(exist_ok=True)
    raw_path = Path('data/raw/Matches.csv')
    raw = load_raw_dataset(str(raw_path))
    clean = clean_and_filter_matches(raw)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    clean.to_csv('data/processed/cleaned_matches_5leagues.csv', index=False)
    frame = compute_all_features(clean)
    train, test, holdout = create_chronological_splits(frame)
    if train.empty or test.empty or holdout.empty:
        raise ValueError('The input must contain training, 2023 test, and post-2023 holdout matches.')
    holdout_label = f"{holdout.MatchDate.min():%Y-%m-%d} to {holdout.MatchDate.max():%Y-%m-%d} Holdout"
    print('Split sizes:', len(train), len(test), len(holdout), flush=True)
    print('Holdout:', holdout_label, flush=True)
    records, predictions, metadata, matrix_paths = [], {}, {}, []
    features = FEATURE_GROUPS['E9_Expanded_Market_Fatigue']
    X_train, y_train = train[features].values, train.target.values
    X_test, X_holdout = test[features].values, holdout[features].values

    def record(exp, model_name, group, frame, probs, figure_name, pred=None):
        truth = frame.target.values
        pred = probs.argmax(axis=1) if pred is None else pred
        metrics = evaluate_predictions(truth, pred, probs, exp)
        cm = metrics.pop('Confusion_Matrix')
        metrics.update(Features=group, Model=model_name, Num_Features=len(FEATURE_GROUPS['E9_Expanded_Market_Fatigue' if 'SMOTE' in group else group]),
                       Matches=len(frame), Date_Min=frame.MatchDate.min().strftime('%Y-%m-%d'),
                       Date_Max=frame.MatchDate.max().strftime('%Y-%m-%d'))
        records.append(metrics)
        predictions[exp + '__y'] = truth.astype(np.int8)
        predictions[exp + '__pred'] = np.asarray(pred, dtype=np.int8)
        predictions[exp + '__prob'] = np.asarray(probs, dtype=np.float64)
        metadata[exp] = {'figure': figure_name, 'model': model_name, 'split': '2023 Benchmark Test' if frame is test else holdout_label}
        plot_and_save_confusion_matrix(cm, f'{model_name}\n{metadata[exp]["split"]} (n={len(frame):,})', str(out / 'figures/confusion_matrices' / figure_name))
        matrix_paths.append('figures/confusion_matrices/' + figure_name)
        print(f'{exp}: accuracy={metrics["Accuracy"]:.6f}, macro F1={metrics["Macro_F1"]:.6f}', flush=True)

    # Feature progression; cached E9 Extra Trees also serves the benchmark.
    progression = {}
    for group in FEATURE_GROUPS:
        key = 'logistic_regression' if group.startswith(('E0_', 'E1_', 'E2_')) else 'extra_trees' if group.startswith('E9_') else 'xgboost'
        model = get_model(key)
        model.fit(train[FEATURE_GROUPS[group]].values, y_train)
        probs = model.predict_proba(test[FEATURE_GROUPS[group]].values)
        progression[group] = model
        record(f'{group}_{key}', key, group, test, probs, f'{group}_{key}.png')

    fitted, test_probs = {}, {}
    for name, key, balanced in MODELS:
        # Never let balanced LR overwrite the ordinary LR member.
        storage_key = 'balanced_logistic_regression' if balanced else key
        model = progression['E9_Expanded_Market_Fatigue'] if key == 'extra_trees' else get_model(key, balanced=balanced)
        if key != 'extra_trees':
            model.fit(X_train, y_train)
        fitted[storage_key] = model
        test_probs[storage_key] = model.predict_proba(X_test)
        record('E9_' + name, name, 'E9_Expanded_Market_Fatigue', test, test_probs[storage_key], 'E9_' + storage_key + '.png')

    for mult in (1.0, 1.15, 1.25, 1.35, 1.5):
        probs = test_probs['xgboost']
        record(f'Draw_Calib_XGB_mult_{mult:.2f}', f'XGBoost (draw multiplier {mult:.2f})', 'E9_Expanded_Market_Fatigue', test,
               probs, f'Draw_Calib_mult_{mult:.2f}.png', tune_draw_threshold(probs, mult))

    ensemble = make_ensemble(fitted)
    probs_test, probs_holdout = ensemble.predict_proba(X_test), ensemble.predict_proba(X_holdout)
    record('Ensemble_5Models_E9', 'Five-model ensemble', 'E9_Expanded_Market_Fatigue', test, probs_test, 'Ensemble_5Models_E9.png')
    record('Ensemble_5Models_DrawCalibrated', 'Ensemble (draw multiplier 1.20)', 'E9_Expanded_Market_Fatigue', test, probs_test,
           'Ensemble_5Models_Calibrated.png', tune_draw_threshold(probs_test, 1.20))
    record('Holdout_Ensemble', 'Five-model ensemble', 'E9_Expanded_Market_Fatigue', holdout, probs_holdout, 'Holdout_Ensemble.png')
    export_predictions(test, probs_test, out / 'error_analysis_test2023.csv')
    export_predictions(holdout, probs_holdout, out / 'predictions_holdout.csv')

    # SMOTE is fitted on training rows only; test and holdout stay untouched.
    imputer = SimpleImputer(strategy='median')
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(imputer.fit_transform(X_train), y_train)
    smote_models = {}
    for name, key, balanced in MODELS:
        if balanced:
            continue
        model = get_model(key)
        model.fit(X_smote, y_smote)
        smote_models[key] = model
        record('SMOTE_' + key, name + ' (SMOTE)', 'E9_Expanded_Market_Fatigue (SMOTE)', test,
               model.predict_proba(X_test), 'SMOTE_' + key + '.png')
    smote_ensemble = make_ensemble(smote_models)
    for subset, X, exp, filename in ((test, X_test, 'SMOTE_Ensemble_5Models_E9', 'SMOTE_Ensemble_5Models_E9.png'),
                                     (holdout, X_holdout, 'SMOTE_Holdout_Ensemble', 'SMOTE_Holdout_Ensemble.png')):
        record(exp, 'Five-model ensemble (SMOTE)', 'E9_Expanded_Market_Fatigue (SMOTE)', subset,
               smote_ensemble.predict_proba(X), filename)

    results = pd.DataFrame(records)
    results.to_csv(out / 'experiment_results.csv', index=False)
    results[results.Experiment.str.startswith('SMOTE_')].to_csv(out / 'smote_experiment_results.csv', index=False)
    np.savez_compressed(out / 'predictions.npz', **predictions)
    (out / 'prediction_index.json').write_text(json.dumps(metadata, indent=2) + '\n')
    pd.DataFrame({'Feature': features, 'XGBoost': fitted['xgboost'].named_steps['clf'].feature_importances_,
                  'RandomForest': fitted['random_forest'].named_steps['clf'].feature_importances_}) \
        .sort_values('XGBoost', ascending=False).to_csv(out / 'feature_importances.csv', index=False)

    packages = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'imbalanced-learn', 'matplotlib', 'seaborn', 'joblib', 'threadpoolctl')
    manifest = {
        'run_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
        'platform': platform.platform(), 'numerical_threads': 4, 'estimator_jobs': 4, 'packages': {p: importlib.metadata.version(p) for p in packages},
        'input': {'file': str(raw_path), 'sha256': sha256(raw_path), 'raw_rows': len(raw)},
        'clean': split_description(clean), 'train': split_description(train), 'test': split_description(test),
        'holdout': split_description(holdout), 'holdout_label': holdout_label,
        'leagues': {k: int(v) for k, v in clean.Division.value_counts().items()},
        'feature_group': 'E9_Expanded_Market_Fatigue', 'features': features,
        'ensemble': {'members': list(ENSEMBLE_KEYS), 'weights': list(ENSEMBLE_WEIGHTS), 'balanced_lr': False},
        'smote': {'random_state': 42, 'train_only': True, 'impute_before_resampling': True, 'scaled_before_resampling': False},
        'source_sha256': {str(p): sha256(p) for p in sorted(Path('src').glob('*.py'))},
    }
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # All diagnostics consume these saved predictions; none independently retrain.
    from src.error_analysis import run_error_analysis
    from src.plot_roc_curves import generate_all_roc_curves
    from src.plot_feature_importance import plot_importances
    from src.refresh_results import refresh_results
    run_error_analysis()
    generate_all_roc_curves()
    plot_importances()
    refresh_results()
    outputs = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'}
    (out / 'artifact_hashes.json').write_text(json.dumps(outputs, indent=2) + '\n')
    print('Complete: tables, predictions, figures, README, and research results regenerated.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run_complete_experiments()
