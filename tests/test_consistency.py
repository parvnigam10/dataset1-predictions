"""Regression checks for ensemble identity and committed-result provenance."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.models import (benchmark_key, build_standard_ensemble, ENSEMBLE_KEYS, ENSEMBLE_WEIGHTS,
                        SoftVotingEnsemble, evaluate_confidence_tiers, tune_draw_threshold)
from src.artifacts import load_predictions, load_metadata, PROB_COLUMNS, sha256
from src.evaluate import evaluate_predictions


class ConstantModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
    def predict_proba(self, x):
        return np.tile(self.probabilities, (len(x), 1))


class EnsembleRegressionTests(unittest.TestCase):
    def test_balanced_lr_cannot_replace_ordinary_lr(self):
        models = {key: ConstantModel([0.7, 0.2, 0.1]) for key in ENSEMBLE_KEYS}
        models[benchmark_key('logistic_regression', True)] = ConstantModel([0.0, 1.0, 0.0])
        self.assertNotEqual(benchmark_key('logistic_regression'), benchmark_key('logistic_regression', True))
        np.testing.assert_allclose(build_standard_ensemble(models).predict_proba([[0]]), [[0.7, 0.2, 0.1]])

    def test_canonical_weight_order(self):
        models = {key: ConstantModel([i / 10, 1 - i / 10, 0]) for i, key in enumerate(ENSEMBLE_KEYS)}
        expected = sum(weight * models[key].predict_proba([[0]]) for key, weight in zip(ENSEMBLE_KEYS, ENSEMBLE_WEIGHTS))
        np.testing.assert_allclose(build_standard_ensemble(models).predict_proba([[0]]), expected)

    def test_invalid_ensemble_configuration_fails(self):
        for models, weights in [([], None), ([ConstantModel([1, 0, 0])], [0]),
                                ([ConstantModel([1, 0, 0])], [-1]),
                                ([ConstantModel([1, 0, 0])], [1, 1])]:
            with self.assertRaises(ValueError):
                SoftVotingEnsemble(models, weights)
        with self.assertRaises(ValueError):
            build_standard_ensemble({})


@unittest.skipUnless(Path('results/run_metadata.json').exists(), 'Run the pipeline or use committed results')
class ExportConsistencyTests(unittest.TestCase):
    def test_every_metric_recomputes_from_its_predictions(self):
        for row in pd.read_csv('results/experiment_results.csv').itertuples():
            with self.subTest(experiment=row.Experiment):
                p = load_predictions(row.Experiment)
                probabilities = p[PROB_COLUMNS].to_numpy()
                multiplier = float(row.Experiment.rsplit('_', 1)[1]) if row.Experiment.startswith('Draw_Calib_XGB_') else 1.20 if row.Experiment == 'Ensemble_5Models_DrawCalibrated' else 1.0
                np.testing.assert_array_equal(p.pred_target, tune_draw_threshold(probabilities, multiplier))
                np.testing.assert_allclose(p.MaxProb, probabilities.max(axis=1), rtol=0, atol=1e-14)
                metrics = evaluate_predictions(p.target, p.pred_target, probabilities)
                for key, value in metrics.items():
                    if key not in ['Experiment', 'Confusion_Matrix']:
                        self.assertAlmostEqual(getattr(row, key), value, places=12)

    def test_error_analysis_is_the_same_five_model_predictions(self):
        canonical = load_predictions('Ensemble_5Models_E9')
        error = pd.read_csv('results/error_analysis_test2023.csv', parse_dates=['MatchDate'])
        pd.testing.assert_frame_equal(error, canonical, check_exact=False, atol=1e-14, rtol=1e-14)
        self.assertEqual(pd.read_csv('results/league_evaluation.csv').Matches.sum(), len(canonical))

    def test_confidence_results_recompute(self):
        meta = load_metadata()
        stored = pd.read_csv('results/confidence_tiers_evaluation.csv')
        for key, name in [('Ensemble_5Models_E9', '2023 Benchmark Test'), ('Holdout_Ensemble', None)]:
            if key == 'Holdout_Ensemble' and not meta['partitions']['holdout']['matches']:
                continue
            p = load_predictions(key)
            expected = pd.DataFrame(evaluate_confidence_tiers(p.target.to_numpy(), p[PROB_COLUMNS].to_numpy()))
            actual = stored[stored.Evaluation_Set == name] if name else stored[stored.Evaluation_Set != '2023 Benchmark Test']
            pd.testing.assert_frame_equal(actual[expected.columns].reset_index(drop=True), expected, check_dtype=False)

    def test_roc_results_recompute_from_canonical_exports(self):
        from src.plot_roc_curves import compute_ovr_roc
        for row in pd.read_csv('results/roc_auc_metrics.csv').itertuples():
            if row.Model == 'Soft-Voting Ensemble (5 Models)':
                name = 'Ensemble_5Models_E9' if row.Evaluation_Set == '2023 Benchmark Test' else 'Holdout_Ensemble'
            elif row.Model == 'SMOTE Soft-Voting Ensemble':
                name = 'SMOTE_Ensemble_5Models_E9'
            else:
                name = 'E9_' + row.Model
            p = load_predictions(name)
            _, _, scores = compute_ovr_roc(p.target.to_numpy(), p[PROB_COLUMNS].to_numpy())
            for col, key in [('AUC_Home', 0), ('AUC_Draw', 1), ('AUC_Away', 2), ('Macro_AUC', 'macro'), ('Micro_AUC', 'micro')]:
                self.assertEqual(getattr(row, col), round(scores[key], 4))

    def test_holdout_dates_and_counts_match_metadata(self):
        meta = load_metadata()
        if not meta['partitions']['holdout']['matches']:
            return
        p = load_predictions('Holdout_Ensemble')
        part = meta['partitions']['holdout']
        self.assertEqual(len(p), part['matches'])
        self.assertEqual(str(p.MatchDate.min().date()), part['start'])
        self.assertEqual(str(p.MatchDate.max().date()), part['end'])
        labels = pd.read_csv('results/experiment_results.csv').Evaluation_Set
        self.assertTrue(any(part['start'] in label and part['end'] in label for label in labels))

    def test_readme_is_generated_from_this_run(self):
        from src.update_readme import START, END, render_results
        actual = Path('README.md').read_text().split(START, 1)[1].split(END, 1)[0].strip()
        self.assertEqual(actual, render_results())

    def test_manifest_and_source_provenance(self):
        manifest = json.loads(Path('results/artifact_manifest.json').read_text())
        observed = {str(p) for p in Path('results').rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
        self.assertEqual(set(manifest['sha256']), observed)
        for path, digest in manifest['sha256'].items():
            self.assertEqual(sha256(path), digest, path)
        self.assertEqual(sha256('README.md'), manifest['readme_sha256'])
        for path, digest in load_metadata()['source_sha256'].items():
            self.assertEqual(sha256(path), digest, path)


if __name__ == '__main__':
    unittest.main()
