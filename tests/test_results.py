"""Regression checks for ensemble identity and cross-artifact consistency."""
import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.evaluate import evaluate_predictions
from src.models import make_ensemble, ENSEMBLE_KEYS, ENSEMBLE_WEIGHTS, tune_draw_threshold

ROOT = Path(__file__).resolve().parents[1]


class FixedModel:
    def __init__(self, probs):
        self.probs = np.array(probs)
    def predict_proba(self, X):
        return np.tile(self.probs, (len(X), 1))


class EnsembleDefinitionTest(unittest.TestCase):
    def test_balanced_lr_cannot_replace_ordinary_lr(self):
        models = {key: FixedModel([.6, .1, .3]) for key in ENSEMBLE_KEYS}
        models['balanced_logistic_regression'] = FixedModel([0, 1, 0])
        ensemble = make_ensemble(models)
        self.assertIs(ensemble.models[-1], models['logistic_regression'])
        self.assertNotIn(models['balanced_logistic_regression'], ensemble.models)
        np.testing.assert_allclose(ensemble.predict_proba([[0]]), [[.6, .1, .3]])
        np.testing.assert_allclose(ensemble.weights, ENSEMBLE_WEIGHTS)


class ArtifactConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = ROOT/'results'
        cls.metrics = pd.read_csv(cls.results/'experiment_results.csv').set_index('Experiment')
        cls.archive = np.load(cls.results/'predictions.npz', allow_pickle=False)
        cls.manifest = json.loads((cls.results/'run_manifest.json').read_text())

    def test_every_metric_recomputes_from_saved_predictions(self):
        self.assertEqual(len(self.metrics), 33)
        self.assertFalse(self.metrics.index.duplicated().any())
        for exp, row in self.metrics.iterrows():
            with self.subTest(experiment=exp):
                y, pred, probs = [self.archive[exp+'__'+suffix] for suffix in ('y','pred','prob')]
                self.assertEqual(len(y), row.Matches)
                self.assertTrue(np.isfinite(probs).all())
                self.assertTrue(((probs >= 0) & (probs <= 1)).all())
                np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
                recomputed = evaluate_predictions(y, pred, probs)
                for key,value in recomputed.items():
                    if key not in ('Experiment','Confusion_Matrix'):
                        self.assertAlmostEqual(value, row[key], places=12, msg=exp+' '+key)
                if exp.startswith('Draw_Calib_XGB_mult_'):
                    expected = tune_draw_threshold(probs, float(exp.rsplit('_',1)[-1]))
                elif exp == 'Ensemble_5Models_DrawCalibrated':
                    expected = tune_draw_threshold(probs, 1.20)
                else:
                    expected = probs.argmax(axis=1)
                np.testing.assert_array_equal(pred, expected)

    def test_ensemble_probabilities_use_ordinary_lr_and_shared_weights(self):
        names = ['Extra Trees', 'XGBoost', 'HistGradientBoosting', 'Random Forest', 'Logistic Regression']
        expected = sum(w * self.archive['E9_' + name + '__prob'] for w, name in zip(ENSEMBLE_WEIGHTS, names))
        np.testing.assert_allclose(self.archive['Ensemble_5Models_E9__prob'], expected, atol=1e-14)

    def test_exported_fixture_predictions_match_ensemble(self):
        for filename,exp,split in [('error_analysis_test2023.csv','Ensemble_5Models_E9','test'),('predictions_holdout.csv','Holdout_Ensemble','holdout')]:
            df = pd.read_csv(self.results/filename)
            self.assertEqual(len(df), self.manifest[split]['matches'])
            np.testing.assert_array_equal(df.target, self.archive[exp+'__y'])
            np.testing.assert_array_equal(df.pred_target, self.archive[exp+'__pred'])
            np.testing.assert_allclose(df[['pred_prob_H','pred_prob_D','pred_prob_A']], self.archive[exp+'__prob'], atol=1e-14)
            self.assertEqual(pd.to_datetime(df.MatchDate).max().strftime('%Y-%m-%d'), self.manifest[split]['date_max'])

    def test_confidence_counts_and_accuracy_recompute(self):
        for _,r in pd.read_csv(self.results/'confidence_tiers_evaluation.csv').iterrows():
            y, probs = self.archive[r.Experiment+'__y'], self.archive[r.Experiment+'__prob']
            mask = probs.max(axis=1) >= r.confidence_threshold
            correct = int((probs.argmax(axis=1)[mask] == y[mask]).sum())
            self.assertEqual(mask.sum(),r.matches)
            self.assertEqual(correct,r.correct_matches)
            self.assertAlmostEqual(r.coverage_pct, 100*mask.mean(), places=10)
            self.assertAlmostEqual(r.accuracy_pct,100*correct/mask.sum(),places=10)

    def test_auc_uses_same_probability_arrays(self):
        aucs = pd.read_csv(self.results/'roc_auc_metrics.csv')
        for _,r in aucs.iterrows():
            if r.Model == 'Soft-Voting Ensemble (5 Models)':
                exp='Ensemble_5Models_E9' if r.Evaluation_Set=='2023 Benchmark Test' else 'Holdout_Ensemble'
            elif r.Model=='SMOTE Soft-Voting Ensemble':
                exp='SMOTE_Ensemble_5Models_E9'
            else:
                exp='E9_'+r.Model
            y,probs=self.archive[exp+'__y'],self.archive[exp+'__prob']
            for c,name in enumerate(('AUC_Home','AUC_Draw','AUC_Away')):
                self.assertAlmostEqual(r[name],roc_auc_score(y==c,probs[:,c]),places=12)

    def test_documentation_uses_current_run(self):
        for filename in ('README.md','RESEARCH_PAPER.md'):
            text=(ROOT/filename).read_text()
            self.assertIn(self.manifest['input']['sha256'],text)
            self.assertIn(f"{self.metrics.loc['Ensemble_5Models_E9'].Accuracy*100:.2f}%",text)
            self.assertIn(self.manifest['holdout']['date_max'],text)
            self.assertNotIn('71.61%',text)
            self.assertNotIn('54.35%',text)
        notebook=json.loads((ROOT/'notebooks/Dataset1_Predictor.ipynb').read_text())
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        self.assertNotIn('.fit(',code)

    def test_artifact_and_source_hashes(self):
        for path, expected in json.loads((self.results/'artifact_hashes.json').read_text()).items():
            self.assertEqual(hashlib.sha256((self.results/path).read_bytes()).hexdigest(),expected,path)
        for path, expected in self.manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected,path)


if __name__=='__main__':
    unittest.main()
