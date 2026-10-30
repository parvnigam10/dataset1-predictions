"""Generate the README result section directly from one run's exported artifacts."""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.artifacts import RESULTS, load_metadata, load_predictions, holdout_label

START, END = '<!-- BEGIN GENERATED RESULTS -->', '<!-- END GENERATED RESULTS -->'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def render_results():
    meta = load_metadata()
    results = pd.read_csv(RESULTS / 'experiment_results.csv')
    score = lambda v: f'{v:.4f}'
    pct = lambda v: f'{v * 100:.2f}%'
    sections = ['## Verified run and data coverage', '',
                f"Input SHA-256: `{meta['dataset']['sha256']}`. The run contains **{meta['dataset']['raw_matches']:,} raw fixtures** and **{meta['dataset']['matches']:,} selected five-league fixtures**.", '',
                table(['Partition', 'First fixture', 'Last fixture', 'Matches'],
                      [[k, v['start'], v['end'], f"{v['matches']:,}"] for k, v in meta['partitions'].items()]), '',
                'These dates describe the available snapshot, not a promised calendar-year endpoint. Full provenance: [run metadata](results/run_metadata.json). Artifact hashes: [manifest](results/artifact_manifest.json).', '',
                '## 2023 feature progression', '',
                'Model families change during the progression, so this is not a controlled feature-only ablation.', '',
                table(['Experiment', 'Columns', 'Accuracy', 'Macro F1', 'Draw recall'],
                      [[r.Experiment, r.Num_Features, pct(r.Accuracy), score(r.Macro_F1), pct(r.Draw_Recall)]
                       for r in results.iloc[:10].itertuples()]), '',
                '## E9 model benchmark', '',
                table(['Model', 'Accuracy', 'Macro F1', 'Home recall', 'Draw recall', 'Away recall'],
                      [[r.Model, pct(r.Accuracy), score(r.Macro_F1), pct(r.Home_Recall), pct(r.Draw_Recall), pct(r.Away_Recall)]
                       for r in results[results.Experiment.str.startswith('E9_') & ~results.Experiment.eq('E9_Expanded_Market_Fatigue_extra_trees')].itertuples()]), '',
                '## Ensemble and SMOTE results', '',
                table(['Experiment', 'Evaluation', 'Accuracy', 'Macro F1', 'Draw recall'],
                      [[r.Experiment, r.Evaluation_Set, pct(r.Accuracy), score(r.Macro_F1), pct(r.Draw_Recall)]
                       for r in results[results.Experiment.isin(['Ensemble_5Models_E9', 'Ensemble_5Models_DrawCalibrated',
                           'Holdout_Ensemble', 'SMOTE_Ensemble_5Models_E9', 'SMOTE_Holdout_Ensemble'])].itertuples()]), '',
                'Detailed baseline/SMOTE comparisons: [CSV](results/smote_vs_baseline_comparison.csv).', '',
                '## XGBoost draw multiplier sweep', '',
                table(['Multiplier', 'Accuracy', 'Macro F1', 'Draw recall'],
                      [[r.Experiment.rsplit('_', 1)[1], pct(r.Accuracy), score(r.Macro_F1), pct(r.Draw_Recall)]
                       for r in results[results.Experiment.str.startswith('Draw_Calib_XGB_')].itertuples()]), '',
                '## Standard ensemble confusion matrix (2023)', '']
    pred = load_predictions('Ensemble_5Models_E9')
    cm = pd.crosstab(pred.target, pred.pred_target).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    sections += [table(['Actual / predicted', 'H', 'D', 'A'], [[name, *cm.iloc[i].tolist()] for i, name in enumerate(['H', 'D', 'A'])]), '',
                 f"There are {len(pred):,} fixtures and {int(pred.is_correct.sum()):,} correct predictions. Error analysis and ROC evaluation use these same prediction exports.", '',
                 '## Confidence filtering', '',
                 table(['Evaluation', 'Confidence at least', 'Selected fixtures', 'Coverage', 'Selected accuracy'],
                       [[r.Evaluation_Set, f'{r.confidence_threshold:.2f}', int(r.matches), f'{r.coverage_pct:.2f}%', f'{r.accuracy_pct:.2f}%']
                        for r in pd.read_csv(RESULTS / 'confidence_tiers_evaluation.csv').itertuples()]), '',
                 '## One-versus-rest ROC-AUC', '',
                 table(['Model', 'Evaluation', 'Home AUC', 'Draw AUC', 'Away AUC', 'Macro AUC'],
                       [[r.Model, r.Evaluation_Set, score(r.AUC_Home), score(r.AUC_Draw), score(r.AUC_Away), score(r.Macro_AUC)]
                        for r in pd.read_csv(RESULTS / 'roc_auc_metrics.csv').itertuples()]), '',
                 '## Top XGBoost feature importances', '',
                 table(['Feature', 'XGBoost importance', 'Random Forest importance'],
                       [[r.Feature, score(r.XGBoost), score(r.RandomForest)] for r in pd.read_csv(RESULTS / 'feature_importances.csv').head(10).itertuples()]), '',
                 'Importance describes the fitted model and does not establish causation.', '',
                 '## Execution environment', '',
                 ', '.join(f'{name} {version}' for name, version in meta['environment'].items()) + '.', '',
                 'See `results/figures/` for generated confusion matrices, ROC curves and feature-importance charts.']
    return '\n'.join(sections)


def update_readme():
    path = Path('README.md')
    text = path.read_text()
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    path.write_text(before + START + '\n\n' + render_results() + '\n\n' + END + after)


if __name__ == '__main__':
    update_readme()
