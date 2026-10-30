"""Generate a result-viewing notebook using the canonical run's artifacts."""
from pathlib import Path
import nbformat as nbf


def create_dataset1_notebook():
    root = Path(__file__).resolve().parents[1]
    cells = [nbf.v4.new_markdown_cell('''# Football Match Prediction — Verified Results

This notebook reads the same recorded artifacts as the README and research report. It does not fit an independent ensemble. Run `python src/run_pipeline.py` from the repository root to regenerate all experiments. Inspect the manifest for input hash, split dates, and software versions before comparing results.
'''), nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image
root = Path.cwd() if (Path.cwd() / 'results').exists() else Path.cwd().parent
results = root / 'results'
manifest = json.loads((results / 'run_manifest.json').read_text())
display(manifest)
''')]
    for title, filename in [('Model metrics','experiment_results.csv'),('Confidence versus coverage','confidence_tiers_evaluation.csv'),
                            ('SMOTE comparison','smote_vs_baseline_comparison.csv'),('ROC-AUC','roc_auc_metrics.csv'),
                            ('League slices','league_slices.csv'),('Empirical Elo quartiles','elo_slices.csv')]:
        cells.extend([nbf.v4.new_markdown_cell('## '+title),nbf.v4.new_code_cell(f"display(pd.read_csv(results / '{filename}'))")])
    cells.extend([nbf.v4.new_markdown_cell('''## Inspect predictions

Class order is H=0, D=1, A=2. The CSVs expose actual outcomes, predicted outcomes, and all three probabilities. The compressed archive stores these arrays for every experiment. Draw-multiplier experiments intentionally use a modified decision rule and retain original probabilities.
'''),nbf.v4.new_code_cell('''display(pd.read_csv(results / 'error_analysis_test2023.csv').head())
archive = np.load(results / 'predictions.npz', allow_pickle=False)
y = archive['Ensemble_5Models_E9__y']
pred = archive['Ensemble_5Models_E9__pred']
print('2023 accuracy from saved decisions:', float((y == pred).mean()))
''')])
    for filename in ['summary/benchmark_comparison.png','summary/confusion_2023_normalized.png','summary/confusion_holdout_normalized.png',
                     'summary/accuracy_coverage.png','roc_curves/roc_ensemble_2023_benchmark.png','roc_curves/roc_ensemble_holdout.png',
                     'feature_importance/xgboost_top15.png']:
        cells.append(nbf.v4.new_code_cell(f"display(Image(filename=str(results / 'figures/{filename}')))") )
    cells.append(nbf.v4.new_markdown_cell('''## Interpretation

Accuracy is the share of correct H/D/A predictions. Macro F1 averages class-specific F1 equally; draw failures therefore matter even when overall accuracy appears reasonable. Confidence-selected accuracy applies only to accepted fixtures, and double chance accepts two outcomes. Neither is directly comparable with full-population 3-way accuracy. The current data ends in 2025; this run supplies no 2026 holdout evidence. See the generated research report for the complete interpretation and limitations.
'''))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    (root/'notebooks').mkdir(exist_ok=True)
    nbf.write(nb,root/'notebooks/Dataset1_Predictor.ipynb')


if __name__=='__main__':
    create_dataset1_notebook()
