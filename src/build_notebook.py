"""Build a read-only notebook from current exports, without copied result numbers."""
import json
from pathlib import Path


def build_notebook():
    cells = []
    def markdown(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True),
                      'execution_count': None, 'outputs': []})
    markdown('# Football outcome prediction\n\nRun `python src/run_all.py` from the repository root to regenerate every result. This notebook reads those same exports. It does not fit a second ensemble or contain handwritten result values.')
    code("from pathlib import Path\nimport json\nimport pandas as pd\nroot = Path.cwd() if Path('results/run_metadata.json').exists() else Path.cwd().parent\nmetadata = json.loads((root / 'results/run_metadata.json').read_text())\nmetadata")
    markdown('## Model comparison\nAccuracy and draw recall describe different objectives. Macro F1 gives equal weight to Home, Draw and Away.')
    code("results = pd.read_csv(root / 'results/experiment_results.csv')\nresults[['Experiment', 'Accuracy', 'Macro_F1', 'Draw_Recall', 'Evaluation_Set']]")
    markdown('## Canonical ensemble confusion matrix\nRows are actual outcomes and columns are predictions. These predictions also supply ROC curves and error analysis.')
    code("p = pd.read_csv(root / 'results/predictions/Ensemble_5Models_E9.csv')\npd.crosstab(p.target, p.pred_target).reindex(index=[0,1,2], columns=[0,1,2], fill_value=0).rename(index={0:'H',1:'D',2:'A'}, columns={0:'H',1:'D',2:'A'})")
    markdown('## Confidence filtering\nSelected accuracy applies only to accepted fixtures. Read the actual holdout dates from metadata. Threshold sweeps are descriptive and do not establish independent validation or probability calibration.')
    code("pd.read_csv(root / 'results/confidence_tiers_evaluation.csv')")
    markdown('## ROC-AUC and SMOTE\nThese files use the exported probabilities from the same fitted models, rather than independent retraining.')
    code("display(pd.read_csv(root / 'results/roc_auc_metrics.csv'))\ndisplay(pd.read_csv(root / 'results/smote_vs_baseline_comparison.csv'))")
    out = Path(__file__).resolve().parents[1] / 'notebooks/Dataset1_Predictor.ipynb'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                              'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}, indent=1) + '\n')


if __name__ == '__main__':
    build_notebook()
