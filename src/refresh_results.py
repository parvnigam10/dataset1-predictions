"""Render documentation and comparisons exclusively from recorded run outputs."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.models import evaluate_double_chance

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pct(x):
    return f'{x * 100:.2f}%'


def refresh_results():
    out = ROOT / 'results'
    manifest = json.loads((out / 'run_manifest.json').read_text())
    archive = np.load(out / 'predictions.npz', allow_pickle=False)
    metrics = pd.read_csv(out / 'experiment_results.csv')
    index = metrics.set_index('Experiment')
    ens = index.loc['Ensemble_5Models_E9']
    hold = index.loc['Holdout_Ensemble']
    test_count = manifest['test']['matches']
    hold_count = manifest['holdout']['matches']
    hold_label = manifest['holdout_label']
    tiers, chances = [], []
    for exp, label in [('Ensemble_5Models_E9', '2023 Benchmark Test'), ('Holdout_Ensemble', hold_label),
                       ('SMOTE_Ensemble_5Models_E9', '2023 Benchmark Test (SMOTE)'),
                       ('SMOTE_Holdout_Ensemble', hold_label + ' (SMOTE)')]:
        y, probs = archive[exp + '__y'], archive[exp + '__prob']
        pred = probs.argmax(axis=1)
        for threshold in (.40, .45, .50, .55, .60, .65, .70):
            mask = probs.max(axis=1) >= threshold
            selected = int(mask.sum())
            correct = int((pred[mask] == y[mask]).sum())
            tiers.append(dict(Evaluation_Set=label, Experiment=exp, confidence_threshold=threshold,
                matches=selected, correct_matches=correct, total_matches=len(y),
                coverage_pct=100 * selected / len(y), accuracy_pct=100 * correct / selected if selected else np.nan))
        chances.append(dict(Experiment=exp, Evaluation_Set=label, **evaluate_double_chance(y, probs)))
    tiers = pd.DataFrame(tiers)
    tiers.to_csv(out / 'confidence_tiers_evaluation.csv', index=False)
    chances = pd.DataFrame(chances)
    chances.to_csv(out / 'double_chance_evaluation.csv', index=False)
    comparisons = []
    for name, baseline, smote in [(r.Model, 'E9_' + r.Model, 'SMOTE_' + key) for key, r in [
        (key, index.loc['E9_' + name]) for name, key in (
            ('Logistic Regression', 'logistic_regression'), ('Random Forest', 'random_forest'),
            ('Extra Trees', 'extra_trees'), ('HistGradientBoosting', 'hist_gbm'),
            ('XGBoost', 'xgboost'), ('Multi-Layer Perceptron (MLP)', 'mlp'))]] + [
        ('Five-model ensemble (2023)', 'Ensemble_5Models_E9', 'SMOTE_Ensemble_5Models_E9'),
        ('Five-model ensemble (holdout)', 'Holdout_Ensemble', 'SMOTE_Holdout_Ensemble')]:
        a, b = index.loc[baseline], index.loc[smote]
        comparisons.append(dict(Model=name, Accuracy_Orig=a.Accuracy, Accuracy_SMOTE=b.Accuracy,
            Acc_Delta=b.Accuracy-a.Accuracy, MacroF1_Orig=a.Macro_F1, MacroF1_SMOTE=b.Macro_F1,
            MacroF1_Delta=b.Macro_F1-a.Macro_F1, DrawRecall_Orig=a.Draw_Recall,
            DrawRecall_SMOTE=b.Draw_Recall, DrawRecall_Delta=b.Draw_Recall-a.Draw_Recall))
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(out / 'smote_vs_baseline_comparison.csv', index=False)

    # Normalized matrices are derived from the exact same prediction arrays.
    fig_dir = out / 'figures/summary'
    fig_dir.mkdir(parents=True, exist_ok=True)
    for exp, name, label in [('Ensemble_5Models_E9', 'confusion_2023_normalized', '2023 benchmark'),
                              ('Holdout_Ensemble', 'confusion_holdout_normalized', hold_label)]:
        cm = confusion_matrix(archive[exp+'__y'], archive[exp+'__pred'], labels=[0, 1, 2])
        norm = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(norm, vmin=0, vmax=1, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{norm[i,j]:.1%}\n(n={cm[i,j]:,})', ha='center', va='center', color='white' if norm[i,j] > .5 else 'black')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=['H', 'D', 'A'], yticklabels=['H', 'D', 'A'],
               xlabel='Predicted outcome', ylabel='Actual outcome', title=f'Five-model ensemble\n{label} (row-normalized)')
        fig.colorbar(im, ax=ax, label='Share of actual class')
        fig.tight_layout(); fig.savefig(fig_dir / (name+'.png'), dpi=200); plt.close(fig)
    benchmark = metrics[metrics.Experiment.str.startswith('E9_') & metrics.Model.str.contains('Regression|Forest|Trees|Boosting|XGBoost|Perceptron')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = [x.replace('Multi-Layer Perceptron (MLP)', 'MLP').replace('Balanced Logistic Regression', 'Balanced LR').replace('Logistic Regression', 'LR') for x in benchmark.Model]
    axes[0].barh(labels, benchmark.Accuracy*100, color='#276b91'); axes[0].set(xlabel='Accuracy (%)', xlim=(0,65), title='2023 benchmark: accuracy')
    axes[1].barh(labels, benchmark.Macro_F1, color='#b66c2d'); axes[1].set(xlabel='Macro F1', xlim=(0,.6), title='2023 benchmark: class balance')
    for ax, vals, fmt in [(axes[0], benchmark.Accuracy*100, '.2f'), (axes[1], benchmark.Macro_F1, '.4f')]:
        for n,v in enumerate(vals): ax.text(v+.005 if ax is axes[1] else v+.5,n,format(v,fmt),va='center',fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir/'benchmark_comparison.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    for exp, label in [('Ensemble_5Models_E9','2023 benchmark'),('Holdout_Ensemble',hold_label)]:
        sub=tiers[tiers.Experiment==exp]
        ax.plot(sub.coverage_pct,sub.accuracy_pct,'o-',label=label)
        for _,r in sub.iterrows(): ax.annotate(f'{r.confidence_threshold:.2f}',(r.coverage_pct,r.accuracy_pct),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Fixtures selected (%)',ylabel='Accuracy on selected fixtures (%)',title='Confidence filtering: accuracy versus coverage')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(fig_dir/'accuracy_coverage.png',dpi=200); plt.close(fig)

    # Generated Markdown tables. No numeric result is hand-copied into documentation.
    progression = metrics.iloc[:10]
    progression_table = table(['Experiment','Model','Features','Accuracy','Macro F1','Draw recall'],
        [[r.Experiment.split('_')[0],r.Model,r.Num_Features,pct(r.Accuracy),f'{r.Macro_F1:.4f}',pct(r.Draw_Recall)] for _,r in progression.iterrows()])
    benchmark_table = table(['Model','Accuracy','Macro F1','Home recall','Draw recall','Away recall'],
        [[r.Model,pct(r.Accuracy),f'{r.Macro_F1:.4f}',pct(r.Home_Recall),pct(r.Draw_Recall),pct(r.Away_Recall)] for _,r in benchmark.iterrows()])
    ensemble_table = table(['Evaluation','Matches','Accuracy','Macro F1','Draw recall'],
        [[label,int(index.loc[exp].Matches),pct(index.loc[exp].Accuracy),f'{index.loc[exp].Macro_F1:.4f}',pct(index.loc[exp].Draw_Recall)] for exp,label in [
            ('Ensemble_5Models_E9','2023 ensemble'),('Ensemble_5Models_DrawCalibrated','2023 ensemble, draw multiplier 1.20'),
            ('Holdout_Ensemble',hold_label),('SMOTE_Ensemble_5Models_E9','2023 SMOTE ensemble'),('SMOTE_Holdout_Ensemble','Holdout SMOTE ensemble')]])
    draw_table = table(['XGBoost draw multiplier','Accuracy','Macro F1','Draw recall'],
        [[r.Experiment.rsplit('_',1)[-1],pct(r.Accuracy),f'{r.Macro_F1:.4f}',pct(r.Draw_Recall)] for _,r in metrics[metrics.Experiment.str.startswith('Draw_Calib')].iterrows()])
    smote_table = table(['Model','Baseline accuracy','SMOTE accuracy','Baseline Macro F1','SMOTE Macro F1','Baseline draw recall','SMOTE draw recall'],
        [[r.Model,pct(r.Accuracy_Orig),pct(r.Accuracy_SMOTE),f'{r.MacroF1_Orig:.4f}',f'{r.MacroF1_SMOTE:.4f}',pct(r.DrawRecall_Orig),pct(r.DrawRecall_SMOTE)] for _,r in comparisons.iterrows()])
    confidence_tables = ''
    for exp,label in [('Ensemble_5Models_E9','2023 benchmark'),('Holdout_Ensemble',hold_label)]:
        confidence_tables += '\n### '+label+'\n\n'+table(['Minimum confidence','Selected','Correct','Coverage','Selected accuracy'],
            [[f'{r.confidence_threshold:.2f}',int(r.matches),int(r.correct_matches),f'{r.coverage_pct:.2f}%',f'{r.accuracy_pct:.2f}%'] for _,r in tiers[tiers.Experiment==exp].iterrows()])+'\n'
    dc_table = table(['Evaluation','Correct / total','Double-chance accuracy'],[[r.Evaluation_Set,f'{r.correct_matches:,} / {r.total_matches:,}',f'{r.accuracy_pct:.2f}%'] for _,r in chances.iloc[:2].iterrows()])
    aucs = pd.read_csv(out/'roc_auc_metrics.csv')
    auc_table = table(['Model','Evaluation','AUC H','AUC D','AUC A','Macro curve AUC','Micro AUC'],[[r.Model,r.Evaluation_Set,*[f'{r[c]:.4f}' for c in ('AUC_Home','AUC_Draw','AUC_Away','Macro_AUC','Micro_AUC')]] for _,r in aucs.iterrows()])
    importance = pd.read_csv(out/'feature_importances.csv').head(15)
    importance_table = table(['Rank','Feature','XGBoost importance','RF impurity importance'],[[n+1,r.Feature,f'{r.XGBoost:.4f}',f'{r.RandomForest:.4f}'] for n,(_,r) in enumerate(importance.iterrows())])
    league = pd.read_csv(out/'league_slices.csv')
    league_table = table(['Division','Matches','Accuracy'],[[r.Division,r.Matches,pct(r.Accuracy)] for _,r in league.iterrows()])
    elo = pd.read_csv(out/'elo_slices.csv')
    elo_table = table(['Empirical |Elo difference| interval','Matches','Accuracy'],[[f'{r.Min_AbsElo:g} to {r.Max_AbsElo:g}',int(r.Matches),pct(r.Accuracy)] for _,r in elo.iterrows()])
    split_table = table(['Partition','Matches','First date','Last date'],[[label,manifest[key]['matches'],manifest[key]['date_min'],manifest[key]['date_max']] for key,label in [('train','Training'),('test','Benchmark test'),('holdout','Temporal holdout')]])
    league_count_table = table(['Division','Selected historical matches'],sorted(manifest['leagues'].items()))
    run_date = manifest['run_utc'][:10]
    data_section = f'''The verified input contains **{manifest['input']['raw_rows']:,} raw rows** and **{manifest['clean']['matches']:,} selected five-league matches**. The selected dates are **{manifest['clean']['date_min']} to {manifest['clean']['date_max']}**.

{split_table}

{league_count_table}

The input is not bundled in Git. Place your own copy at `data/raw/Matches.csv`. The exact snapshot used here has SHA-256:

`{manifest['input']['sha256']}`

The available holdout has **{hold_count:,} matches**, ending **{manifest['holdout']['date_max']}**. The earlier claim of 4,502 holdout matches extending into 2026 is superseded; this run does not evaluate 2026 fixtures.
'''
    methodology = '''The target encoding is H=0, D=1, A=2. Training uses matches through 2022; 2023 is the benchmark; all later available matches form the holdout. Model parameters are fixed by `src/models.py`; the random seed is 42. Rolling goals, shots, corners, cards, venue statistics, rest, and H2H use past fixtures. Supplied Elo/form and closing odds are assumed to be available before kickoff; their upstream provenance is not independently established by this code.

The E9 feature set contains 56 features, including market probabilities, Elo gaps, rolling form and goals, rest/congestion, and H2H. The weighted ensemble combines **Extra Trees 0.35, XGBoost 0.20, HistGradientBoosting 0.20, Random Forest 0.15, and ordinary logistic regression 0.10**. Balanced LR is a separate benchmark and cannot overwrite the ordinary LR fit.

SMOTE resamples only imputed training data. This preserves the existing experimental method: distance-based SMOTE is applied before scaling, which can make large-scale features dominate neighbor selection; a scale-aware variant requires a separately validated experiment. Test and holdout data are never resampled. The draw multiplier changes the decision rule, not the stored raw probability estimates, and is not evidence of probability calibration.

Feature sets, draw multipliers and confidence thresholds are reported descriptively. Their comparison on the benchmark is not a separate validation procedure, and repeated inspection of the holdout means it should not be treated as a pristine final test. Empirical 54% results do not establish a mathematical accuracy ceiling or prove that higher results imply leakage.
'''
    results_section = f'''## Feature progression

{progression_table}

These are measured comparisons for the recorded data snapshot. Adding features does not guarantee an accuracy increase; results also depend on the chosen classifier.

## Model benchmark (2023, {test_count:,} matches)

{benchmark_table}

![Benchmark accuracy and Macro F1](results/figures/summary/benchmark_comparison.png)

Accuracy measures the share of correct outcomes. Macro F1 gives H, D, and A equal weight; low draw recall can therefore produce a low Macro F1 despite reasonable overall accuracy.

## Ensemble and temporal holdout

{ensemble_table}

The baseline ensemble obtains **{pct(ens.Accuracy)} accuracy and {ens.Macro_F1:.4f} Macro F1** on all 2023 fixtures. It obtains **{pct(hold.Accuracy)} accuracy** on the available holdout. These measurements describe this snapshot and do not prove absence of temporal degradation or probability calibration.

![2023 normalized confusion matrix](results/figures/summary/confusion_2023_normalized.png)

Rows are actual classes and columns are predicted classes. The diagonal gives per-class recall; the draw row shows the baseline's difficulty predicting draws.

![Holdout normalized confusion matrix](results/figures/summary/confusion_holdout_normalized.png)

The holdout uses the same trained ensemble, with histories updated sequentially from earlier completed matches. Its dates and sample count differ from the former 2026 claim.

## Draw decision multiplier (XGBoost)

{draw_table}

Multiplying the draw score can raise draw recall and Macro F1 while lowering accuracy. These are sensitivity results on the benchmark; a deployment multiplier must be selected on separate validation data.

## SMOTE comparison

{smote_table}

The same five-model membership and weights are used for the baseline and SMOTE ensembles. SMOTE changes training balance; it does not improve every metric for every classifier.

## Confidence filtering

Confidence is the maximum predicted class probability. A threshold accepts only fixtures meeting that score; the displayed accuracy applies only to that subset.
{confidence_tables}

![Accuracy versus coverage](results/figures/summary/accuracy_coverage.png)

Point labels are confidence thresholds. Higher selected accuracy usually comes with fewer accepted matches. These empirical confidence scores have not been demonstrated to be calibrated probabilities.

## Double chance

{dc_table}

Double chance accepts two of the three outcomes (1X, X2, or 12). It is a different, easier target than a single H/D/A prediction and must not be presented as 3-way accuracy.

## ROC-AUC

{auc_table}

ROC measures ranking over possible thresholds. Draw AUC near 0.5 indicates weak discrimination. Macro curve AUC is the area under the interpolated mean ROC curve; micro AUC pools the one-vs-rest decisions. AUC does not establish probability calibration or a universal accuracy ceiling.

![Ensemble ROC](results/figures/roc_curves/roc_ensemble_2023_benchmark.png)

![Holdout ROC](results/figures/roc_curves/roc_ensemble_holdout.png)

![Model ROC comparison](results/figures/roc_curves/roc_multimodel_comparison.png)

![SMOTE ROC comparison](results/figures/roc_curves/roc_smote_vs_baseline.png)

All ROC curves use the saved probabilities from the run that produced the benchmark table.

## Feature importance

{importance_table}

![XGBoost importance](results/figures/feature_importance/xgboost_top15.png)

![Random Forest importance](results/figures/feature_importance/randomforest_top15.png)

Importances describe these fitted models, not causal effects. Correlated market features can share importance; RF values in the table are aligned to the XGBoost-ranked rows, while its figure uses RF's own ranking.

## Error slices (same five-model ensemble)

{league_table}

{elo_table}

The Elo intervals above are empirical quartiles calculated from the 2023 fixtures. They replace the former incorrect fixed-range labels. Small slice differences require uncertainty analysis before strong conclusions.
'''
    quickstart = '''## Reproduce or inspect

Use Python 3.12 and a clean environment:

```bash
python -m venv .venv
# Windows PowerShell:
.\\.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-reproducible.txt
# Place the verified input at data/raw/Matches.csv, then run:
python src/run_pipeline.py
python -m unittest discover -s tests -v
```

One command trains all baseline/progression/SMOTE models, exports predictions, and regenerates all tables, figures, README, research results, and the notebook. The legacy SMOTE entry point now invokes the same complete run. Diagnostic scripts only read saved outputs, avoiding a second model definition. Small numerical changes across operating systems or numerical libraries remain possible; compare the manifest and input hash before attributing differences to models.

Key outputs:

- `results/experiment_results.csv`: full precision metrics with sample counts and dates.
- `results/predictions.npz`: actual labels, decisions, and H/D/A probabilities for every experiment; inspect using `numpy.load(..., allow_pickle=False)`.
- `results/error_analysis_test2023.csv` and `results/predictions_holdout.csv`: per-fixture baseline ensemble predictions, probabilities, and actual classes.
- `results/run_manifest.json`: input SHA-256, package versions, split counts/dates, features, ensemble weights, and source hashes.
- `results/artifact_hashes.json`: hashes of the generated result files.
- `notebooks/Dataset1_Predictor.ipynb`: exploration of these exact saved outputs; no independent fitting.
- `RESEARCH_PAPER.md`: regenerated technical report using the same result tables.

Generated numeric documentation comes from `src/refresh_results.py`; edit its template instead of hand-copying results. The raw and processed input datasets remain excluded by `.gitignore`.
'''
    fixes = '''## What was corrected

- README tables and figures now derive from one recorded run, replacing mixed historical summaries.
- Ordinary and balanced logistic regression have separate storage keys and confusion matrices.
- Error analysis, ROC curves, and benchmarks use the same five-model ensemble and saved probabilities.
- Holdout dates/counts reflect the available data instead of an unsupported 2026 extension.
- Old figures from superseded E8/three-model runs are removed rather than mixed with regenerated figures.
- Unsupported claims of a universal 54–56% ceiling, guaranteed zero degradation, and proven probability calibration are removed.

The former research write-up contained results from multiple runs and unsupported literature comparisons. Its empirical report is regenerated here; the previous text remains available in Git history. This correction does not certify the accuracy of the earlier external literature claims or update separate PDF/PPT files outside this repository.
'''
    header = f'# Football Match Outcome Prediction — Dataset 1\n\nVerified results generated on **{run_date}** from the recorded input snapshot.\n\n'
    (ROOT/'README.md').write_text(header+'## Data and evaluation scope\n\n'+data_section+'\n## Method\n\n'+methodology+'\n'+results_section+'\n'+quickstart+'\n'+fixes, encoding='utf-8')
    # A factual technical report replaces the inconsistent empirical narrative.
    report = f'''# Football Match Outcome Prediction: Reconciled Experimental Report

## Abstract

We compare pre-match classifiers across five major European leagues using chronological evaluation. On the {test_count:,}-fixture 2023 benchmark, the fixed five-model ensemble achieves {pct(ens.Accuracy)} accuracy and {ens.Macro_F1:.4f} Macro F1. The available {hold_count:,}-fixture holdout ends on {manifest['holdout']['date_max']} and obtains {pct(hold.Accuracy)} accuracy. Draw imbalance, confidence filtering and SMOTE are evaluated as descriptive experiments. All empirical tables and plots below are regenerated from one set of saved predictions. This report supersedes the earlier inconsistent results narrative.

## Dataset and split

{data_section}
## Methodology

{methodology}
## Metric definitions

For class c, precision = TP_c/(TP_c+FP_c), recall = TP_c/(TP_c+FN_c), and F1_c = 2 precision_c recall_c/(precision_c+recall_c), with zero assigned when undefined. Macro F1 is the arithmetic mean of the three class F1 scores. Accuracy is the number of correct single-outcome predictions divided by all evaluated fixtures. Coverage is accepted fixtures divided by all fixtures. Selected accuracy is correct accepted predictions divided by accepted fixtures.

The ensemble probability is p(c|x) = sum_m w_m p_m(c|x). Its decision is argmax_c p(c|x). A draw decision multiplier M replaces the draw score by M p(D|x) for the argmax only. Confidence filtering accepts max_c p(c|x) >= tau. Double chance selects the pair with largest summed probability.

{results_section}
## Discussion and limits

The benchmark emphasizes a persistent draw-recognition problem. Changing the decision rule and training balance alters the accuracy/recall trade-off. Confidence filtering improves accuracy only on accepted fixtures and does not solve full-population prediction. Results depend on the supplied historical data and software environment. This experiment does not estimate a mathematical accuracy limit, establish causality, demonstrate betting profitability, or independently verify the provenance of supplied ratings and odds. Validation-selected settings, uncertainty intervals, and a future untouched test set are needed for stronger generalization claims.

{quickstart}
{fixes}
'''
    (ROOT/'RESEARCH_PAPER.md').write_text(report, encoding='utf-8')
    from src.build_notebook import create_dataset1_notebook
    create_dataset1_notebook()
    print('Documentation regenerated from full-precision outputs.')


if __name__ == '__main__':
    import os, sys
    os.chdir(ROOT)
    refresh_results()
