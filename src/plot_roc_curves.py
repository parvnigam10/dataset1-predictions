"""
plot_roc_curves.py
Computes and plots Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC)
for 3-way match outcome prediction (Home Win, Draw, Away Win) using One-vs-Rest (OvR).

Generates:
1. results/figures/roc_curves/roc_ensemble_2023_benchmark.png (Ensemble on 2023 Benchmark Test)
2. results/figures/roc_curves/roc_ensemble_holdout_2024_2026.png (Ensemble on 2024-2026 Holdout)
3. results/figures/roc_curves/roc_multimodel_comparison.png (Macro-average comparison across all architectures)
4. results/figures/roc_curves/roc_smote_vs_baseline.png (Impact of SMOTE on ROC curves)
5. results/roc_auc_metrics.csv (Complete numerical tracking of class-wise and macro/micro AUCs)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from src.artifacts import RESULTS, PROB_COLUMNS, load_predictions, load_metadata, holdout_label

CLASS_NAMES = ['Home Win (H)', 'Draw (D)', 'Away Win (A)']
CLASS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']


def compute_ovr_roc(y_true, y_prob):
    """
    Computes One-vs-Rest ROC curve and AUC for each class, as well as micro and macro averages.
    """
    n_classes = 3
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Class-wise ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc


def plot_single_model_roc(fpr, tpr, roc_auc, title, filepath):
    """Plots multi-class OvR ROC curves with macro and micro averages."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.figure(figsize=(8, 7))
    
    # Micro and macro
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f"Micro-average ROC (AUC = {roc_auc['micro']:.3f})",
        color='deeppink', linestyle=':', linewidth=3
    )
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f"Macro-average ROC (AUC = {roc_auc['macro']:.3f})",
        color='navy', linestyle=':', linewidth=3
    )
    
    # Per-class curves
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        plt.plot(
            fpr[i], tpr[i], color=color, linewidth=2,
            label=f"{name} (AUC = {roc_auc[i]:.3f})"
        )
        
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random Chance (AUC = 0.500)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=14)
    plt.legend(loc="lower right", fontsize=10.5, framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"Saved: {filepath}")


def generate_all_roc_curves():
    """All curves consume exported probabilities from the single training run."""
    out_dir = RESULTS / 'figures/roc_curves'
    out_dir.mkdir(parents=True, exist_ok=True)
    records, curves = [], {}

    def add(name, display, evaluation):
        frame = load_predictions(name)
        fpr, tpr, scores = compute_ovr_roc(frame.target.to_numpy(), frame[PROB_COLUMNS].to_numpy())
        records.append({'Model': display, 'Evaluation_Set': evaluation,
                        'AUC_Home': round(scores[0], 4), 'AUC_Draw': round(scores[1], 4),
                        'AUC_Away': round(scores[2], 4), 'Macro_AUC': round(scores['macro'], 4),
                        'Micro_AUC': round(scores['micro'], 4)})
        return fpr, tpr, scores

    for display in ['Extra Trees', 'XGBoost', 'HistGradientBoosting', 'Random Forest',
                    'Logistic Regression', 'Multi-Layer Perceptron (MLP)']:
        curves[display] = add('E9_' + display, display, '2023 Benchmark Test')
    base = add('Ensemble_5Models_E9', 'Soft-Voting Ensemble (5 Models)', '2023 Benchmark Test')
    curves['Standard ensemble'] = base
    plot_single_model_roc(*base, 'Five-model ensemble (2023)', str(out_dir / 'roc_ensemble_2023_benchmark.png'))
    metadata = load_metadata()
    if metadata['partitions']['holdout']['matches']:
        ho = add('Holdout_Ensemble', 'Soft-Voting Ensemble (5 Models)', holdout_label(metadata))
        plot_single_model_roc(*ho, holdout_label(metadata), str(out_dir / 'roc_ensemble_holdout.png'))
    plt.figure(figsize=(9, 7))
    for display, (fpr, tpr, scores) in curves.items():
        plt.plot(fpr['macro'], tpr['macro'], label=f"{display} ({scores['macro']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('2023 macro-average ROC comparison')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_multimodel_comparison.png', dpi=200)
    plt.close()
    if (RESULTS / 'predictions/SMOTE_Ensemble_5Models_E9.csv').exists():
        smote = add('SMOTE_Ensemble_5Models_E9', 'SMOTE Soft-Voting Ensemble', '2023 Benchmark Test')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for axis, key, title in [(axes[0], 'macro', 'Macro-average ROC'), (axes[1], 1, 'Draw ROC')]:
            for name, (fpr, tpr, scores) in [('Original', base), ('SMOTE', smote)]:
                axis.plot(fpr[key], tpr[key], label=f"{name} ({scores[key]:.3f})")
            axis.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axis.set(xlabel='False positive rate', ylabel='True positive rate', title=title)
            axis.legend()
            axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / 'roc_smote_vs_baseline.png', dpi=200)
        plt.close(fig)
    pd.DataFrame(records).to_csv(RESULTS / 'roc_auc_metrics.csv', index=False)


if __name__ == '__main__':
    generate_all_roc_curves()
