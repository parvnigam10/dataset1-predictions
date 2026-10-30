"""
plot_roc_curves.py
Computes and plots Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC)
for 3-way match outcome prediction (Home Win, Draw, Away Win) using One-vs-Rest (OvR).

Generates:
1. results/figures/roc_curves/roc_ensemble_2023_benchmark.png (Ensemble on 2023 Benchmark Test)
2. results/figures/roc_curves/roc_ensemble_holdout.png (Ensemble on the available post-2023 holdout)
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

import json

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
    print("=" * 80)
    print("GENERATING COMPREHENSIVE MULTI-CLASS ROC CURVES & AUC METRICS")
    print("=" * 80)
    
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    out_dir = 'results/figures/roc_curves'
    os.makedirs(out_dir, exist_ok=True)
    
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    archive = np.load('results/predictions.npz', allow_pickle=False)
    manifest = json.loads(open('results/run_manifest.json').read())
    models_config = [
        ('Extra Trees', 'extra_trees'), ('XGBoost', 'xgboost'),
        ('HistGradientBoosting', 'hist_gbm'), ('Random Forest', 'random_forest'),
        ('Logistic Regression', 'logistic_regression'), ('Multi-Layer Perceptron (MLP)', 'mlp'),
    ]
    y_test = archive['Ensemble_5Models_E9__y']
    y_holdout = archive['Holdout_Ensemble__y']
    model_test_probs = {}
    auc_records = []
    for display_name, key in models_config:
        probs = archive['E9_' + display_name + '__prob']
        model_test_probs[key] = probs
        _, _, scores = compute_ovr_roc(y_test, probs)
        auc_records.append({'Model': display_name, 'Evaluation_Set': '2023 Benchmark Test',
            'AUC_Home': scores[0], 'AUC_Draw': scores[1], 'AUC_Away': scores[2],
            'Macro_AUC': scores['macro'], 'Micro_AUC': scores['micro']})

    # 3. Soft-Voting Ensemble (2023 Benchmark Test)
    ens_prob_test = archive['Ensemble_5Models_E9__prob']
    ens_fpr_test, ens_tpr_test, ens_auc_test = compute_ovr_roc(y_test, ens_prob_test)
    
    auc_records.append({
        'Model': 'Soft-Voting Ensemble (5 Models)',
        'Evaluation_Set': '2023 Benchmark Test',
        'AUC_Home': ens_auc_test[0],
        'AUC_Draw': ens_auc_test[1],
        'AUC_Away': ens_auc_test[2],
        'Macro_AUC': ens_auc_test['macro'],
        'Micro_AUC': ens_auc_test['micro'],
    })
    
    plot_single_model_roc(
        ens_fpr_test, ens_tpr_test, ens_auc_test,
        title='One-vs-Rest ROC Curves: 5-Model Ensemble (2023 Benchmark Test)',
        filepath=f"{out_dir}/roc_ensemble_2023_benchmark.png"
    )
    
    # 4. Soft-Voting Ensemble (2024-2026 Holdout Set)
    ens_prob_holdout = archive['Holdout_Ensemble__prob']
    ens_fpr_ho, ens_tpr_ho, ens_auc_ho = compute_ovr_roc(y_holdout, ens_prob_holdout)
    
    auc_records.append({
        'Model': 'Soft-Voting Ensemble (5 Models)',
        'Evaluation_Set': manifest['holdout_label'],
        'AUC_Home': ens_auc_ho[0],
        'AUC_Draw': ens_auc_ho[1],
        'AUC_Away': ens_auc_ho[2],
        'Macro_AUC': ens_auc_ho['macro'],
        'Micro_AUC': ens_auc_ho['micro'],
    })
    
    plot_single_model_roc(
        ens_fpr_ho, ens_tpr_ho, ens_auc_ho,
        title=f"Ensemble ROC: {manifest['holdout']['date_min']} to {manifest['holdout']['date_max']}",
        filepath=f"{out_dir}/roc_ensemble_holdout.png"
    )
    
    # 5. Multi-Model Macro ROC Comparison Plot
    plt.figure(figsize=(9, 7))
    model_colors = {
        'Soft-Voting Ensemble': ('#e41a1c', ens_fpr_test['macro'], ens_tpr_test['macro'], ens_auc_test['macro']),
    }
    
    for display_name, m_key in models_config:
        f_p, t_p, a_p = compute_ovr_roc(y_test, model_test_probs[m_key])
        model_colors[display_name] = (None, f_p['macro'], t_p['macro'], a_p['macro'])
        
    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']
    for idx, (m_label, (_, f_macro, t_macro, a_macro)) in enumerate(model_colors.items()):
        lw = 2.8 if 'Ensemble' in m_label else 1.8
        plt.plot(
            f_macro, t_macro, label=f"{m_label} (Macro AUC = {a_macro:.3f})",
            linewidth=lw, color=palette[idx % len(palette)]
        )
        
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random Chance (AUC = 0.500)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Multi-Model Comparison: Macro-Average ROC Curves (2023 Benchmark)', fontsize=13, fontweight='bold', pad=14)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_multimodel_comparison.png", dpi=200)
    plt.close()
    print(f"Saved: {out_dir}/roc_multimodel_comparison.png")

    # 6. SMOTE vs Baseline ROC Comparison
    smote_prob_test = archive['SMOTE_Ensemble_5Models_E9__prob']
    sm_fpr, sm_tpr, sm_auc = compute_ovr_roc(y_test, smote_prob_test)
    
    auc_records.append({
        'Model': 'SMOTE Soft-Voting Ensemble',
        'Evaluation_Set': '2023 Benchmark Test',
        'AUC_Home': sm_auc[0],
        'AUC_Draw': sm_auc[1],
        'AUC_Away': sm_auc[2],
        'Macro_AUC': sm_auc['macro'],
        'Micro_AUC': sm_auc['micro'],
    })

    # Plot SMOTE vs Baseline (showing Draw class specifically and Macro average)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: Macro-Average
    axes[0].plot(ens_fpr_test['macro'], ens_tpr_test['macro'], color='#1f77b4', linewidth=2.5, label=f"Original Ensemble (AUC = {ens_auc_test['macro']:.3f})")
    axes[0].plot(sm_fpr['macro'], sm_tpr['macro'], color='#e377c2', linewidth=2.5, linestyle='--', label=f"SMOTE Ensemble (AUC = {sm_auc['macro']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    axes[0].set_title('Macro-Average ROC: Original vs SMOTE', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=10.5)
    
    # Right subplot: Draw Class (Class 1) Discrimination
    axes[1].plot(ens_fpr_test[1], ens_tpr_test[1], color='#ff7f0e', linewidth=2.5, label=f"Original Ensemble: Draw (AUC = {ens_auc_test[1]:.3f})")
    axes[1].plot(sm_fpr[1], sm_tpr[1], color='#2ca02c', linewidth=2.5, linestyle='--', label=f"SMOTE Ensemble: Draw (AUC = {sm_auc[1]:.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
    axes[1].set_title('Minority Class Discrimination: Draw (D) ROC', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=10.5)
    
    plt.suptitle('Comparison: Original Baseline vs SMOTE-Balanced Ensemble (2023 Test)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_smote_vs_baseline.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir}/roc_smote_vs_baseline.png")

    # 7. Save AUC metrics to CSV
    auc_df = pd.DataFrame(auc_records)
    csv_path = 'results/roc_auc_metrics.csv'
    auc_df.to_csv(csv_path, index=False)
    print(f"\nSaved numerical AUC metrics to {csv_path}")
    print("\n" + "=" * 80)
    print("ROC-AUC SCORES SUMMARY TABLE")
    print("=" * 80)
    print(auc_df.to_string(index=False))


if __name__ == '__main__':
    generate_all_roc_curves()


