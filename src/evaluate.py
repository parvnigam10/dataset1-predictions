"""
evaluate.py
Evaluation metrics, confusion matrix plotting, and experiment tracking module.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

CLASS_LABELS = ['H', 'D', 'A']
CLASS_NAMES = ['Home Win', 'Draw', 'Away Win']


def evaluate_predictions(y_true, y_pred, y_prob=None, experiment_name=""):
    """
    Computes complete evaluation metrics for 3-way outcome classification.
    Returns dictionary of metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics: 0=H, 1=D, 2=A
    precisions = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    metrics = {
        'Experiment': experiment_name,
        'Accuracy': float(acc),
        'Macro_F1': float(macro_f1),
        'Weighted_F1': float(weighted_f1),
        'Home_Precision': float(precisions[0]),
        'Home_Recall': float(recalls[0]),
        'Home_F1': float(f1s[0]),
        'Draw_Precision': float(precisions[1]),
        'Draw_Recall': float(recalls[1]),
        'Draw_F1': float(f1s[1]),
        'Away_Precision': float(precisions[2]),
        'Away_Recall': float(recalls[2]),
        'Away_F1': float(f1s[2]),
        'Confusion_Matrix': cm
    }
    return metrics


def plot_and_save_confusion_matrix(cm, title, filepath):
    """Plots and saves confusion matrix heatmap."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        cbar=False, annot_kws={'size': 14}
    )
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Outcome', fontsize=11)
    plt.ylabel('Actual Outcome', fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()


def print_evaluation_summary(metrics):
    """Prints a clean formatted summary of evaluation results."""
    print("=" * 60)
    print(f" EXPERIMENT EVALUATION: {metrics['Experiment']}")
    print("=" * 60)
    print(f" Overall Accuracy : {metrics['Accuracy']*100:.2f}%")
    print(f" Macro F1-Score   : {metrics['Macro_F1']:.4f}")
    print(f" Weighted F1-Score: {metrics['Weighted_F1']:.4f}")
    print("-" * 60)
    print(f" {'Outcome':15s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s}")
    print("-" * 60)
    print(f" {'Home Win (H)':15s} | {metrics['Home_Precision']*100:9.2f}% | {metrics['Home_Recall']*100:9.2f}% | {metrics['Home_F1']*100:9.2f}%")
    print(f" {'Draw (D)':15s} | {metrics['Draw_Precision']*100:9.2f}% | {metrics['Draw_Recall']*100:9.2f}% | {metrics['Draw_F1']*100:9.2f}%")
    print(f" {'Away Win (A)':15s} | {metrics['Away_Precision']*100:9.2f}% | {metrics['Away_Recall']*100:9.2f}% | {metrics['Away_F1']*100:9.2f}%")
    print("-" * 60)
    print(" Confusion Matrix (Actual rows x Predicted cols):")
    cm = metrics['Confusion_Matrix']
    print(f"        Pred H   Pred D   Pred A")
    print(f" Act H: {cm[0,0]:6d}   {cm[0,1]:6d}   {cm[0,2]:6d}")
    print(f" Act D: {cm[1,0]:6d}   {cm[1,1]:6d}   {cm[1,2]:6d}")
    print(f" Act A: {cm[2,0]:6d}   {cm[2,1]:6d}   {cm[2,2]:6d}")
    print("=" * 60 + "\n")

