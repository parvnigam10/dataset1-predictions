"""
plot_feature_importance.py
Plots importances exported from the exact E9 benchmark fits; does not retrain.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns



def plot_importances():
    os.makedirs('results/figures/feature_importance', exist_ok=True)
    
    imp_df = pd.read_csv('results/feature_importances.csv')

    # Plot Top 15 XGBoost Feature Importances
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df.head(15), x='XGBoost', y='Feature', color='#23679b')
    plt.title('Top 15 Most Important Pre-Match Features (XGBoost)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Weight', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance/xgboost_top15.png', dpi=200)
    plt.close()
    
    # Plot Top 15 Random Forest Feature Importances
    imp_df_rf = imp_df.sort_values(by='RandomForest', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df_rf.head(15), x='RandomForest', y='Feature', color='#ad5b2a')
    plt.title('Top 15 Most Important Pre-Match Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Gini Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance/randomforest_top15.png', dpi=200)
    plt.close()
    print("Saved feature importance plots.")

if __name__ == '__main__':
    plot_importances()

