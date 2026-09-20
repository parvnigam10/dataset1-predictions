"""
plot_feature_importance.py
Extracts and plots feature importances from XGBoost and Random Forest on E7_Selected.
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

from src.data_loader import load_raw_dataset, clean_and_filter_matches, create_chronological_splits
from src.features import compute_all_features, FEATURE_GROUPS
from src.models import get_model


def plot_importances():
    os.makedirs('results/figures/feature_importance', exist_ok=True)
    
    clean_df = pd.read_csv('data/processed/cleaned_matches_5leagues.csv')
    clean_df['MatchDate'] = pd.to_datetime(clean_df['MatchDate'])
    
    feat_df = compute_all_features(clean_df)
    train_df, test_df, _ = create_chronological_splits(feat_df, train_end_year=2022, test_year=2023)
    
    features = FEATURE_GROUPS['E9_Expanded_Market_Fatigue']
    X_train = train_df[features].values
    y_train = train_df['target'].values
    
    # Train XGBoost
    xgb_pipeline = get_model('xgboost')
    xgb_pipeline.fit(X_train, y_train)
    xgb_model = xgb_pipeline.named_steps['clf']
    xgb_importances = xgb_model.feature_importances_
    
    # Train Random Forest
    rf_pipeline = get_model('random_forest')
    rf_pipeline.fit(X_train, y_train)
    rf_model = rf_pipeline.named_steps['clf']
    rf_importances = rf_model.feature_importances_
    
    # Create DataFrame
    imp_df = pd.DataFrame({
        'Feature': features,
        'XGBoost': xgb_importances,
        'RandomForest': rf_importances
    }).sort_values(by='XGBoost', ascending=False)
    
    imp_df.to_csv('results/feature_importances.csv', index=False)
    print("Saved feature importances to results/feature_importances.csv")
    
    # Plot Top 15 XGBoost Feature Importances
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df.head(15), x='XGBoost', y='Feature', palette='viridis')
    plt.title('Top 15 Most Important Pre-Match Features (XGBoost)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Weight', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance/xgboost_top15.png', dpi=200)
    plt.close()
    
    # Plot Top 15 Random Forest Feature Importances
    imp_df_rf = imp_df.sort_values(by='RandomForest', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df_rf.head(15), x='RandomForest', y='Feature', palette='magma')
    plt.title('Top 15 Most Important Pre-Match Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Gini Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/figures/feature_importance/randomforest_top15.png', dpi=200)
    plt.close()
    print("Saved feature importance plots.")

if __name__ == '__main__':
    plot_importances()
