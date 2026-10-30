"""
error_analysis.py
Performs slice-based error analysis on the 2023 test set:
- Misclassification breakdowns across match conditions
- Elo disparity vs error rate
- Market favorite upset rate
- Draw classification confusion breakdown
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from src.features import compute_all_features, FEATURE_GROUPS
from src.data_loader import create_chronological_splits
from src.models import get_model


def run_error_analysis():
    clean_df = pd.read_csv('data/processed/cleaned_matches_5leagues.csv', low_memory=False)
    clean_df['MatchDate'] = pd.to_datetime(clean_df['MatchDate'])
    
    feat_df = compute_all_features(clean_df)
    train_df, test_df, _ = create_chronological_splits(feat_df, train_end_year=2022, test_year=2023)
    
    features = FEATURE_GROUPS['E9_Expanded_Market_Fatigue']
    X_train = train_df[features].values
    y_train = train_df['target'].values
    X_test = test_df[features].values
    y_test = test_df['target'].values
    
    from src.models import SoftVotingEnsemble
    et = get_model('extra_trees')
    et.fit(X_train, y_train)
    xgb_clf = get_model('xgboost')
    xgb_clf.fit(X_train, y_train)
    hgb = get_model('hist_gbm')
    hgb.fit(X_train, y_train)
    rf = get_model('random_forest')
    rf.fit(X_train, y_train)
    lr = get_model('logistic_regression', balanced=True)
    lr.fit(X_train, y_train)
    
    ensemble = SoftVotingEnsemble(
        models=[et, xgb_clf, hgb, rf, lr],
        weights=[0.35, 0.20, 0.20, 0.15, 0.10]
    )
    
    test_df = test_df.copy()
    probs = ensemble.predict_proba(X_test)
    test_df['pred_target'] = ensemble.predict(X_test)
    test_df['pred_prob_H'] = probs[:, 0]
    test_df['pred_prob_D'] = probs[:, 1]
    test_df['pred_prob_A'] = probs[:, 2]
    test_df['is_correct'] = (test_df['target'] == test_df['pred_target']).astype(int)
    
    print("=" * 70)
    print("PHASE 18: SLICE-BASED ERROR ANALYSIS & 70%+ ACCURACY DEEP-DIVE")
    print("=" * 70)
    
    # 1. Error Rate by League / Division
    print("\n1. Accuracy by League / Division:")
    league_perf = test_df.groupby('Division')['is_correct'].agg(['count', 'mean']).rename(columns={'count': 'Matches', 'mean': 'Accuracy'})
    league_perf['Accuracy'] = (league_perf['Accuracy'] * 100).round(2)
    print(league_perf)
    
    # 2. Performance by Elo Disparity
    test_df['AbsEloDiff'] = test_df['EloDiff'].abs()
    test_df['Elo_Tier'] = pd.qcut(test_df['AbsEloDiff'], q=4, labels=['Balanced (|Diff|<75)', 'Mild Disparity (75-150)', 'Substantial (150-250)', 'Heavy Favorite (>250)'])
    print("\n2. Accuracy by Pre-Match Elo Disparity:")
    elo_perf = test_df.groupby('Elo_Tier', observed=False)['is_correct'].agg(['count', 'mean']).rename(columns={'count': 'Matches', 'mean': 'Accuracy'})
    elo_perf['Accuracy'] = (elo_perf['Accuracy'] * 100).round(2)
    print(elo_perf)
    
    # 3. Draw Misclassification Breakdown
    draws_only = test_df[test_df['FTResult'] == 'D']
    print(f"\n3. Analysis of Actual Draws (Total = {len(draws_only)}):")
    pred_draw_dist = draws_only['pred_target'].map({0: 'Predicted Home Win', 1: 'Correctly Predicted Draw', 2: 'Predicted Away Win'}).value_counts()
    for k, v in pred_draw_dist.items():
        print(f"  {k:26s}: {v:4d} ({v/len(draws_only)*100:.1f}%)")
        
    # 4. Where does the model achieve 70%+ accuracy? (Confidence Tiers)
    test_df['MaxProb'] = test_df[['pred_prob_H', 'pred_prob_D', 'pred_prob_A']].max(axis=1)
    test_df['Confidence_Tier'] = pd.cut(test_df['MaxProb'], bins=[0.0, 0.45, 0.55, 0.65, 1.0], labels=['Low (<45%)', 'Medium (45-55%)', 'High (55-65%)', 'Very High (>65%)'])
    print("\n4. Calibration / Confidence vs Accuracy (The 70%+ Threshold):")
    conf_perf = test_df.groupby('Confidence_Tier', observed=False)['is_correct'].agg(['count', 'mean']).rename(columns={'count': 'Matches', 'mean': 'Accuracy'})
    conf_perf['Accuracy'] = (conf_perf['Accuracy'] * 100).round(2)
    print(conf_perf)
    
    # 5. Double Chance evaluation
    p_1X = probs[:, 0] + probs[:, 1]
    p_X2 = probs[:, 1] + probs[:, 2]
    p_12 = probs[:, 0] + probs[:, 2]
    
    dc_correct = 0
    for i in range(len(y_test)):
        best_dc = np.argmax([p_1X[i], p_X2[i], p_12[i]])
        actual = y_test[i]
        if (best_dc == 0 and actual in [0, 1]) or (best_dc == 1 and actual in [1, 2]) or (best_dc == 2 and actual in [0, 2]):
            dc_correct += 1
    print(f"\n5. Double Chance (1X, X2, 12) Accuracy: {dc_correct / len(y_test) * 100:.2f}% ({dc_correct}/{len(y_test)})")
    
    # Save error analysis summary
    test_df[['MatchDate', 'Division', 'HomeTeam', 'AwayTeam', 'FTResult', 'pred_target', 'is_correct', 'EloDiff', 'MaxProb']].to_csv('results/error_analysis_test2023.csv', index=False)
    print("\nSaved detailed prediction breakdown to results/error_analysis_test2023.csv")


if __name__ == '__main__':
    run_error_analysis()
