"""Slice the canonical five-model predictions. Never refit a different ensemble."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.artifacts import RESULTS, load_predictions


def run_error_analysis():
    frame = load_predictions('Ensemble_5Models_E9')
    frame.to_csv(RESULTS / 'error_analysis_test2023.csv', index=False, float_format='%.17g')
    league = frame.groupby('Division').is_correct.agg(Matches='size', Correct='sum', Accuracy='mean')
    league['Accuracy'] *= 100
    league.to_csv(RESULTS / 'league_evaluation.csv')
    tiers, edges = pd.qcut(frame.EloDiff.abs(), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], retbins=True)
    quartiles = frame.assign(Quartile=tiers).groupby('Quartile', observed=False).is_correct.agg(
        Matches='size', Correct='sum', Accuracy='mean')
    quartiles['Accuracy'] *= 100
    quartiles['Lower_abs_Elo'] = edges[:-1]
    quartiles['Upper_abs_Elo'] = edges[1:]
    quartiles.to_csv(RESULTS / 'elo_quartile_evaluation.csv')
    draws = frame[frame.target == 1]
    print(league.to_string())
    print(quartiles.to_string())
    print('Actual draws by predicted label (0=H, 1=D, 2=A):', draws.pred_target.value_counts().sort_index().to_dict())


if __name__ == '__main__':
    run_error_analysis()
