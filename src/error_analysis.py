"""Derive error slices from the canonical five-model prediction export."""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run_error_analysis():
    os.chdir(ROOT)
    test = pd.read_csv('results/error_analysis_test2023.csv')
    test.groupby('Division').is_correct.agg(Matches='count', Accuracy='mean').to_csv('results/league_slices.csv')
    test['AbsEloDiff'] = test.EloDiff.abs()
    # These are empirical quartiles, not the old hard-coded 75/150/250 cutoffs.
    test['Elo_Quartile'] = pd.qcut(test.AbsEloDiff, q=4)
    slices = test.groupby('Elo_Quartile', observed=False).agg(Matches=('is_correct', 'size'),
        Accuracy=('is_correct', 'mean'), Min_AbsElo=('AbsEloDiff', 'min'), Max_AbsElo=('AbsEloDiff', 'max'))
    slices.to_csv('results/elo_slices.csv')
    draws = test[test.FTResult == 'D']
    draws.pred_target.value_counts().reindex([0, 1, 2], fill_value=0).rename_axis('Predicted_Target').to_csv('results/draw_breakdown.csv', header=['Matches'])
    print('Error slices regenerated from the main ensemble predictions.')


if __name__ == '__main__':
    run_error_analysis()
