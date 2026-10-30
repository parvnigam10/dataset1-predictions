import pandas as pd
import numpy as np
from collections import defaultdict

def test_feature_engineering():
    print("Testing feature engineering logic...")
    df = pd.read_csv('data/raw/Matches.csv', low_memory=False)
    maj = ['D1', 'E0', 'F1', 'I1', 'SP1']
    df = df[df['Division'].isin(maj)].copy()
    
    # Chronological sort
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values(by=['MatchDate', 'Division']).reset_index(drop=True)
    
    print(f"Total rows to process: {len(df)}")
    
    # Let's inspect team names and sample matches
    print("Sample rows:")
    print(df[['Division', 'MatchDate', 'HomeTeam', 'AwayTeam', 'FTHome', 'FTAway', 'FTResult']].head(5))
    
if __name__ == '__main__':
    test_feature_engineering()

