import pandas as pd
import numpy as np

def inspect_dataset():
    filepath = 'data/raw/Matches.csv'
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Dataset Shape: {df.shape}")
    print("\nColumns and Data Types:")
    for col, dtype, n_null in zip(df.columns, df.dtypes, df.isnull().sum()):
        print(f"  {col:15s} | {str(dtype):10s} | Nulls: {n_null:6d} ({n_null/len(df)*100:.1f}%)")
        
    print(f"\nOverall Date Range: {df['MatchDate'].min()} to {df['MatchDate'].max()}")
    print("\nOverall Target (FTResult) Distribution:")
    print(df['FTResult'].value_counts(dropna=False))
    
    major_leagues = ['D1', 'E0', 'F1', 'I1', 'SP1']
    df_maj = df[df['Division'].isin(major_leagues)].copy()
    print(f"\nTotal matches in 5 Major Leagues ({major_leagues}): {len(df_maj)}")
    print(df_maj['Division'].value_counts())
    print(f"5 Major Leagues Date Range: {df_maj['MatchDate'].min()} to {df_maj['MatchDate'].max()}")
    
    print("\nTarget Distribution in 5 Major Leagues:")
    counts = df_maj['FTResult'].value_counts()
    pcts = df_maj['FTResult'].value_counts(normalize=True) * 100
    for res in ['H', 'D', 'A']:
        if res in counts:
            print(f"  {res}: {counts[res]} ({pcts[res]:.2f}%)")
            
    print("\nChecking duplicates in 5 Major Leagues:")
    dups = df_maj.duplicated(subset=['Division', 'MatchDate', 'HomeTeam', 'AwayTeam']).sum()
    print(f"  Duplicates by (Division, MatchDate, HomeTeam, AwayTeam): {dups}")
    
    print("\nOdds coverage in 5 Major Leagues:")
    print(f"  OddHome available: {df_maj['OddHome'].notnull().sum()} ({df_maj['OddHome'].notnull().mean()*100:.1f}%)")
    print(f"  HomeElo available: {df_maj['HomeElo'].notnull().sum()} ({df_maj['HomeElo'].notnull().mean()*100:.1f}%)")
    print(f"  HomeShots available: {df_maj['HomeShots'].notnull().sum()} ({df_maj['HomeShots'].notnull().mean()*100:.1f}%)")
    
if __name__ == '__main__':
    inspect_dataset()

