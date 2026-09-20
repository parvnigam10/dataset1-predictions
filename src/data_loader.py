"""
data_loader.py
Data loading, cleaning, and preprocessing module for Football Match Outcome Prediction.
"""

import os
import pandas as pd
import numpy as np

MAJOR_LEAGUES = ['D1', 'E0', 'F1', 'I1', 'SP1']

LEAGUE_NAMES = {
    'D1': 'Germany Bundesliga',
    'E0': 'England Premier League',
    'F1': 'France Ligue 1',
    'I1': 'Italy Serie A',
    'SP1': 'Spain La Liga'
}

TARGET_MAP = {'H': 0, 'D': 1, 'A': 2}
TARGET_NAMES = ['Home Win (H)', 'Draw (D)', 'Away Win (A)']
TARGET_CLASSES = ['H', 'D', 'A']


def load_raw_dataset(filepath: str = 'data/raw/Matches.csv') -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    return df


def clean_and_filter_matches(df: pd.DataFrame, leagues: list = None) -> pd.DataFrame:
    """
    Clean matches dataset:
    - Filter for specified leagues (defaults to 5 major European leagues)
    - Drop rows where target (FTResult) is null
    - Parse MatchDate into datetime
    - Sort chronologically by MatchDate and Division
    - Remove duplicate matches
    - Standardize team names
    """
    if leagues is None:
        leagues = MAJOR_LEAGUES
        
    df = df[df['Division'].isin(leagues)].copy()
    
    # Drop rows without target
    df = df[df['FTResult'].isin(['H', 'D', 'A'])].copy()
    
    # Date parsing
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    
    # Strip whitespace in team names
    df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip()
    df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip()
    
    # Check and drop duplicates
    df = df.drop_duplicates(subset=['Division', 'MatchDate', 'HomeTeam', 'AwayTeam']).copy()
    
    # Chronological sort
    df = df.sort_values(by=['MatchDate', 'Division']).reset_index(drop=True)
    
    # Add target integer column
    df['target'] = df['FTResult'].map(TARGET_MAP)
    df['Year'] = df['MatchDate'].dt.year
    
    return df


def create_chronological_splits(
    df: pd.DataFrame,
    train_end_year: int = 2022,
    test_year: int = 2023,
    holdout_after_test: bool = True
):
    """
    Create chronological train, validation, and test splits.
    - Train: All matches up to train_end_year (allows historical burn-in)
    - Test: Matches in test_year (e.g. 2023) to benchmark against previous experiments
    - Holdout: Matches beyond test_year (e.g. 2024-2026) as extra unseen validation
    """
    train_mask = df['Year'] <= train_end_year
    test_mask = df['Year'] == test_year
    holdout_mask = df['Year'] > test_year
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    holdout_df = df[holdout_mask].copy() if holdout_after_test else None
    
    return train_df, test_df, holdout_df
