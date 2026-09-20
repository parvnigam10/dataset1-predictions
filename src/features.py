"""
features.py
Feature engineering module for Football Match Outcome Prediction.
Guarantees ZERO lookahead data leakage by shifting historical aggregations strictly prior to match t.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all leakage-free pre-match features across the chronological dataset.
    Returns dataframe with newly engineered features.
    """
    # Ensure chronologically sorted
    df = df.sort_values(by=['MatchDate', 'Division']).reset_index(drop=True).copy()
    n = len(df)
    
    # 1. Elo Features (pre-match Elo ratings are given in dataset)
    df['HomeElo'] = df['HomeElo'].fillna(1500.0)
    df['AwayElo'] = df['AwayElo'].fillna(1500.0)
    df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    df['EloTotal'] = df['HomeElo'] + df['AwayElo']
    df['EloAdvantage'] = df['EloDiff'] / (df['EloTotal'] + 1e-6)
    # Theoretical Elo win probabilities
    df['EloWinProbHome'] = 1.0 / (1.0 + 10.0 ** (-df['EloDiff'] / 400.0))
    df['EloWinProbAway'] = 1.0 / (1.0 + 10.0 ** (df['EloDiff'] / 400.0))
    
    # 2. Form Features (Form3, Form5 provided pre-match in dataset)
    df['Form3Home'] = df['Form3Home'].fillna(4.0)
    df['Form5Home'] = df['Form5Home'].fillna(7.0)
    df['Form3Away'] = df['Form3Away'].fillna(4.0)
    df['Form5Away'] = df['Form5Away'].fillna(7.0)
    df['Form3Diff'] = df['Form3Home'] - df['Form3Away']
    df['Form5Diff'] = df['Form5Home'] - df['Form5Away']
    df['FormMomentumHome'] = df['Form3Home'] - (df['Form5Home'] - df['Form3Home'])
    df['FormMomentumAway'] = df['Form3Away'] - (df['Form5Away'] - df['Form3Away'])
    df['FormMomentumDiff'] = df['FormMomentumHome'] - df['FormMomentumAway']
    
    # Precompute pre-match rest days, congestion, and Head-to-Head (H2H) history
    last_match_date = {}
    h2h_history = defaultdict(lambda: deque(maxlen=6))
    
    days_rest_home = np.zeros(n, dtype=np.float32)
    days_rest_away = np.zeros(n, dtype=np.float32)
    h2h_home_win_rate = np.zeros(n, dtype=np.float32)
    h2h_draw_rate = np.zeros(n, dtype=np.float32)
    h2h_goal_diff = np.zeros(n, dtype=np.float32)
    
    DEFAULT_REST = 7.0
    dates = pd.to_datetime(df['MatchDate']).values
    home_teams = df['HomeTeam'].values
    away_teams = df['AwayTeam'].values
    ft_homes = df['FTHome'].values
    ft_aways = df['FTAway'].values
    ft_results = df['FTResult'].values
    
    for i in range(n):
        ht = home_teams[i]
        at = away_teams[i]
        dt = dates[i]
        
        # 1. Pre-match rest days (strictly prior to current match)
        if ht in last_match_date:
            d = (pd.to_datetime(dt) - pd.to_datetime(last_match_date[ht])).days
            days_rest_home[i] = min(max(float(d), 2.0), 30.0)
        else:
            days_rest_home[i] = DEFAULT_REST
            
        if at in last_match_date:
            d = (pd.to_datetime(dt) - pd.to_datetime(last_match_date[at])).days
            days_rest_away[i] = min(max(float(d), 2.0), 30.0)
        else:
            days_rest_away[i] = DEFAULT_REST
            
        # 2. Pre-match H2H history between this pair of teams
        pair_key = tuple(sorted([ht, at]))
        past_encounters = h2h_history[pair_key]
        if len(past_encounters) >= 2:
            h_wins = sum(1 for m in past_encounters if (m['home'] == ht and m['res'] == 'H') or (m['away'] == ht and m['res'] == 'A'))
            draws = sum(1 for m in past_encounters if m['res'] == 'D')
            gd = sum((m['hg'] - m['ag']) if m['home'] == ht else (m['ag'] - m['hg']) for m in past_encounters)
            
            h2h_home_win_rate[i] = h_wins / len(past_encounters)
            h2h_draw_rate[i] = draws / len(past_encounters)
            h2h_goal_diff[i] = gd / len(past_encounters)
        else:
            h2h_home_win_rate[i] = 0.45
            h2h_draw_rate[i] = 0.26
            h2h_goal_diff[i] = 0.20
            
        last_match_date[ht] = dt
        last_match_date[at] = dt
        
        hg = ft_homes[i] if not np.isnan(ft_homes[i]) else 1.0
        ag = ft_aways[i] if not np.isnan(ft_aways[i]) else 1.0
        h2h_history[pair_key].append({
            'home': ht,
            'away': at,
            'res': ft_results[i],
            'hg': hg,
            'ag': ag
        })
        
    df['DaysRestHome'] = days_rest_home
    df['DaysRestAway'] = days_rest_away
    df['RestAdvantage'] = days_rest_home - days_rest_away
    df['CongestedHome'] = (days_rest_home <= 3.5).astype(float)
    df['CongestedAway'] = (days_rest_away <= 3.5).astype(float)
    df['H2H_HomeWinRate'] = h2h_home_win_rate
    df['H2H_DrawRate'] = h2h_draw_rate
    df['H2H_GoalDiff'] = h2h_goal_diff

    # 3. Dynamic Rolling Features strictly prior to match t
    # Data structures for tracking match history per team
    team_history = defaultdict(lambda: deque(maxlen=10))
    venue_home_history = defaultdict(lambda: deque(maxlen=20))
    venue_away_history = defaultdict(lambda: deque(maxlen=20))
    
    # Pre-allocate arrays for efficiency
    feat_arrays = {
        'HomeGoalsScored3': np.zeros(n, dtype=np.float32),
        'HomeGoalsConceded3': np.zeros(n, dtype=np.float32),
        'AwayGoalsScored3': np.zeros(n, dtype=np.float32),
        'AwayGoalsConceded3': np.zeros(n, dtype=np.float32),
        'HomeGoalsScored5': np.zeros(n, dtype=np.float32),
        'HomeGoalsConceded5': np.zeros(n, dtype=np.float32),
        'AwayGoalsScored5': np.zeros(n, dtype=np.float32),
        'AwayGoalsConceded5': np.zeros(n, dtype=np.float32),
        'HomeGoalsScored10': np.zeros(n, dtype=np.float32),
        'HomeGoalsConceded10': np.zeros(n, dtype=np.float32),
        'AwayGoalsScored10': np.zeros(n, dtype=np.float32),
        'AwayGoalsConceded10': np.zeros(n, dtype=np.float32),
        'HomeWinRateVenue': np.zeros(n, dtype=np.float32),
        'HomeGoalsScoredVenue': np.zeros(n, dtype=np.float32),
        'HomeGoalsConcededVenue': np.zeros(n, dtype=np.float32),
        'AwayWinRateVenue': np.zeros(n, dtype=np.float32),
        'AwayGoalsScoredVenue': np.zeros(n, dtype=np.float32),
        'AwayGoalsConcededVenue': np.zeros(n, dtype=np.float32),
        'HomeShotsL5': np.zeros(n, dtype=np.float32),
        'AwayShotsL5': np.zeros(n, dtype=np.float32),
        'HomeSOTL5': np.zeros(n, dtype=np.float32),
        'AwaySOTL5': np.zeros(n, dtype=np.float32),
        'HomeCornersL5': np.zeros(n, dtype=np.float32),
        'AwayCornersL5': np.zeros(n, dtype=np.float32),
        'HomeFoulsL5': np.zeros(n, dtype=np.float32),
        'AwayFoulsL5': np.zeros(n, dtype=np.float32),
        'HomeCardPointsL5': np.zeros(n, dtype=np.float32),
        'AwayCardPointsL5': np.zeros(n, dtype=np.float32),
        'HomeAvgOpponentEloL5': np.zeros(n, dtype=np.float32),
        'AwayAvgOpponentEloL5': np.zeros(n, dtype=np.float32),
    }
    
    # Default priors when history is short
    DEFAULT_GOALS_SCORED = 1.35
    DEFAULT_GOALS_CONCEDED = 1.35
    DEFAULT_WIN_RATE_HOME = 0.45
    DEFAULT_WIN_RATE_AWAY = 0.28
    DEFAULT_SHOTS = 12.0
    DEFAULT_SOT = 4.5
    DEFAULT_CORNERS = 5.0
    DEFAULT_FOULS = 13.0
    DEFAULT_CARD_POINTS = 4.0
    DEFAULT_ELO = 1500.0
    
    # Iterate chronologically through matches
    home_teams = df['HomeTeam'].values
    away_teams = df['AwayTeam'].values
    home_elos = df['HomeElo'].values
    away_elos = df['AwayElo'].values
    ft_homes = df['FTHome'].values
    ft_aways = df['FTAway'].values
    ft_results = df['FTResult'].values
    
    home_shots_arr = df['HomeShots'].fillna(DEFAULT_SHOTS).values
    away_shots_arr = df['AwayShots'].fillna(DEFAULT_SHOTS).values
    home_sot_arr = df['HomeTarget'].fillna(DEFAULT_SOT).values
    away_sot_arr = df['AwayTarget'].fillna(DEFAULT_SOT).values
    home_corners_arr = df['HomeCorners'].fillna(DEFAULT_CORNERS).values
    away_corners_arr = df['AwayCorners'].fillna(DEFAULT_CORNERS).values
    home_fouls_arr = df['HomeFouls'].fillna(DEFAULT_FOULS).values
    away_fouls_arr = df['AwayFouls'].fillna(DEFAULT_FOULS).values
    home_yellow_arr = df['HomeYellow'].fillna(1.5).values
    away_yellow_arr = df['AwayYellow'].fillna(1.5).values
    home_red_arr = df['HomeRed'].fillna(0.1).values
    away_red_arr = df['AwayRed'].fillna(0.1).values
    
    for i in range(n):
        ht = home_teams[i]
        at = away_teams[i]
        h_elo = home_elos[i]
        a_elo = away_elos[i]
        
        # 1. Historical goals & stats for Home Team (strictly past matches)
        h_hist = team_history[ht]
        if len(h_hist) >= 3:
            h_last3 = list(h_hist)[-3:]
            feat_arrays['HomeGoalsScored3'][i] = np.mean([m['goals_scored'] for m in h_last3])
            feat_arrays['HomeGoalsConceded3'][i] = np.mean([m['goals_conceded'] for m in h_last3])
        else:
            feat_arrays['HomeGoalsScored3'][i] = DEFAULT_GOALS_SCORED
            feat_arrays['HomeGoalsConceded3'][i] = DEFAULT_GOALS_CONCEDED
            
        if len(h_hist) >= 5:
            h_last5 = list(h_hist)[-5:]
            feat_arrays['HomeGoalsScored5'][i] = np.mean([m['goals_scored'] for m in h_last5])
            feat_arrays['HomeGoalsConceded5'][i] = np.mean([m['goals_conceded'] for m in h_last5])
            feat_arrays['HomeShotsL5'][i] = np.mean([m['shots'] for m in h_last5])
            feat_arrays['HomeSOTL5'][i] = np.mean([m['sot'] for m in h_last5])
            feat_arrays['HomeCornersL5'][i] = np.mean([m['corners'] for m in h_last5])
            feat_arrays['HomeFoulsL5'][i] = np.mean([m['fouls'] for m in h_last5])
            feat_arrays['HomeCardPointsL5'][i] = np.mean([m['card_points'] for m in h_last5])
            feat_arrays['HomeAvgOpponentEloL5'][i] = np.mean([m['opp_elo'] for m in h_last5])
        else:
            feat_arrays['HomeGoalsScored5'][i] = DEFAULT_GOALS_SCORED
            feat_arrays['HomeGoalsConceded5'][i] = DEFAULT_GOALS_CONCEDED
            feat_arrays['HomeShotsL5'][i] = DEFAULT_SHOTS
            feat_arrays['HomeSOTL5'][i] = DEFAULT_SOT
            feat_arrays['HomeCornersL5'][i] = DEFAULT_CORNERS
            feat_arrays['HomeFoulsL5'][i] = DEFAULT_FOULS
            feat_arrays['HomeCardPointsL5'][i] = DEFAULT_CARD_POINTS
            feat_arrays['HomeAvgOpponentEloL5'][i] = DEFAULT_ELO
            
        if len(h_hist) >= 10:
            h_last10 = list(h_hist)[-10:]
            feat_arrays['HomeGoalsScored10'][i] = np.mean([m['goals_scored'] for m in h_last10])
            feat_arrays['HomeGoalsConceded10'][i] = np.mean([m['goals_conceded'] for m in h_last10])
        else:
            feat_arrays['HomeGoalsScored10'][i] = DEFAULT_GOALS_SCORED
            feat_arrays['HomeGoalsConceded10'][i] = DEFAULT_GOALS_CONCEDED
            
        # 2. Historical goals & stats for Away Team (strictly past matches)
        a_hist = team_history[at]
        if len(a_hist) >= 3:
            a_last3 = list(a_hist)[-3:]
            feat_arrays['AwayGoalsScored3'][i] = np.mean([m['goals_scored'] for m in a_last3])
            feat_arrays['AwayGoalsConceded3'][i] = np.mean([m['goals_conceded'] for m in a_last3])
        else:
            feat_arrays['AwayGoalsScored3'][i] = DEFAULT_GOALS_SCORED
            feat_arrays['AwayGoalsConceded3'][i] = DEFAULT_GOALS_CONCEDED
            
        if len(a_hist) >= 5:
            a_last5 = list(a_hist)[-5:]
            feat_arrays['AwayGoalsScored5'][i] = np.mean([m['goals_scored'] for m in a_last5])
            feat_arrays['AwayGoalsConceded5'][i] = np.mean([m['goals_conceded'] for m in a_last5])
            feat_arrays['AwayShotsL5'][i] = np.mean([m['shots'] for m in a_last5])
            feat_arrays['AwaySOTL5'][i] = np.mean([m['sot'] for m in a_last5])
            feat_arrays['AwayCornersL5'][i] = np.mean([m['corners'] for m in a_last5])
            feat_arrays['AwayFoulsL5'][i] = np.mean([m['fouls'] for m in a_last5])
            feat_arrays['AwayCardPointsL5'][i] = np.mean([m['card_points'] for m in a_last5])
            feat_arrays['AwayAvgOpponentEloL5'][i] = np.mean([m['opp_elo'] for m in a_last5])
        else:
            feat_arrays['AwayGoalsScored5'][i] = DEFAULT_GOALS_SCORED
            feat_arrays['AwayGoalsConceded5'][i] = DEFAULT_GOALS_CONCEDED
            feat_arrays['AwayShotsL5'][i] = DEFAULT_SHOTS
            feat_arrays['AwaySOTL5'][i] = DEFAULT_SOT
            feat_arrays['AwayCornersL5'][i] = DEFAULT_CORNERS
            feat_arrays['AwayFoulsL5'][i] = DEFAULT_FOULS
            feat_arrays['AwayCardPointsL5'][i] = DEFAULT_CARD_POINTS
            feat_arrays['AwayAvgOpponentEloL5'][i] = DEFAULT_ELO
            
        if len(a_hist) >= 10:
            a_last10 = list(a_hist)[-10:]
            feat_arrays['AwayGoalsScored10'][i] = np.mean([m['goals_scored'] for m in a_last10])
            feat_arrays['AwayGoalsConceded10'][i] = np.mean([m['goals_conceded'] for m in a_last10])
        else:
            feat_arrays['AwayGoalsScored10'][i] = DEFAULT_GOALS_SCORED
            feat_arrays['AwayGoalsConceded10'][i] = DEFAULT_GOALS_CONCEDED
            
        # 3. Venue-specific history
        h_venue = venue_home_history[ht]
        if len(h_venue) >= 3:
            feat_arrays['HomeWinRateVenue'][i] = np.mean([m['is_win'] for m in h_venue])
            feat_arrays['HomeGoalsScoredVenue'][i] = np.mean([m['goals_scored'] for m in h_venue])
            feat_arrays['HomeGoalsConcededVenue'][i] = np.mean([m['goals_conceded'] for m in h_venue])
        else:
            feat_arrays['HomeWinRateVenue'][i] = DEFAULT_WIN_RATE_HOME
            feat_arrays['HomeGoalsScoredVenue'][i] = DEFAULT_GOALS_SCORED + 0.2
            feat_arrays['HomeGoalsConcededVenue'][i] = DEFAULT_GOALS_CONCEDED - 0.2
            
        a_venue = venue_away_history[at]
        if len(a_venue) >= 3:
            feat_arrays['AwayWinRateVenue'][i] = np.mean([m['is_win'] for m in a_venue])
            feat_arrays['AwayGoalsScoredVenue'][i] = np.mean([m['goals_scored'] for m in a_venue])
            feat_arrays['AwayGoalsConcededVenue'][i] = np.mean([m['goals_conceded'] for m in a_venue])
        else:
            feat_arrays['AwayWinRateVenue'][i] = DEFAULT_WIN_RATE_AWAY
            feat_arrays['AwayGoalsScoredVenue'][i] = DEFAULT_GOALS_SCORED - 0.2
            feat_arrays['AwayGoalsConcededVenue'][i] = DEFAULT_GOALS_CONCEDED + 0.2
            
        # 4. POST-CALCULATION UPDATE: Update histories with match i result
        hg = ft_homes[i] if not np.isnan(ft_homes[i]) else 1.0
        ag = ft_aways[i] if not np.isnan(ft_aways[i]) else 1.0
        res = ft_results[i]
        
        # Using red card multiplier = 3 as explicitly recommended in Paper Table 7
        h_card = home_yellow_arr[i] + 3.0 * home_red_arr[i]
        a_card = away_yellow_arr[i] + 3.0 * away_red_arr[i]
        
        team_history[ht].append({
            'goals_scored': hg,
            'goals_conceded': ag,
            'shots': home_shots_arr[i],
            'sot': home_sot_arr[i],
            'corners': home_corners_arr[i],
            'fouls': home_fouls_arr[i],
            'card_points': h_card,
            'opp_elo': a_elo
        })
        team_history[at].append({
            'goals_scored': ag,
            'goals_conceded': hg,
            'shots': away_shots_arr[i],
            'sot': away_sot_arr[i],
            'corners': away_corners_arr[i],
            'fouls': away_fouls_arr[i],
            'card_points': a_card,
            'opp_elo': h_elo
        })
        
        venue_home_history[ht].append({
            'is_win': 1.0 if res == 'H' else 0.0,
            'goals_scored': hg,
            'goals_conceded': ag
        })
        venue_away_history[at].append({
            'is_win': 1.0 if res == 'A' else 0.0,
            'goals_scored': ag,
            'goals_conceded': hg
        })
        
    # Assign arrays to DataFrame
    for k, v in feat_arrays.items():
        df[k] = v
        
    # Compute relative differences and matchup features
    df['GoalsScoredDiff3'] = df['HomeGoalsScored3'] - df['AwayGoalsScored3']
    df['GoalsConcededDiff3'] = df['HomeGoalsConceded3'] - df['AwayGoalsConceded3']
    df['GoalsScoredDiff5'] = df['HomeGoalsScored5'] - df['AwayGoalsScored5']
    df['GoalsConcededDiff5'] = df['HomeGoalsConceded5'] - df['AwayGoalsConceded5']
    df['GoalsScoredDiff10'] = df['HomeGoalsScored10'] - df['AwayGoalsScored10']
    df['GoalsConcededDiff10'] = df['HomeGoalsConceded10'] - df['AwayGoalsConceded10']
    df['GoalDiffMargin5'] = (df['HomeGoalsScored5'] - df['HomeGoalsConceded5']) - (df['AwayGoalsScored5'] - df['AwayGoalsConceded5'])
    df['HomeAttack_vs_AwayDefence'] = df['HomeGoalsScored5'] - df['AwayGoalsConceded5']
    df['AwayAttack_vs_HomeDefence'] = df['AwayGoalsScored5'] - df['HomeGoalsConceded5']
    df['NetMatchupExpectation'] = df['HomeAttack_vs_AwayDefence'] - df['AwayAttack_vs_HomeDefence']
    
    # Recency momentum trend (L3 vs L10)
    df['RecencyMomentumHome'] = df['HomeGoalsScored3'] - df['HomeGoalsScored10']
    df['RecencyMomentumAway'] = df['AwayGoalsScored3'] - df['AwayGoalsScored10']
    df['RecencyMomentumDiff'] = df['RecencyMomentumHome'] - df['RecencyMomentumAway']
    
    # Opponent-Adjusted Performance (Section 32.C from User README)
    df['OpponentAdjGoalsScoredHome'] = df['HomeGoalsScored5'] * (df['HomeAvgOpponentEloL5'] / 1500.0)
    df['OpponentAdjGoalsScoredAway'] = df['AwayGoalsScored5'] * (df['AwayAvgOpponentEloL5'] / 1500.0)
    df['OpponentAdjGoalsDiff'] = df['OpponentAdjGoalsScoredHome'] - df['OpponentAdjGoalsScoredAway']
    df['OpponentAdjFormHome'] = df['Form5Home'] * (df['HomeAvgOpponentEloL5'] / 1500.0)
    df['OpponentAdjFormAway'] = df['Form5Away'] * (df['AwayAvgOpponentEloL5'] / 1500.0)
    df['OpponentAdjFormDiff'] = df['OpponentAdjFormHome'] - df['OpponentAdjFormAway']
    
    # Venue differences
    df['VenueWinRateDiff'] = df['HomeWinRateVenue'] - df['AwayWinRateVenue']
    df['VenueGoalsScoredDiff'] = df['HomeGoalsScoredVenue'] - df['AwayGoalsScoredVenue']
    df['VenueGoalsConcededDiff'] = df['HomeGoalsConcededVenue'] - df['AwayGoalsConcededVenue']
    
    # Match statistics differences & Table 7 totals
    df['ShotsDiffL5'] = df['HomeShotsL5'] - df['AwayShotsL5']
    df['ShotsTotalL5'] = df['HomeShotsL5'] + df['AwayShotsL5']
    df['SOTDiffL5'] = df['HomeSOTL5'] - df['AwaySOTL5']
    df['CornersDiffL5'] = df['HomeCornersL5'] - df['AwayCornersL5']
    df['CornersTotalL5'] = df['HomeCornersL5'] + df['AwayCornersL5']
    df['FoulsDiffL5'] = df['HomeFoulsL5'] - df['AwayFoulsL5']
    df['CardPointsDiffL5'] = df['HomeCardPointsL5'] - df['AwayCardPointsL5']
    df['HomeShotAccL5'] = df['HomeSOTL5'] / (df['HomeShotsL5'] + 1e-5)
    df['AwayShotAccL5'] = df['AwaySOTL5'] / (df['AwayShotsL5'] + 1e-5)
    df['ShotAccDiffL5'] = df['HomeShotAccL5'] - df['AwayShotAccL5']
    
    # Game Dominance Index from Table 7 in Paper
    df['GameDominanceIndex'] = (df['CornersDiffL5'] + df['ShotsDiffL5']) / 2.0
    df['DominanceIndexL5'] = df['GameDominanceIndex']
    
    # 4. Betting Odds & Market Expectations (Table 7 derived odds)
    odd_h = df['OddHome'].fillna(2.5).clip(1.05, 30.0)
    odd_d = df['OddDraw'].fillna(3.3).clip(1.05, 30.0)
    odd_a = df['OddAway'].fillna(3.2).clip(1.05, 30.0)
    
    raw_prob_h = 1.0 / odd_h
    raw_prob_d = 1.0 / odd_d
    raw_prob_a = 1.0 / odd_a
    prob_total = raw_prob_h + raw_prob_d + raw_prob_a
    
    df['ImpliedProbHome'] = raw_prob_h
    df['ImpliedProbDraw'] = raw_prob_d
    df['ImpliedProbAway'] = raw_prob_a
    df['BookmakerMargin'] = prob_total - 1.0
    df['NormProbHome'] = raw_prob_h / prob_total
    df['NormProbDraw'] = raw_prob_d / prob_total
    df['NormProbAway'] = raw_prob_a / prob_total
    df['OddsRatio_H_A'] = odd_a / (odd_h + 1e-5)
    
    # Double Chance Odds from Table 7
    df['1XOdd'] = 1.0 / (raw_prob_h + raw_prob_d + 1e-5)
    df['X2Odd'] = 1.0 / (raw_prob_a + raw_prob_d + 1e-5)
    df['12Odd'] = 1.0 / (raw_prob_h + raw_prob_a + 1e-5)
    
    # Draw Likelihood metric from Paper Table 7
    # Derived from EloDifference, Form5Difference, and ImpliedProbDraw
    elo_draw_penalty = np.exp(-np.abs(df['EloDiff']) / 400.0)
    form_draw_penalty = np.exp(-np.abs(df['Form5Diff']) / 15.0)
    df['DrawLikelihood'] = df['NormProbDraw'] * elo_draw_penalty * form_draw_penalty
    
    # 5. Market Handicap & Over/Under Total Signals (strictly pre-match market lines)
    elo_implied_handi = -(df['EloDiff'] / 400.0) * 0.75
    df['MarketHandiSize'] = df['HandiSize'].fillna(elo_implied_handi)
    df['MarketHandiDiscrepancy'] = df['MarketHandiSize'] - elo_implied_handi
    
    df['MarketOver25'] = df['Over25'].fillna(1.95)
    df['MarketUnder25'] = df['Under25'].fillna(1.95)
    df['MarketImpliedOver25'] = 1.0 / df['MarketOver25']
    df['MarketImpliedUnder25'] = 1.0 / df['MarketUnder25']
    df['MarketGoalExpectancyRatio'] = df['MarketImpliedOver25'] / (df['MarketImpliedUnder25'] + 1e-5)
    
    # Bookmaker spread and variance
    max_h = df['MaxHome'].fillna(odd_h)
    max_d = df['MaxDraw'].fillna(odd_d)
    max_a = df['MaxAway'].fillna(odd_a)
    df['OddsSpreadHome'] = max_h - odd_h
    df['OddsSpreadDraw'] = max_d - odd_d
    df['OddsSpreadAway'] = max_a - odd_a
    
    # Cross-feature interactions
    df['EloDiff_x_ImpliedOver25'] = df['EloDiff'] * df['MarketImpliedOver25']
    df['NormHome_x_NormDraw'] = df['NormProbHome'] * df['NormProbDraw']

    # Division encoding (one-hot for league context)
    div_dummies = pd.get_dummies(df['Division'], prefix='League', dtype=float)
    df = pd.concat([df, div_dummies], axis=1)
    
    return df


# Feature groups definitions for experimental progression E0 -> E9
FEATURE_GROUPS = {
    'E0_Baseline': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1'
    ],
    'E1_Elo': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'HomeElo', 'AwayElo', 'EloDiff', 'EloAdvantage', 'EloWinProbHome', 'EloWinProbAway'
    ],
    'E2_Form': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'HomeElo', 'AwayElo', 'EloDiff', 'EloAdvantage', 'EloWinProbHome', 'EloWinProbAway',
        'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away', 'Form3Diff', 'Form5Diff',
        'FormMomentumHome', 'FormMomentumAway', 'FormMomentumDiff'
    ],
    'E3_Goals': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'HomeElo', 'AwayElo', 'EloDiff', 'EloAdvantage', 'EloWinProbHome', 'EloWinProbAway',
        'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away', 'Form3Diff', 'Form5Diff',
        'FormMomentumHome', 'FormMomentumAway', 'FormMomentumDiff',
        'HomeGoalsScored3', 'HomeGoalsConceded3', 'AwayGoalsScored3', 'AwayGoalsConceded3',
        'HomeGoalsScored5', 'HomeGoalsConceded5', 'AwayGoalsScored5', 'AwayGoalsConceded5',
        'GoalsScoredDiff3', 'GoalsConcededDiff3', 'GoalsScoredDiff5', 'GoalsConcededDiff5',
        'GoalDiffMargin5', 'HomeAttack_vs_AwayDefence', 'AwayAttack_vs_HomeDefence'
    ],
    'E4_Venue': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'HomeElo', 'AwayElo', 'EloDiff', 'EloAdvantage', 'EloWinProbHome', 'EloWinProbAway',
        'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away', 'Form3Diff', 'Form5Diff',
        'FormMomentumHome', 'FormMomentumAway', 'FormMomentumDiff',
        'HomeGoalsScored3', 'HomeGoalsConceded3', 'AwayGoalsScored3', 'AwayGoalsConceded3',
        'HomeGoalsScored5', 'HomeGoalsConceded5', 'AwayGoalsScored5', 'AwayGoalsConceded5',
        'GoalsScoredDiff3', 'GoalsConcededDiff3', 'GoalsScoredDiff5', 'GoalsConcededDiff5',
        'GoalDiffMargin5', 'HomeAttack_vs_AwayDefence', 'AwayAttack_vs_HomeDefence',
        'HomeWinRateVenue', 'HomeGoalsScoredVenue', 'HomeGoalsConcededVenue',
        'AwayWinRateVenue', 'AwayGoalsScoredVenue', 'AwayGoalsConcededVenue',
        'VenueWinRateDiff', 'VenueGoalsScoredDiff', 'VenueGoalsConcededDiff'
    ],
    'E5_MatchStats': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'HomeElo', 'AwayElo', 'EloDiff', 'EloAdvantage', 'EloWinProbHome', 'EloWinProbAway',
        'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away', 'Form3Diff', 'Form5Diff',
        'FormMomentumHome', 'FormMomentumAway', 'FormMomentumDiff',
        'HomeGoalsScored3', 'HomeGoalsConceded3', 'AwayGoalsScored3', 'AwayGoalsConceded3',
        'HomeGoalsScored5', 'HomeGoalsConceded5', 'AwayGoalsScored5', 'AwayGoalsConceded5',
        'GoalsScoredDiff3', 'GoalsConcededDiff3', 'GoalsScoredDiff5', 'GoalsConcededDiff5',
        'GoalDiffMargin5', 'HomeAttack_vs_AwayDefence', 'AwayAttack_vs_HomeDefence',
        'HomeWinRateVenue', 'HomeGoalsScoredVenue', 'HomeGoalsConcededVenue',
        'AwayWinRateVenue', 'AwayGoalsScoredVenue', 'AwayGoalsConcededVenue',
        'VenueWinRateDiff', 'VenueGoalsScoredDiff', 'VenueGoalsConcededDiff',
        'HomeShotsL5', 'AwayShotsL5', 'ShotsDiffL5',
        'HomeSOTL5', 'AwaySOTL5', 'SOTDiffL5',
        'HomeCornersL5', 'AwayCornersL5', 'CornersDiffL5',
        'HomeFoulsL5', 'AwayFoulsL5', 'FoulsDiffL5',
        'HomeCardPointsL5', 'AwayCardPointsL5', 'CardPointsDiffL5',
        'HomeShotAccL5', 'AwayShotAccL5', 'ShotAccDiffL5', 'DominanceIndexL5'
    ],
    'E6_Odds': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'HomeElo', 'AwayElo', 'EloDiff', 'EloAdvantage', 'EloWinProbHome', 'EloWinProbAway',
        'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away', 'Form3Diff', 'Form5Diff',
        'FormMomentumHome', 'FormMomentumAway', 'FormMomentumDiff',
        'HomeGoalsScored3', 'HomeGoalsConceded3', 'AwayGoalsScored3', 'AwayGoalsConceded3',
        'HomeGoalsScored5', 'HomeGoalsConceded5', 'AwayGoalsScored5', 'AwayGoalsConceded5',
        'GoalsScoredDiff3', 'GoalsConcededDiff3', 'GoalsScoredDiff5', 'GoalsConcededDiff5',
        'GoalDiffMargin5', 'HomeAttack_vs_AwayDefence', 'AwayAttack_vs_HomeDefence',
        'HomeWinRateVenue', 'HomeGoalsScoredVenue', 'HomeGoalsConcededVenue',
        'AwayWinRateVenue', 'AwayGoalsScoredVenue', 'AwayGoalsConcededVenue',
        'VenueWinRateDiff', 'VenueGoalsScoredDiff', 'VenueGoalsConcededDiff',
        'HomeShotsL5', 'AwayShotsL5', 'ShotsDiffL5',
        'HomeSOTL5', 'AwaySOTL5', 'SOTDiffL5',
        'HomeCornersL5', 'AwayCornersL5', 'CornersDiffL5',
        'HomeFoulsL5', 'AwayFoulsL5', 'FoulsDiffL5',
        'HomeCardPointsL5', 'AwayCardPointsL5', 'CardPointsDiffL5',
        'HomeShotAccL5', 'AwayShotAccL5', 'ShotAccDiffL5', 'DominanceIndexL5',
        'ImpliedProbHome', 'ImpliedProbDraw', 'ImpliedProbAway',
        'BookmakerMargin', 'NormProbHome', 'NormProbDraw', 'NormProbAway', 'OddsRatio_H_A'
    ],
    'E7_Selected': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'EloDiff', 'EloAdvantage', 'EloWinProbHome',
        'Form5Diff', 'FormMomentumDiff',
        'GoalsScoredDiff5', 'GoalsConcededDiff5', 'GoalDiffMargin5',
        'HomeAttack_vs_AwayDefence', 'AwayAttack_vs_HomeDefence',
        'VenueWinRateDiff', 'VenueGoalsScoredDiff', 'VenueGoalsConcededDiff',
        'ShotsDiffL5', 'SOTDiffL5', 'CornersDiffL5', 'ShotAccDiffL5', 'DominanceIndexL5',
        'NormProbHome', 'NormProbDraw', 'NormProbAway', 'BookmakerMargin', 'OddsRatio_H_A'
    ],
    'E8_Paper_Advanced': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'EloDiff', 'EloAdvantage', 'EloWinProbHome',
        'Form5Diff', 'FormMomentumDiff',
        'GoalsScoredDiff5', 'GoalsConcededDiff5', 'GoalDiffMargin5',
        'HomeAttack_vs_AwayDefence', 'AwayAttack_vs_HomeDefence', 'NetMatchupExpectation',
        'RecencyMomentumDiff',
        'OpponentAdjGoalsDiff', 'OpponentAdjFormDiff',
        'VenueWinRateDiff', 'VenueGoalsScoredDiff', 'VenueGoalsConcededDiff',
        'ShotsDiffL5', 'ShotsTotalL5', 'SOTDiffL5', 'CornersDiffL5', 'CornersTotalL5',
        'ShotAccDiffL5', 'GameDominanceIndex', 'CardPointsDiffL5',
        'NormProbHome', 'NormProbDraw', 'NormProbAway', 'BookmakerMargin', 'OddsRatio_H_A',
        '1XOdd', 'X2Odd', '12Odd', 'DrawLikelihood'
    ],
    'E9_Expanded_Market_Fatigue': [
        'League_D1', 'League_E0', 'League_F1', 'League_I1', 'League_SP1',
        'EloDiff', 'EloAdvantage', 'EloWinProbHome',
        'Form5Diff', 'FormMomentumDiff',
        'GoalsScoredDiff5', 'GoalsConcededDiff5', 'GoalDiffMargin5',
        'HomeAttack_vs_AwayDefence', 'AwayAttack_vs_HomeDefence', 'NetMatchupExpectation',
        'RecencyMomentumDiff',
        'OpponentAdjGoalsDiff', 'OpponentAdjFormDiff',
        'VenueWinRateDiff', 'VenueGoalsScoredDiff', 'VenueGoalsConcededDiff',
        'ShotsDiffL5', 'ShotsTotalL5', 'SOTDiffL5', 'CornersDiffL5', 'CornersTotalL5',
        'ShotAccDiffL5', 'GameDominanceIndex', 'CardPointsDiffL5',
        'NormProbHome', 'NormProbDraw', 'NormProbAway', 'BookmakerMargin', 'OddsRatio_H_A',
        '1XOdd', 'X2Odd', '12Odd', 'DrawLikelihood',
        'DaysRestHome', 'DaysRestAway', 'RestAdvantage', 'CongestedHome', 'CongestedAway',
        'H2H_HomeWinRate', 'H2H_DrawRate', 'H2H_GoalDiff',
        'MarketHandiSize', 'MarketHandiDiscrepancy',
        'MarketImpliedOver25', 'MarketGoalExpectancyRatio',
        'OddsSpreadHome', 'OddsSpreadDraw', 'OddsSpreadAway',
        'EloDiff_x_ImpliedOver25', 'NormHome_x_NormDraw'
    ]
}

