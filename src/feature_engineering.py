"""Create richer model features from cleaned match data.

Adds: recent home/away form, head-to-head, simple Elo rating, days-rest, and bookmaker odds-derived features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from datetime import timedelta


def _average(values):
    return float(sum(values)) / len(values) if values else 0.0


def _exp_weighted_avg(values, decay=0.6):
    # exponential decay: more recent matches have higher weight
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    # newest value should have largest weight -> reverse powers
    weights = decay ** np.arange(len(arr) - 1, -1, -1)
    w = weights / weights.sum()
    return float((arr * w).sum())


def _points_for_score(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 3
    if home_goals == away_goals:
        return 1
    return 0


def _goal_diff(home_goals: int, away_goals: int) -> int:
    return home_goals - away_goals


def _init_team_stats() -> dict:
    return {
        'home_points': [],
        'home_goals_for': [],
        'home_goals_against': [],
        'home_goal_diff': [],
        'away_points': [],
        'away_goals_for': [],
        'away_goals_against': [],
        'away_goal_diff': [],
        'all_points': [],
        'all_goals_for': [],
        'all_goals_against': [],
        'all_goal_diff': [],
        'last_date': None,
        'elo': 1500.0,
    }


def _init_h2h_stats() -> dict:
    return {
        'matches': 0,
        'home_points': [],
        'away_points': [],
        'home_goals': [],
        'away_goals': [],
    }


def _extract_odds_from_row(row: pd.Series):
    # prefer stronger market odds when available
    keys_sets = [
        ('B365H', 'B365D', 'B365A'),
        ('PSH', 'PSD', 'PSA'),
        ('BbAvH', 'BbAvD', 'BbAvA'),
        ('AvgH', 'AvgD', 'AvgA'),
        ('BWH', 'BWD', 'BWA'),
        ('WHH', 'WHD', 'WHA'),
    ]
    for h, d, a in keys_sets:
        if h in row.index and d in row.index and a in row.index:
            try:
                oh = float(row[h])
                od = float(row[d])
                oa = float(row[a])
                return oh, od, oa
            except Exception:
                continue
    return None, None, None


def compute_team_history_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute pre-match team statistics from past results only, plus rest, elo and odds."""
    df = df.sort_values('date').reset_index(drop=True).copy()
    # ensure date dtype
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    history_records = []
    team_history = {}
    head2head_history = {}

    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        date = row['date'] if not pd.isna(row['date']) else pd.Timestamp.today()
        h2h_key = (home_team, away_team)

        home_stats = team_history.get(home_team, _init_team_stats())
        away_stats = team_history.get(away_team, _init_team_stats())
        h2h_stats = head2head_history.get(h2h_key, _init_h2h_stats())

        # days since last match
        home_last = home_stats.get('last_date')
        away_last = away_stats.get('last_date')
        home_days_rest = (date - home_last).days if home_last is not None else 7
        away_days_rest = (date - away_last).days if away_last is not None else 7
        rest_diff = home_days_rest - away_days_rest

        # Elo current
        home_elo = home_stats.get('elo', 1500.0)
        away_elo = away_stats.get('elo', 1500.0)

        # odds features
        oh, od, oa = _extract_odds_from_row(row)
        implied_h = implied_d = implied_a = 0.0
        if oh and od and oa:
            # convert to implied probabilities and normalize
            probs = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
            probs = probs / probs.sum()
            implied_h, implied_d, implied_a = probs.tolist()

        odds_available = int(oh is not None and od is not None and oa is not None)
        market_probs = np.array([implied_h, implied_d, implied_a], dtype=float)
        market_max_prob = float(market_probs.max()) if odds_available else 0.0
        if odds_available:
            sorted_probs = np.sort(market_probs)
            market_prob_spread = float(sorted_probs[-1] - sorted_probs[-2])
        else:
            market_prob_spread = 0.0

        home_recent_home_points = _exp_weighted_avg(home_stats['home_points'])
        away_recent_away_points = _exp_weighted_avg(away_stats['away_points'])
        home_recent_home_goal_diff = _exp_weighted_avg(home_stats['home_goal_diff'])
        away_recent_away_goal_diff = _exp_weighted_avg(away_stats['away_goal_diff'])

        history_records.append({
            'home_advantage': 1,
            'home_days_rest': float(home_days_rest),
            'away_days_rest': float(away_days_rest),
            'rest_diff': float(rest_diff),
            'home_elo': float(home_elo),
            'away_elo': float(away_elo),
            'elo_diff': float(home_elo - away_elo),
            'implied_h_prob': float(implied_h),
            'implied_d_prob': float(implied_d),
            'implied_a_prob': float(implied_a),
            'odds_available': float(odds_available),
            'market_max_prob': float(market_max_prob),
            'market_prob_spread': float(market_prob_spread),
            'home_recent_home_points': home_recent_home_points,
            'home_recent_home_goals_for': _exp_weighted_avg(home_stats['home_goals_for']),
            'home_recent_home_goals_against': _exp_weighted_avg(home_stats['home_goals_against']),
            'home_recent_home_goal_diff': home_recent_home_goal_diff,
            'home_recent_away_points': _exp_weighted_avg(home_stats['away_points']),
            'home_recent_away_goals_for': _exp_weighted_avg(home_stats['away_goals_for']),
            'home_recent_away_goals_against': _exp_weighted_avg(home_stats['away_goals_against']),
            'home_recent_away_goal_diff': _exp_weighted_avg(home_stats['away_goal_diff']),
            'home_overall_points': _exp_weighted_avg(home_stats['all_points']),
            'home_overall_goals_for': _exp_weighted_avg(home_stats['all_goals_for']),
            'home_overall_goals_against': _exp_weighted_avg(home_stats['all_goals_against']),
            'home_overall_goal_diff': _exp_weighted_avg(home_stats['all_goal_diff']),
            'away_recent_away_points': away_recent_away_points,
            'away_recent_away_goals_for': _exp_weighted_avg(away_stats['away_goals_for']),
            'away_recent_away_goals_against': _exp_weighted_avg(away_stats['away_goals_against']),
            'away_recent_away_goal_diff': away_recent_away_goal_diff,
            'away_recent_home_points': _exp_weighted_avg(away_stats['home_points']),
            'away_recent_home_goals_for': _exp_weighted_avg(away_stats['home_goals_for']),
            'away_recent_home_goals_against': _exp_weighted_avg(away_stats['home_goals_against']),
            'away_recent_home_goal_diff': _exp_weighted_avg(away_stats['home_goal_diff']),
            'away_overall_points': _exp_weighted_avg(away_stats['all_points']),
            'away_overall_goals_for': _exp_weighted_avg(away_stats['all_goals_for']),
            'away_overall_goals_against': _exp_weighted_avg(away_stats['all_goals_against']),
            'away_overall_goal_diff': _exp_weighted_avg(away_stats['all_goal_diff']),
            'home_recent_form_diff': float(home_recent_home_points - away_recent_away_points),
            'home_recent_goal_diff_diff': float(home_recent_home_goal_diff - away_recent_away_goal_diff),
            'h2h_matches': h2h_stats['matches'],
            'h2h_home_avg_points': _average(h2h_stats['home_points']),
            'h2h_away_avg_points': _average(h2h_stats['away_points']),
            'h2h_home_avg_goals': _average(h2h_stats['home_goals']),
            'h2h_away_avg_goals': _average(h2h_stats['away_goals']),
        })

        # update histories
        home_points = _points_for_score(row['home_goals'], row['away_goals'])
        away_points = _points_for_score(row['away_goals'], row['home_goals'])
        home_diff = _goal_diff(row['home_goals'], row['away_goals'])
        away_diff = _goal_diff(row['away_goals'], row['home_goals'])

        home_stats['home_points'].append(home_points)
        home_stats['home_goals_for'].append(row['home_goals'])
        home_stats['home_goals_against'].append(row['away_goals'])
        home_stats['home_goal_diff'].append(home_diff)
        away_stats['away_points'].append(away_points)
        away_stats['away_goals_for'].append(row['away_goals'])
        away_stats['away_goals_against'].append(row['home_goals'])
        away_stats['away_goal_diff'].append(away_diff)

        home_stats['all_points'].append(home_points)
        home_stats['all_goals_for'].append(row['home_goals'])
        home_stats['all_goals_against'].append(row['away_goals'])
        home_stats['all_goal_diff'].append(home_diff)
        away_stats['all_points'].append(away_points)
        away_stats['all_goals_for'].append(row['away_goals'])
        away_stats['all_goals_against'].append(row['home_goals'])
        away_stats['all_goal_diff'].append(away_diff)

        # update elo ratings (simple)
        # expected score
        diff = home_stats['elo'] - away_stats['elo'] + 40.0  # home advantage
        exp_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        exp_away = 1.0 - exp_home
        # actual
        act_home = 1.0 if home_points == 3 else (0.5 if home_points == 1 else 0.0)
        k = 20.0
        home_stats['elo'] = home_stats.get('elo', 1500.0) + k * (act_home - exp_home)
        away_stats['elo'] = away_stats.get('elo', 1500.0) + k * ((1 - act_home) - exp_away)

        # update h2h
        h2h_stats['matches'] += 1
        h2h_stats['home_points'].append(home_points)
        h2h_stats['away_points'].append(away_points)
        h2h_stats['home_goals'].append(row['home_goals'])
        h2h_stats['away_goals'].append(row['away_goals'])

        # update last date
        home_stats['last_date'] = date
        away_stats['last_date'] = date

        # trim windows
        if len(home_stats['home_points']) > window:
            home_stats['home_points'].pop(0)
            home_stats['home_goals_for'].pop(0)
            home_stats['home_goals_against'].pop(0)
            home_stats['home_goal_diff'].pop(0)
        if len(away_stats['away_points']) > window:
            away_stats['away_points'].pop(0)
            away_stats['away_goals_for'].pop(0)
            away_stats['away_goals_against'].pop(0)
            away_stats['away_goal_diff'].pop(0)

        team_history[home_team] = home_stats
        team_history[away_team] = away_stats
        head2head_history[h2h_key] = h2h_stats

    return pd.DataFrame(history_records)


def build_features(df: pd.DataFrame, competition_encoder: OneHotEncoder | None = None):
    """Return feature matrix X, labels y, and the fitted competition encoder."""
    df = df.sort_values('date').reset_index(drop=True)
    stats_df = compute_team_history_stats(df)
    feature_df = pd.concat([df, stats_df], axis=1)

    numeric_cols = [
        'home_advantage',
        'home_days_rest',
        'away_days_rest',
        'rest_diff',
        'home_elo',
        'away_elo',
        'elo_diff',
        'implied_h_prob',
        'implied_d_prob',
        'implied_a_prob',
        'odds_available',
        'market_max_prob',
        'market_prob_spread',
        'home_recent_home_points',
        'home_recent_home_goal_diff',
        'home_recent_home_goals_for',
        'home_recent_home_goals_against',
        'home_recent_away_points',
        'home_recent_away_goal_diff',
        'home_recent_away_goals_for',
        'home_recent_away_goals_against',
        'home_recent_form_diff',
        'home_recent_goal_diff_diff',
        'home_overall_points',
        'home_overall_goal_diff',
        'home_overall_goals_for',
        'home_overall_goals_against',
        'away_recent_away_points',
        'away_recent_away_goal_diff',
        'away_recent_away_goals_for',
        'away_recent_away_goals_against',
        'away_recent_home_points',
        'away_recent_home_goal_diff',
        'away_recent_home_goals_for',
        'away_recent_home_goals_against',
        'away_overall_points',
        'away_overall_goal_diff',
        'away_overall_goals_for',
        'away_overall_goals_against',
        'h2h_matches',
        'h2h_home_avg_points',
        'h2h_away_avg_points',
        'h2h_home_avg_goals',
        'h2h_away_avg_goals',
    ]

    X_numeric = feature_df[numeric_cols].fillna(0).astype(float).to_numpy()

    if competition_encoder is None:
        try:
            competition_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            competition_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_competition = competition_encoder.fit_transform(feature_df[['competition']])
    else:
        X_competition = competition_encoder.transform(feature_df[['competition']])

    X = np.hstack([X_numeric, X_competition])
    y = feature_df['result']
    return X, y, competition_encoder


def build_match_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    competition: str,
    competition_encoder: OneHotEncoder,
    window: int = 5,
) -> np.ndarray:
    """Build features for a future match from past history."""
    df = df.sort_values('date').reset_index(drop=True)
    # reuse compute_team_history_stats to obtain rolling stats and current elo/rest
    stats_df = compute_team_history_stats(df, window=window)
    # find last row for the matchup teams (latest overall)
    merged = pd.concat([df, stats_df], axis=1)
    # take the last available stats for each team
    last_home = merged[merged['home_team'] == home_team]
    last_away = merged[merged['away_team'] == away_team]

    # default zeros
    if not last_home.empty:
        home_row = last_home.iloc[-1]
    else:
        home_row = pd.Series({c: 0 for c in stats_df.columns})
    if not last_away.empty:
        away_row = last_away.iloc[-1]
    else:
        away_row = pd.Series({c: 0 for c in stats_df.columns})

    h2h_key = (home_team, away_team)
    # compute h2h from stats_df
    h2h_rows = stats_df.copy()
    # find last h2h record if exists elsewhere
    h2h_match = h2h_rows[(df['home_team'] == home_team) & (df['away_team'] == away_team)]
    if not h2h_match.empty:
        h2h_row = h2h_match.iloc[-1]
    else:
        h2h_row = pd.Series({c: 0 for c in stats_df.columns})

    feature_vector = [
        1.0,
        float(home_row.get('home_days_rest', 7)),
        float(away_row.get('away_days_rest', 7)),
        float(home_row.get('rest_diff', 0)),
        float(home_row.get('home_elo', 1500.0)),
        float(away_row.get('away_elo', 1500.0)),
        float(home_row.get('elo_diff', home_row.get('home_elo', 1500.0) - away_row.get('away_elo', 1500.0))),
        float(home_row.get('implied_h_prob', 0.0)),
        float(home_row.get('implied_d_prob', 0.0)),
        float(home_row.get('implied_a_prob', 0.0)),
        float(home_row.get('odds_available', 0.0)),
        float(home_row.get('market_max_prob', 0.0)),
        float(home_row.get('market_prob_spread', 0.0)),
        float(home_row.get('home_recent_home_points', 0.0)),
        float(home_row.get('home_recent_home_goal_diff', 0.0)),
        float(home_row.get('home_recent_home_goals_for', 0.0)),
        float(home_row.get('home_recent_home_goals_against', 0.0)),
        float(home_row.get('home_recent_away_points', 0.0)),
        float(home_row.get('home_recent_away_goal_diff', 0.0)),
        float(home_row.get('home_recent_away_goals_for', 0.0)),
        float(home_row.get('home_recent_away_goals_against', 0.0)),
        float(home_row.get('home_recent_form_diff', 0.0)),
        float(home_row.get('home_recent_goal_diff_diff', 0.0)),
        float(home_row.get('home_overall_points', 0.0)),
        float(home_row.get('home_overall_goal_diff', 0.0)),
        float(home_row.get('home_overall_goals_for', 0.0)),
        float(home_row.get('home_overall_goals_against', 0.0)),
        float(away_row.get('away_recent_away_points', 0.0)),
        float(away_row.get('away_recent_away_goal_diff', 0.0)),
        float(away_row.get('away_recent_away_goals_for', 0.0)),
        float(away_row.get('away_recent_away_goals_against', 0.0)),
        float(away_row.get('away_recent_home_points', 0.0)),
        float(away_row.get('away_recent_home_goal_diff', 0.0)),
        float(away_row.get('away_recent_home_goals_for', 0.0)),
        float(away_row.get('away_recent_home_goals_against', 0.0)),
        float(away_row.get('away_overall_points', 0.0)),
        float(away_row.get('away_overall_goal_diff', 0.0)),
        float(away_row.get('away_overall_goals_for', 0.0)),
        float(away_row.get('away_overall_goals_against', 0.0)),
        float(h2h_row.get('h2h_matches', 0.0)),
        float(h2h_row.get('h2h_home_avg_points', 0.0)),
        float(h2h_row.get('h2h_away_avg_points', 0.0)),
        float(h2h_row.get('h2h_home_avg_goals', 0.0)),
        float(h2h_row.get('h2h_away_avg_goals', 0.0)),
    ]

    competition_vector = competition_encoder.transform(
        pd.DataFrame({'competition': [competition]})
    )
    return np.hstack([np.array([feature_vector], dtype=float), competition_vector])
