"""Clean raw match data for model training."""

import pandas as pd


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_columns = [
        'date',
        'home_team',
        'away_team',
        'home_goals',
        'away_goals',
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=required_columns)

    df['home_team'] = df['home_team'].astype(str).str.strip()
    df['away_team'] = df['away_team'].astype(str).str.strip()
    df['competition'] = df.get('competition', pd.Series(['Unknown'] * len(df))).fillna('Unknown').astype(str).str.strip()

    df['home_goals'] = df['home_goals'].astype(int)
    df['away_goals'] = df['away_goals'].astype(int)
    return df


def add_result_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def label(row):
        if row['home_goals'] > row['away_goals']:
            return 'home'
        if row['home_goals'] < row['away_goals']:
            return 'away'
        return 'draw'

    df['result'] = df.apply(label, axis=1)
    return df
