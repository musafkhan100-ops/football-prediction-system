"""Data collection helpers for football match prediction."""

from pathlib import Path

import pandas as pd


COLUMN_NAME_MAPPING = {
    'Date': 'date',
    'HomeTeam': 'home_team',
    'AwayTeam': 'away_team',
    'FTHG': 'home_goals',
    'FTAG': 'away_goals',
    'Div': 'competition',
    'League': 'competition',
    'Comp': 'competition',
    'HTHG': 'home_half_goals',
    'HTAG': 'away_half_goals',
}


def _normalize_columns(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    for original, normalized in COLUMN_NAME_MAPPING.items():
        if original in df.columns:
            rename_map[original] = normalized
        elif original.lower() in [c.lower() for c in df.columns]:
            matching = [c for c in df.columns if c.lower() == original.lower()]
            if matching:
                rename_map[matching[0]] = normalized

    df = df.rename(columns=rename_map)
    if 'competition' not in df.columns:
        stem = source_path.stem.lower()
        if 'e0' in stem or 'premier' in stem:
            df['competition'] = 'Premier League'
        else:
            df['competition'] = 'Unknown'

    return df


def _read_csv_file(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = _normalize_columns(df, csv_file)
    return df


def load_match_data(path: str) -> pd.DataFrame:
    """Load match data from a CSV file or all CSV files in a folder."""
    raw_path = Path(path)
    if raw_path.is_dir():
        # ignore ingestion artifacts like merged_matches.csv or matches.csv
        csv_files = sorted(
            [p for p in raw_path.glob('*.csv') if p.name.lower() not in ('matches.csv', 'merged_matches.csv')]
        )
        if not csv_files:
            raise ValueError(f'No CSV files found in directory: {raw_path}')

        frames = []
        for csv_file in csv_files:
            frames.append(_read_csv_file(csv_file))

        df = pd.concat(frames, ignore_index=True, sort=False)
    elif raw_path.is_file():
        df = _read_csv_file(raw_path)
    else:
        raise FileNotFoundError(f'Path not found: {raw_path}')

    if df.empty:
        raise ValueError(f'No data loaded from: {raw_path}')

    df = df.drop_duplicates(ignore_index=True)
    validate_match_data(df)
    return df


def validate_match_data(df: pd.DataFrame) -> None:
    """Validate the columns we need for prediction."""
    required_columns = [
        'date',
        'home_team',
        'away_team',
        'home_goals',
        'away_goals',
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
