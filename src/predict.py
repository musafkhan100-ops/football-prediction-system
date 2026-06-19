"""Predict match outcome using a trained model."""

import argparse
from pathlib import Path

import joblib
from .data_collection import load_match_data
from .data_cleaning import clean_match_data
from .feature_engineering import build_match_features


def predict_match(
    history_path: str,
    home_team: str,
    away_team: str,
    competition: str,
    model_path: str = 'models/match_model.joblib',
    encoder_path: str = 'models/competition_encoder.joblib',
):
    if home_team.strip().lower() == away_team.strip().lower():
        raise ValueError('Home team and away team must be different.')

    model_file = Path(model_path)
    encoder_file = Path(encoder_path)
    if not model_file.exists() or not encoder_file.exists():
        raise FileNotFoundError(
            'Trained model or encoder not found. Run model_training.py first.'
        )

    model = joblib.load(model_file)
    competition_encoder = joblib.load(encoder_file)

    df = load_match_data(history_path)
    df = clean_match_data(df)

    features = build_match_features(
        df,
        home_team=home_team.strip(),
        away_team=away_team.strip(),
        competition=competition.strip(),
        competition_encoder=competition_encoder,
    )

    prediction = model.predict(features)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]

    # model may use numeric labels (0=home,1=draw,2=away) or string labels
    label_map_inv = {0: 'home', 1: 'draw', 2: 'away'}
    try:
        # if prediction is numeric, map back
        if isinstance(prediction, (int, float)) or (hasattr(prediction, 'dtype') and str(prediction.dtype).startswith('int')):
            pred_label = label_map_inv.get(int(prediction), 'draw')
        else:
            pred_label = str(prediction)
    except Exception:
        pred_label = str(prediction)

    if pred_label == 'home':
        prediction_name = home_team.strip()
    elif pred_label == 'away':
        prediction_name = away_team.strip()
    else:
        prediction_name = 'Draw'

    return prediction, prediction_name, proba, model


def main():
    parser = argparse.ArgumentParser(description='Predict a football match result.')
    parser.add_argument('history_path', help='Path to historical match CSV file or folder containing CSV files.')
    parser.add_argument('home_team', help='Name of the home team.')
    parser.add_argument('away_team', help='Name of the away team.')
    parser.add_argument('--competition', default='Unknown', help='Competition name.')
    parser.add_argument('--model-path', default='models/match_model.joblib', help='Path to trained model file.')
    parser.add_argument('--encoder-path', default='models/competition_encoder.joblib', help='Path to saved competition encoder.')
    args = parser.parse_args()

    prediction, prediction_name, proba, model = predict_match(
        args.history_path,
        args.home_team,
        args.away_team,
        args.competition,
        args.model_path,
        args.encoder_path,
    )

    print(f'Home: {args.home_team}')
    print(f'Away: {args.away_team}')
    print(f'Competition: {args.competition}')
    print(f'Predicted result: {prediction_name}')
    if proba is not None:
        # model.classes_ might be numeric or string; map probabilities to canonical labels
        class_map = {}
        for cls, p in zip(model.classes_, proba):
            if isinstance(cls, (int, float)):
                canonical = label_map_inv.get(int(cls), str(cls))
            else:
                canonical = str(cls)
            class_map[canonical] = p

        label_names = {
            'home': args.home_team.strip(),
            'away': args.away_team.strip(),
            'draw': 'Draw',
        }
        print('Prediction probabilities:')
        for label in ['home', 'draw', 'away']:
            score = class_map.get(label, 0.0)
            print(f'  {label_names[label]}: {score:.3f}')


if __name__ == '__main__':
    main()
