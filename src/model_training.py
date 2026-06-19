"""Train and evaluate a football match outcome model using time-based splits.

This implements the following fixes required by the project brief:
- time-based train/test split (first 80% train, last 20% test)
- numeric label mapping: home=0, draw=1, away=2
- no shuffled/stratified CV; TimeSeriesSplit used for CV summaries
- odds-only baseline and hybrid (0.6 odds + 0.4 ML) evaluation
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, log_loss

from .data_collection import load_match_data
from .data_cleaning import add_result_label, clean_match_data
from .feature_engineering import build_features, compute_team_history_stats
from .merge_data import merge_data


def train_model(
    data_path: str,
    model_output_dir: str = 'models',
):
    input_path = Path(data_path)
    if input_path.is_dir():
        merged_path = input_path / 'merged_matches.csv'
        if not merged_path.exists():
            # preserve existing behavior for users who want a merged CSV
            try:
                print(f'Creating merged file at {merged_path}')
                merge_data(str(input_path), output_path=str(merged_path))
            except Exception:
                pass

    df = load_match_data(data_path)
    print(f'Loaded {len(df)} rows from {data_path}')

    df = clean_match_data(df)
    df = add_result_label(df)

    # ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)

    # features and labels
    X, y, competition_encoder = build_features(df)

    # numeric label mapping required by XGBoost and for consistent evaluation
    label_map = {'home': 0, 'draw': 1, 'away': 2}
    y_num = y.map(label_map).astype(int)

    # time-based CV for summary statistics (no shuffling)
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        'LightGBM': LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=7, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', n_jobs=-1, verbosity=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=500, random_state=42, n_jobs=-1, verbosity=0),
        'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
    }

    results_cv = {}
    for name, model in models.items():
        try:
            acc_cv = cross_val_score(model, X, y_num, cv=tscv, scoring='accuracy', n_jobs=-1)
        except Exception:
            acc_cv = np.array([np.nan])
        try:
            probs_cv = cross_val_predict(model, X, y_num, cv=tscv, method='predict_proba', n_jobs=-1)
            ll_cv = log_loss(y_num, probs_cv, labels=[0, 1, 2])
        except Exception:
            ll_cv = np.nan
        results_cv[name] = {'acc_mean': float(np.nanmean(acc_cv)), 'acc_std': float(np.nanstd(acc_cv)), 'log_loss': float(ll_cv)}

    # time-based train/test split: first 80% train, last 20% test
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_num.iloc[:split_index], y_num.iloc[split_index:]

    # compute implied probabilities for baseline and hybrid (from stats)
    stats_df = compute_team_history_stats(df)
    stats_df = stats_df.reset_index(drop=True)
    odds_available = stats_df['odds_available'] == 1
    coverage = float(odds_available.mean())

    stats_test = stats_df.iloc[split_index:].reset_index(drop=True)
    # baseline odds probabilities (columns implied_h_prob, implied_d_prob, implied_a_prob)
    odds_probs_test = np.vstack([
        stats_test['implied_h_prob'].fillna(0).to_numpy(),
        stats_test['implied_d_prob'].fillna(0).to_numpy(),
        stats_test['implied_a_prob'].fillna(0).to_numpy(),
    ]).T

    # avoid rows with zero-sum by normalizing where needed
    row_sums = odds_probs_test.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() > 0
    odds_probs_test[mask] = odds_probs_test[mask] / row_sums[mask]

    # baseline predictions
    baseline_preds = np.argmax(odds_probs_test, axis=1)
    baseline_acc = accuracy_score(y_test.to_numpy(), baseline_preds)
    try:
        baseline_ll = log_loss(y_test.to_numpy(), odds_probs_test, labels=[0, 1, 2])
    except Exception:
        baseline_ll = float('nan')

    # train each model and evaluate on the held-out chronological test set
    results_test = {}
    output_dir = Path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs_test = model.predict_proba(X_test)
        preds_test = np.argmax(probs_test, axis=1)
        acc_test = accuracy_score(y_test, preds_test)
        ll_test = log_loss(y_test, probs_test, labels=[0, 1, 2])

        # hybrid: 0.6 * odds + 0.4 * ML
        hybrid_probs = 0.6 * odds_probs_test + 0.4 * probs_test
        hybrid_preds = np.argmax(hybrid_probs, axis=1)
        hybrid_acc = accuracy_score(y_test, hybrid_preds)
        hybrid_ll = log_loss(y_test, hybrid_probs, labels=[0, 1, 2])

        results_test[name] = {
            'test_accuracy': float(acc_test),
            'test_log_loss': float(ll_test),
            'hybrid_accuracy': float(hybrid_acc),
            'hybrid_log_loss': float(hybrid_ll),
        }

        # persist model + encoder for the best model (by hybrid log loss)
        model_file = output_dir / f'{name.lower()}_model.joblib'
        joblib.dump(model, model_file)

    # save competition encoder as well
    joblib.dump(competition_encoder, output_dir / 'competition_encoder.joblib')

    # print summary
    print('\nDATASET:')
    print('  rows:', len(df))
    print('  seasons (unique competitions):', df['competition'].unique().tolist())
    print('  odds_coverage:', coverage)

    print('\nBASELINE (odds-only) ON TEST:')
    print(f'  accuracy: {baseline_acc:.4f}, log_loss: {baseline_ll:.4f}')

    print('\nCV SUMMARY (TimeSeriesSplit):')
    for name, stats in results_cv.items():
        print(f'  {name}: acc={stats["acc_mean"]:.4f} ± {stats["acc_std"]:.4f}, log_loss={stats["log_loss"]:.4f}')

    print('\nTEST RESULTS:')
    for name in results_test:
        rt = results_test[name]
        print(f'  {name}: test_acc={rt["test_accuracy"]:.4f}, test_logloss={rt["test_log_loss"]:.4f}, hybrid_acc={rt["hybrid_accuracy"]:.4f}, hybrid_logloss={rt["hybrid_log_loss"]:.4f}')

    print('\nSaved models to:', str(output_dir))
    return results_cv, results_test, {'baseline_acc': baseline_acc, 'baseline_logloss': baseline_ll}


def main():
    parser = argparse.ArgumentParser(description='Train a football match outcome model.')
    parser.add_argument('data_path', help='Path to raw match data CSV file or folder containing CSV files.')
    parser.add_argument('--model-output', default='models/match_model.joblib', help='Path (file) to save one model; parent folder used to save all models.')
    parser.add_argument('--encoder-output', default='models/competition_encoder.joblib', help='(unused) kept for compatibility')
    args = parser.parse_args()

    model_output_dir = Path(args.model_output).parent
    train_model(args.data_path, model_output_dir=str(model_output_dir))


if __name__ == '__main__':
    main()
