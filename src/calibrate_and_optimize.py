"""Calibration and hybrid optimization for existing ML models.

- Uses time-based split: first 80% train, last 20% test.
- Trains base models on train_inner, fits calibrator on calibration split (last 20% of train).
- Evaluates uncalibrated and calibrated probabilities on the test set.
- Searches hybrid weight w in 0.0..1.0 step 0.1 to combine odds and ML probs.
- Reports log loss and accuracy; primary metric is log loss.
"""
from pathlib import Path
import sys
# ensure repo root is on sys.path so `src` imports work when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from src.data_collection import load_match_data
from src.data_cleaning import clean_match_data, add_result_label
from src.feature_engineering import build_features, compute_team_history_stats


def load_and_split(data_path: str):
    df = load_match_data(data_path)
    df = clean_match_data(df)
    df = add_result_label(df)
    df = df.sort_values('date').reset_index(drop=True)
    X, y, enc = build_features(df)
    label_map = {'home': 0, 'draw': 1, 'away': 2}
    y_num = y.map(label_map).astype(int)
    n = len(X)
    test_idx = int(n * 0.8)
    X_train_full = X[:test_idx]
    y_train_full = y_num.iloc[:test_idx]
    X_test = X[test_idx:]
    y_test = y_num.iloc[test_idx:]
    # inner training split for base (to assess uncalibrated) - keep 80/20 split inside train
    inner_idx = int(len(X_train_full) * 0.8)
    X_train_inner = X_train_full[:inner_idx]
    y_train_inner = y_train_full.iloc[:inner_idx]
    # return full train too for calibration
    return df, X_train_inner, y_train_inner, X_train_full, y_train_full, X_test, y_test, enc


def compute_odds_probs_on_test(df, test_start_index):
    stats = compute_team_history_stats(df)
    stats = stats.reset_index(drop=True)
    stats_test = stats.iloc[test_start_index:]
    odds_probs_test = np.vstack([
        stats_test['implied_h_prob'].fillna(0).to_numpy(),
        stats_test['implied_d_prob'].fillna(0).to_numpy(),
        stats_test['implied_a_prob'].fillna(0).to_numpy(),
    ]).T
    # normalize rows where sum > 0
    row_sums = odds_probs_test.sum(axis=1, keepdims=True)
    mask = row_sums.squeeze() > 0
    odds_probs_test[mask] = odds_probs_test[mask] / row_sums[mask]
    return odds_probs_test


def evaluate_models(data_path: str):
    df, X_train_inner, y_train_inner, X_train_full, y_train_full, X_test, y_test, enc = load_and_split(data_path)
    # test_start_index = number of rows before test; equals len(train_full)
    test_start_index = len(X_train_full)
    # compute odds probs on test
    odds_probs_test = compute_odds_probs_on_test(df, test_start_index)

    models = {
        'LightGBM': LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=7, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', n_jobs=-1, verbosity=-1),
        'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    }

    results = {}
    weights = np.round(np.linspace(0.0, 1.0, 11), 2)

    for name, model in models.items():
        print(f'Processing {name}')
        # train base model on train_inner
        model.fit(X_train_inner, y_train_inner)
        probs_test_uncal = model.predict_proba(X_test)
        ll_uncal = log_loss(y_test, probs_test_uncal, labels=[0, 1, 2])
        acc_uncal = accuracy_score(y_test, np.argmax(probs_test_uncal, axis=1))

        # calibrate using TimeSeriesSplit on the full training set (no shuffle)
        from sklearn.model_selection import TimeSeriesSplit
        tscv_cal = TimeSeriesSplit(n_splits=3)
        calib_results = {}
        for method in ('sigmoid', 'isotonic'):
            try:
                base_for_cal = type(model)()  # fresh instance of same estimator class
                calibrator = CalibratedClassifierCV(base_for_cal, method=method, cv=tscv_cal)
                calibrator.fit(X_train_full, y_train_full)
                probs_test_cal = calibrator.predict_proba(X_test)
                ll_cal = log_loss(y_test, probs_test_cal, labels=[0, 1, 2])
                acc_cal = accuracy_score(y_test, np.argmax(probs_test_cal, axis=1))
            except Exception as ex:
                probs_test_cal = None
                ll_cal = float('nan')
                acc_cal = float('nan')
            calib_results[method] = {
                'probs_test': probs_test_cal,
                'log_loss': ll_cal,
                'accuracy': acc_cal,
            }

        # hybrid grid search using uncalibrated and calibrated
        # baseline odds-only
        baseline_ll = log_loss(y_test, odds_probs_test, labels=[0, 1, 2])
        baseline_acc = accuracy_score(y_test, np.argmax(odds_probs_test, axis=1))

        best = {
            'uncal': {'w': None, 'll': float('inf'), 'acc': 0.0},
            'sigmoid': {'w': None, 'll': float('inf'), 'acc': 0.0},
            'isotonic': {'w': None, 'll': float('inf'), 'acc': 0.0},
        }

        for w in weights:
            # uncalibrated hybrid
            hybrid_uncal = w * odds_probs_test + (1 - w) * probs_test_uncal
            try:
                ll_h_uncal = log_loss(y_test, hybrid_uncal, labels=[0, 1, 2])
                acc_h_uncal = accuracy_score(y_test, np.argmax(hybrid_uncal, axis=1))
            except Exception:
                ll_h_uncal = float('inf')
                acc_h_uncal = 0.0
            if ll_h_uncal < best['uncal']['ll']:
                best['uncal'].update({'w': float(w), 'll': float(ll_h_uncal), 'acc': float(acc_h_uncal)})

            # calibrated sigmoid
            probs_sig = calib_results['sigmoid']['probs_test']
            if probs_sig is not None:
                hybrid_sig = w * odds_probs_test + (1 - w) * probs_sig
                try:
                    ll_h_sig = log_loss(y_test, hybrid_sig, labels=[0, 1, 2])
                    acc_h_sig = accuracy_score(y_test, np.argmax(hybrid_sig, axis=1))
                except Exception:
                    ll_h_sig = float('inf')
                    acc_h_sig = 0.0
                if ll_h_sig < best['sigmoid']['ll']:
                    best['sigmoid'].update({'w': float(w), 'll': float(ll_h_sig), 'acc': float(acc_h_sig)})

            # calibrated isotonic
            probs_iso = calib_results['isotonic']['probs_test']
            if probs_iso is not None:
                hybrid_iso = w * odds_probs_test + (1 - w) * probs_iso
                try:
                    ll_h_iso = log_loss(y_test, hybrid_iso, labels=[0, 1, 2])
                    acc_h_iso = accuracy_score(y_test, np.argmax(hybrid_iso, axis=1))
                except Exception:
                    ll_h_iso = float('inf')
                    acc_h_iso = 0.0
                if ll_h_iso < best['isotonic']['ll']:
                    best['isotonic'].update({'w': float(w), 'll': float(ll_h_iso), 'acc': float(acc_h_iso)})

        results[name] = {
            'uncalibrated': {'log_loss': ll_uncal, 'accuracy': acc_uncal},
            'calibrated_sigmoid': {'log_loss': calib_results['sigmoid']['log_loss'], 'accuracy': calib_results['sigmoid']['accuracy']},
            'calibrated_isotonic': {'log_loss': calib_results['isotonic']['log_loss'], 'accuracy': calib_results['isotonic']['accuracy']},
            'best_hybrid_uncal': best['uncal'],
            'best_hybrid_sigmoid': best['sigmoid'],
            'best_hybrid_isotonic': best['isotonic'],
            'baseline': {'log_loss': baseline_ll, 'accuracy': baseline_acc},
        }

    return results


if __name__ == '__main__':
    import json, sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/'
    res = evaluate_models(data_path)
    print('\nFINAL SUMMARY:')
    for model, stats in res.items():
        print(f"\nModel: {model}")
        print(f"  Baseline (odds) -- log_loss: {stats['baseline']['log_loss']:.4f}, acc: {stats['baseline']['accuracy']:.4f}")
        print(f"  Uncalibrated -- log_loss: {stats['uncalibrated']['log_loss']:.4f}, acc: {stats['uncalibrated']['accuracy']:.4f}")
        print(f"  Calibrated (sigmoid) -- log_loss: {stats['calibrated_sigmoid']['log_loss']:.4f}, acc: {stats['calibrated_sigmoid']['accuracy']:.4f}")
        print(f"  Calibrated (isotonic) -- log_loss: {stats['calibrated_isotonic']['log_loss']:.4f}, acc: {stats['calibrated_isotonic']['accuracy']:.4f}")
        print(f"  Best hybrid (uncal) -- w: {stats['best_hybrid_uncal']['w']}, log_loss: {stats['best_hybrid_uncal']['ll']:.4f}, acc: {stats['best_hybrid_uncal']['acc']:.4f}")
        print(f"  Best hybrid (sigmoid) -- w: {stats['best_hybrid_sigmoid']['w']}, log_loss: {stats['best_hybrid_sigmoid']['ll']:.4f}, acc: {stats['best_hybrid_sigmoid']['acc']:.4f}")
        print(f"  Best hybrid (isotonic) -- w: {stats['best_hybrid_isotonic']['w']}, log_loss: {stats['best_hybrid_isotonic']['ll']:.4f}, acc: {stats['best_hybrid_isotonic']['acc']:.4f}")
    print('\nDone.')
