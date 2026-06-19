from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from src.data_collection import load_match_data
from src.data_cleaning import clean_match_data, add_result_label
from src.feature_engineering import build_features, compute_team_history_stats

path = Path('data/raw/')
df = load_match_data(str(path))
raw_rows = len(df)
all_files = sorted(path.glob('*.csv'))
seasons = [p.stem for p in all_files]
clean_df = clean_match_data(df)
clean_df = add_result_label(clean_df)
X, y, encoder = build_features(clean_df)
le = LabelEncoder().fit(y)
y_num = le.transform(y)
models = {
    'LightGBM': LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=7, subsample=0.8, colsample_bytree=0.8, class_weight='balanced', n_jobs=-1, verbosity=-1),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=500, random_state=42, n_jobs=-1, verbosity=0),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in models.items():
    acc = cross_val_score(model, X, y_num, cv=cv, scoring='accuracy', n_jobs=-1)
    try:
        probs = cross_val_predict(model, X, y_num, cv=cv, method='predict_proba', n_jobs=-1)
        ll = log_loss(y_num, probs, labels=np.arange(len(le.classes_)))
    except Exception:
        ll = None
    results[name] = {
        'accuracy_mean': float(np.mean(acc)),
        'accuracy_std': float(np.std(acc)),
        'log_loss': float(ll) if ll is not None else None,
    }
stats_df = compute_team_history_stats(clean_df)
feature_df = pd.concat([clean_df.sort_values('date').reset_index(drop=True), stats_df], axis=1)
coverage = float((feature_df['odds_available'] == 1).mean())
preds = []
for _, row in feature_df.iterrows():
    if row['odds_available'] == 1:
        idx = int(np.argmax([row['implied_h_prob'], row['implied_d_prob'], row['implied_a_prob']]))
        preds.append(['home', 'draw', 'away'][idx])
    else:
        preds.append('home')
baseline_acc_all = accuracy_score(feature_df['result'], preds)
preds_cov = []
for _, row in feature_df[feature_df['odds_available'] == 1].iterrows():
    idx = int(np.argmax([row['implied_h_prob'], row['implied_d_prob'], row['implied_a_prob']]))
    preds_cov.append(['home', 'draw', 'away'][idx])
accuracy_cov = accuracy_score(feature_df.loc[feature_df['odds_available'] == 1, 'result'], preds_cov)
print('rows', raw_rows)
print('season_count', len(seasons))
print('seasons', seasons)
print('raw_columns', df.columns.tolist())
print('label_classes', le.classes_.tolist())
print('coverage', coverage)
print('baseline_acc_all', baseline_acc_all)
print('baseline_acc_covered', accuracy_cov)
print('results', results)
