import pandas as pd
import numpy as np
import json
import sys
sys.path.insert(0, '/root/大模型提取特征')
from src.data.baseline import run_baseline, get_baseline_feature_names
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import xgboost as xgb

tx_df = pd.read_csv('data/transactions.csv')
open_df = pd.read_csv('data/accounts.csv')
close_df = pd.read_csv('data/close.csv')

print('=== Step 1: Baseline features ===')
baseline_df = run_baseline(open_df, tx_df, close_df)
feat_names = get_baseline_feature_names()
available = [c for c in baseline_df.columns if c in feat_names]
y = baseline_df['is_laundering'].values
print('Shape:', baseline_df.shape, 'Features:', len(available))

def calc_prauc(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in skf.split(X, y):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, tree_method='hist', random_state=42, eval_metric='aucpr')
        m.fit(X[tr], y[tr], verbose=False)
        scores.append(average_precision_score(y[te], m.predict_proba(X[te])[:,1]))
    return np.mean(scores)

print('\n=== Step 2: Baseline PR-AUC ===')
X_base = baseline_df[available].values
bp = calc_prauc(X_base, y)
print('Baseline PR-AUC: %.6f' % bp)

print('\n=== Step 3: Add new features ===')
with open('output/features.json') as f:
    features = json.load(f)

src_col = None
for c in ['From_Account', 'src', 'From']:
    if c in baseline_df.columns:
        src_col = c
        break
dst_col = None
for c in ['To_Account', 'dst', 'To']:
    if c in baseline_df.columns:
        dst_col = c
        break

if src_col is None:
    tx_sorted = tx_df.sort_values('timestamp').reset_index(drop=True)
    if len(tx_sorted) > len(baseline_df):
        tx_sorted = tx_sorted.drop_duplicates(subset='tx_id', keep='first').head(len(baseline_df))
    src_series = tx_sorted['src'].values[:len(baseline_df)]
    dst_series = tx_sorted['dst'].values[:len(baseline_df)]
    print('Using tx_df src/dst (aligned by sort order)')
else:
    src_series = baseline_df[src_col].values
    dst_series = baseline_df[dst_col].values
    print('Using baseline_df columns:', src_col, dst_col)

for feat in features:
    name = feat['feature_name']
    local_ns = {'tx_df': tx_df.copy(), 'pd': pd, 'np': np}
    exec(feat['code'], local_ns)
    result = local_ns['result']
    val_col = [c for c in result.columns if c != 'wallet_id'][0]
    src_map = result.set_index('wallet_id')[val_col].to_dict()
    baseline_df[name + '_src'] = pd.Series(src_series).map(src_map).fillna(0).values
    baseline_df[name + '_dst'] = pd.Series(dst_series).map(src_map).fillna(0).values
    print('  Added', name)

new_cols = list(available)
for feat in features:
    new_cols.append(feat['feature_name'] + '_src')
    new_cols.append(feat['feature_name'] + '_dst')
print('Total features:', len(new_cols))

print('\n=== Step 4: Combined PR-AUC ===')
X_new = baseline_df[new_cols].values
cp = calc_prauc(X_new, y)
print('Combined PR-AUC: %.6f' % cp)

print('\n========== RESULT ==========')
print('Baseline PR-AUC:  %.6f (%d features)' % (bp, len(available)))
print('Combined PR-AUC:  %.6f (%d features)' % (cp, len(new_cols)))
print('Delta PR-AUC:     %+.6f' % (cp - bp))
print('Improvement:      %+.2f%%' % ((cp - bp) / bp * 100))
