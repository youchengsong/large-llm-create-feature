import sys
sys.path.insert(0, '/root/大模型提取特征')

import pandas as pd
import numpy as np
from src.data.loader import load_all_tables
from src.data.baseline import run_baseline

open_df, tx_df, close_df = load_all_tables("data")
print(f"tx_df shape: {tx_df.shape}")
print(f"tx_df columns: {list(tx_df.columns)}")
print()

label_cols = [c for c in tx_df.columns if 'risk' in c.lower() or 'launder' in c.lower() or 'abnormal' in c.lower() or 'victim' in c.lower() or 'label' in c.lower()]
print(f"Label-like columns in tx_df: {label_cols}")
for col in label_cols:
    vc = tx_df[col].value_counts()
    print(f"  {col}: pos={vc.get(1,0)}, neg={vc.get(0,0)}, nan={tx_df[col].isna().sum()}")
print()

baseline_df = run_baseline(open_df, tx_df, close_df)
label = baseline_df["is_laundering"].astype(int)
print(f"Label distribution:")
print(f"  Total: {len(label)}")
print(f"  Positive (1): {label.sum()} ({label.mean()*100:.4f}%)")
print(f"  Negative (0): {(label==0).sum()} ({(1-label.mean())*100:.4f}%)")
print()

feat_cols = [c for c in baseline_df.columns if c != "is_laundering"]
print(f"Baseline feature count: {len(feat_cols)}")

suspect = [c for c in feat_cols if 'risk' in c.lower() or 'launder' in c.lower() or 'abnormal' in c.lower() or 'victim' in c.lower()]
print(f"Suspect features (label-like names): {suspect}")
print()

# Check if any feature has very high correlation with label
print("Top 10 features by |correlation| with label:")
corrs = []
for col in feat_cols:
    c = baseline_df[col].fillna(0).corr(label)
    if not np.isnan(c):
        corrs.append((col, abs(c)))
corrs.sort(key=lambda x: -x[1])
for name, c in corrs[:10]:
    print(f"  {name}: {c:.4f}")
