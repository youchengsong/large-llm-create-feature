import sys
sys.path.insert(0, '/root/大模型提取特征')

from src.data.loader import load_all_tables
from src.data.baseline import run_baseline
import pandas as pd
import numpy as np

open_df, tx_df, close_df = load_all_tables("data")
baseline_df = run_baseline(open_df, tx_df, close_df)
cols = list(baseline_df.columns)
print(f"Total columns: {len(cols)}")

leaky = [c for c in cols if 'laundering' in c.lower() or 'is_risk' in c.lower() or 'abnormal' in c.lower() or 'victim' in c.lower()]
print(f"Leaky columns remaining: {leaky}")

label = baseline_df["is_laundering"]
feat_cols = [c for c in cols if c != "is_laundering"]
print(f"Feature columns: {len(feat_cols)}")
print(f"Label dist: pos={int(label.sum())}, neg={int((label==0).sum())}, total={len(label)}")
print(f"Positive rate: {label.mean()*100:.4f}%")

print("\nTop 15 features by |correlation| with label:")
corrs = []
for col in feat_cols:
    c = baseline_df[col].fillna(0).corr(label)
    if not np.isnan(c):
        corrs.append((col, abs(c), c))
corrs.sort(key=lambda x: -x[1])
for name, absc, c in corrs[:15]:
    print(f"  {name}: |corr|={absc:.4f} (raw={c:.4f})")

print("\nBottom 5 features by |correlation|:")
for name, absc, c in corrs[-5:]:
    print(f"  {name}: |corr|={absc:.6f}")
