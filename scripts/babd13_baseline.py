"""BABD-13 Baseline: 149 features, XGBoost 5-fold CV with CR@K"""
import json, time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

print("Loading BABD-13 ...")
df = pd.read_csv("/root/autodl-tmp/archive/BABD-13.csv")
print(f"Shape: {df.shape}")

ILLICIT = {0, 2, 6, 7, 8, 9, 11}
y = df["label"].isin(ILLICIT).astype(int).values
feature_cols = [c for c in df.columns if c not in ("account", "label")]

# SW is categorical (SA/WA) -> encode as 0/1
df["SW"] = (df["SW"] == "WA").astype(int)

X = df[feature_cols].values.astype(np.float32)
print(f"Features: {len(feature_cols)}, Positive: {y.sum()}, Negative: {(1-y).sum()}")
print(f"Positive rate: {y.mean():.4f}, scale_pos_weight: {(1-y).mean()/y.mean():.2f}")

import xgboost as xgb

spw = float((1-y).sum()/y.sum())
params = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=spw,
    eval_metric="logloss", random_state=42, n_jobs=-1,
    use_label_encoder=False,
)

K_VALUES = [0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.05]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    t0 = time.time()
    clf = xgb.XGBClassifier(**params)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    prob = clf.predict_proba(X[te])[:, 1]

    rec = recall_score(y[te], pred)
    prec = precision_score(y[te], pred)
    f1 = f1_score(y[te], pred)
    auc = roc_auc_score(y[te], prob)

    cr = {}
    total_pos = y[te].sum()
    n_te = len(te)
    sorted_idx = np.argsort(-prob)
    for k in K_VALUES:
        topk = int(np.ceil(n_te * k))
        cr[f"CR@{k*100:.1f}%"] = float(y[te][sorted_idx[:topk]].sum()) / total_pos

    row = {"fold": fold, "recall": rec, "precision": prec, "f1": f1, "auc": auc, **cr}
    results.append(row)
    cr_str = " ".join(f"CR@{k*100:.1f}%={cr[f'CR@{k*100:.1f}%']:.4f}" for k in K_VALUES)
    print(f"  Fold {fold}: recall={rec:.4f} prec={prec:.4f} f1={f1:.4f} auc={auc:.4f} {cr_str} ({time.time()-t0:.0f}s)")

print("\n汇总:")
summary = {}
for metric in ["recall", "precision", "f1", "auc"] + [f"CR@{k*100:.1f}%" for k in K_VALUES]:
    vals = [r[metric] for r in results]
    summary[metric] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    print(f"  {metric}: {summary[metric]}")

# Feature importance
imp = np.zeros(len(feature_cols))
for fold, (tr, te) in enumerate(skf.split(X, y)):
    clf = xgb.XGBClassifier(**params)
    clf.fit(X[tr], y[tr])
    imp += clf.feature_importances_
imp /= 5
top20 = np.argsort(-imp)[:20]
print("\nTop 20 features:")
for i, idx in enumerate(top20):
    print(f"  {i+1}. {feature_cols[idx]}: {imp[idx]:.4f}")

out = {"experiment": "BABD-13 baseline (149 features)", "n_features": len(feature_cols),
       "scale_pos_weight": round(spw, 2), "folds": results, "summary": summary,
       "top20_features": [{"name": feature_cols[idx], "importance": float(imp[idx])} for idx in top20]}
with open("/root/大模型提取特征/outputs/babd13_baseline.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved to /root/大模型提取特征/outputs/babd13_baseline.json")
