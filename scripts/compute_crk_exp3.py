"""补算 CR@K: 只跑实验 3 (95+18 KnowFeat 特征)，实验 1/2 已完成。"""
import sys
sys.path.insert(0, "/root/大模型提取特征")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

BASELINE_EXCLUDE_FEATURES = {
    "is_laundering", "is_risk", "tx_id", "src", "dst",
    "Min_Received", "Min_Sent", "S_src_counterparty_index",
    "Timestamp_Hour", "S_dst_counterparty_index",
    "1H_frequency_From", "1H_frequency_To",
    "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent",
}
LABEL_ID_COLS = {"is_laundering", "is_risk", "tx_id", "src", "dst"}


def _k_label(k):
    pct = k * 100
    return f"CR@{pct:g}%"


def capture_rate_at_k(y_true, y_prob, k_pcts=(0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.05)):
    n = len(y_true)
    total_pos = y_true.sum()
    if total_pos == 0:
        return {_k_label(k): 0.0 for k in k_pcts}
    indices = np.argsort(y_prob)[::-1]
    results = {}
    for k in k_pcts:
        top_k = max(1, int(n * k))
        captured = y_true[indices[:top_k]].sum()
        results[_k_label(k)] = float(captured / total_pos)
    return results


def cv_evaluate_with_crk(X, y, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    recalls, precisions, f1s, aucs = [], [], [], []
    cr_accum = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=59.0, eval_metric="logloss",
            random_state=seed, verbosity=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()

        recall = tp / (tp + ((y_pred == 0) & (y_val == 1)).sum()) if tp > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        auc = roc_auc_score(y_val, y_prob)

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        aucs.append(auc)

        cr = capture_rate_at_k(y_val, y_prob)
        for k, v in cr.items():
            cr_accum.setdefault(k, []).append(v)

        cr_str = " ".join(f"{k}={v:.4f}" for k, v in cr.items())
        print(f"  Fold {fold+1}: recall={recall:.4f} prec={precision:.4f} {cr_str}")

    result = {
        "recall": f"{np.mean(recalls):.4f}±{np.std(recalls):.4f}",
        "precision": f"{np.mean(precisions):.4f}±{np.std(precisions):.4f}",
        "f1": f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}",
        "auc": f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}",
    }
    for k, v in cr_accum.items():
        result[k] = f"{np.mean(v):.4f}±{np.std(v):.4f}"
    result["_raw"] = {
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "f1": float(np.mean(f1s)),
        "auc": float(np.mean(aucs)),
        **{k: float(np.mean(v)) for k, v in cr_accum.items()},
    }
    return result


# ────────────────────── 加载数据 ──────────────────────

print("=" * 60)
print("加载 baseline_cache.parquet ...")
df = pd.read_parquet("/root/大模型提取特征/outputs/baseline_cache.parquet")
print(f"数据: {df.shape[0]} 行, {df.shape[1]} 列")

label_col = "is_risk" if "is_risk" in df.columns else "is_laundering"
y = df[label_col].astype(int).values
print(f"标签列: {label_col}, 正样本: {y.sum()}, 负样本: {len(y)-y.sum()}")

all_cols = set(df.columns)
slim_cols = sorted([c for c in all_cols if c not in BASELINE_EXCLUDE_FEATURES])
X_slim = df[slim_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# ────────────────────── 实验 3: 95 + 18 KnowFeat ──────────────────────

print(f"\n{'='*60}")
print("实验 3: 精简特征 (95) + KnowFeat 18 新特征")
print("=" * 60)

print("加载 Stage 2 特征代码...")
features_path = Path("/root/大模型提取特征/outputs/stage2_features.json")
with open(features_path) as f:
    stage2_features = json.load(f)

print("加载原始交易数据...")
data_root = Path("/root/大模型提取特征/data")
open_df = pd.read_csv(data_root / "accounts.csv")
tx_df = pd.read_csv(data_root / "transactions.csv")
close_df = pd.read_csv(data_root / "close.csv")
print(f"  accounts: {open_df.shape}, transactions: {tx_df.shape}, close: {close_df.shape}")

X_knowfeat = X_slim.copy()

# baseline_cache 没有 src 列，从 transactions.csv 获取
src_series = tx_df["src"]
print(f"  Join key: src from tx_df, unique wallets: {src_series.nunique()}")
assert len(src_series) == len(X_knowfeat), f"tx_df ({len(src_series)}) != baseline ({len(X_knowfeat)})"

success_count = 0
for feat in stage2_features:
    name = feat["name"]
    code = feat["code"]
    try:
        exec_ns = {}
        exec(code, exec_ns)
        func_name = [k for k in exec_ns if k.startswith("feature_")][0]
        feat_df = exec_ns[func_name](open_df, tx_df, close_df)

        if "wallet_id" in feat_df.columns:
            # wallet 级别 → 交易级别: 通过 src merge
            feat_map = feat_df.set_index("wallet_id")[name]
            feat_values = src_series.map(feat_map).values
            feat_values = pd.to_numeric(pd.Series(feat_values), errors="coerce").fillna(0).values
        else:
            feat_values = pd.to_numeric(feat_df[name], errors="coerce").fillna(0).values
            if len(feat_values) != len(X_knowfeat):
                print(f"  [SKIP] {name}: 长度不匹配 ({len(feat_values)} vs {len(X_knowfeat)})")
                continue

        X_knowfeat[name] = feat_values
        success_count += 1
        print(f"  [OK] {name} ({feat_df.shape[0]} wallets)")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

print(f"\n成功加载 {success_count}/{len(stage2_features)} 个 KnowFeat 特征")
print(f"最终特征矩阵: {X_knowfeat.shape}")

result_knowfeat = cv_evaluate_with_crk(X_knowfeat, y)
print(f"\n汇总: {json.dumps({k:v for k,v in result_knowfeat.items() if k != '_raw'}, indent=2)}")

# 保存
out_path = Path("/root/大模型提取特征/outputs/crk_results_exp3.json")
with open(out_path, "w") as f:
    json.dump(result_knowfeat, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 {out_path}")
