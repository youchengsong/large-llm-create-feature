"""补算 CR@K 指标：基线 (95/105) + KnowFeat (95+18) 三组实验。"""
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
        fn = ((y_pred == 0) & (y_val == 1)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
full_cols = sorted([c for c in all_cols if c not in LABEL_ID_COLS])
slim_cols = sorted([c for c in all_cols if c not in BASELINE_EXCLUDE_FEATURES])

X_full = df[full_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
X_slim = df[slim_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# ────────────────────── 实验 1 & 2 已有结果，跳过 ──────────────────────

import os
SKIP_12 = os.environ.get("SKIP_EXP12", "1") == "1"

if not SKIP_12:
    print(f"\n{'='*60}")
    print(f"实验 1: 全量特征 ({len(full_cols)} 个)")
    print("=" * 60)
    result_full = cv_evaluate_with_crk(X_full, y)
    print(f"\n汇总: {json.dumps({k:v for k,v in result_full.items() if k != '_raw'}, indent=2)}")

    print(f"\n{'='*60}")
    print(f"实验 2: 精简特征 ({len(slim_cols)} 个)")
    print("=" * 60)
    result_slim = cv_evaluate_with_crk(X_slim, y)
    print(f"\n汇总: {json.dumps({k:v for k,v in result_slim.items() if k != '_raw'}, indent=2)}")
else:
    print("\n跳过实验 1 & 2（已有结果），直接跑实验 3")
    result_full = None
    result_slim = None

# ────────────────────── 实验 3: 95 + 18 KnowFeat 特征 ──────────────────────

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

# baseline_cache 是交易级别 (1.38M)，特征代码输出是 wallet 级别 (~43K)
# parquet 中没有 src 列，从 tx_df 取（行序一致）
src_series = tx_df["src"]
print(f"  Join key: tx_df.src, 行数: {len(src_series)}, unique wallets: {src_series.nunique()}")
assert len(src_series) == len(X_knowfeat), f"tx_df 行数 {len(src_series)} != parquet 行数 {len(X_knowfeat)}"

success_count = 0
for feat in stage2_features:
    name = feat["name"]
    code = feat["code"]
    try:
        exec_ns = {}
        exec(code, exec_ns)
        func_name = [k for k in exec_ns if k.startswith("feature_")][0]
        feat_df = exec_ns[func_name](open_df, tx_df, close_df)

        merge_df = pd.DataFrame({"wallet_id": src_series.values})
        merged = merge_df.merge(feat_df, on="wallet_id", how="left")
        feat_values = pd.to_numeric(merged[name], errors="coerce").fillna(0).values

        X_knowfeat[name] = feat_values
        success_count += 1
        print(f"  [OK] {name} ({feat_df.shape[0]} wallets)")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

print(f"\n成功加载 {success_count}/{len(stage2_features)} 个 KnowFeat 特征")
print(f"最终特征矩阵: {X_knowfeat.shape}")

result_knowfeat = cv_evaluate_with_crk(X_knowfeat, y)
print(f"\n汇总: {json.dumps({k:v for k,v in result_knowfeat.items() if k != '_raw'}, indent=2)}")

# ────────────────────── 汇总对比 ──────────────────────

print(f"\n{'='*60}")
print("CR@K 汇总对比")
print("=" * 60)
cr_keys = [f"CR@{k}%" for k in ["0.1", "0.2", "0.3", "0.5", "1", "3", "5"]]
header = f"{'实验':<25} {'recall':<10} {'prec':<10}" + "".join(f" {k:<10}" for k in cr_keys)
print(header)
print("-" * len(header))
all_results = []
if result_full is not None:
    all_results.append(("105 全量特征", result_full))
if result_slim is not None:
    all_results.append(("95 精简特征", result_slim))
all_results.append(("95+18 KnowFeat", result_knowfeat))
for label, res in all_results:
    r = res["_raw"]
    cr_vals = "".join(f" {r.get(k, 0):<10.4f}" for k in cr_keys)
    print(f"{label:<25} {r['recall']:<10.4f} {r['precision']:<10.4f}{cr_vals}")

# 保存结果
output = {"knowfeat_95_18": result_knowfeat}
if result_full is not None:
    output["full_105"] = result_full
if result_slim is not None:
    output["slim_95"] = result_slim
out_path = Path("/root/大模型提取特征/outputs/crk_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 {out_path}")
