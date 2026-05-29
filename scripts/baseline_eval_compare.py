"""对比 106列全量 vs 95列精简基线的模型评价指标。"""
import sys
sys.path.insert(0, "/root/大模型提取特征")

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from pathlib import Path

BASELINE_EXCLUDE_FEATURES = {
    "is_laundering", "is_risk", "tx_id", "src", "dst",
    "Min_Received", "Min_Sent", "S_src_counterparty_index",
    "Timestamp_Hour", "S_dst_counterparty_index",
    "1H_frequency_From", "1H_frequency_To",
    "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent",
}

LABEL_ID_COLS = {"is_laundering", "is_risk", "tx_id", "src", "dst"}

EXCLUDED_10 = BASELINE_EXCLUDE_FEATURES - LABEL_ID_COLS


def capture_rate_at_k(y_true, y_prob, k_pcts=(0.01, 0.03, 0.05)):
    """CR@K: Top K% 样本中捕获的正样本占总正样本的比例。"""
    n = len(y_true)
    total_pos = y_true.sum()
    if total_pos == 0:
        return {f"CR@{int(k*100)}%": 0.0 for k in k_pcts}
    indices = np.argsort(y_prob)[::-1]
    results = {}
    for k in k_pcts:
        top_k = max(1, int(n * k))
        captured = y_true[indices[:top_k]].sum()
        results[f"CR@{int(k*100)}%"] = float(captured / total_pos)
    return results


def cv_evaluate(X, y, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    recalls, precisions, f1s, aucs = [], [], [], []
    cr_accum = {}
    from sklearn.metrics import roc_auc_score

    for train_idx, val_idx in skf.split(X, y):
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

    result = {
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "f1": float(np.mean(f1s)),
        "auc": float(np.mean(aucs)),
        "recall_std": float(np.std(recalls)),
        "precision_std": float(np.std(precisions)),
        "f1_std": float(np.std(f1s)),
        "auc_std": float(np.std(aucs)),
    }
    for k, v in cr_accum.items():
        result[k] = float(np.mean(v))
        result[f"{k}_std"] = float(np.std(v))
    return result


print("加载 baseline_cache.parquet ...")
df = pd.read_parquet("/root/大模型提取特征/outputs/baseline_cache.parquet")
print(f"原始 parquet: {df.shape[0]} 行, {df.shape[1]} 列")

all_cols = set(df.columns)
label_id_in_data = LABEL_ID_COLS & all_cols
excluded_10_in_data = EXCLUDED_10 & all_cols

print(f"\n=== 列分析 ===")
print(f"总列数: {len(all_cols)}")
print(f"标签/ID列 (在数据中): {sorted(label_id_in_data)} ({len(label_id_in_data)}个)")
print(f"排除的10个特征 (在数据中): {sorted(excluded_10_in_data)} ({len(excluded_10_in_data)}个)")

# 确定标签列
if "is_risk" in df.columns:
    label_col = "is_risk"
elif "is_laundering" in df.columns:
    label_col = "is_laundering"
else:
    raise ValueError("找不到标签列")

y = df[label_col].astype(int).values
print(f"标签列: {label_col}, 正样本: {y.sum()}, 负样本: {len(y) - y.sum()}, 比例: {y.sum()/len(y):.4f}")

# 全量特征（去掉标签/ID）
full_cols = sorted([c for c in all_cols if c not in LABEL_ID_COLS])
# 精简特征（去掉标签/ID + 10个排除特征）
slim_cols = sorted([c for c in all_cols if c not in BASELINE_EXCLUDE_FEATURES])

print(f"\n全量特征: {len(full_cols)} 个")
print(f"精简特征: {len(slim_cols)} 个 (去掉了 {len(full_cols) - len(slim_cols)} 个)")

X_full = df[full_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
X_slim = df[slim_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"\n=== 跑基线评价 (5-fold CV) ===")
print(f"全量特征 ({len(full_cols)}个) 开始评估...")
result_full = cv_evaluate(X_full, y)
print(f"全量: recall={result_full['recall']:.4f}±{result_full['recall_std']:.4f}  "
      f"precision={result_full['precision']:.4f}±{result_full['precision_std']:.4f}  "
      f"f1={result_full['f1']:.4f}±{result_full['f1_std']:.4f}  "
      f"auc={result_full['auc']:.4f}±{result_full['auc_std']:.4f}")
print(f"  CR@1%={result_full.get('CR@1%', 0):.4f}  CR@3%={result_full.get('CR@3%', 0):.4f}  CR@5%={result_full.get('CR@5%', 0):.4f}")

print(f"\n精简特征 ({len(slim_cols)}个) 开始评估...")
result_slim = cv_evaluate(X_slim, y)
print(f"精简: recall={result_slim['recall']:.4f}±{result_slim['recall_std']:.4f}  "
      f"precision={result_slim['precision']:.4f}±{result_slim['precision_std']:.4f}  "
      f"f1={result_slim['f1']:.4f}±{result_slim['f1_std']:.4f}  "
      f"auc={result_slim['auc']:.4f}±{result_slim['auc_std']:.4f}")
print(f"  CR@1%={result_slim.get('CR@1%', 0):.4f}  CR@3%={result_slim.get('CR@3%', 0):.4f}  CR@5%={result_slim.get('CR@5%', 0):.4f}")

print(f"\n=== 对比 ===")
for metric in ["recall", "precision", "f1", "auc"]:
    delta = result_slim[metric] - result_full[metric]
    print(f"  {metric}: {result_full[metric]:.4f} → {result_slim[metric]:.4f} (Δ={delta:+.4f})")

# 保存结果
output = {
    "parquet_shape": list(df.shape),
    "label_col": label_col,
    "pos_count": int(y.sum()),
    "neg_count": int(len(y) - y.sum()),
    "full_feature_count": len(full_cols),
    "slim_feature_count": len(slim_cols),
    "excluded_features": sorted(excluded_10_in_data),
    "full_baseline": result_full,
    "slim_baseline": result_slim,
}
out_path = Path("/root/大模型提取特征/outputs/baseline_compare.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 {out_path}")
