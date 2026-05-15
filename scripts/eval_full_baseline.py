"""用已有的完整基线特征文件评估 PR-AUC，检查新特征重复，计算去重后增量。"""
import json
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG


def compute_prauc(X, y, label=""):
    cfg = DEFAULT_CONFIG.xgboost
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    scores = []
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    for train_idx, val_idx in skf.split(X_arr, y_arr):
        model = XGBClassifier(
            scale_pos_weight=cfg.scale_pos_weight,
            eval_metric=cfg.eval_metric,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        model.fit(X_arr[train_idx], y_arr[train_idx], verbose=False)
        y_prob = model.predict_proba(X_arr[val_idx])[:, 1]
        scores.append(average_precision_score(y_arr[val_idx], y_prob))
    mean_score = float(np.mean(scores))
    print(f"[{label}] PR-AUC = {mean_score:.6f} ({X_arr.shape[1]} features)")
    return mean_score


def main():
    # ===== 1. 加载已有的完整基线 =====
    print("=" * 60)
    print("Step 1: 加载完整基线 (output/full_baseline_features_original.csv)")
    print("=" * 60)
    full_df = pd.read_csv("output/full_baseline_features_original.csv")
    non_feat = {"id", "amount", "is_laundering", "account_is_laundering_From",
                "account_is_laundering_To", "src", "dst"}
    y = full_df["is_laundering"].astype(int)
    feature_cols = [c for c in full_df.columns if c not in non_feat]
    X_full = full_df[feature_cols].fillna(0)
    print(f"基线: {len(feature_cols)} 特征, {len(y)} 行\n")

    prauc_full = compute_prauc(X_full, y, f"完整基线({len(feature_cols)}特征)")
    print()

    # ===== 2. 执行新特征 =====
    print("=" * 60)
    print("Step 2: 执行新特征")
    print("=" * 60)
    open_df = pd.read_csv("inputs/SimECNY/open/accounts.csv")
    tx_df = pd.read_csv("inputs/SimECNY/transaction/wallet_temporal_transactions_1105_1450_temp.csv", on_bad_lines="skip")
    close_df = pd.read_csv("inputs/SimECNY/close/new_close_3.csv")

    with open("output/features.json") as f:
        new_features = json.load(f)

    new_vals = {}
    failures = []
    for feat in new_features:
        name, code = feat["feature_name"], feat["code"]
        try:
            lv = {"tx_df": tx_df.copy(), "open_df": open_df.copy(),
                  "close_df": close_df.copy(), "pd": pd, "np": np}
            exec(code, {"pd": pd, "np": np, "__builtins__": __builtins__}, lv)
            r = lv.get("result")
            if not isinstance(r, pd.DataFrame) or "wallet_id" not in r.columns:
                failures.append((name, "bad result"))
                continue
            vc = name if name in r.columns else ("feature_value" if "feature_value" in r.columns else None)
            if vc is None:
                other = [c for c in r.columns if c != "wallet_id"]
                vc = other[0] if len(other) == 1 else None
            if vc is None:
                failures.append((name, "no value col"))
                continue
            mapped = tx_df["src"].map(r.set_index("wallet_id")[vc]).fillna(0)
            new_vals[name] = mapped.values
        except Exception as e:
            failures.append((name, str(e)[:60]))

    print(f"成功: {len(new_vals)}/{len(new_features)}, 失败: {len(failures)}")
    for n, e in failures:
        print(f"  {n}: {e}")
    print()

    # ===== 3. 去重 =====
    print("=" * 60)
    print("Step 3: 相关性去重 (|corr| > 0.95)")
    print("=" * 60)
    dupes = {}
    for nn, nv in new_vals.items():
        if len(nv) != len(X_full):
            continue
        ns = pd.Series(nv).fillna(0)
        if ns.std() == 0:
            dupes[nn] = [("constant", 0.0)]
            continue
        for bc in feature_cols:
            bs = X_full[bc].fillna(0)
            if bs.std() == 0:
                continue
            corr = np.corrcoef(ns.values, bs.values)[0, 1]
            if abs(corr) > 0.95:
                dupes.setdefault(nn, []).append((bc, corr))

    if dupes:
        print(f"重复: {len(dupes)}")
        for n, cs in sorted(dupes.items()):
            for bc, cr in cs:
                print(f"  {n} <-> {bc}: {cr:.4f}")
    else:
        print("无重复")
    non_dup = set(new_vals) - set(dupes)
    print(f"去重后: {len(non_dup)}/{len(new_vals)}\n")

    # ===== 4. 增量 =====
    print("=" * 60)
    print("Step 4: 增量 PR-AUC")
    print("=" * 60)
    X_dedup = X_full.copy()
    for n in sorted(non_dup):
        if len(new_vals[n]) == len(X_dedup):
            X_dedup[n] = new_vals[n]
    prauc_dedup = compute_prauc(X_dedup, y, f"基线+去重({X_dedup.shape[1]}特征)")

    X_all = X_full.copy()
    for n, v in new_vals.items():
        if len(v) == len(X_all):
            X_all[n] = v
    prauc_all = compute_prauc(X_all, y, f"基线+全部({X_all.shape[1]}特征)")

    print(f"\n{'='*60}")
    print("最终对比")
    print("=" * 60)
    print(f"  完整基线 ({len(feature_cols)}特征):         PR-AUC = {prauc_full:.6f}")
    print(f"  +全部新特征 ({X_all.shape[1]}特征):   PR-AUC = {prauc_all:.6f}  ({prauc_all-prauc_full:+.6f})")
    print(f"  +去重新特征 ({X_dedup.shape[1]}特征): PR-AUC = {prauc_dedup:.6f}  ({prauc_dedup-prauc_full:+.6f})")


if __name__ == "__main__":
    main()
