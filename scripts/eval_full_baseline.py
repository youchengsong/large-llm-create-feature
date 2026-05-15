"""
评估完整基线特征的 PR-AUC，检查新特征与基线重复，计算去重后增量。
"""
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
    print(f"[{label}] PR-AUC = {mean_score:.6f} (5-fold, {X_arr.shape[1]} features)")
    return mean_score


def main():
    # ===== 1. 加载完整基线特征（原始值，未归一化）=====
    print("=" * 60)
    print("Step 1: 加载完整基线特征（原始值）")
    print("=" * 60)
    full_df = pd.read_csv("output/full_baseline_features_original.csv")

    non_feature_cols = {"id", "amount", "is_laundering", "account_is_laundering_From", "account_is_laundering_To"}
    feature_cols = [c for c in full_df.columns if c not in non_feature_cols]
    y = full_df["is_laundering"].astype(int)
    X_full = full_df[feature_cols].fillna(0)

    print(f"完整基线: {len(feature_cols)} 个特征, {len(y)} 行")
    print(f"正样本比例: {y.mean()*100:.2f}%")
    print()

    # ===== 2. 完整基线 PR-AUC =====
    print("=" * 60)
    print("Step 2: 完整基线 PR-AUC")
    print("=" * 60)
    prauc_full = compute_prauc(X_full, y, f"完整基线({len(feature_cols)}特征)")
    print()

    # ===== 3. 加载新特征并执行 =====
    print("=" * 60)
    print("Step 3: 执行52个新特征代码")
    print("=" * 60)

    open_df = pd.read_csv("inputs/SimECNY/open/accounts.csv")
    tx_df = pd.read_csv(
        "inputs/SimECNY/transaction/wallet_temporal_transactions_1105_1450_temp.csv",
        on_bad_lines="skip",
    )
    close_df = pd.read_csv("inputs/SimECNY/close/new_close_3.csv")

    with open("output/features.json", "r") as f:
        new_features = json.load(f)

    new_feature_values = {}
    exec_failures = []
    for feat in new_features:
        name = feat["feature_name"]
        code = feat["code"]
        try:
            local_vars = {
                "tx_df": tx_df.copy(),
                "open_df": open_df.copy(),
                "close_df": close_df.copy(),
                "pd": pd,
                "np": np,
            }
            exec(code, {"pd": pd, "np": np, "__builtins__": __builtins__}, local_vars)
            result = local_vars.get("result")
            if result is not None and isinstance(result, pd.DataFrame):
                if "wallet_id" not in result.columns:
                    exec_failures.append((name, f"no wallet_id, cols={list(result.columns)[:3]}"))
                    continue
                if name in result.columns:
                    val_col = name
                elif "feature_value" in result.columns:
                    val_col = "feature_value"
                else:
                    other_cols = [c for c in result.columns if c != "wallet_id"]
                    if len(other_cols) == 1:
                        val_col = other_cols[0]
                    else:
                        exec_failures.append((name, f"can't identify value col: {other_cols}"))
                        continue
                wallet_feat = result.set_index("wallet_id")[val_col]
                tx_feat = tx_df["src"].map(wallet_feat).fillna(0)
                new_feature_values[name] = tx_feat.values
            else:
                exec_failures.append((name, f"result type: {type(result)}"))
        except Exception as e:
            exec_failures.append((name, str(e)[:80]))

    print(f"成功执行: {len(new_feature_values)}/{len(new_features)}")
    if exec_failures:
        print(f"失败: {len(exec_failures)}")
        for name, err in exec_failures:
            print(f"  {name}: {err}")
    print()

    # ===== 4. 相关性去重 =====
    print("=" * 60)
    print("Step 4: 相关性去重 (|corr| > 0.95 视为重复)")
    print("=" * 60)

    duplicated = {}
    for new_name, new_vals in new_feature_values.items():
        if len(new_vals) != len(X_full):
            continue
        new_s = pd.Series(new_vals).fillna(0)
        if new_s.std() == 0:
            duplicated[new_name] = [("constant_feature", 0.0)]
            continue
        for base_col in feature_cols:
            base_s = X_full[base_col].fillna(0)
            if base_s.std() == 0:
                continue
            corr = np.corrcoef(new_s.values, base_s.values)[0, 1]
            if abs(corr) > 0.95:
                if new_name not in duplicated:
                    duplicated[new_name] = []
                duplicated[new_name].append((base_col, corr))

    if duplicated:
        print(f"发现 {len(duplicated)} 个重复特征:")
        for name, corrs in sorted(duplicated.items()):
            for base_col, corr in corrs:
                print(f"  {name} <-> {base_col}: corr={corr:.4f}")
    else:
        print("未发现重复特征")

    non_dup_names = set(new_feature_values.keys()) - set(duplicated.keys())
    print(f"\n去重后新特征: {len(non_dup_names)}/{len(new_feature_values)}")
    print()

    # ===== 5. 增量 PR-AUC =====
    print("=" * 60)
    print("Step 5: 增量 PR-AUC")
    print("=" * 60)

    X_combined = X_full.copy()
    added = 0
    for name in sorted(non_dup_names):
        vals = new_feature_values[name]
        if len(vals) == len(X_combined):
            X_combined[name] = vals
            added += 1

    print(f"添加 {added} 个去重新特征")
    prauc_combined = compute_prauc(X_combined, y, f"完整基线+去重新特征({X_combined.shape[1]}特征)")

    # 也测试所有新特征（不去重）
    X_all = X_full.copy()
    added_all = 0
    for name, vals in new_feature_values.items():
        if len(vals) == len(X_all):
            X_all[name] = vals
            added_all += 1
    prauc_all = compute_prauc(X_all, y, f"完整基线+全部新特征({X_all.shape[1]}特征)")

    print()
    print("=" * 60)
    print("最终对比")
    print("=" * 60)
    print(f"  完整基线 ({len(feature_cols)} 特征):                PR-AUC = {prauc_full:.6f}")
    print(f"  完整基线+全部新特征 ({X_all.shape[1]} 特征):    PR-AUC = {prauc_all:.6f}  (增量 {prauc_all - prauc_full:+.6f})")
    print(f"  完整基线+去重新特征 ({X_combined.shape[1]} 特征):  PR-AUC = {prauc_combined:.6f}  (增量 {prauc_combined - prauc_full:+.6f})")


if __name__ == "__main__":
    main()
