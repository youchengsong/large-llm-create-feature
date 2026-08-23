"""OpenFE baseline wrapper: 自动搜索 top-K 变换特征。

用法:
    python scripts/run_openfe.py --dataset credit_g --n-features 30
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_REGISTRY = {
    "credit_g": {"csv": "data/public/credit_g.csv", "target": "class"},
    "diabetes": {"csv": "data/public/diabetes.csv", "target": "class"},
    "blood_transfusion": {"csv": "data/public/blood_transfusion.csv", "target": "Class"},
    "heart": {"csv": "data/public/heart.csv", "target": "HeartDisease"},
    "bank_marketing": {"csv": "data/public/bank_marketing.csv", "target": "y"},
    "breast_w": {"csv": "data/public/breast_w.csv", "target": "Class"},
    "adult": {"csv": "data/public/adult.csv", "target": "class"},
    "wine_quality": {"csv": "data/public/wine_quality.csv", "target": "class"},
    "simecny": {"csv": None, "target": "is_risk"},
}


def load_dataset(name: str):
    reg = DATASET_REGISTRY[name]
    target = reg["target"]
    if name == "simecny":
        df = pd.read_parquet(PROJECT_ROOT / "outputs" / "baseline_cache.parquet")
    else:
        df = pd.read_csv(PROJECT_ROOT / reg["csv"])

    for col in df.columns:
        if col != target and (df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col])):
            df[col] = df[col].astype("category").cat.codes
    df = df.fillna(-999)
    return df, target


def main():
    parser = argparse.ArgumentParser(description="Run OpenFE baseline")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--n-features", type=int, default=30,
                        help="Number of top features to select")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "openfe")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, target = load_dataset(args.dataset)
    baseline_exclude = {"Min_Received", "Min_Sent", "S_src_counterparty_index",
                        "Timestamp_Hour", "S_dst_counterparty_index",
                        "1H_frequency_From", "1H_frequency_To",
                        "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent"}
    df = df.drop(columns=[c for c in baseline_exclude if c in df.columns], errors="ignore")
    y = df[target]
    X = df.drop(columns=[target])

    # 无泄漏协议：使用与 KnowFeat / evaluate_holdout 共享的冻结划分。
    # OpenFE 只在 D_dev 上 fit，再 transform 到 dev+test 输出全量 CSV，
    # 最终由 evaluate_holdout 在冻结 test 上评分，保证与其他方法同一 test 集。
    from knowfeat.data.split import get_or_make_split
    split_cache = PROJECT_ROOT / "outputs" / "splits" / f"{args.dataset}_split.json"
    y_int = pd.to_numeric(y, errors="coerce")
    if y_int.isna().any():
        from sklearn.preprocessing import LabelEncoder
        y_int = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
    split = get_or_make_split(len(df), labels=y_int.astype(int), cache_path=split_cache)
    X_train, y_train = X.iloc[split.dev_idx], y.iloc[split.dev_idx]
    X_test, y_test = X.iloc[split.test_idx], y.iloc[split.test_idx]

    print(f"Running OpenFE on {args.dataset} ({len(df)} rows, {X.shape[1]} features)")

    from openfe import OpenFE, transform

    ofe = OpenFE()
    features = ofe.fit(
        data=X_train, label=y_train,
        n_jobs=1,
        verbose=False,
    )

    top_features = features[:args.n_features]
    print(f"OpenFE found {len(features)} features, selecting top {len(top_features)}")

    X_train_new, X_test_new = transform(X_train, X_test, top_features, n_jobs=1)

    original_cols = set(X.columns)
    new_cols = [c for c in X_train_new.columns if c not in original_cols]

    if new_cols:
        full_new = pd.concat([X_train_new[new_cols], X_test_new[new_cols]])
        full_new = full_new.sort_index()
        out_path = args.output_dir / f"{args.dataset}_features.csv"
        full_new.to_csv(out_path, index=False)
        print(f"Generated {len(new_cols)} new features → {out_path}")
    else:
        print("No new features generated.")


if __name__ == "__main__":
    main()
