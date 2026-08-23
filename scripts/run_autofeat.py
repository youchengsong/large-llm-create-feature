"""AutoFeat baseline wrapper: 多项式特征扩展 + L1 选择。

用法:
    python scripts/run_autofeat.py --dataset credit_g
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(pd.Series, "ravel"):
    pd.Series.ravel = lambda self, order="C": self.to_numpy().ravel(order)

import sklearn.utils.validation as _skval
_orig_check_array = _skval.check_array
def _patched_check_array(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)
_skval.check_array = _patched_check_array

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
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes
    df = df.fillna(0)
    for col in df.columns:
        if pd.api.types.is_extension_array_dtype(df[col]):
            df[col] = df[col].to_numpy(dtype="float64", na_value=0)
    df = df.replace([np.inf, -np.inf], 0)
    return df, target


def main():
    parser = argparse.ArgumentParser(description="Run AutoFeat baseline")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "autofeat")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, target = load_dataset(args.dataset)
    baseline_exclude = {"Min_Received", "Min_Sent", "S_src_counterparty_index",
                        "Timestamp_Hour", "S_dst_counterparty_index",
                        "1H_frequency_From", "1H_frequency_To",
                        "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent"}
    df = df.drop(columns=[c for c in baseline_exclude if c in df.columns], errors="ignore")
    y = df[target].values
    X = df.drop(columns=[target])

    # AutoFeat needs numeric label
    if hasattr(y[0], 'dtype') and y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Use the same frozen split as KnowFeat and holdout evaluation.  AutoFeat
    # sees labels only on D_dev; the learned transformations are then applied
    # to the untouched D_test rows.
    from knowfeat.data.split import get_or_make_split
    split_cache = PROJECT_ROOT / "outputs" / "splits" / f"{args.dataset}_split.json"
    split = get_or_make_split(
        len(df), labels=pd.Series(y), cache_path=split_cache, seed=args.seed,
    )
    X_train = X.iloc[split.dev_idx]
    X_test = X.iloc[split.test_idx]
    y_train = y[split.dev_idx]

    print(f"Running AutoFeat on {args.dataset} ({len(df)} rows, {X.shape[1]} features)")

    from autofeat import AutoFeatClassifier

    steps = 1 if X.shape[1] > 50 or len(df) > 50000 else 2
    afc = AutoFeatClassifier(
        feateng_steps=steps,
        verbose=0,
    )
    print(f"  feateng_steps={steps} (cols={X.shape[1]}, rows={len(df)})")
    X_train_new = afc.fit_transform(X_train, y_train)
    X_test_new = afc.transform(X_test)

    original_cols = set(X.columns)
    new_cols = [c for c in X_train_new.columns if c not in original_cols]

    if new_cols:
        full_new = pd.concat([
            pd.DataFrame(X_train_new[new_cols], index=X_train.index),
            pd.DataFrame(X_test_new[new_cols], index=X_test.index),
        ])
        full_new = full_new.sort_index()
        out_path = args.output_dir / f"{args.dataset}_features.csv"
        full_new.to_csv(out_path, index=False)
        print(f"Generated {len(new_cols)} new features → {out_path}")
    else:
        print("No new features generated.")


if __name__ == "__main__":
    main()
