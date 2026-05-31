"""AutoFeat baseline wrapper: 多项式特征扩展 + L1 选择。

用法:
    python scripts/run_autofeat.py --dataset credit_g
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_REGISTRY = {
    "credit_g": {"csv": "data/public/credit_g.csv", "target": "class"},
    "diabetes": {"csv": "data/public/diabetes.csv", "target": "class"},
    "blood_transfusion": {"csv": "data/public/blood_transfusion.csv", "target": "Class"},
    "heart": {"csv": "data/public/heart.csv", "target": "HeartDisease"},
    "bank_marketing": {"csv": "data/public/bank_marketing.csv", "target": "y"},
    "breast_w": {"csv": "data/public/breast_w.csv", "target": "Class"},
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
    parser = argparse.ArgumentParser(description="Run AutoFeat baseline")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "autofeat")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, target = load_dataset(args.dataset)
    y = df[target].values
    X = df.drop(columns=[target])

    # AutoFeat needs numeric label
    if hasattr(y[0], 'dtype') and y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )

    print(f"Running AutoFeat on {args.dataset} ({len(df)} rows, {X.shape[1]} features)")

    from autofeat import AutoFeatClassifier

    afc = AutoFeatClassifier(
        feateng_steps=2,
        verbose=0,
    )
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
