"""CAAFE baseline wrapper: 在指定数据集上运行 CAAFE 生成特征代码并输出增强后的 CSV。

用法:
    python scripts/run_caafe.py --dataset credit_g --iterations 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "CAAFE"))

# Stub out tabpfn to avoid heavy dependency
import types
sys.modules["tabpfn"] = types.ModuleType("tabpfn")
sys.modules["tabpfn.scripts"] = types.ModuleType("tabpfn.scripts")
sys.modules["tabpfn.scripts.transformer_prediction_interface"] = types.ModuleType(
    "tabpfn.scripts.transformer_prediction_interface"
)

from caafe.caafe import generate_features
from caafe.run_llm_code import run_llm_code

DATASET_REGISTRY = {
    "credit_g": {
        "csv": PROJECT_ROOT / "data" / "public" / "credit_g.csv",
        "target": "class",
        "description": "German credit dataset. Classify customers as good or bad credit risks based on attributes like credit history, purpose, employment, etc.",
    },
    "diabetes": {
        "csv": PROJECT_ROOT / "data" / "public" / "diabetes.csv",
        "target": "class",
        "description": "Pima Indians diabetes dataset. Predict whether a patient has diabetes based on diagnostic measurements like glucose, BMI, age, etc.",
    },
    "blood_transfusion": {
        "csv": PROJECT_ROOT / "data" / "public" / "blood_transfusion.csv",
        "target": "Class",
        "description": "Blood transfusion service center dataset. Predict whether a donor donated blood in March 2007 based on recency, frequency, monetary, and time features.",
    },
    "heart": {
        "csv": PROJECT_ROOT / "data" / "public" / "heart.csv",
        "target": "HeartDisease",
        "description": "Heart disease dataset. Predict whether a patient has heart disease based on age, sex, chest pain type, resting blood pressure, cholesterol, etc.",
    },
    "bank_marketing": {
        "csv": PROJECT_ROOT / "data" / "public" / "bank_marketing.csv",
        "target": "y",
        "description": "Bank marketing dataset. Predict whether a client will subscribe to a term deposit based on demographic and campaign contact information.",
    },
    "breast_w": {
        "csv": PROJECT_ROOT / "data" / "public" / "breast_w.csv",
        "target": "Class",
        "description": "Wisconsin breast cancer dataset. Classify tumors as benign or malignant based on cell nucleus features like clump thickness, uniformity, etc.",
    },
    "simecny": {
        "csv": None,
        "target": "is_risk",
        "description": "Digital RMB anti-money-laundering dataset. Identify risky transactions based on transaction amount, frequency, time patterns, wallet attributes, and network features.",
    },
}


def load_dataset(name: str) -> tuple[pd.DataFrame, str]:
    reg = DATASET_REGISTRY[name]
    target = reg["target"]
    if name == "simecny":
        df = pd.read_parquet(PROJECT_ROOT / "outputs" / "baseline_cache.parquet")
    else:
        df = pd.read_csv(reg["csv"])
    return df, target


def build_ds_tuple(df: pd.DataFrame, target: str, description: str):
    """Build the ds tuple that CAAFE expects:
    (name, X_tensor, y_tensor, categorical_indices, attribute_names+[target], description)
    """
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == "object" or hasattr(X[col], "cat"):
            X[col] = X[col].astype("category").cat.codes
    X = X.fillna(-999)

    cat_indices = []
    attr_names = list(feature_cols) + [target]

    return (None, None, None, cat_indices, attr_names, description)


def main():
    parser = argparse.ArgumentParser(description="Run CAAFE baseline")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterative LLM feature generation rounds")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "caafe")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, target = load_dataset(args.dataset)
    reg = DATASET_REGISTRY[args.dataset]
    ds = build_ds_tuple(df, target, reg["description"])

    df_for_caafe = df.copy()
    for col in df_for_caafe.columns:
        if col != target and (df_for_caafe[col].dtype == "object" or hasattr(df_for_caafe[col], "cat")):
            df_for_caafe[col] = df_for_caafe[col].astype("category").cat.codes
    df_for_caafe = df_for_caafe.fillna(-999)

    print(f"Running CAAFE on {args.dataset} ({len(df)} rows, {len(df.columns)-1} features)")
    print(f"Model: {args.model}, iterations: {args.iterations}")

    code, prompt, messages = generate_features(
        ds=ds,
        df=df_for_caafe,
        model=args.model,
        iterative=args.iterations,
        metric_used="auc",
        iterative_method="logistic",
        display_method="print",
    )

    code_path = args.output_dir / f"{args.dataset}_code.py"
    code_path.write_text(code or "", encoding="utf-8")
    print(f"Generated code saved to {code_path}")

    if code and code.strip():
        df_enhanced = run_llm_code(code, df_for_caafe.drop(columns=[target]))
        original_cols = set(df.columns) - {target}
        new_cols = [c for c in df_enhanced.columns if c not in original_cols]
        if new_cols:
            features_df = df_enhanced[new_cols]
            out_path = args.output_dir / f"{args.dataset}_features.csv"
            features_df.to_csv(out_path, index=False)
            print(f"Generated {len(new_cols)} new features → {out_path}")
        else:
            print("No new features generated.")
    else:
        print("No code generated.")


if __name__ == "__main__":
    main()
