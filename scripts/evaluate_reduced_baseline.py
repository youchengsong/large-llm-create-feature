"""Evaluate generated features from a three-feature baseline without ID leakage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from knowfeat.data.split import make_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "results" / "reduced_baseline_analysis.json"
MODEL_SEEDS = (42, 123, 456, 789, 2024)
EXCLUDE_COLUMNS = {
    "row_id", "account", "wallet_id", "tx_id", "tx_id_fe",
    "label", "class", "is_risk", "is_illicit", "is_laundering",
}

DATASETS = {
    "credit_g": {
        "csv": "data/public/credit_g.csv",
        "target": "class",
        "core": ["duration", "credit_amount", "age"],
        "features": {
            "OpenFE": "outputs/openfe/credit_g_features.csv",
            "CAAFE": "outputs/caafe/credit_g_features.csv",
            "FeatLLM": "outputs/featllm/credit_g_features.csv",
            "KnowFeat": "outputs/credit_g/knowfeat_features.csv",
        },
    },
    "heart": {
        "csv": "data/public/heart.csv",
        "target": "HeartDisease",
        "core": ["Age", "RestingBP", "Cholesterol"],
        "features": {
            "OpenFE": "outputs/openfe/heart_features.csv",
            "CAAFE": "outputs/caafe/heart_features.csv",
            "FeatLLM": "outputs/featllm/heart_features.csv",
            "KnowFeat": "outputs/heart/knowfeat_features.csv",
        },
    },
    "diabetes": {
        "csv": "data/public/diabetes.csv",
        "target": "class",
        "core": ["Glucose", "BMI", "Age"],
        "features": {
            "OpenFE": "outputs/openfe/diabetes_features.csv",
            "CAAFE": "outputs/caafe/diabetes_features.csv",
            "FeatLLM": "outputs/featllm/diabetes_features.csv",
            "KnowFeat": "outputs/diabetes/knowfeat_features.csv",
        },
    },
    "wine_quality": {
        "csv": "data/public/wine_quality.csv",
        "target": "class",
        "core": ["alcohol", "volatile_acidity", "pH"],
        "features": {
            "OpenFE": "outputs/openfe/wine_quality_features.csv",
            "KnowFeat": "outputs/wine_quality/knowfeat_features.csv",
        },
    },
}


def prepare_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers and encode any remaining categorical columns."""
    frame = frame.drop(
        columns=[column for column in frame.columns if column in EXCLUDE_COLUMNS],
        errors="ignore",
    ).copy()
    for column in frame:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype("category").cat.codes
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(-999)
    frame.columns = [re.sub(r"[\[\]<>]", "_", str(column)) for column in frame.columns]
    return frame


def evaluate_auc(X: pd.DataFrame, y: pd.Series) -> dict:
    split = make_split(len(y), labels=y, seed=42, test_size=0.2)
    X_dev, X_test = X.iloc[split.dev_idx], X.iloc[split.test_idx]
    y_dev, y_test = y.iloc[split.dev_idx], y.iloc[split.test_idx]
    scale_pos_weight = float((y_dev == 0).sum() / max((y_dev == 1).sum(), 1))
    values = []
    for seed in MODEL_SEEDS:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="logloss",
            enable_categorical=True,
            verbosity=0,
        )
        model.fit(X_dev, y_dev)
        values.append(float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])))
    return {
        "mean": float(np.mean(values)),
        "seed_std": float(np.std(values)),
        "values": values,
        "n_features": int(X.shape[1]),
    }


def evaluate_all() -> dict:
    results = {
        "protocol": {
            "split_seed": 42,
            "test_size": 0.2,
            "model_seeds": list(MODEL_SEEDS),
            "identifier_columns_excluded": sorted(EXCLUDE_COLUMNS),
        },
        "datasets": {},
    }
    for dataset, config in DATASETS.items():
        raw = pd.read_csv(PROJECT_ROOT / config["csv"])
        target = raw[config["target"]]
        if not pd.api.types.is_numeric_dtype(target):
            target = pd.Series(LabelEncoder().fit_transform(target.astype(str)))
        target = target.astype(int).reset_index(drop=True)
        core = prepare_numeric(raw[config["core"]])

        dataset_results = {"No-FE": evaluate_auc(core, target)}
        for method, relative_path in config["features"].items():
            generated = pd.read_csv(PROJECT_ROOT / relative_path)
            if len(generated) != len(raw):
                raise ValueError(
                    f"{dataset}/{method} has {len(generated)} rows; expected {len(raw)}"
                )
            generated = prepare_numeric(generated)
            overlap = set(core.columns).intersection(generated.columns)
            if overlap:
                raise ValueError(f"{dataset}/{method} duplicates core columns: {sorted(overlap)}")
            dataset_results[method] = evaluate_auc(
                pd.concat([core, generated], axis=1),
                target,
            )
        results["datasets"][dataset] = dataset_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = evaluate_all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for dataset, methods in results["datasets"].items():
        values = " ".join(f"{method}={result['mean']:.4f}" for method, result in methods.items())
        print(f"{dataset}: {values}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
