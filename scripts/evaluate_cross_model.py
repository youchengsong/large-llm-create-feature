"""Check whether KnowFeat gains persist across tree-model families.

The experiment compares the original feature matrix (No-FE) with the same
matrix augmented by the already-generated KnowFeat features. Every estimator
uses the frozen development/test partition; only model seeds vary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from knowfeat.data.split import get_or_make_split
from scripts.evaluate_holdout import MODEL_SEEDS, load_dataset, prepare_xy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "results" / "cross_model_analysis.json"
DATASETS = (
    "credit_g", "diabetes", "blood_transfusion", "heart",
    "bank_marketing", "breast_w", "adult", "wine_quality",
)
MODEL_NAMES = ("XGBoost", "LightGBM", "CatBoost")
FEATURE_FILES = {
    dataset: PROJECT_ROOT / "outputs" / dataset / "knowfeat_features.csv"
    for dataset in DATASETS
}


def load_comparison_data(dataset: str):
    """Load aligned No-FE and KnowFeat matrices without IDs or labels."""
    if dataset not in FEATURE_FILES:
        raise ValueError(f"unsupported dataset: {dataset}")
    feature_path = FEATURE_FILES[dataset]
    if not feature_path.exists():
        raise FileNotFoundError(f"KnowFeat matrix is missing: {feature_path}")

    baseline_frame, target = load_dataset(dataset)
    augmented_frame, augmented_target = load_dataset(dataset, str(feature_path))
    if target != augmented_target:
        raise ValueError(f"target mismatch for {dataset}")
    baseline, labels = prepare_xy(baseline_frame, target)
    augmented, augmented_labels = prepare_xy(augmented_frame, target)
    if not labels.equals(augmented_labels):
        raise ValueError(f"label alignment mismatch for {dataset}")
    if len(augmented) != len(baseline):
        raise ValueError(f"row alignment mismatch for {dataset}")
    if augmented.shape[1] <= baseline.shape[1]:
        raise ValueError(f"KnowFeat added no usable columns for {dataset}")
    return baseline, augmented, labels


def _make_estimator(model_name: str, seed: int, scale_pos_weight: float, n_jobs: int):
    common = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": seed,
    }
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **common, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            enable_categorical=True, verbosity=0, n_jobs=n_jobs,
        )
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **common, num_leaves=63, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            verbosity=-1, deterministic=True, force_col_wise=True, n_jobs=n_jobs,
        )
    if model_name == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=common["n_estimators"], depth=common["max_depth"],
            learning_rate=common["learning_rate"], random_seed=seed,
            scale_pos_weight=scale_pos_weight, bootstrap_type="Bernoulli",
            subsample=0.8, rsm=0.8, loss_function="Logloss", eval_metric="AUC",
            verbose=False, allow_writing_files=False, thread_count=n_jobs,
        )
    raise ValueError(f"unsupported model: {model_name}")


def evaluate_matrix(X, labels, split, model_name: str, seeds: Iterable[int], n_jobs: int) -> dict[str, Any]:
    """Fit one estimator family on the frozen split for each model seed."""
    X_dev, X_test = X.iloc[split.dev_idx], X.iloc[split.test_idx]
    y_dev, y_test = labels.iloc[split.dev_idx], labels.iloc[split.test_idx]
    scale_pos_weight = float((y_dev == 0).sum() / max((y_dev == 1).sum(), 1))
    values = []
    for seed in seeds:
        estimator = _make_estimator(model_name, int(seed), scale_pos_weight, n_jobs)
        estimator.fit(X_dev, y_dev)
        probability = estimator.predict_proba(X_test)[:, 1]
        values.append(float(roc_auc_score(y_test, probability)))
    return {
        "mean": float(np.mean(values)),
        "seed_std": float(np.std(values)),
        "values": values,
        "n_features": int(X.shape[1]),
    }


def summarize_model(dataset_results: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    """Summarize paired KnowFeat-minus-No-FE AUC deltas across datasets."""
    deltas = np.asarray([
        result["KnowFeat"]["mean"] - result["No-FE"]["mean"]
        for result in dataset_results.values()
    ], dtype=float)
    tolerance = 1e-12
    if np.all(np.abs(deltas) <= tolerance):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(deltas, alternative="two-sided", method="auto")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "n_datasets": int(len(deltas)),
        "wins": int(np.sum(deltas > tolerance)),
        "ties": int(np.sum(np.abs(deltas) <= tolerance)),
        "losses": int(np.sum(deltas < -tolerance)),
        "mean_auc_delta": float(np.mean(deltas)),
        "median_auc_delta": float(np.median(deltas)),
        "min_auc_delta": float(np.min(deltas)),
        "max_auc_delta": float(np.max(deltas)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for position, original_index in enumerate(order):
        running_max = max(running_max, min(1.0, (total - position) * p_values[original_index]))
        adjusted[original_index] = running_max
    return adjusted


def evaluate_all(
    datasets: Iterable[str] = DATASETS,
    models: Iterable[str] = MODEL_NAMES,
    seeds: Iterable[int] = MODEL_SEEDS,
    n_jobs: int = 4,
) -> dict[str, Any]:
    datasets = tuple(datasets)
    models = tuple(models)
    seeds = tuple(int(seed) for seed in seeds)
    results: dict[str, Any] = {
        "protocol": {
            "comparison": "No-FE versus No-FE plus frozen KnowFeat features",
            "split_seed": 42,
            "test_size": 0.2,
            "model_seeds": list(seeds),
            "primary_metric": "frozen-test AUC",
            "estimator_budget": {
                "n_estimators_or_iterations": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
        },
        "datasets": {},
        "models": {},
        "inputs": {},
    }

    for dataset in datasets:
        baseline, augmented, labels = load_comparison_data(dataset)
        feature_path = FEATURE_FILES[dataset]
        results["inputs"][dataset] = {
            "feature_matrix": str(feature_path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "n_rows": int(len(labels)),
            "baseline_features": int(baseline.shape[1]),
            "augmented_features": int(augmented.shape[1]),
        }
        split_path = PROJECT_ROOT / "outputs" / "splits" / f"{dataset}_split.json"
        split = get_or_make_split(len(labels), labels=labels, cache_path=split_path)
        results["datasets"][dataset] = {}
        for model_name in models:
            no_fe = evaluate_matrix(baseline, labels, split, model_name, seeds, n_jobs)
            knowfeat = evaluate_matrix(augmented, labels, split, model_name, seeds, n_jobs)
            results["datasets"][dataset][model_name] = {
                "No-FE": no_fe,
                "KnowFeat": knowfeat,
                "auc_delta": knowfeat["mean"] - no_fe["mean"],
            }
            print(
                f"{dataset:20s} {model_name:8s} No-FE={no_fe['mean']:.4f} "
                f"KnowFeat={knowfeat['mean']:.4f} "
                f"delta={knowfeat['mean'] - no_fe['mean']:+.4f}", flush=True,
            )

    summaries = []
    for model_name in models:
        by_dataset = {
            dataset: results["datasets"][dataset][model_name]
            for dataset in datasets
        }
        summary = summarize_model(by_dataset)
        results["models"][model_name] = summary
        summaries.append(summary)
    adjusted = _holm_adjust([summary["wilcoxon_p_value"] for summary in summaries])
    for model_name, p_value in zip(models, adjusted):
        results["models"][model_name]["wilcoxon_p_value_holm"] = p_value
        results["models"][model_name]["significant_at_0.05"] = p_value < 0.05
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(MODEL_SEEDS))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = evaluate_all(args.datasets, args.models, args.seeds, args.n_jobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
