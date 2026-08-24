"""Reproduce public-dataset ranks and paired significance tests for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "eval_holdout"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "results" / "public_holdout_analysis.json"

DATASETS = (
    "credit_g",
    "diabetes",
    "blood_transfusion",
    "heart",
    "bank_marketing",
    "breast_w",
    "adult",
    "wine_quality",
)
METHODS = ("No-FE", "OpenFE", "AutoFeat", "CAAFE", "FeatLLM", "LLM-FE", "KnowFeat")
EXPECTED_MISSING = {
    ("bank_marketing", "AutoFeat"),
    ("breast_w", "LLM-FE"),
    ("adult", "AutoFeat"),
    ("wine_quality", "LLM-FE"),
}


def load_public_results(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, dict[str, dict[str, Any]]]:
    """Load and validate the exact frozen-holdout results used by the paper."""
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASETS:
        dataset_results: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            path = results_dir / f"{dataset}_{method}.json"
            if not path.exists():
                if (dataset, method) in EXPECTED_MISSING:
                    continue
                raise FileNotFoundError(f"required paper result is missing: {path}")

            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol") != "frozen_holdout_bootstrap":
                raise ValueError(f"unexpected protocol in {path}")
            if payload.get("split_seed") != 42 or payload.get("test_size") != 0.2:
                raise ValueError(f"unexpected frozen split in {path}")

            auc = payload.get("metrics", {}).get("auc", {})
            ci95 = auc.get("bootstrap_ci95")
            if auc.get("test_point") is None or not isinstance(ci95, list) or len(ci95) != 2:
                raise ValueError(f"incomplete AUC result in {path}")
            dataset_results[method] = {
                "auc": float(auc["test_point"]),
                "seed_std": float(auc["test_seed_std"]),
                "bootstrap_ci95": [float(ci95[0]), float(ci95[1])],
                "n_features": int(payload["n_features"]),
                "n_dev": int(payload["n_dev"]),
                "n_test": int(payload["n_test"]),
                "source": str(path.relative_to(PROJECT_ROOT)),
            }
        results[dataset] = dataset_results
    return results


def average_ranks(results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, float]:
    """Rank methods per dataset, averaging ties and excluding missing runs."""
    ranks: dict[str, list[float]] = {method: [] for method in METHODS}
    for dataset in DATASETS:
        available = [method for method in METHODS if method in results[dataset]]
        aucs = [results[dataset][method]["auc"] for method in available]
        dataset_ranks = rankdata(-np.asarray(aucs), method="average")
        for method, rank in zip(available, dataset_ranks):
            ranks[method].append(float(rank))
    return {method: float(np.mean(values)) for method, values in ranks.items()}


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for position, original_index in enumerate(order):
        candidate = min(1.0, (total - position) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def compare_knowfeat(results: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Run two-sided paired Wilcoxon tests across shared datasets."""
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for baseline in METHODS:
        if baseline == "KnowFeat":
            continue
        shared = [
            dataset for dataset in DATASETS
            if baseline in results[dataset] and "KnowFeat" in results[dataset]
        ]
        differences = np.asarray([
            results[dataset]["KnowFeat"]["auc"] - results[dataset][baseline]["auc"]
            for dataset in shared
        ])
        test = wilcoxon(differences, alternative="two-sided", method="auto")
        raw_p_values.append(float(test.pvalue))
        comparisons.append({
            "baseline": baseline,
            "n_datasets": len(shared),
            "datasets": shared,
            "median_auc_delta": float(np.median(differences)),
            "wilcoxon_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
        })

    for comparison, adjusted in zip(comparisons, _holm_adjust(raw_p_values)):
        comparison["p_value_holm"] = adjusted
        comparison["significant_at_0.05"] = adjusted < 0.05
    return comparisons


def analyze(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, Any]:
    results = load_public_results(results_dir)
    return {
        "protocol": {
            "statistical_unit": "dataset-level frozen-test AUC",
            "test": "two-sided paired Wilcoxon signed-rank",
            "multiple_testing": "Holm family-wise correction across six KnowFeat comparisons",
            "alpha": 0.05,
        },
        "datasets": list(DATASETS),
        "methods": list(METHODS),
        "average_ranks": average_ranks(results),
        "knowfeat_comparisons": compare_knowfeat(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = analyze(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Average ranks:")
    for method, rank in sorted(summary["average_ranks"].items(), key=lambda item: item[1]):
        print(f"  {method:10s} {rank:.4f}")
    print("KnowFeat pairwise tests (Holm-adjusted):")
    for comparison in summary["knowfeat_comparisons"]:
        print(
            f"  vs {comparison['baseline']:8s} n={comparison['n_datasets']} "
            f"p={comparison['p_value']:.4f} p_holm={comparison['p_value_holm']:.4f}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
