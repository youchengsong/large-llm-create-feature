"""Analyze feature-set and frozen-test stability across independent LLM runs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "llm_stability"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "results" / "llm_stability_analysis.json"
DEFAULT_DATASETS = ("heart", "credit_g")


def _load_run(run_dir: Path, dataset: str) -> dict[str, Any]:
    paths = {
        name: run_dir / filename for name, filename in {
            "summary": "generic_summary.json",
            "features": "generic_features.json",
            "holdout": "holdout.json",
        }.items()
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"incomplete stability run: {path}")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    features = json.loads(paths["features"].read_text(encoding="utf-8"))
    holdout = json.loads(paths["holdout"].read_text(encoding="utf-8"))
    summary_names = [str(name) for name in summary.get("accepted_features", [])]
    feature_names = [str(feature["name"]) for feature in features]
    if summary_names != feature_names:
        raise ValueError(f"feature metadata mismatch in {run_dir}")
    if summary.get("total_accepted") != len(feature_names):
        raise ValueError(f"accepted-feature count mismatch in {run_dir}")
    if holdout.get("dataset") != dataset:
        raise ValueError(f"holdout dataset mismatch in {run_dir}")
    if holdout.get("protocol") != "frozen_holdout_bootstrap":
        raise ValueError(f"unexpected holdout protocol in {run_dir}")
    if holdout.get("split_seed") != 42 or holdout.get("test_size") != 0.2:
        raise ValueError(f"unexpected frozen split in {run_dir}")
    return {
        "run": run_dir.name,
        "features": feature_names,
        "accepted_count": len(feature_names),
        "auc": float(holdout["metrics"]["auc"]["test_point"]),
        "elapsed_sec": float(summary.get("elapsed_sec", 0.0)),
    }


def analyze_stability(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    expected_runs: int = 3,
) -> dict[str, Any]:
    """Require complete independent runs and aggregate identity/performance stability."""
    output: dict[str, Any] = {
        "protocol": {
            "independent_llm_runs_per_dataset": expected_runs,
            "feature_identity_metric": "pairwise Jaccard similarity on accepted feature names",
            "performance_metric": "frozen-test AUC",
            "split_seed": 42,
            "test_size": 0.2,
        },
        "datasets": {},
    }
    for dataset in datasets:
        dataset_dir = Path(results_dir) / dataset
        run_dirs = sorted(
            (path for path in dataset_dir.glob("run_*") if path.is_dir()),
            key=lambda path: int(path.name.removeprefix("run_")),
        )
        if len(run_dirs) != expected_runs:
            raise ValueError(f"{dataset}: expected {expected_runs} runs, found {len(run_dirs)}")
        runs = [_load_run(run_dir, dataset) for run_dir in run_dirs]
        feature_sets = [set(run["features"]) for run in runs]
        pairwise = []
        for left, right in itertools.combinations(range(len(runs)), 2):
            union = feature_sets[left] | feature_sets[right]
            similarity = len(feature_sets[left] & feature_sets[right]) / len(union) if union else 1.0
            pairwise.append({
                "runs": [runs[left]["run"], runs[right]["run"]],
                "jaccard": float(similarity),
            })
        aucs = np.asarray([run["auc"] for run in runs], dtype=float)
        counts = np.asarray([run["accepted_count"] for run in runs], dtype=float)
        jaccards = [pair["jaccard"] for pair in pairwise]
        output["datasets"][dataset] = {
            "n_runs": len(runs),
            "runs": runs,
            "accepted_count_mean": float(np.mean(counts)),
            "accepted_count_std": float(np.std(counts)),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "auc_range": [float(np.min(aucs)), float(np.max(aucs))],
            "pairwise_jaccard": pairwise,
            "pairwise_jaccard_mean": float(np.mean(jaccards)),
            "pairwise_jaccard_std": float(np.std(jaccards)),
            "common_features": sorted(set.intersection(*feature_sets)),
            "union_feature_count": len(set.union(*feature_sets)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze_stability(args.results_dir, args.datasets, args.expected_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for dataset, summary in result["datasets"].items():
        print(
            f"{dataset}: AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}, "
            f"Jaccard={summary['pairwise_jaccard_mean']:.3f}±"
            f"{summary['pairwise_jaccard_std']:.3f}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
