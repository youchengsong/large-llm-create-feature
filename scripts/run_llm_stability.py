"""Run independent KnowFeat generations and frozen-holdout evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.analyze_llm_stability import analyze_stability
from scripts.evaluate_holdout import evaluate_holdout


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_stability"
DEFAULT_ANALYSIS = PROJECT_ROOT / "paper" / "results" / "llm_stability_analysis.json"
DEFAULT_DATASETS = ("heart", "credit_g")


def _require_api_key() -> None:
    if not os.environ.get("KNOWFEAT_API_KEY"):
        raise RuntimeError("KNOWFEAT_API_KEY is required for independent LLM stability runs")


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_stability(
    datasets: tuple[str, ...], runs: int, output_dir: Path, max_rounds: int,
    max_features: int, temperature: float, n_bootstrap: int, force: bool,
) -> dict:
    _require_api_key()
    for dataset in datasets:
        for run_id in range(1, runs + 1):
            run_dir = output_dir / dataset / f"run_{run_id}"
            holdout_path = run_dir / "holdout.json"
            if holdout_path.exists() and not force:
                print(f"Skipping completed {dataset}/run_{run_id}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            _run([
                sys.executable, "scripts/run_knowfeat_public.py", "--dataset", dataset,
                "--max-rounds", str(max_rounds), "--max-features", str(max_features),
                "--seed", "42", "--temperature", str(temperature),
                "--result-dir", str(run_dir),
            ])
            _run([
                sys.executable, "scripts/knowfeat_to_csv.py", "--dataset", dataset,
                "--features-dir", str(run_dir),
            ])
            result = evaluate_holdout(
                dataset_name=dataset,
                method_name=f"KnowFeat-stability-run-{run_id}",
                features_csv=str(run_dir / "knowfeat_features.csv"),
                n_bootstrap=n_bootstrap,
            )
            holdout_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return analyze_stability(output_dir, datasets, expected_runs=runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-output", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=9999)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    analysis = run_stability(
        datasets=tuple(args.datasets), runs=args.runs, output_dir=args.output_dir,
        max_rounds=args.max_rounds, max_features=args.max_features,
        temperature=args.temperature, n_bootstrap=args.n_bootstrap, force=args.force,
    )
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.analysis_output}")


if __name__ == "__main__":
    main()
