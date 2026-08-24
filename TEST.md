# Test Guide

Run the unit suite from the repository root:

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

The repository `pytest.ini` limits collection to `tests/`, so scripts that call
external APIs are not imported as tests. The split tests cover reproducible
stratified partitions, cache invalidation by size/seed/test ratio, malformed
cache recovery, duplicate indices, bounds, complete dev/test coverage, and
Stage 1/Stage 2 development-row subsetting.

For baseline experiments, inspect `outputs/splits/<dataset>_split.json` before
holdout evaluation. It must be the same cache used by feature generation and
`scripts/evaluate_holdout.py`.

Reproduce and validate the manuscript statistics:

```bash
PYTHONPATH=. .venv/bin/python scripts/analyze_public_results.py
PYTHONPATH=. .venv/bin/python scripts/evaluate_reduced_baseline.py
PYTHONPATH=. .venv/bin/python scripts/evaluate_cross_model.py
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_public_results_analysis.py tests/test_reduced_baseline.py \
  tests/test_cross_model_analysis.py tests/test_llm_stability_analysis.py
```

The public-results test verifies all required frozen-holdout JSON files,
average ranks, paired Wilcoxon/Holm conclusions, and bootstrap confidence
interval availability. The reduced-baseline test guards against reintroducing
identifier or label columns.
The cross-model test verifies all 8 datasets and 3 model families, checks the
stored feature-matrix hashes, and requires its XGBoost column to reproduce the
versioned holdout AUC exactly. Stability-analysis tests require three complete,
independent run directories and validate feature-set Jaccard aggregation,
frozen-split metadata, and API-key fail-fast behavior.

Compile-check the manuscript after changing paper sources:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
