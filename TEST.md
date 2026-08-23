# Test Guide

Run the unit suite from the repository root:

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

The repository `pytest.ini` limits collection to `tests/`, so scripts that call
external APIs are not imported as tests. The split tests cover reproducible
stratified partitions, cache invalidation by size/seed/test ratio, malformed
cache recovery, duplicate indices, bounds, and complete dev/test coverage.

For baseline experiments, inspect `outputs/splits/<dataset>_split.json` before
holdout evaluation. It must be the same cache used by feature generation and
`scripts/evaluate_holdout.py`.
