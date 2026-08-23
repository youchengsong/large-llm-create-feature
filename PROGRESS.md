# Progress

## 2026-08-23: verification and holdout protocol repair

- Problem: legacy verification callers used `*_fallback` names after the
  configuration refactor; recall-mode L3 ignored the configured precision
  floor; post-hoc selection ran on every pipeline invocation; and split caches
  were reused based only on row count. Baseline scripts also generated some
  features with independent or full-data splits.
- Resolution: added deprecated config/calibration aliases, restored recall-mode
  precision enforcement, made post-hoc pruning deterministic and opt-in,
  validated complete cached partitions (including seed and test ratio), and
  moved AutoFeat, CAAFE, and FeatLLM to the shared frozen split protocol.
  Updated holdout-derived paper tables and the average-rank figure source.
- Prevention: run `PYTHONPATH=. .venv/bin/pytest -q`; keep
  `outputs/splits/<dataset>_split.json` as the single split cache; use the
  explicit post-hoc flag only for experiments that require a second selection
  pass.
- Commit: `6c1d287`

The full suite currently has pre-existing failures in unrelated translation and
retry-behavior changes in the dirty worktree; the focused verification,
orchestration, and split suites pass.
