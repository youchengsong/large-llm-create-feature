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

## 2026-08-23: reproducible paper submission package

- Problem: the paper mixed stale aggregate claims with unversioned holdout
  outputs, lacked a reproducible Wilcoxon/Holm analysis, and used reduced
  baselines that could retain identifier columns. The compiled PDF also
  contained a Type 3 figure font and an overflowing author line.
- Resolution: versioned all frozen-holdout inputs, added deterministic public
  and reduced-baseline analysis scripts, excluded identifiers and labels from
  every reduced method, aligned the manuscript and rebuttal with the resulting
  statistics, and regenerated a clean 12-page PDF with embedded TrueType/Type 1
  fonts. Added wine-quality assets and regression tests for the shared protocol.
- Prevention: derive paper tables and figures from versioned JSON, keep feature
  eligibility rules centralized in the dataset loader, run the full test suite,
  and check LaTeX logs plus `pdffonts` before submission.
- Commit: `4b68b5e`

The full suite passes with 627 tests; the remaining nine warnings are Python
3.14 `multiprocessing.fork()` deprecation warnings in tool tests.
