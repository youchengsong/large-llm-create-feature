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

## 2026-08-24: cross-model validation and matrix traceability

- Problem: several released KnowFeat feature matrices had drifted from the
  matrices that produced the versioned frozen-holdout JSON, so a new
  classifier-family experiment could silently evaluate different inputs.
  Holdout evaluation also allowed XGBoost to consume every host CPU, causing
  severe oversubscription when evaluations ran concurrently.
- Resolution: restored and versioned all eight paper matrices, regenerated the
  two stale holdout records and all dependent paper analyses, and required the
  cross-model XGBoost column to reproduce every paper AUC exactly. Added
  LightGBM/CatBoost evaluation with matrix SHA-256 hashes, capped model jobs,
  updated the manuscript, and added an isolated three-run LLM stability runner
  with feature-name Jaccard and frozen-test AUC aggregation.
- Prevention: treat the feature matrix as a versioned manuscript input, store
  its hash with derived results, and test matrix-to-JSON reproduction before
  reporting secondary analyses. Repeated LLM runs must use isolated run
  directories and cannot be summarized until all three runs are complete.
- Commit: `fc28682`

The full suite passes with 634 tests. Actual independent LLM calls remain
pending because the project environment has no `KNOWFEAT_API_KEY`; the runner
fails before creating partial output when credentials are absent.

## 2026-08-24: EDBT template tracking and preamble consistency

- Problem: the EDBT variant (`main_edbt.tex`, acmart sigconf) shares every body
  file with the IEEEtran variant but was never tracked, so a fresh clone could
  not build the actual submission target. Its preamble had also drifted from the
  shared body: it loaded `algpseudocode` while `method.tex` writes Algorithm 1
  in `algorithmic` syntax (`\STATE`, `\FOR`, `\TO`), and it omitted `amssymb`
  even though `related_work.tex` uses `\checkmark` eight times in Table 1.
  `acmart.cls` provides `amsmath` but not `amssymb`, so both Algorithm 1 and
  Table 1 would fail to compile.
- Resolution: tracked `main_edbt.tex`, `acmart.cls`, `edbt-macros.tex`, and
  `ACM-Reference-Format.bst`; switched the EDBT preamble to `algorithmic`; and
  added `amssymb`. Verified the remaining shared-body dependencies
  (`booktabs`, `xcolor`, `graphicx`, `hyperref`, `balance`) are supplied by
  `acmart.cls`, and that no IEEEtran-specific command appears in the body.
- Prevention: when two document classes share body files, diff both preambles
  against the commands the body actually uses rather than against each other.
  A shared body makes the classes look interchangeable when they are not.
- Commit: `07200e9`

Not verified by compilation: this environment has no LaTeX toolchain and cannot
install one (`apt` requires root). The fixes were derived by reading
`acmart.cls` and the shared body sources, and still need one `pdflatex` run to
confirm page count and fonts.

The LLM stability experiment remains incomplete. With
`KNOWFEAT_BASE_URL=https://api.apexin.ai/v1` and
`KNOWFEAT_MODEL=deepseek-v4-flash` a single-round smoke run finished in 1164 s
and passed L1/L2, but a full run needs roughly 3.2 hours per dataset-run and
about 19 hours for all six runs, which exceeds this environment's background
task limit. Do not use `deepseek-v4-pro`: it is a reasoning model whose chain of
thought consumes the entire `max_tokens` budget, leaving empty content and
stalling the agent at `generated=0`.

