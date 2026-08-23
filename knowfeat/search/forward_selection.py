"""Forward selection: greedy search for feature combinations from the candidate pool, solving the problem of individual features with insufficient incremental gain but effective in combination."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from knowfeat.config import KnowFeatConfig, VerificationConfig
from knowfeat.verification.model_eval import ModelEvalResult, evaluate_feature

logger = logging.getLogger(__name__)


@dataclass
class CandidateFeature:
    """Candidate feature information."""
    name: str
    series: pd.Series
    dimension: list[str] = field(default_factory=list)
    code: str = ""
    explanation: str = ""
    reasoning: str = ""
    knowledge_source: str = ""
    data_fields_used: list[str] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
    computation_steps: str = ""


@dataclass
class SelectionResult:
    """Forward selection result."""
    accepted: list[CandidateFeature]
    rejected: list[CandidateFeature]
    eval_details: dict[str, ModelEvalResult]


def forward_select(
    baseline_X: pd.DataFrame,
    label: pd.Series,
    candidates: list[CandidateFeature],
    cfg: Optional[KnowFeatConfig] = None,
    vcfg: Optional[VerificationConfig] = None,
    seed: int = 42,
    spw_override: float = None,
) -> SelectionResult:
    """Greedy forward selection: evaluate candidate features one by one, accept if delta_recall > 0.

    Parameters
    ----------
    baseline_X : pd.DataFrame
        Current accepted feature matrix (baseline + features that passed L3).
    label : pd.Series
        Binary label (0/1).
    candidates : list[CandidateFeature]
        Candidate feature list (passed L2 but not L3 individual evaluation).
    cfg : KnowFeatConfig, optional
        Top-level configuration.
    vcfg : VerificationConfig, optional
        Verification configuration. Takes priority over cfg.verification.
    seed : int
        Random seed.

    Returns
    -------
    SelectionResult
        accepted: list of accepted candidate features (in acceptance order).
        rejected: list of non-accepted candidate features.
        eval_details: mapping of each candidate feature name -> ModelEvalResult.
    """
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    if not candidates:
        logger.info("Forward selection: candidate pool empty, skipping")
        return SelectionResult(accepted=[], rejected=[], eval_details={})

    current_X = baseline_X.copy()
    accepted: list[CandidateFeature] = []
    rejected: list[CandidateFeature] = []
    eval_details: dict[str, ModelEvalResult] = {}

    logger.info("Forward selection started: %d candidate features", len(candidates))

    for i, cand in enumerate(candidates):
        feat = pd.to_numeric(cand.series, errors="coerce").fillna(0)

        result = evaluate_feature(
            baseline_X=current_X,
            new_feature=feat.rename(cand.name),
            label=label,
            vcfg=vcfg,
            seed=seed,
            spw_override=spw_override,
        )
        eval_details[cand.name] = result

        if result.passed:
            accepted.append(cand)
            current_X[cand.name] = feat.values
            if result.accept_metric == "auc":
                delta_str = f"delta_auc={result.delta_auc:+.4f}"
            else:
                delta_str = f"delta_recall={result.delta_recall:+.4f}"
            logger.info(
                "Forward selection [%d/%d] %s -> accepted (%s)",
                i + 1, len(candidates), cand.name, delta_str,
            )
        else:
            rejected.append(cand)
            logger.info(
                "Forward selection [%d/%d] %s -> rejected (%s)",
                i + 1, len(candidates), cand.name,
                "; ".join(result.reject_reasons),
            )

    logger.info(
        "Forward selection completed: %d accepted, %d rejected (out of %d candidates)",
        len(accepted), len(rejected), len(candidates),
    )

    return SelectionResult(
        accepted=accepted,
        rejected=rejected,
        eval_details=eval_details,
    )


def posthoc_select(
    baseline_X: pd.DataFrame,
    label: pd.Series,
    accepted_features: dict[str, "CandidateFeature"],
    cfg: "KnowFeatConfig | None" = None,
    vcfg: "VerificationConfig | None" = None,
    seed: int = 42,
    spw_override: float = None,
) -> list[str]:
    """Post-hoc greedy forward selection over all accepted features.

    Starts from the original baseline (no generated features) and greedily
    adds the feature that most improves the target metric.  Stops when no
    remaining feature improves performance.  This removes redundant features
    that individually passed L3 but hurt when combined.

    Returns the list of selected feature names (in selection order).
    """
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    if not accepted_features:
        return []

    # Dict insertion order is part of the experiment's reproducibility
    # contract; do not use a set for the candidate queue.
    feat_names = list(accepted_features.keys())
    feat_series = {}
    for name, cand in accepted_features.items():
        feat_series[name] = pd.to_numeric(cand.series, errors="coerce").fillna(0)

    original_baseline = baseline_X[[c for c in baseline_X.columns if c not in feat_names]].copy()

    selected: list[str] = []
    remaining = list(feat_names)
    current_X = original_baseline.copy()

    logger.info("Post-hoc selection started: %d accepted features to prune", len(feat_names))

    while remaining:
        best_name = None
        best_delta = -np.inf
        for name in remaining:
            result = evaluate_feature(
                baseline_X=current_X,
                new_feature=feat_series[name].rename(name),
                label=label,
                vcfg=vcfg,
                seed=seed,
                spw_override=spw_override,
            )
            if vcfg.accept_metric == "auc":
                delta = result.delta_auc
            else:
                delta = result.delta_recall

            if result.passed and delta > best_delta:
                best_delta = delta
                best_name = name

        if best_name is not None and best_delta > 0:
            selected.append(best_name)
            current_X[best_name] = feat_series[best_name].values
            remaining.remove(best_name)
            logger.info(
                "Post-hoc [%d] selected %s (delta=%.4f, remaining=%d)",
                len(selected), best_name, best_delta, len(remaining),
            )
        else:
            logger.info(
                "Post-hoc selection stopped: best delta=%.4f <= 0, %d features pruned",
                best_delta, len(remaining),
            )
            break

    logger.info(
        "Post-hoc selection: %d / %d features retained",
        len(selected), len(feat_names),
    )
    return selected
