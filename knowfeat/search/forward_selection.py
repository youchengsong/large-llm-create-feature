"""前向选择：从候选池中贪心搜索特征组合，解决单特征增量不够但组合有效的问题。"""

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
    """候选特征信息。"""
    name: str
    series: pd.Series
    dimension: list[str] = field(default_factory=list)
    code: str = ""
    explanation: str = ""
    reasoning: str = ""
    knowledge_source: str = ""
    data_fields_used: list[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    """前向选择结果。"""
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
) -> SelectionResult:
    """贪心前向选择：逐个评估候选特征，delta_recall > 0 则接受。

    Parameters
    ----------
    baseline_X : pd.DataFrame
        当前已接受的特征矩阵（含基线 + 已通过 L3 的特征）。
    label : pd.Series
        二分类标签（0/1）。
    candidates : list[CandidateFeature]
        候选特征列表（通过 L2 但未通过 L3 单独评估的特征）。
    cfg : KnowFeatConfig, optional
        顶层配置。
    vcfg : VerificationConfig, optional
        验证配置。优先于 cfg.verification。
    seed : int
        随机种子。

    Returns
    -------
    SelectionResult
        accepted: 被接受的候选特征列表（按接受顺序）。
        rejected: 未被接受的候选特征列表。
        eval_details: 每个候选特征名 → ModelEvalResult。
    """
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    if not candidates:
        logger.info("前向选择: 候选池为空，跳过")
        return SelectionResult(accepted=[], rejected=[], eval_details={})

    current_X = baseline_X.copy()
    accepted: list[CandidateFeature] = []
    rejected: list[CandidateFeature] = []
    eval_details: dict[str, ModelEvalResult] = {}

    logger.info("前向选择开始: %d 个候选特征", len(candidates))

    for i, cand in enumerate(candidates):
        feat = pd.to_numeric(cand.series, errors="coerce").fillna(0)

        result = evaluate_feature(
            baseline_X=current_X,
            new_feature=feat.rename(cand.name),
            label=label,
            vcfg=vcfg,
            seed=seed,
        )
        eval_details[cand.name] = result

        if result.passed:
            accepted.append(cand)
            current_X[cand.name] = feat.values
            logger.info(
                "前向选择 [%d/%d] %s → 接受 (delta_recall=%+.4f)",
                i + 1, len(candidates), cand.name, result.delta_recall,
            )
        else:
            rejected.append(cand)
            logger.info(
                "前向选择 [%d/%d] %s → 拒绝 (%s)",
                i + 1, len(candidates), cand.name,
                "; ".join(result.reject_reasons),
            )

    logger.info(
        "前向选择完成: %d 接受, %d 拒绝 (共 %d 候选)",
        len(accepted), len(rejected), len(candidates),
    )

    return SelectionResult(
        accepted=accepted,
        rejected=rejected,
        eval_details=eval_details,
    )
