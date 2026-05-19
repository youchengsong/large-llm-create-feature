"""特征验证流水线：L1 → L2(IV+KS) → L3(F1/Recall/Precision) → 打分 + 前向选择。"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from src.config import DEFAULT_CONFIG, ValidationThresholds
from src.evaluation.sandbox import execute_feature_code, validate_output
from src.evaluation.metrics import (
    compute_iv,
    compute_ks,
    compute_redundancy,
    compute_mutual_info,
    compute_cv_metrics,
    compute_delta_metrics,
    compute_shap_rank,
    compute_permutation_importance,
    normalize_score,
)


def run_l1_validation(
    code: str,
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str]:
    """L1 代码验证：沙盒执行 + 输出格式检查。返回 (feature_df, error_msg)。"""
    feature_df, error = execute_feature_code(code, open_df, tx_df, close_df)
    if error:
        return None, f"执行失败: {error}"

    valid, reason = validate_output(feature_df)
    if not valid:
        return None, f"输出验证失败: {reason}"

    return feature_df, ""


def run_l2_validation(
    feature: pd.Series,
    label: pd.Series,
    existing_features: pd.DataFrame,
    thresholds: ValidationThresholds | None = None,
) -> tuple[bool, dict]:
    """L2 统计验证：IV + KS + 冗余度 + 缺失率。返回 (passed, metrics_dict)。"""
    if thresholds is None:
        thresholds = DEFAULT_CONFIG.validation

    iv = compute_iv(feature, label)
    ks = compute_ks(feature, label)
    redundancy = compute_redundancy(feature, existing_features)
    missing_rate = float(feature.isna().mean())

    metrics = {
        "iv": iv,
        "ks": ks,
        "redundancy": redundancy,
        "missing_rate": missing_rate,
    }

    passed = (
        iv >= thresholds.iv_min
        and ks >= thresholds.ks_min
        and redundancy < thresholds.corr_max
        and missing_rate <= thresholds.missing_rate_max
    )

    return passed, metrics


def run_l3_validation(
    X_existing: pd.DataFrame,
    new_feature: pd.Series,
    y: pd.Series,
    baseline_metrics: dict,
) -> tuple[bool, dict]:
    """L3 模型验证（Recall优先）。

    接受条件：
    - delta_recall > 0（Recall必须提升）
    - new_precision >= 0.95（Precision绝对值不低于95%）
    """
    delta_result = compute_delta_metrics(X_existing, new_feature, y)

    X_all = X_existing.copy()
    feat_name = new_feature.name if new_feature.name else "new_feature"
    X_all[feat_name] = new_feature.values

    shap_rank = compute_shap_rank(X_all, y, feat_name)

    metrics = {
        **delta_result,
        "shap_rank": shap_rank,
        "total_features": X_all.shape[1],
    }

    delta_recall = delta_result["delta_recall"]
    aug_precision = delta_result["aug_precision"]

    passed = (
        delta_recall > 0
        and aug_precision >= 0.95
    )

    return passed, metrics


def compute_composite_score(
    shap_rank: int,
    iv: float,
    redundancy: float,
    delta_f1: float,
    all_scores: dict[str, list[float]],
) -> float:
    """加权打分。"""
    weights = DEFAULT_CONFIG.scoring

    inv_rank = 1.0 / shap_rank
    inv_redundancy = 1.0 - redundancy

    all_inv_ranks = [1.0 / r for r in all_scores.get("shap_rank", [shap_rank])]
    all_ivs = all_scores.get("iv", [iv])
    all_inv_reds = [1.0 - r for r in all_scores.get("redundancy", [redundancy])]
    all_deltas = all_scores.get("delta_f1", [delta_f1])

    norm_shap = normalize_score(inv_rank, all_inv_ranks)
    norm_iv = normalize_score(iv, all_ivs)
    norm_red = normalize_score(inv_redundancy, all_inv_reds)
    norm_delta = normalize_score(delta_f1, all_deltas)

    score = (
        weights.shap * norm_shap
        + weights.iv * norm_iv
        + weights.redundancy * norm_red
        + weights.delta_f1 * norm_delta
    )

    return float(score)


def _check_accept_condition(current_metrics: dict, new_metrics: dict) -> bool:
    """Forward Selection接受条件（Recall优先，与L3一致）。

    要求：
    - delta_recall > 0（Recall必须提升）
    - new_precision >= 0.95（Precision绝对值不低于95%）
    """
    delta_recall = new_metrics["recall"] - current_metrics["recall"]

    return (
        delta_recall > 0
        and new_metrics["precision"] >= 0.95
    )


def forward_select_features(
    X_baseline: pd.DataFrame,
    candidate_features: dict[str, np.ndarray],
    y: pd.Series,
    baseline_metrics: dict,
) -> tuple[list[str], dict]:
    """前向选择（Recall优先）：逐个添加特征，按Recall增量排序。

    接受条件（与L3一致）：
    - delta_recall > 0
    - delta_f1 >= -0.001
    - delta_precision >= -0.001

    返回 (selected_names, final_metrics)。
    """
    if not candidate_features:
        return [], baseline_metrics

    remaining = set(candidate_features.keys())
    selected = []
    X_current = X_baseline.copy()
    current_metrics = baseline_metrics.copy()

    logger.info("Forward selection: %d candidates, baseline Recall=%.6f, F1=%.6f, Precision=%.4f",
                len(remaining), current_metrics["recall"], current_metrics["f1"], current_metrics["precision"])

    while remaining:
        best_name = None
        best_score = -float("inf")
        best_metrics = None

        for name in remaining:
            X_try = X_current.copy()
            X_try[name] = candidate_features[name]
            m = compute_cv_metrics(X_try, y)
            if _check_accept_condition(current_metrics, m):
                score = (m["recall"] - current_metrics["recall"]) + 0.3 * (m["f1"] - current_metrics["f1"])
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_metrics = m

        if best_name is None:
            logger.info("Forward selection stopped: no feature meets accept condition")
            break

        selected.append(best_name)
        X_current[best_name] = candidate_features[best_name]
        current_metrics = best_metrics
        remaining.discard(best_name)

        logger.info("Forward selection added %s: Recall=%.4f, F1=%.6f, Precision=%.4f",
                    best_name, best_metrics["recall"], best_metrics["f1"], best_metrics["precision"])

    final_metrics = compute_cv_metrics(X_current, y)
    logger.info("Forward selection done: %d features selected, final Recall=%.4f, F1=%.6f, Precision=%.4f",
                len(selected), final_metrics["recall"], final_metrics["f1"], final_metrics["precision"])

    return selected, final_metrics


def backward_eliminate_pending(
    X_accepted: pd.DataFrame,
    pending_features: dict[str, np.ndarray],
    y: pd.Series,
    baseline_metrics: dict | None = None,
) -> tuple[set[str], set[str]]:
    """前向选择替代反向淘汰：从候选池中选出能提升 F1 的特征。"""
    if not pending_features:
        return set(), set()

    if baseline_metrics is None:
        baseline_metrics = compute_cv_metrics(X_accepted, y)

    selected, final_metrics = forward_select_features(
        X_accepted, pending_features, y, baseline_metrics,
    )

    accepted = set(selected)
    rejected = set(pending_features.keys()) - accepted

    logger.info("Pending pool evaluation: %d accepted, %d rejected",
                len(accepted), len(rejected))

    return accepted, rejected
