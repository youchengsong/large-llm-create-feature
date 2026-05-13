"""特征验证流水线：L1 代码验证 → L2 统计验证 → L3 模型验证 → 加权打分。"""

import pandas as pd

from src.config import DEFAULT_CONFIG
from src.evaluation.sandbox import execute_feature_code, validate_output
from src.evaluation.metrics import (
    compute_mutual_info,
    compute_redundancy,
    compute_cramers_v,
    compute_bootstrap_chi2,
    compute_delta_prauc,
    compute_shap_rank,
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
) -> tuple[bool, dict]:
    """L2 统计验证：MI、冗余度、Cramer's V、Bootstrap χ²。返回 (passed, metrics_dict)。"""
    thresholds = DEFAULT_CONFIG.validation

    mi = compute_mutual_info(feature, label)
    redundancy = compute_redundancy(feature, existing_features)
    cramers_v = compute_cramers_v(feature, label)
    bootstrap_p = compute_bootstrap_chi2(feature, label)

    metrics = {
        "mi": mi,
        "redundancy": redundancy,
        "cramers_v": cramers_v,
        "bootstrap_p": bootstrap_p,
    }

    passed = (
        mi >= thresholds.mi_min
        and redundancy < thresholds.corr_max
        and cramers_v >= thresholds.cramers_v_min
        and bootstrap_p < thresholds.bootstrap_p_max
    )

    return passed, metrics


def run_l3_validation(
    X_existing: pd.DataFrame,
    new_feature: pd.Series,
    y: pd.Series,
    baseline_prauc: float,
) -> tuple[bool, dict]:
    """L3 模型验证：ΔPR-AUC + SHAP 排名。返回 (passed, metrics_dict)。"""
    thresholds = DEFAULT_CONFIG.validation

    delta = compute_delta_prauc(X_existing, new_feature, y)

    X_all = X_existing.copy()
    feat_name = new_feature.name if new_feature.name else "new_feature"
    X_all[feat_name] = new_feature.values

    shap_rank = compute_shap_rank(X_all, y, feat_name)

    metrics = {
        "delta_prauc": delta,
        "shap_rank": shap_rank,
        "total_features": X_all.shape[1],
    }

    passed = delta >= thresholds.delta_prauc_min

    return passed, metrics


def compute_composite_score(
    shap_rank: int,
    mi: float,
    redundancy: float,
    delta_prauc: float,
    all_scores: dict[str, list[float]],
) -> float:
    """加权打分。all_scores 包含各指标的历史值列表，用于归一化。"""
    weights = DEFAULT_CONFIG.scoring

    inv_rank = 1.0 / shap_rank
    inv_redundancy = 1.0 - redundancy

    all_inv_ranks = [1.0 / r for r in all_scores.get("shap_rank", [shap_rank])]
    all_mis = all_scores.get("mi", [mi])
    all_inv_reds = [1.0 - r for r in all_scores.get("redundancy", [redundancy])]
    all_deltas = all_scores.get("delta_prauc", [delta_prauc])

    norm_shap = normalize_score(inv_rank, all_inv_ranks)
    norm_mi = normalize_score(mi, all_mis)
    norm_red = normalize_score(inv_redundancy, all_inv_reds)
    norm_delta = normalize_score(delta_prauc, all_deltas)

    score = (
        weights.shap * norm_shap
        + weights.mi * norm_mi
        + weights.redundancy * norm_red
        + weights.delta_prauc * norm_delta
    )

    return float(score)
