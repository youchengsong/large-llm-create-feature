"""特征验证流水线：L1 → L2 → L3 → 打分 + 缓刑池反向淘汰。"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from src.config import DEFAULT_CONFIG
from src.evaluation.sandbox import execute_feature_code, validate_output
from src.evaluation.metrics import (
    compute_mutual_info,
    compute_redundancy,
    compute_cramers_v,
    compute_bootstrap_chi2,
    compute_cv_prauc,
    compute_delta_prauc,
    compute_shap_rank,
    normalize_score,
)


def calibrate_l2_thresholds(
    baseline_features: pd.DataFrame,
    label: pd.Series,
    percentile: float = 10.0,
) -> dict:
    """基于基线特征的指标分布，自动校准 L2 阈值。

    对每个基线特征计算 MI 和 Cramer's V，取 p-percentile 作为新阈值。
    返回 dict 包含校准后的阈值和基线指标统计。
    """
    mis = []
    cramers_vs = []

    for col in baseline_features.columns:
        feature = baseline_features[col].fillna(0)
        if feature.nunique() <= 1:
            continue
        mi = compute_mutual_info(feature, label)
        cv = compute_cramers_v(feature, label)
        mis.append(mi)
        cramers_vs.append(cv)

    if not mis:
        logger.warning("No valid baseline features for calibration, using defaults")
        return {
            "mi_min": DEFAULT_CONFIG.validation.mi_min,
            "cramers_v_min": DEFAULT_CONFIG.validation.cramers_v_min,
            "baseline_stats": {},
        }

    mi_arr = np.array(mis)
    cv_arr = np.array(cramers_vs)

    mi_threshold = float(max(np.percentile(mi_arr, percentile), 1e-6))
    cv_threshold = float(max(np.percentile(cv_arr, percentile), 0.001))

    stats = {
        "n_features": len(mis),
        "mi": {"min": float(mi_arr.min()), "p10": float(np.percentile(mi_arr, 10)),
               "median": float(np.median(mi_arr)), "max": float(mi_arr.max())},
        "cramers_v": {"min": float(cv_arr.min()), "p10": float(np.percentile(cv_arr, 10)),
                      "median": float(np.median(cv_arr)), "max": float(cv_arr.max())},
    }

    logger.info(
        "L2 threshold calibration: MI >= %.6f (was %.6f), Cramer's V >= %.4f (was %.4f), "
        "based on %d baseline features (p%.0f)",
        mi_threshold, DEFAULT_CONFIG.validation.mi_min,
        cv_threshold, DEFAULT_CONFIG.validation.cramers_v_min,
        len(mis), percentile,
    )

    return {
        "mi_min": mi_threshold,
        "cramers_v_min": cv_threshold,
        "baseline_stats": stats,
    }


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
    mi_min: float | None = None,
    cramers_v_min: float | None = None,
) -> tuple[bool, dict]:
    """L2 统计验证：MI、冗余度、Cramer's V、Bootstrap χ²。返回 (passed, metrics_dict)。

    mi_min / cramers_v_min 可由调用方传入（自动校准值），未传则用默认配置。
    """
    thresholds = DEFAULT_CONFIG.validation
    effective_mi_min = mi_min if mi_min is not None else thresholds.mi_min
    effective_cv_min = cramers_v_min if cramers_v_min is not None else thresholds.cramers_v_min

    mi = compute_mutual_info(feature, label)
    redundancy = compute_redundancy(feature, existing_features)
    cramers_v = compute_cramers_v(feature, label)
    bootstrap_p = compute_bootstrap_chi2(feature, label)

    metrics = {
        "mi": mi,
        "redundancy": redundancy,
        "cramers_v": cramers_v,
        "bootstrap_p": bootstrap_p,
        "thresholds_used": {
            "mi_min": effective_mi_min,
            "cramers_v_min": effective_cv_min,
            "corr_max": thresholds.corr_max,
            "bootstrap_p_max": thresholds.bootstrap_p_max,
        },
    }

    passed = (
        mi >= effective_mi_min
        and redundancy < thresholds.corr_max
        and cramers_v >= effective_cv_min
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


def backward_eliminate_pending(
    X_accepted: pd.DataFrame,
    pending_features: dict[str, np.ndarray],
    y: pd.Series,
) -> tuple[set[str], set[str]]:
    """反向淘汰：把缓刑池特征全部加入，从最弱的开始逐个尝试剔除。

    返回 (accepted_names, rejected_names)。
    accepted_names: 剔除后分数下降的特征（有组合贡献）
    rejected_names: 剔除后分数不降的特征（没有贡献）
    """
    if not pending_features:
        return set(), set()

    X_full = X_accepted.copy()
    for name, values in pending_features.items():
        X_full[name] = values

    score_full = compute_cv_prauc(X_full, y)
    logger.info("Backward elimination: full score=%.6f with %d accepted + %d pending",
                score_full, X_accepted.shape[1], len(pending_features))

    import shap
    from xgboost import XGBClassifier

    cfg = DEFAULT_CONFIG.xgboost
    model = XGBClassifier(
        scale_pos_weight=cfg.scale_pos_weight,
        eval_metric=cfg.eval_metric,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    model.fit(X_full, y, verbose=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_full)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance = dict(zip(X_full.columns, mean_abs))

    pending_names_sorted = sorted(
        pending_features.keys(),
        key=lambda n: importance.get(n, 0),
    )

    current_score = score_full
    rejected = set()
    X_current = X_full.copy()

    for name in pending_names_sorted:
        X_without = X_current.drop(columns=[name])
        score_without = compute_cv_prauc(X_without, y)

        if score_without >= current_score:
            rejected.add(name)
            X_current = X_without
            current_score = score_without
            logger.info("  Eliminated %s: score %.6f -> %.6f (no loss)",
                        name, current_score, score_without)
        else:
            logger.info("  Kept %s: removing would drop score %.6f -> %.6f",
                        name, current_score, score_without)

    accepted = set(pending_features.keys()) - rejected
    return accepted, rejected
