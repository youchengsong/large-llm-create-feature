"""特征验证流水线：L1 → L2 → L3 → 打分 + 缓刑池反向淘汰。"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from src.config import DEFAULT_CONFIG, ValidationThresholds
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
    """L2 统计验证：MI、冗余度、Cramer's V、Bootstrap χ²。返回 (passed, metrics_dict)。"""
    if thresholds is None:
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


def backward_eliminate_pending(
    X_accepted: pd.DataFrame,
    pending_features: dict[str, np.ndarray],
    y: pd.Series,
    baseline_prauc: float = 0.0,
) -> tuple[set[str], set[str]]:
    """两阶段筛选：反向淘汰去噪 + 单特征增量验证。

    阶段 1（反向淘汰）：把缓刑池特征全部加入已接受特征，从 SHAP 最弱的开始
    逐个剔除，移除对组合无贡献的噪音特征。

    阶段 2（增量验证）：对阶段 1 幸存的每个特征，单独加入原始 baseline
    计算 delta_prauc。只有 delta > 0 的特征才被最终接受。

    返回 (accepted_names, rejected_names)。
    """
    if not pending_features:
        return set(), set()

    # ── 阶段 1：反向淘汰，清除组合中无贡献的噪音 ──
    X_full = X_accepted.copy()
    for name, values in pending_features.items():
        X_full[name] = values

    score_full = compute_cv_prauc(X_full, y)
    logger.info("Backward elimination phase 1: full score=%.6f (baseline=%.6f) "
                "with %d accepted + %d pending",
                score_full, baseline_prauc, X_accepted.shape[1], len(pending_features))

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
        tree_method=cfg.tree_method,
        device=cfg.device,
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
    phase1_rejected = set()
    X_current = X_full.copy()

    for name in pending_names_sorted:
        X_without = X_current.drop(columns=[name])
        score_without = compute_cv_prauc(X_without, y)

        if score_without >= current_score:
            phase1_rejected.add(name)
            X_current = X_without
            current_score = score_without
            logger.info("  Phase 1 eliminated %s: score %.6f -> %.6f (no loss)",
                        name, current_score, score_without)
        else:
            logger.info("  Phase 1 kept %s: removing would drop score %.6f -> %.6f",
                        name, current_score, score_without)

    phase1_survivors = set(pending_features.keys()) - phase1_rejected
    logger.info("Phase 1 done: %d survivors, %d eliminated",
                len(phase1_survivors), len(phase1_rejected))

    if not phase1_survivors:
        return set(), set(pending_features.keys())

    # ── 阶段 2：对幸存特征逐个验证对 baseline 的增量 ──
    logger.info("Phase 2: verifying %d survivors against baseline (prauc=%.6f)",
                len(phase1_survivors), baseline_prauc)

    accepted = set()
    phase2_rejected = set()

    for name in phase1_survivors:
        feat_series = pd.Series(pending_features[name], name=name)
        delta = compute_delta_prauc(X_accepted, feat_series, y)
        if delta > 0:
            accepted.add(name)
            logger.info("  Phase 2 accepted %s: delta_prauc=+%.6f", name, delta)
        else:
            phase2_rejected.add(name)
            logger.info("  Phase 2 rejected %s: delta_prauc=%.6f (not positive)",
                        name, delta)

    rejected = phase1_rejected | phase2_rejected
    logger.info("Final: %d accepted, %d rejected (phase1: %d, phase2: %d)",
                len(accepted), len(rejected), len(phase1_rejected), len(phase2_rejected))
    return accepted, rejected
