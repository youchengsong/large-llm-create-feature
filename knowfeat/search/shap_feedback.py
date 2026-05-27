"""SHAP 反馈：计算特征 SHAP 重要性、按维度聚合贡献、复合打分。

前向选择后对当前模型计算 SHAP 值，按维度聚合贡献并按贡献反比
分配下一轮预算。同时为每个特征计算复合分数（用于排序和跨维度
传播优先级，不影响接受判定）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from knowfeat.config import KnowFeatConfig, SearchConfig, VerificationConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureScore:
    """单个特征的 SHAP 分析与复合打分结果。"""
    name: str
    dimensions: list[str]
    shap_importance: float
    shap_rank: int
    iv: float
    redundancy: float
    delta_recall: float
    composite_score: float


@dataclass
class ShapFeedback:
    """一轮 SHAP 反馈的完整结果。"""
    feature_scores: list[FeatureScore]
    dimension_contributions: dict[str, float]
    shap_values_mean: dict[str, float]


def compute_shap_importance(
    X: pd.DataFrame,
    label: pd.Series,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """计算每个特征的平均绝对 SHAP 值（基于 XGBoost 内置 tree SHAP）。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵（含基线 + 已接受特征）。
    label : pd.Series
        二分类标签（0/1）。
    n_folds : int
        CV 折数。
    seed : int
        随机种子。

    Returns
    -------
    dict[str, float]
        {特征名: 平均 |SHAP| 值}。
    """
    if X.shape[1] == 0:
        return {}

    y = label.values.astype(int)
    X_clean = X.copy()
    for col in X_clean.columns:
        X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
    X_clean = X_clean.fillna(0)

    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale = neg_count / pos_count if pos_count > 0 else 1.0

    shap_accum = np.zeros(X_clean.shape[1])
    n_samples = 0

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(X_clean, y):
        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train = y[train_idx]

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_train, y_train)

        booster = clf.get_booster()
        import xgboost as xgb
        dval = xgb.DMatrix(X_val, feature_names=list(X_clean.columns))
        shap_vals = booster.predict(dval, pred_contribs=True)

        # pred_contribs returns shape (n_samples, n_features + 1), last col is bias
        shap_vals = shap_vals[:, :-1]
        shap_accum += np.abs(shap_vals).sum(axis=0)
        n_samples += len(val_idx)

    mean_abs_shap = shap_accum / max(n_samples, 1)
    return dict(zip(X_clean.columns, mean_abs_shap))


def aggregate_dimension_contributions(
    shap_importance: dict[str, float],
    feature_dimensions: dict[str, list[str]],
    dimensions: list[str] | None = None,
) -> dict[str, float]:
    """按维度聚合 SHAP 贡献。

    多维度标签的特征，其 SHAP 值会被均分到各所属维度。

    Parameters
    ----------
    shap_importance : {特征名: SHAP 值}。
    feature_dimensions : {特征名: [维度列表]}。
    dimensions : 全部维度列表。默认从 feature_dimensions 中收集。

    Returns
    -------
    dict[str, float]
        {维度名: 聚合 SHAP 贡献}。
    """
    if dimensions is None:
        dims_set: set[str] = set()
        for dl in feature_dimensions.values():
            dims_set.update(dl)
        dimensions = sorted(dims_set)

    contrib: dict[str, float] = {d: 0.0 for d in dimensions}

    for feat, shap_val in shap_importance.items():
        dims = feature_dimensions.get(feat, [])
        if not dims:
            continue
        share = shap_val / len(dims)
        for d in dims:
            if d in contrib:
                contrib[d] += share

    return contrib


def compute_composite_score(
    shap_rank: int,
    total_features: int,
    iv: float,
    redundancy: float,
    delta_recall: float,
    cfg: KnowFeatConfig | None = None,
    search_cfg: SearchConfig | None = None,
) -> float:
    """计算特征复合打分。

    30% SHAP rank(1/rank) + 20% IV + 20% (1-redundancy) + 30% Delta Recall

    Parameters
    ----------
    shap_rank : 特征在 SHAP 排名中的位置（1 = 最重要）。
    total_features : 总特征数（用于归一化 SHAP rank）。
    iv : 特征 IV 值。
    redundancy : 冗余度（0-1，与已有特征最大相关系数）。
    delta_recall : 特征带来的 delta-recall。
    cfg : 顶层配置。
    search_cfg : 搜索配置，优先于 cfg.search。

    Returns
    -------
    float
        复合分数。
    """
    if search_cfg is None:
        search_cfg = cfg.search if cfg is not None else SearchConfig()

    w_shap = search_cfg.score_weight_shap
    w_iv = search_cfg.score_weight_iv
    w_novelty = search_cfg.score_weight_novelty
    w_dr = search_cfg.score_weight_delta_recall

    shap_component = (1.0 / max(shap_rank, 1)) if total_features > 0 else 0.0
    novelty_component = max(0.0, 1.0 - redundancy)

    return (
        w_shap * shap_component
        + w_iv * iv
        + w_novelty * novelty_component
        + w_dr * delta_recall
    )


def analyze_shap_feedback(
    X: pd.DataFrame,
    label: pd.Series,
    feature_dimensions: dict[str, list[str]],
    feature_iv: dict[str, float] | None = None,
    feature_redundancy: dict[str, float] | None = None,
    feature_delta_recall: dict[str, float] | None = None,
    cfg: KnowFeatConfig | None = None,
    search_cfg: SearchConfig | None = None,
    vcfg: VerificationConfig | None = None,
    seed: int = 42,
) -> ShapFeedback:
    """完整 SHAP 反馈分析：计算 SHAP → 聚合维度贡献 → 特征复合打分。

    Parameters
    ----------
    X : pd.DataFrame
        当前特征矩阵（基线 + 已接受特征）。
    label : pd.Series
        二分类标签。
    feature_dimensions : {特征名: [维度列表]}。
        每个已接受特征的维度归属。基线特征可不包含。
    feature_iv : {特征名: IV 值}，可选。
    feature_redundancy : {特征名: 冗余度}，可选。
    feature_delta_recall : {特征名: delta_recall}，可选。
    cfg : 顶层配置。
    search_cfg : 搜索配置。
    vcfg : 验证配置。
    seed : 随机种子。

    Returns
    -------
    ShapFeedback
        包含 feature_scores、dimension_contributions、shap_values_mean。
    """
    if search_cfg is None:
        search_cfg = cfg.search if cfg is not None else SearchConfig()
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    if feature_iv is None:
        feature_iv = {}
    if feature_redundancy is None:
        feature_redundancy = {}
    if feature_delta_recall is None:
        feature_delta_recall = {}

    n_folds = vcfg.cv_folds

    shap_importance = compute_shap_importance(X, label, n_folds=n_folds, seed=seed)

    dim_contributions = aggregate_dimension_contributions(
        shap_importance, feature_dimensions, search_cfg.dimensions,
    )

    sorted_feats = sorted(
        shap_importance.items(), key=lambda x: x[1], reverse=True,
    )
    rank_map = {name: rank + 1 for rank, (name, _) in enumerate(sorted_feats)}
    total_features = len(sorted_feats)

    feature_scores: list[FeatureScore] = []
    for feat_name, shap_val in sorted_feats:
        dims = feature_dimensions.get(feat_name, [])
        rank = rank_map[feat_name]
        iv = feature_iv.get(feat_name, 0.0)
        redundancy = feature_redundancy.get(feat_name, 0.0)
        dr = feature_delta_recall.get(feat_name, 0.0)

        score = compute_composite_score(
            shap_rank=rank,
            total_features=total_features,
            iv=iv,
            redundancy=redundancy,
            delta_recall=dr,
            search_cfg=search_cfg,
        )
        feature_scores.append(FeatureScore(
            name=feat_name,
            dimensions=dims,
            shap_importance=shap_val,
            shap_rank=rank,
            iv=iv,
            redundancy=redundancy,
            delta_recall=dr,
            composite_score=score,
        ))

    logger.info(
        "SHAP 反馈: %d 特征, 维度贡献=%s",
        len(feature_scores),
        {d: f"{v:.4f}" for d, v in dim_contributions.items()},
    )

    return ShapFeedback(
        feature_scores=feature_scores,
        dimension_contributions=dim_contributions,
        shap_values_mean=dict(shap_importance),
    )


def format_shap_summary(feedback: ShapFeedback, top_k: int = 10) -> str:
    """格式化 SHAP 反馈摘要文本（供 prompt 注入）。

    Parameters
    ----------
    feedback : ShapFeedback
        SHAP 反馈结果。
    top_k : int
        显示排名前 K 的特征。

    Returns
    -------
    str
        格式化的 Markdown 文本。
    """
    lines: list[str] = []

    lines.append("## 当前特征 SHAP 重要性反馈")
    lines.append("")
    lines.append("### 维度贡献分布")
    for dim, contrib in sorted(
        feedback.dimension_contributions.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        lines.append(f"- {dim}: {contrib:.4f}")

    lines.append("")
    lines.append(f"### 特征重要性排名（Top {top_k}）")
    lines.append("| 排名 | 特征名 | SHAP值 | 维度 | 复合分 |")
    lines.append("|------|--------|--------|------|--------|")
    for fs in feedback.feature_scores[:top_k]:
        dims_str = "/".join(fs.dimensions) if fs.dimensions else "-"
        lines.append(
            f"| {fs.shap_rank} | {fs.name} | {fs.shap_importance:.4f} | "
            f"{dims_str} | {fs.composite_score:.4f} |"
        )

    return "\n".join(lines)
