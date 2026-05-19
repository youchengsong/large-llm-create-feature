"""SHAP 反馈生成：训练 XGBoost 计算全局 SHAP 重要性，生成反馈文本供 LLM prompt 使用。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG, ISLANDS


def generate_shap_feedback(X_all: pd.DataFrame, y: pd.Series) -> dict:
    """训练 XGBoost + SHAP，返回全局特征重要性反馈。

    Returns:
        {
            "top_10_features": [(name, normalized_importance), ...],
            "weak_dimensions": [dimension_name, ...],
            "feature_importance": {name: normalized_importance, ...},
            "model_type": "XGBoost",
        }
    """
    if X_all.empty or X_all.shape[1] == 0:
        return {
            "top_10_features": [],
            "weak_dimensions": [ic.dimension for ic in ISLANDS],
            "feature_importance": {},
            "model_type": "XGBoost",
        }

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
    model.fit(X_all, y, verbose=False)

    try:
        explainer = shap.TreeExplainer(model)
        max_shap_samples = 50000
        if len(X_all) > max_shap_samples:
            X_shap = X_all.sample(n=max_shap_samples, random_state=42)
        else:
            X_shap = X_all
        shap_values = explainer.shap_values(X_shap)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
    except Exception:
        mean_abs = model.feature_importances_

    total = mean_abs.sum()
    if total > 0:
        normalized = mean_abs / total
    else:
        normalized = np.zeros_like(mean_abs)

    columns = list(X_all.columns)
    feature_importance = {col: float(normalized[i]) for i, col in enumerate(columns)}

    sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])
    top_10 = sorted_features[:10]

    dim_importance: dict[str, float] = {}
    for ic in ISLANDS:
        dim_vals = [feature_importance[f] for f in columns if _feature_belongs_to_dimension(f, ic.fields)]
        dim_importance[ic.dimension] = sum(dim_vals) if dim_vals else 0.0

    if dim_importance:
        mean_dim = np.mean(list(dim_importance.values()))
        weak_dimensions = [d for d, v in dim_importance.items() if v < mean_dim]
    else:
        weak_dimensions = [ic.dimension for ic in ISLANDS]

    return {
        "top_10_features": top_10,
        "weak_dimensions": weak_dimensions,
        "feature_importance": feature_importance,
        "model_type": "XGBoost",
    }


def _feature_belongs_to_dimension(feature_name: str, dimension_fields: list[str]) -> bool:
    """判断特征名是否与某维度的字段相关（简单子串匹配）。"""
    fn_lower = feature_name.lower()
    return any(field.lower() in fn_lower for field in dimension_fields)


def shap_feedback_to_prompt(feedback: dict, island_dimensions: list[str] | None = None) -> str:
    """将 SHAP 反馈转为 prompt 文本。"""
    lines = ["## SHAP 特征重要性反馈（下游模型: XGBoost）\n"]

    top_features = feedback.get("top_10_features", [])
    if top_features:
        lines.append("当前最重要的特征（归一化 SHAP 值）：")
        for name, imp in top_features:
            lines.append(f"  - {name}: {imp:.4f}")
    else:
        lines.append("暂无已接受的特征。")

    weak = feedback.get("weak_dimensions", [])
    dim_name_map = {ic.dimension: ic.name for ic in ISLANDS}
    if weak:
        weak_names = [dim_name_map.get(d, d) for d in weak]
        lines.append(f"\n薄弱维度（需要更多探索）：{', '.join(weak_names)}")

    if island_dimensions:
        current_dims = [dim_name_map.get(d, d) for d in island_dimensions]
        lines.append(f"\n当前岛屿关注维度：{', '.join(current_dims)}")

    return "\n".join(lines)


def compute_island_budgets(
    dim_importance: dict[str, float],
    island_dimensions: list[str],
    B_total: int = 12,
    b_min: int = 1,
    b_max: int = 5,
) -> dict[str, int]:
    """根据 SHAP 维度重要性的反比分配每个岛屿的候选预算。

    薄弱维度获得更多候选名额，已饱和维度获得更少。
    """
    eps = 0.01
    raw_weights = {}
    for dim in island_dimensions:
        imp = dim_importance.get(dim, 0.0)
        raw_weights[dim] = 1.0 / (imp + eps)

    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        return {dim: B_total // len(island_dimensions) for dim in island_dimensions}

    budgets = {}
    for dim in island_dimensions:
        raw = raw_weights[dim] / total_weight * B_total
        budgets[dim] = max(b_min, min(b_max, round(raw)))

    current_total = sum(budgets.values())
    diff = B_total - current_total
    if diff != 0:
        sorted_dims = sorted(raw_weights.keys(), key=lambda d: -raw_weights[d])
        for dim in sorted_dims:
            if diff == 0:
                break
            if diff > 0 and budgets[dim] < b_max:
                budgets[dim] += 1
                diff -= 1
            elif diff < 0 and budgets[dim] > b_min:
                budgets[dim] -= 1
                diff += 1

    return budgets
