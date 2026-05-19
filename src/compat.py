"""XGBoost + SHAP 兼容性修复。

XGBoost 3.2 GPU 模型的 base_score 返回带方括号的字符串（如 '[1.4586365E-1]'），
shap 0.49.1 解析时崩溃。此模块在创建 TreeExplainer 前修复该属性。
"""

from __future__ import annotations

import re

import shap


def _fix_xgb_base_score(model) -> None:
    """修复 XGBoost 模型的 base_score 格式，使其兼容 shap。"""
    try:
        bs = model.get_params().get("base_score")
        if bs is None:
            raw = getattr(model, "base_score_", None)
            if raw is None:
                return
            bs = raw

        if isinstance(bs, str):
            cleaned = re.sub(r"[\[\]\s]", "", bs)
            model.base_score = float(cleaned)
    except Exception:
        pass


def safe_tree_explainer(model, **kwargs) -> shap.TreeExplainer:
    """创建 TreeExplainer，自动处理 XGBoost 3.2 + CUDA 兼容性问题。"""
    _fix_xgb_base_score(model)
    return shap.TreeExplainer(model, **kwargs)
