"""L4 稳定性验证：Nogueira 稳定性指数 + Bootstrap SHAP 排名。"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG


def compute_nogueira_stability(selection_sets: list[set], n_features: int) -> float:
    """计算 Nogueira 稳定性指数。

    Args:
        selection_sets: 每次 bootstrap 选出的特征名集合列表
        n_features: 总特征数

    Returns:
        稳定性指数，范围 [-1, 1]，越高越稳定
    """
    M = len(selection_sets)
    p = n_features
    if M < 2 or p < 2:
        return 0.0

    all_features = sorted({f for s in selection_sets for f in s})
    feat_to_idx = {f: i for i, f in enumerate(all_features)}
    n_observed = len(all_features)

    indicator = np.zeros((M, n_observed), dtype=int)
    for i, s in enumerate(selection_sets):
        for f in s:
            indicator[i, feat_to_idx[f]] = 1

    k_bar = indicator.sum(axis=1).mean()
    denominator = (k_bar / p) * (1 - k_bar / p)
    if denominator == 0:
        return 0.0

    # Average sample variance across all p features (unobserved features have variance 0)
    numerator = np.sum(indicator.var(axis=0, ddof=1)) / p

    return float(1 - numerator / denominator)


def run_l4_stability(
    X_final: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 10,
    top_k: int = 20,
) -> dict:
    """10 次 bootstrap 训练 XGBoost + SHAP，收集 top-k 特征排名，计算稳定性指数。

    Args:
        X_final: 完整特征矩阵
        y: 标签
        n_bootstrap: bootstrap 次数
        top_k: 每次取 top-k 特征

    Returns:
        {stability_index, per_round_rankings, pass}
    """
    import shap

    cfg = DEFAULT_CONFIG.xgboost
    top_k = min(top_k, X_final.shape[1])

    selection_sets = []
    per_round_rankings = []
    rng = np.random.RandomState(42)

    for i in range(n_bootstrap):
        X_boot, y_boot = resample(
            X_final, y, stratify=y, random_state=rng.randint(0, 2**31)
        )

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
        model.fit(X_boot, y_boot, verbose=False)

        explainer = shap.TreeExplainer(model)
        shap_sample_size = min(5000, len(X_boot))
        if shap_sample_size < len(X_boot):
            sample_idx = rng.choice(len(X_boot), shap_sample_size, replace=False)
            X_shap = X_boot.iloc[sample_idx] if isinstance(X_boot, pd.DataFrame) else X_boot[sample_idx]
        else:
            X_shap = X_boot
        shap_values = explainer.shap_values(X_shap)
        mean_abs = np.mean(np.abs(shap_values), axis=0)

        sorted_indices = np.argsort(-mean_abs)[:top_k]
        top_features = [X_final.columns[idx] for idx in sorted_indices]

        selection_sets.append(set(top_features))
        per_round_rankings.append(top_features)

    stability_index = compute_nogueira_stability(selection_sets, X_final.shape[1])

    return {
        "stability_index": stability_index,
        "per_round_rankings": per_round_rankings,
        "pass": stability_index > 0.8,
    }
