"""5 组增量贡献实验：基线、增量、独立、随机对照、消融。"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG


def _cv_prauc(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5
) -> tuple[float, float]:
    """5 折分层 CV 计算 PR-AUC，返回 (mean, std)。"""
    cfg = DEFAULT_CONFIG.xgboost
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = XGBClassifier(
            scale_pos_weight=cfg.scale_pos_weight,
            eval_metric=cfg.eval_metric,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        model.fit(X[train_idx], y[train_idx], verbose=False)
        pred = model.predict_proba(X[val_idx])[:, 1]
        scores.append(average_precision_score(y[val_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def _compute_shap_ranks(X: pd.DataFrame, y: np.ndarray) -> dict[str, int]:
    """训练 XGBoost + SHAP，返回 {feature_name: rank(1-based)}。"""
    import shap

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
    model.fit(X, y, verbose=False)
    explainer = shap.TreeExplainer(model)
    shap_sample_size = min(5000, len(X))
    X_shap = X.sample(n=shap_sample_size, random_state=42) if shap_sample_size < len(X) else X
    shap_values = explainer.shap_values(X_shap)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(-mean_abs)
    ranks = {}
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[X.columns[idx]] = rank
    return ranks


def run_ablation_experiments(
    X_baseline: pd.DataFrame,
    X_llm: pd.DataFrame,
    y: pd.Series,
    baseline_feature_names: list[str],
    llm_feature_names: list[str],
) -> dict:
    """运行 5 组增量贡献实验。

    Args:
        X_baseline: 人工基线特征矩阵
        X_llm: LLM 生成特征矩阵
        y: 标签
        baseline_feature_names: 基线特征名列表
        llm_feature_names: LLM 特征名列表

    Returns:
        包含 5 组实验结果的字典
    """
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    has_baseline = len(baseline_feature_names) > 0
    X_base = X_baseline[baseline_feature_names].values if has_baseline else np.empty((len(y_arr), 0))
    X_llm_arr = X_llm[llm_feature_names].values

    results = {}

    # 实验 1：仅人工特征
    if has_baseline:
        mean1, std1 = _cv_prauc(X_base, y_arr)
    else:
        mean1, std1 = 0.0, 0.0
    results["exp1_baseline"] = {"prauc_mean": mean1, "prauc_std": std1}

    # 实验 2：人工 + LLM 特征
    if has_baseline:
        X_aug = np.column_stack([X_base, X_llm_arr])
    else:
        X_aug = X_llm_arr
    mean2, std2 = _cv_prauc(X_aug, y_arr)
    results["exp2_augmented"] = {"prauc_mean": mean2, "prauc_std": std2}

    # 实验 3：仅 LLM 特征
    mean3, std3 = _cv_prauc(X_llm_arr, y_arr)
    results["exp3_llm_only"] = {"prauc_mean": mean3, "prauc_std": std3}

    # 实验 4：人工 + 等量随机特征
    rng = np.random.RandomState(42)
    n_llm = X_llm_arr.shape[1]
    X_random = rng.randn(X_aug.shape[0], n_llm)
    if has_baseline:
        X_random_aug = np.column_stack([X_base, X_random])
    else:
        X_random_aug = X_random
    mean4, std4 = _cv_prauc(X_random_aug, y_arr)
    results["exp4_random_control"] = {"prauc_mean": mean4, "prauc_std": std4}

    # 实验 5：消融 — 逐个移除 LLM 特征
    ablation = {}
    for i, fname in enumerate(llm_feature_names):
        cols_keep = [j for j in range(n_llm) if j != i]
        if cols_keep:
            if has_baseline:
                X_ablated = np.column_stack([X_base, X_llm_arr[:, cols_keep]])
            else:
                X_ablated = X_llm_arr[:, cols_keep]
        else:
            if has_baseline:
                X_ablated = X_base
            else:
                ablation[fname] = {
                    "prauc_mean": 0.0,
                    "prauc_std": 0.0,
                    "contribution": mean2,
                }
                continue
        m, s = _cv_prauc(X_ablated, y_arr)
        ablation[fname] = {
            "prauc_mean": m,
            "prauc_std": s,
            "contribution": mean2 - m,
        }
    results["exp5_ablation"] = ablation

    # SHAP 排名（在完整模型中）
    all_feature_names = (baseline_feature_names + llm_feature_names) if has_baseline else llm_feature_names
    X_full_df = pd.DataFrame(X_aug, columns=all_feature_names)
    shap_ranks = _compute_shap_ranks(X_full_df, y_arr)
    results["shap_ranks"] = shap_ranks

    # 汇总指标
    results["summary"] = {
        "delta_prauc": mean2 - mean1,
        "llm_independent_prauc": mean3,
        "random_control_delta": mean4 - mean1,
        "most_critical_feature": max(
            ablation, key=lambda f: ablation[f]["contribution"]
        ) if ablation else None,
    }

    return results
