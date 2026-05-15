"""特征评估指标：MI、冗余度、Cramer's V、Bootstrap χ²、ΔPR-AUC、SHAP 排名、归一化。"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG


def compute_mutual_info(feature: pd.Series, label: pd.Series) -> float:
    """计算特征与标签的互信息。"""
    X = feature.values.reshape(-1, 1)
    mi = mutual_info_classif(X, label, random_state=42)[0]
    return float(mi)


def compute_redundancy(feature: pd.Series, existing_features: pd.DataFrame) -> float:
    """计算特征与已有特征的最大 |corr|。无已有特征时返回 0。"""
    if existing_features is None or existing_features.empty:
        return 0.0
    max_corr = 0.0
    for col in existing_features.columns:
        corr = abs(feature.corr(existing_features[col]))
        if np.isnan(corr):
            continue
        max_corr = max(max_corr, corr)
    return float(max_corr)


def compute_cramers_v(feature: pd.Series, label: pd.Series) -> float:
    """计算 Cramer's V（连续特征先分箱）。"""
    if feature.nunique() > 10:
        binned = pd.qcut(feature, q=10, duplicates="drop")
    else:
        binned = feature
    contingency = pd.crosstab(binned, label)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(contingency)
    n = len(feature)
    k = min(contingency.shape) - 1
    if k == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def compute_bootstrap_chi2(
    feature: pd.Series, label: pd.Series, n_resamples: int = 2500
) -> float:
    """Bootstrap 卡方检验，返回平均 p 值。"""
    if feature.nunique() > 10:
        binned = pd.qcut(feature, q=10, duplicates="drop")
    else:
        binned = feature

    bin_codes = binned.astype("category").cat.codes.values
    label_arr = label.values.astype(int)
    n_bins = int(bin_codes.max()) + 1
    n_labels = int(label_arr.max()) + 1
    n = len(feature)

    rng = np.random.RandomState(42)
    p_values = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        b = bin_codes[idx]
        l = label_arr[idx]
        ct = np.zeros((n_bins, n_labels), dtype=np.int64)
        np.add.at(ct, (b, l), 1)
        row_mask = ct.sum(axis=1) > 0
        col_mask = ct.sum(axis=0) > 0
        ct_trimmed = ct[np.ix_(row_mask, col_mask)]
        if ct_trimmed.shape[0] < 2 or ct_trimmed.shape[1] < 2:
            p_values.append(1.0)
            continue
        _, p, _, _ = chi2_contingency(ct_trimmed)
        p_values.append(p)
    return float(np.mean(p_values))


def compute_cv_prauc(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """5 折 CV 计算 PR-AUC，用于缓刑池批量评估。"""
    cfg = DEFAULT_CONFIG.xgboost
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    if X_arr.ndim != 2 or X_arr.shape[1] == 0:
        return float(y_arr.mean())

    params = dict(
        scale_pos_weight=cfg.scale_pos_weight,
        eval_metric=cfg.eval_metric,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        tree_method=cfg.tree_method,
        device=cfg.device,
        tree_method=cfg.tree_method,
        device=cfg.device,
    )

    scores = []
    for train_idx, val_idx in cv.split(X_arr, y_arr):
        model = XGBClassifier(**params)
        model.fit(X_arr[train_idx], y_arr[train_idx], verbose=False)
        pred = model.predict_proba(X_arr[val_idx])[:, 1]
        scores.append(average_precision_score(y_arr[val_idx], pred))

    return float(np.mean(scores))


def compute_delta_prauc(
    X_existing: pd.DataFrame, new_feature: pd.Series, y: pd.Series
) -> float:
    """XGBoost 5 折 CV 增量 PR-AUC（含新特征 vs 不含）。"""
    cfg = DEFAULT_CONFIG.xgboost
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    X_base = X_existing.values if isinstance(X_existing, pd.DataFrame) else np.asarray(X_existing)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    has_base = X_base.ndim == 2 and X_base.shape[1] > 0
    if has_base:
        X_aug = np.column_stack([X_base, new_feature.values])
    else:
        X_aug = new_feature.values.reshape(-1, 1)

    base_scores = []
    aug_scores = []

    params = dict(
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

    for train_idx, val_idx in cv.split(X_aug, y_arr):
        if has_base:
            model_base = XGBClassifier(**params)
            model_base.fit(X_base[train_idx], y_arr[train_idx], verbose=False)
            pred_base = model_base.predict_proba(X_base[val_idx])[:, 1]
            base_scores.append(average_precision_score(y_arr[val_idx], pred_base))
        else:
            positive_rate = y_arr.mean()
            base_scores.append(float(positive_rate))

        model_aug = XGBClassifier(**params)
        model_aug.fit(X_aug[train_idx], y_arr[train_idx], verbose=False)
        pred_aug = model_aug.predict_proba(X_aug[val_idx])[:, 1]
        aug_scores.append(average_precision_score(y_arr[val_idx], pred_aug))

    return float(np.mean(aug_scores) - np.mean(base_scores))


def compute_shap_rank(
    X_all: pd.DataFrame, y: pd.Series, feature_name: str
) -> int:
    """训练 XGBoost 并用 SHAP 计算指定特征的重要性排名（1-based，1=最重要）。"""
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
        tree_method=cfg.tree_method,
        device=cfg.device,
    )
    model.fit(X_all, y, verbose=False)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    sorted_indices = np.argsort(-mean_abs)
    columns = list(X_all.columns)
    if feature_name not in columns:
        return len(columns)
    feat_idx = columns.index(feature_name)
    rank = int(np.where(sorted_indices == feat_idx)[0][0]) + 1
    return rank


def normalize_score(value: float, all_values: list[float]) -> float:
    """Min-max 归一化到 [0, 1]。"""
    min_v = min(all_values)
    max_v = max(all_values)
    if max_v == min_v:
        return 0.5
    return (value - min_v) / (max_v - min_v)
