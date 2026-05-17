"""特征评估指标：IV、KS、冗余度、F1/Recall/Precision、Permutation Importance、SHAP。"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG


def compute_iv(feature: pd.Series, label: pd.Series, n_bins: int = 10) -> float:
    """计算 Information Value (IV)。风控行业标准特征筛选指标。"""
    feat = feature.fillna(0)
    if feat.nunique() <= 2:
        binned = feat
    else:
        try:
            binned = pd.qcut(feat, q=n_bins, duplicates="drop")
        except ValueError:
            binned = pd.cut(feat, bins=n_bins, duplicates="drop")

    df = pd.DataFrame({"bin": binned, "label": label})
    grouped = df.groupby("bin", observed=True)["label"]

    total_good = (label == 0).sum()
    total_bad = (label == 1).sum()

    if total_good == 0 or total_bad == 0:
        return 0.0

    iv = 0.0
    for _, group in grouped:
        good = (group == 0).sum()
        bad = (group == 1).sum()
        good_pct = max(good / total_good, 1e-10)
        bad_pct = max(bad / total_bad, 1e-10)
        woe = np.log(good_pct / bad_pct)
        iv += (good_pct - bad_pct) * woe

    return float(iv)


def compute_ks(feature: pd.Series, label: pd.Series) -> float:
    """计算 KS 统计量：正负样本累积分布的最大分离度。"""
    feat = feature.fillna(0).values
    lab = label.values.astype(int)

    pos = feat[lab == 1]
    neg = feat[lab == 0]

    if len(pos) == 0 or len(neg) == 0:
        return 0.0

    all_vals = np.sort(np.unique(feat))
    if len(all_vals) > 1000:
        all_vals = np.percentile(feat, np.linspace(0, 100, 1000))

    pos_cdf = np.searchsorted(np.sort(pos), all_vals, side="right") / len(pos)
    neg_cdf = np.searchsorted(np.sort(neg), all_vals, side="right") / len(neg)

    return float(np.max(np.abs(pos_cdf - neg_cdf)))


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


def compute_mutual_info(feature: pd.Series, label: pd.Series) -> float:
    """计算特征与标签的互信息。"""
    from sklearn.feature_selection import mutual_info_classif
    X = feature.values.reshape(-1, 1)
    mi = mutual_info_classif(X, label, random_state=42)[0]
    return float(mi)


def _get_xgb_params() -> dict:
    cfg = DEFAULT_CONFIG.xgboost
    return dict(
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


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """找到使 F1 最大化的分类阈值。"""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


def compute_cv_metrics(
    X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
) -> dict:
    """5 折 CV 计算 F1, Recall, Precision, PR-AUC。返回各指标的均值。"""
    cfg = DEFAULT_CONFIG.xgboost
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    if X_arr.ndim != 2 or X_arr.shape[1] == 0:
        pos_rate = float(y_arr.mean())
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0, "prauc": pos_rate}

    params = _get_xgb_params()
    f1_scores, recall_scores, precision_scores, prauc_scores = [], [], [], []

    for train_idx, val_idx in cv.split(X_arr, y_arr):
        model = XGBClassifier(**params)
        model.fit(X_arr[train_idx], y_arr[train_idx], verbose=False)

        y_prob_train = model.predict_proba(X_arr[train_idx])[:, 1]
        thr = _find_best_threshold(y_arr[train_idx], y_prob_train)

        y_prob = model.predict_proba(X_arr[val_idx])[:, 1]
        y_val = y_arr[val_idx]
        y_pred = (y_prob >= thr).astype(int)

        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        prauc_scores.append(average_precision_score(y_val, y_prob))

    return {
        "f1": float(np.mean(f1_scores)),
        "recall": float(np.mean(recall_scores)),
        "precision": float(np.mean(precision_scores)),
        "prauc": float(np.mean(prauc_scores)),
    }


def compute_delta_metrics(
    X_existing: pd.DataFrame, new_feature: pd.Series, y: pd.Series
) -> dict:
    """增量评估：加入新特征后 F1/Recall/Precision 的变化。"""
    cfg = DEFAULT_CONFIG.xgboost
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    X_base = X_existing.values if isinstance(X_existing, pd.DataFrame) else np.asarray(X_existing)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    has_base = X_base.ndim == 2 and X_base.shape[1] > 0
    if has_base:
        X_aug = np.column_stack([X_base, new_feature.values])
    else:
        X_aug = new_feature.values.reshape(-1, 1)

    params = _get_xgb_params()

    base_f1, base_recall, base_prec = [], [], []
    aug_f1, aug_recall, aug_prec = [], [], []

    for train_idx, val_idx in cv.split(X_aug, y_arr):
        y_val = y_arr[val_idx]

        if has_base:
            m_base = XGBClassifier(**params)
            m_base.fit(X_base[train_idx], y_arr[train_idx], verbose=False)
            prob_base_train = m_base.predict_proba(X_base[train_idx])[:, 1]
            thr_base = _find_best_threshold(y_arr[train_idx], prob_base_train)
            prob_base = m_base.predict_proba(X_base[val_idx])[:, 1]
            pred_base = (prob_base >= thr_base).astype(int)
            base_f1.append(f1_score(y_val, pred_base, zero_division=0))
            base_recall.append(recall_score(y_val, pred_base, zero_division=0))
            base_prec.append(precision_score(y_val, pred_base, zero_division=0))
        else:
            base_f1.append(0.0)
            base_recall.append(0.0)
            base_prec.append(0.0)

        m_aug = XGBClassifier(**params)
        m_aug.fit(X_aug[train_idx], y_arr[train_idx], verbose=False)
        prob_aug_train = m_aug.predict_proba(X_aug[train_idx])[:, 1]
        thr_aug = _find_best_threshold(y_arr[train_idx], prob_aug_train)
        prob_aug = m_aug.predict_proba(X_aug[val_idx])[:, 1]
        pred_aug = (prob_aug >= thr_aug).astype(int)
        aug_f1.append(f1_score(y_val, pred_aug, zero_division=0))
        aug_recall.append(recall_score(y_val, pred_aug, zero_division=0))
        aug_prec.append(precision_score(y_val, pred_aug, zero_division=0))

    return {
        "base_f1": float(np.mean(base_f1)),
        "base_recall": float(np.mean(base_recall)),
        "base_precision": float(np.mean(base_prec)),
        "aug_f1": float(np.mean(aug_f1)),
        "aug_recall": float(np.mean(aug_recall)),
        "aug_precision": float(np.mean(aug_prec)),
        "delta_f1": float(np.mean(aug_f1) - np.mean(base_f1)),
        "delta_recall": float(np.mean(aug_recall) - np.mean(base_recall)),
        "delta_precision": float(np.mean(aug_prec) - np.mean(base_prec)),
    }


def compute_cv_prauc(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """5 折 CV 计算 PR-AUC（向后兼容）。"""
    return compute_cv_metrics(X, y)["prauc"]


def compute_shap_rank(
    X_all: pd.DataFrame, y: pd.Series, feature_name: str,
    max_samples: int = 50000,
) -> int:
    """训练 XGBoost 并用 SHAP 计算指定特征的重要性排名（1-based，1=最重要）。"""
    import shap

    params = _get_xgb_params()
    model = XGBClassifier(**params)
    model.fit(X_all, y, verbose=False)

    explainer = shap.TreeExplainer(model)
    if len(X_all) > max_samples:
        X_sample = X_all.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_all
    shap_values = explainer.shap_values(X_sample)
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    sorted_indices = np.argsort(-mean_abs)
    columns = list(X_all.columns)
    if feature_name not in columns:
        return len(columns)
    feat_idx = columns.index(feature_name)
    rank = int(np.where(sorted_indices == feat_idx)[0][0]) + 1
    return rank


def compute_permutation_importance(
    X_all: pd.DataFrame, y: pd.Series, feature_name: str, n_repeats: int = 5
) -> float:
    """计算单个特征的 Permutation Importance（打乱后 F1 下降量）。"""
    params = _get_xgb_params()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    X_arr = X_all.values if isinstance(X_all, pd.DataFrame) else np.asarray(X_all)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    columns = list(X_all.columns)
    feat_idx = columns.index(feature_name)

    importances = []
    for train_idx, val_idx in cv.split(X_arr, y_arr):
        model = XGBClassifier(**params)
        model.fit(X_arr[train_idx], y_arr[train_idx], verbose=False)

        y_prob = model.predict_proba(X_arr[val_idx])[:, 1]
        thr = _find_best_threshold(y_arr[val_idx], y_prob)
        y_pred = (y_prob >= thr).astype(int)
        base_f1 = f1_score(y_arr[val_idx], y_pred, zero_division=0)

        rng = np.random.RandomState(42)
        perm_scores = []
        for _ in range(n_repeats):
            X_perm = X_arr[val_idx].copy()
            rng.shuffle(X_perm[:, feat_idx])
            y_prob_perm = model.predict_proba(X_perm)[:, 1]
            y_pred_perm = (y_prob_perm >= thr).astype(int)
            perm_f1 = f1_score(y_arr[val_idx], y_pred_perm, zero_division=0)
            perm_scores.append(base_f1 - perm_f1)

        importances.append(np.mean(perm_scores))

    return float(np.mean(importances))


def normalize_score(value: float, all_values: list[float]) -> float:
    """Min-max 归一化到 [0, 1]。"""
    min_v = min(all_values)
    max_v = max(all_values)
    if max_v == min_v:
        return 0.5
    return (value - min_v) / (max_v - min_v)
