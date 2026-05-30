"""L3 模型效果验证：XGBoost 5-fold StratifiedKFold CV delta-recall。"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from knowfeat.config import KnowFeatConfig, VerificationConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelEvalResult:
    """L3 模型效果验证结果。"""
    passed: bool
    baseline_recall: float
    new_recall: float
    delta_recall: float
    baseline_f1: float
    new_f1: float
    delta_f1: float
    new_precision: float
    reject_reasons: list[str]
    cr_at_k: dict[str, float] | None = None


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """计算 recall, precision, f1。"""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(recall), float(precision), float(f1)


def capture_rate_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_pcts: list[float] = (0.001, 0.002, 0.003, 0.005, 0.01, 0.03, 0.05),
) -> dict[str, float]:
    """CR@K: Top K% 样本中捕获的正样本占总正样本的比例。"""
    n = len(y_true)
    total_pos = y_true.sum()
    if total_pos == 0:
        return {f"CR@{k*100:g}%": 0.0 for k in k_pcts}
    indices = np.argsort(y_prob)[::-1]
    results = {}
    for k in k_pcts:
        top_k = max(1, int(n * k))
        captured = y_true[indices[:top_k]].sum()
        results[f"CR@{k*100:g}%"] = float(captured / total_pos)
    return results


def _cv_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    compute_cr: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, dict[str, float]]:
    """XGBoost StratifiedKFold CV，返回 (mean_recall, mean_precision, mean_f1[, cr@k])。"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    recalls, precisions, f1s = [], [], []
    cr_accum: dict[str, list[float]] = {}

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        r, p, f = _compute_metrics(y_val, y_pred)
        recalls.append(r)
        precisions.append(p)
        f1s.append(f)

        if compute_cr:
            y_prob = clf.predict_proba(X_val)[:, 1]
            cr = capture_rate_at_k(y_val, y_prob)
            for k, v in cr.items():
                cr_accum.setdefault(k, []).append(v)

    base = (
        float(np.mean(recalls)),
        float(np.mean(precisions)),
        float(np.mean(f1s)),
    )
    if compute_cr:
        cr_mean = {k: float(np.mean(v)) for k, v in cr_accum.items()}
        return (*base, cr_mean)
    return base


def evaluate_feature(
    baseline_X: pd.DataFrame,
    new_feature: pd.Series,
    label: pd.Series,
    cfg: Optional[KnowFeatConfig] = None,
    vcfg: Optional[VerificationConfig] = None,
    seed: int = 42,
) -> ModelEvalResult:
    """L3 模型效果验证：比较加入新特征前后的 delta-recall。

    Parameters
    ----------
    baseline_X : pd.DataFrame
        基线特征矩阵（已接受的特征 + 原始基线特征）。
    new_feature : pd.Series
        待评估的新特征列。
    label : pd.Series
        二分类标签（0/1）。
    cfg : KnowFeatConfig, optional
        顶层配置。
    vcfg : VerificationConfig, optional
        验证配置。优先于 cfg.verification。
    seed : int
        随机种子。

    Returns
    -------
    ModelEvalResult
        passed=True 表示通过 L3 验证。
    """
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    y = label.values.astype(int)

    baseline_X_clean = baseline_X.copy()
    for col in baseline_X_clean.columns:
        baseline_X_clean[col] = pd.to_numeric(baseline_X_clean[col], errors="coerce")
    baseline_X_clean = baseline_X_clean.fillna(0)

    feat_name = new_feature.name if new_feature.name else "new_feature"
    new_col = pd.to_numeric(new_feature, errors="coerce").fillna(0)
    augmented_X = baseline_X_clean.copy()
    augmented_X[feat_name] = new_col.values

    if baseline_X_clean.shape[1] == 0:
        b_recall, b_precision, b_f1 = 0.0, 0.0, 0.0
    else:
        b_recall, b_precision, b_f1 = _cv_evaluate(
            baseline_X_clean, y, n_folds=vcfg.cv_folds, seed=seed,
        )
    n_recall, n_precision, n_f1 = _cv_evaluate(
        augmented_X, y, n_folds=vcfg.cv_folds, seed=seed,
    )

    delta_recall = n_recall - b_recall
    delta_f1 = n_f1 - b_f1

    reasons = []
    if delta_recall <= vcfg.delta_recall_min:
        reasons.append(
            f"delta_recall={delta_recall:+.4f} <= {vcfg.delta_recall_min}"
        )
    if n_precision < vcfg.precision_min:
        reasons.append(
            f"precision={n_precision:.4f} < {vcfg.precision_min}"
        )

    passed = len(reasons) == 0

    logger.info(
        "L3 模型验证 %s: baseline_recall=%.4f new_recall=%.4f "
        "delta_recall=%+.4f precision=%.4f delta_f1=%+.4f%s",
        "通过" if passed else "未通过",
        b_recall, n_recall, delta_recall, n_precision, delta_f1,
        "" if passed else f" | 原因: {'; '.join(reasons)}",
    )

    return ModelEvalResult(
        passed=passed,
        baseline_recall=b_recall,
        new_recall=n_recall,
        delta_recall=delta_recall,
        baseline_f1=b_f1,
        new_f1=n_f1,
        delta_f1=delta_f1,
        new_precision=n_precision,
        reject_reasons=reasons,
    )
