"""多模型评估：验证 Agent 生成特征的模型无关性。

在 XGBoost / LightGBM / RandomForest / LogisticRegression 上分别评估
基线特征 vs 基线+Agent特征，证明增量来自特征本身而非特定模型的拟合。
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
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


def _build_model(model_name: str, scale_pos_weight: float = 19.0):
    """构建指定名称的分类模型。"""
    cfg = DEFAULT_CONFIG.xgboost

    if model_name == "xgboost":
        return XGBClassifier(
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

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight={0: 1.0, 1: scale_pos_weight},
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight={0: 1.0, 1: scale_pos_weight},
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            solver="saga",
        )

    raise ValueError(f"Unknown model: {model_name}")


def _cv_evaluate(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scale_pos_weight: float = 19.0,
    need_scale: bool = False,
) -> dict:
    """5-fold CV 评估单个模型，返回均值和标准差。"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1s, recalls, precisions, praucs = [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if need_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model = _build_model(model_name, scale_pos_weight)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_prob_train = model.predict_proba(X_train)[:, 1]
        thr = _find_best_threshold(y_train, y_prob_train)
        y_pred = (y_prob >= thr).astype(int)

        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        praucs.append(average_precision_score(y_val, y_prob))

    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "prauc_mean": float(np.mean(praucs)),
        "prauc_std": float(np.std(praucs)),
    }


MODELS = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]
NEEDS_SCALE = {"logistic_regression"}


def evaluate_multi_model(
    X_baseline: pd.DataFrame,
    X_agent_features: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """在多个模型上评估基线 vs 基线+Agent特征。

    Args:
        X_baseline: 基线特征矩阵
        X_agent_features: Agent 生成的特征矩阵（只包含新特征列）
        y: 标签

    Returns:
        {model_name: {baseline: {...}, augmented: {...}, delta: {...}}}
    """
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    X_base = X_baseline.values if isinstance(X_baseline, pd.DataFrame) else np.asarray(X_baseline)

    X_aug = np.column_stack([X_base, X_agent_features.values])

    pos_rate = y_arr.mean()
    spw = (1 - pos_rate) / pos_rate

    results = {}
    for model_name in MODELS:
        logger.info("Evaluating %s ...", model_name)
        need_scale = model_name in NEEDS_SCALE

        try:
            base_metrics = _cv_evaluate(model_name, X_base, y_arr,
                                        scale_pos_weight=spw, need_scale=need_scale)
            aug_metrics = _cv_evaluate(model_name, X_aug, y_arr,
                                       scale_pos_weight=spw, need_scale=need_scale)

            delta = {
                "delta_f1": aug_metrics["f1_mean"] - base_metrics["f1_mean"],
                "delta_recall": aug_metrics["recall_mean"] - base_metrics["recall_mean"],
                "delta_precision": aug_metrics["precision_mean"] - base_metrics["precision_mean"],
                "delta_prauc": aug_metrics["prauc_mean"] - base_metrics["prauc_mean"],
            }

            results[model_name] = {
                "baseline": base_metrics,
                "augmented": aug_metrics,
                "delta": delta,
            }
            logger.info("  %s: ΔRecall=%.4f, ΔF1=%.4f",
                        model_name, delta["delta_recall"], delta["delta_f1"])
        except Exception as e:
            logger.error("  %s failed: %s", model_name, e)
            results[model_name] = {"error": str(e)}

    return results
