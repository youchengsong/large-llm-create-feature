"""L3 model performance verification: XGBoost 5-fold StratifiedKFold CV, supports delta-recall (AML) / delta-AUC (public datasets)."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from knowfeat.config import KnowFeatConfig, VerificationConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelEvalResult:
    """L3 model performance verification result."""
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
    baseline_auc: float = 0.0
    new_auc: float = 0.0
    delta_auc: float = 0.0
    accept_metric: str = "recall"


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    """Compute recall, precision, f1."""
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
    """CR@K: proportion of positive samples captured in top K% samples relative to total positives."""
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


@dataclass
class _CVMetrics:
    recall: float
    precision: float
    f1: float
    auc: float
    cr_at_k: dict[str, float] | None = None


EVAL_SEEDS = [42, 123, 456, 789, 2024]


def _cv_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    compute_cr: bool = False,
    spw_override: float = None,
) -> _CVMetrics:
    """80/20 train_test_split x 5 seeds, consistent with evaluate.py."""
    from sklearn.metrics import roc_auc_score

    recalls, precisions, f1s, aucs = [], [], [], []
    cr_accum: dict[str, list[float]] = {}

    for s in EVAL_SEEDS:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=s, stratify=y,
        )

        if spw_override is not None:
            spw = spw_override
        else:
            spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=s,
            verbosity=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        r, p, f = _compute_metrics(y_val, y_pred)
        recalls.append(r)
        precisions.append(p)
        f1s.append(f)
        aucs.append(float(roc_auc_score(y_val, y_prob)))

        if compute_cr:
            cr = capture_rate_at_k(y_val, y_prob)
            for k, v in cr.items():
                cr_accum.setdefault(k, []).append(v)

    return _CVMetrics(
        recall=float(np.mean(recalls)),
        precision=float(np.mean(precisions)),
        f1=float(np.mean(f1s)),
        auc=float(np.mean(aucs)),
        cr_at_k={k: float(np.mean(v)) for k, v in cr_accum.items()} if compute_cr else None,
    )


def evaluate_feature(
    baseline_X: pd.DataFrame,
    new_feature: pd.Series,
    label: pd.Series,
    cfg: Optional[KnowFeatConfig] = None,
    vcfg: Optional[VerificationConfig] = None,
    seed: int = 42,
    spw_override: float = None,
) -> ModelEvalResult:
    """L3 model performance verification: compare delta-recall before and after adding a new feature.

    Parameters
    ----------
    baseline_X : pd.DataFrame
        Baseline feature matrix (accepted features + original baseline features).
    new_feature : pd.Series
        New feature column to evaluate.
    label : pd.Series
        Binary label (0/1).
    cfg : KnowFeatConfig, optional
        Top-level configuration.
    vcfg : VerificationConfig, optional
        Verification configuration. Takes priority over cfg.verification.
    seed : int
        Random seed.

    Returns
    -------
    ModelEvalResult
        passed=True means L3 verification passed.
    """
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    y = label.values.astype(int)

    baseline_X_clean = baseline_X.copy()
    for col in baseline_X_clean.columns:
        baseline_X_clean[col] = pd.to_numeric(baseline_X_clean[col], errors="coerce")
    baseline_X_clean = baseline_X_clean.fillna(-999)

    feat_name = new_feature.name if new_feature.name else "new_feature"
    new_col = pd.to_numeric(new_feature, errors="coerce").fillna(-999)
    augmented_X = baseline_X_clean.copy()
    augmented_X[feat_name] = new_col.values

    if baseline_X_clean.shape[1] == 0:
        base_m = _CVMetrics(recall=0.0, precision=0.0, f1=0.0, auc=0.0)
    else:
        base_m = _cv_evaluate(
            baseline_X_clean, y, n_folds=vcfg.cv_folds, seed=seed,
            spw_override=spw_override,
        )
    new_m = _cv_evaluate(
        augmented_X, y, n_folds=vcfg.cv_folds, seed=seed,
        spw_override=spw_override,
    )

    delta_recall = new_m.recall - base_m.recall
    delta_f1 = new_m.f1 - base_m.f1
    delta_auc = new_m.auc - base_m.auc

    reasons = []
    metric = vcfg.accept_metric

    if metric == "auc":
        if delta_auc <= vcfg.delta_auc_min:
            reasons.append(
                f"delta_auc={delta_auc:+.4f} <= {vcfg.delta_auc_min}"
            )
    else:
        if delta_recall <= vcfg.delta_recall_min:
            reasons.append(
                f"delta_recall={delta_recall:+.4f} <= {vcfg.delta_recall_min}"
            )
        # Recall-based AML selection has an explicit precision floor.  AUC
        # mode intentionally does not apply this classification threshold.
        if new_m.precision < vcfg.precision_min:
            reasons.append(
                f"precision={new_m.precision:.4f} < {vcfg.precision_min}"
            )

    passed = len(reasons) == 0

    if metric == "auc":
        logger.info(
            "L3 model verification %s [metric=auc]: baseline_auc=%.4f new_auc=%.4f "
            "delta_auc=%+.4f delta_f1=%+.4f%s",
            "passed" if passed else "failed",
            base_m.auc, new_m.auc, delta_auc, delta_f1,
            "" if passed else f" | reasons: {'; '.join(reasons)}",
        )
    else:
        logger.info(
            "L3 model verification %s [metric=recall]: baseline_recall=%.4f new_recall=%.4f "
            "delta_recall=%+.4f precision=%.4f delta_f1=%+.4f%s",
            "passed" if passed else "failed",
            base_m.recall, new_m.recall, delta_recall, new_m.precision, delta_f1,
            "" if passed else f" | reasons: {'; '.join(reasons)}",
        )

    return ModelEvalResult(
        passed=passed,
        baseline_recall=base_m.recall,
        new_recall=new_m.recall,
        delta_recall=delta_recall,
        baseline_f1=base_m.f1,
        new_f1=new_m.f1,
        delta_f1=delta_f1,
        new_precision=new_m.precision,
        reject_reasons=reasons,
        baseline_auc=base_m.auc,
        new_auc=new_m.auc,
        delta_auc=delta_auc,
        accept_metric=metric,
    )
