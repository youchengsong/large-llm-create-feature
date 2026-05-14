"""L2 阈值自动校准：用基线特征的指标分布来设定合理的验证门槛。"""

import logging

import numpy as np
import pandas as pd

from src.config import ValidationThresholds
from src.evaluation.metrics import compute_mutual_info, compute_cramers_v, compute_bootstrap_chi2

logger = logging.getLogger(__name__)


def calibrate_l2_thresholds(
    baseline_features: pd.DataFrame,
    label: pd.Series,
    percentile: float = 10.0,
) -> ValidationThresholds:
    """根据基线特征的 MI / Cramer's V / bootstrap p 分布自动校准 L2 阈值。

    对每个基线特征计算三项指标，然后取 percentile 分位数作为阈值下限。
    这样新特征只需要达到"不比最弱的基线特征差太多"即可通过 L2。

    Args:
        baseline_features: 基线特征 DataFrame（不含标签列）
        label: 标签 Series
        percentile: 用于设定阈值的分位数（默认 10，即 p10）

    Returns:
        校准后的 ValidationThresholds
    """
    if baseline_features.empty or baseline_features.shape[1] == 0:
        logger.warning("No baseline features for calibration, using defaults")
        return ValidationThresholds()

    mi_values = []
    cramers_v_values = []
    bootstrap_p_values = []

    for col in baseline_features.columns:
        feature = baseline_features[col].fillna(0)
        if feature.nunique() <= 1:
            continue

        try:
            mi = compute_mutual_info(feature, label)
            mi_values.append(mi)
        except Exception as e:
            logger.debug("MI computation failed for %s: %s", col, e)

        try:
            cv = compute_cramers_v(feature, label)
            cramers_v_values.append(cv)
        except Exception as e:
            logger.debug("Cramer's V computation failed for %s: %s", col, e)

        try:
            bp = compute_bootstrap_chi2(feature, label, n_resamples=500)
            bootstrap_p_values.append(bp)
        except Exception as e:
            logger.debug("Bootstrap chi2 failed for %s: %s", col, e)

    if not mi_values or not cramers_v_values:
        logger.warning("Calibration failed (no valid metrics), using defaults")
        return ValidationThresholds()

    mi_arr = np.array(mi_values)
    cv_arr = np.array(cramers_v_values)

    calibrated_mi = float(np.percentile(mi_arr, percentile))
    calibrated_cv = float(np.percentile(cv_arr, percentile))

    calibrated_mi = max(calibrated_mi * 0.5, 1e-6)
    calibrated_cv = max(calibrated_cv * 0.5, 1e-4)

    thresholds = ValidationThresholds(
        mi_min=calibrated_mi,
        corr_max=0.9,
        cramers_v_min=calibrated_cv,
        delta_prauc_min=0.005,
        bootstrap_p_max=0.05,
    )

    logger.info(
        "L2 thresholds calibrated from %d baseline features:\n"
        "  MI: baseline p%d=%.6f -> threshold=%.6f (was 0.001)\n"
        "  Cramer's V: baseline p%d=%.6f -> threshold=%.6f (was 0.05)\n"
        "  Redundancy: %.2f (unchanged)\n"
        "  Bootstrap p: %.2f (unchanged)",
        len(mi_values),
        percentile, float(np.percentile(mi_arr, percentile)), calibrated_mi,
        percentile, float(np.percentile(cv_arr, percentile)), calibrated_cv,
        thresholds.corr_max,
        thresholds.bootstrap_p_max,
    )

    logger.info(
        "Baseline MI distribution: min=%.6f, p25=%.6f, median=%.6f, p75=%.6f, max=%.6f",
        mi_arr.min(), np.percentile(mi_arr, 25), np.median(mi_arr),
        np.percentile(mi_arr, 75), mi_arr.max(),
    )
    logger.info(
        "Baseline Cramer's V distribution: min=%.6f, p25=%.6f, median=%.6f, p75=%.6f, max=%.6f",
        cv_arr.min(), np.percentile(cv_arr, 25), np.median(cv_arr),
        np.percentile(cv_arr, 75), cv_arr.max(),
    )

    return thresholds
