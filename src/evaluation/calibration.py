"""L2 阈值自动校准：噪声基线法 —— 用随机特征的指标分布设定验证门槛。"""

import logging

import numpy as np
import pandas as pd

from src.config import ValidationThresholds
from src.evaluation.metrics import compute_iv, compute_ks

logger = logging.getLogger(__name__)


def calibrate_l2_thresholds(
    baseline_features: pd.DataFrame,
    label: pd.Series,
    n_random: int = 100,
    noise_percentile: float = 95.0,
) -> ValidationThresholds:
    """噪声基线校准：生成随机特征，计算其 IV/KS 分布，取 p95 作为阈值。

    任何候选特征的 IV/KS 超过随机噪声的 95 分位，即认为有真实信号。

    Args:
        baseline_features: 基线特征 DataFrame（用于参考，非校准依据）
        label: 标签 Series
        n_random: 随机特征数量（默认 100）
        noise_percentile: 噪声分位数（默认 95，即 p < 0.05 假阳性率）

    Returns:
        校准后的 ValidationThresholds
    """
    if label is None or len(label) == 0:
        logger.warning("No label for calibration, using defaults")
        return ValidationThresholds()

    rng = np.random.RandomState(42)
    n = len(label)
    noise_ivs = []
    noise_ks_vals = []

    for i in range(n_random):
        noise = pd.Series(rng.randn(n), name=f"noise_{i}")
        try:
            noise_ivs.append(compute_iv(noise, label))
        except Exception:
            noise_ivs.append(0.0)
        try:
            noise_ks_vals.append(compute_ks(noise, label))
        except Exception:
            noise_ks_vals.append(0.0)

    iv_arr = np.array(noise_ivs)
    ks_arr = np.array(noise_ks_vals)

    calibrated_iv = max(float(np.percentile(iv_arr, noise_percentile)), 0.001)
    calibrated_ks = max(float(np.percentile(ks_arr, noise_percentile)), 0.001)

    thresholds = ValidationThresholds(
        iv_min=calibrated_iv,
        ks_min=calibrated_ks,
        corr_max=0.7,
        missing_rate_max=0.5,
    )

    logger.info(
        "L2 thresholds calibrated from %d random noise features:\n"
        "  Noise IV distribution: min=%.6f, median=%.6f, p95=%.6f, max=%.6f\n"
        "  Noise KS distribution: min=%.6f, median=%.6f, p95=%.6f, max=%.6f\n"
        "  -> IV threshold: %.6f\n"
        "  -> KS threshold: %.6f",
        n_random,
        iv_arr.min(), np.median(iv_arr), np.percentile(iv_arr, 95), iv_arr.max(),
        ks_arr.min(), np.median(ks_arr), np.percentile(ks_arr, 95), ks_arr.max(),
        calibrated_iv, calibrated_ks,
    )

    if not baseline_features.empty:
        baseline_ivs = []
        baseline_ks_vals = []
        for col in baseline_features.columns:
            feat = baseline_features[col].fillna(0)
            if feat.nunique() <= 1:
                continue
            try:
                baseline_ivs.append(compute_iv(feat, label))
            except Exception:
                pass
            try:
                baseline_ks_vals.append(compute_ks(feat, label))
            except Exception:
                pass

        if baseline_ivs:
            b_iv = np.array(baseline_ivs)
            logger.info(
                "Baseline IV distribution (reference): min=%.6f, p25=%.6f, median=%.6f, p75=%.6f, max=%.6f",
                b_iv.min(), np.percentile(b_iv, 25), np.median(b_iv),
                np.percentile(b_iv, 75), b_iv.max(),
            )
        if baseline_ks_vals:
            b_ks = np.array(baseline_ks_vals)
            logger.info(
                "Baseline KS distribution (reference): min=%.6f, p25=%.6f, median=%.6f, p75=%.6f, max=%.6f",
                b_ks.min(), np.percentile(b_ks, 25), np.median(b_ks),
                np.percentile(b_ks, 75), b_ks.max(),
            )

    return thresholds
