"""L2 statistical verification: IV/KS computation + noise calibration thresholds + redundancy detection + missing rate."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from knowfeat.config import KnowFeatConfig, VerificationConfig

logger = logging.getLogger(__name__)


@dataclass
class StatResult:
    """L2 statistical verification result."""
    passed: bool
    iv: float
    ks: float
    missing_rate: float
    redundancy: float
    iv_threshold: float
    ks_threshold: float
    reject_reasons: list[str]


@dataclass
class NoiseThresholds:
    """Noise calibration dynamic thresholds."""
    iv_threshold: float
    ks_threshold: float
    iv_distribution: np.ndarray
    ks_distribution: np.ndarray


def compute_iv(feature: pd.Series, label: pd.Series, n_bins: int = 10) -> float:
    """Compute Information Value (WoE binning method).

    Parameters
    ----------
    feature : pd.Series
        Feature values (numeric).
    label : pd.Series
        Binary label (0/1).
    n_bins : int
        Number of bins, default 10.

    Returns
    -------
    float
        IV value. Returns 0.0 if feature is constant or binning fails.
    """
    mask = feature.notna() & label.notna()
    feat = feature[mask].values.astype(float)
    lab = label[mask].values.astype(int)

    if len(feat) == 0 or lab.sum() == 0 or (1 - lab).sum() == 0:
        return 0.0

    try:
        bins = pd.qcut(feat, q=n_bins, duplicates="drop")
    except ValueError:
        bins = None

    if bins is not None:
        bin_codes = bins.codes
        unique_bins = np.unique(bin_codes[bin_codes >= 0])
    else:
        unique_bins = np.array([])

    if len(unique_bins) < 2:
        # qcut binning failed (sparse feature), fallback to zero/non-zero + re-bin non-zero portion
        zero_mask = feat == 0
        if zero_mask.all() or (~zero_mask).all():
            return 0.0
        nonzero_feat = feat[~zero_mask]
        try:
            nz_bins = pd.qcut(nonzero_feat, q=min(n_bins - 1, len(np.unique(nonzero_feat))),
                              duplicates="drop")
            nz_codes = nz_bins.codes + 1  # non-zero portion bins start from 1
        except (ValueError, TypeError):
            nz_codes = np.ones(len(nonzero_feat), dtype=int)
        bin_codes = np.zeros(len(feat), dtype=int)  # 0 = zero-value bin
        bin_codes[~zero_mask] = nz_codes
        unique_bins = np.unique(bin_codes[bin_codes >= 0])
        if len(unique_bins) < 2:
            return 0.0

    total_good = (lab == 0).sum()
    total_bad = (lab == 1).sum()

    iv = 0.0
    for b in unique_bins:
        in_bin = bin_codes == b
        good_i = ((lab == 0) & in_bin).sum()
        bad_i = ((lab == 1) & in_bin).sum()

        good_pct = max(good_i / total_good, 1e-10)
        bad_pct = max(bad_i / total_bad, 1e-10)

        woe = np.log(good_pct / bad_pct)
        iv += (good_pct - bad_pct) * woe

    return float(iv)


def compute_ks(feature: pd.Series, label: pd.Series) -> float:
    """Compute KS statistic (maximum difference between positive and negative cumulative distributions).

    Parameters
    ----------
    feature : pd.Series
        Feature values (numeric).
    label : pd.Series
        Binary label (0/1).

    Returns
    -------
    float
        KS value.
    """
    mask = feature.notna() & label.notna()
    feat = feature[mask].values.astype(float)
    lab = label[mask].values.astype(int)

    if len(feat) == 0 or lab.sum() == 0 or (1 - lab).sum() == 0:
        return 0.0

    if np.ptp(feat) == 0:
        return 0.0

    order = np.argsort(feat)
    lab_sorted = lab[order]

    cum_good = np.cumsum(lab_sorted == 0) / (lab_sorted == 0).sum()
    cum_bad = np.cumsum(lab_sorted == 1) / (lab_sorted == 1).sum()

    return float(np.max(np.abs(cum_good - cum_bad)))


def compute_missing_rate(feature: pd.Series) -> float:
    """Compute missing rate."""
    if len(feature) == 0:
        return 1.0
    return float(feature.isna().sum() / len(feature))


def compute_redundancy(
    feature: pd.Series,
    existing_features: pd.DataFrame,
) -> float:
    """Compute maximum absolute correlation coefficient with existing features.

    Parameters
    ----------
    feature : pd.Series
        New feature to evaluate.
    existing_features : pd.DataFrame
        Accepted feature matrix (one feature per column).

    Returns
    -------
    float
        Maximum absolute Pearson correlation coefficient. Returns 0.0 if no existing features.
    """
    if existing_features is None or existing_features.empty:
        return 0.0

    feat = feature.values.astype(float)
    mask = ~np.isnan(feat)
    if mask.sum() < 10:
        return 0.0

    max_corr = 0.0
    for col in existing_features.columns:
        other = existing_features[col].values.astype(float)
        valid = mask & ~np.isnan(other)
        if valid.sum() < 10:
            continue
        corr = np.corrcoef(feat[valid], other[valid])[0, 1]
        if np.isnan(corr):
            continue
        max_corr = max(max_corr, abs(corr))

    return float(max_corr)


def calibrate_noise_thresholds(
    label: pd.Series,
    n_samples: int,
    n_random: int = 100,
    percentile: float = 95.0,
    iv_floor: float = 0.005,
    ks_floor: float = 0.005,
    seed: int = 42,
    *,
    noise_multiplier: float = 3.0,
    iv_fallback: Optional[float] = None,
    ks_fallback: Optional[float] = None,
) -> NoiseThresholds:
    """Generate random noise features, take IV/KS high percentile x multiplier as dynamic thresholds.

    Threshold = max(noise_p95 x multiplier, floor), adaptive to different positive rates.

    ``iv_fallback`` and ``ks_fallback`` are deprecated aliases retained for
    callers using the pre-calibration API.  They take precedence over the
    corresponding floor when supplied.
    """
    if iv_fallback is not None:
        iv_floor = iv_fallback
    if ks_fallback is not None:
        ks_floor = ks_fallback

    rng = np.random.RandomState(seed)
    iv_list = []
    ks_list = []

    for _ in range(n_random):
        noise = pd.Series(rng.randn(n_samples))
        iv_list.append(compute_iv(noise, label))
        ks_list.append(compute_ks(noise, label))

    iv_arr = np.array(iv_list)
    ks_arr = np.array(ks_list)

    iv_p = float(np.percentile(iv_arr, percentile))
    ks_p = float(np.percentile(ks_arr, percentile))
    iv_thresh = max(iv_p * noise_multiplier, iv_floor)
    ks_thresh = max(ks_p, ks_floor)

    logger.info(
        "Noise calibration completed: IV p%.0f=%.4f x%.0f->threshold=%.4f, KS p%.0f=%.4f ->threshold=%.4f",
        percentile, iv_p, noise_multiplier, iv_thresh,
        percentile, ks_p, ks_thresh,
    )

    return NoiseThresholds(
        iv_threshold=iv_thresh,
        ks_threshold=ks_thresh,
        iv_distribution=iv_arr,
        ks_distribution=ks_arr,
    )


def validate_feature(
    feature: pd.Series,
    label: pd.Series,
    existing_features: Optional[pd.DataFrame] = None,
    thresholds: Optional[NoiseThresholds] = None,
    cfg: Optional[KnowFeatConfig] = None,
    vcfg: Optional[VerificationConfig] = None,
) -> StatResult:
    """L2 statistical verification: combined IV/KS/missing rate/redundancy assessment.

    Parameters
    ----------
    feature : pd.Series
        Feature column to verify.
    label : pd.Series
        Binary label (0/1), same length as feature.
    existing_features : pd.DataFrame, optional
        Accepted feature matrix, used for redundancy calculation.
    thresholds : NoiseThresholds, optional
        Noise calibration thresholds. If not provided, uses fallback values from config.
    cfg : KnowFeatConfig, optional
        Top-level configuration.
    vcfg : VerificationConfig, optional
        Verification configuration. Takes priority over cfg.verification.

    Returns
    -------
    StatResult
        passed=True means L2 verification passed.
    """
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    iv_thresh = thresholds.iv_threshold if thresholds else vcfg.iv_min_floor
    ks_thresh = thresholds.ks_threshold if thresholds else vcfg.ks_min_floor

    iv = compute_iv(feature, label)
    ks = compute_ks(feature, label)
    missing_rate = compute_missing_rate(feature)
    redundancy = compute_redundancy(feature, existing_features)

    reasons = []
    if iv <= iv_thresh:
        reasons.append(f"IV={iv:.4f} <= threshold {iv_thresh:.4f}")
    if ks <= ks_thresh:
        reasons.append(f"KS={ks:.4f} <= threshold {ks_thresh:.4f}")
    if missing_rate > vcfg.missing_rate_max:
        reasons.append(
            f"missing_rate={missing_rate:.2%} > {vcfg.missing_rate_max:.0%} "
            f"(缺失率={missing_rate:.2%})"
        )
    if redundancy > vcfg.redundancy_max:
        reasons.append(
            f"redundancy={redundancy:.4f} > {vcfg.redundancy_max} "
            f"(冗余度={redundancy:.4f})"
        )

    passed = len(reasons) == 0

    logger.info(
        "L2 statistical verification %s: IV=%.4f KS=%.4f missing_rate=%.2f%% redundancy=%.4f%s",
        "passed" if passed else "failed",
        iv, ks, missing_rate * 100, redundancy,
        "" if passed else f" | reasons: {'; '.join(reasons)}",
    )

    return StatResult(
        passed=passed,
        iv=iv,
        ks=ks,
        missing_rate=missing_rate,
        redundancy=redundancy,
        iv_threshold=iv_thresh,
        ks_threshold=ks_thresh,
        reject_reasons=reasons,
    )
