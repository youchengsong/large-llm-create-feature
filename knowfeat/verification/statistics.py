"""L2 统计验证：IV/KS 计算 + 噪声校准阈值 + 冗余度检测 + 缺失率。"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from knowfeat.config import KnowFeatConfig, VerificationConfig

logger = logging.getLogger(__name__)


@dataclass
class StatResult:
    """L2 统计验证结果。"""
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
    """噪声校准动态阈值。"""
    iv_threshold: float
    ks_threshold: float
    iv_distribution: np.ndarray
    ks_distribution: np.ndarray


def compute_iv(feature: pd.Series, label: pd.Series, n_bins: int = 10) -> float:
    """计算 Information Value（WoE 分箱法）。

    Parameters
    ----------
    feature : pd.Series
        特征值（数值型）。
    label : pd.Series
        二分类标签（0/1）。
    n_bins : int
        分箱数，默认 10。

    Returns
    -------
    float
        IV 值。特征全为常数或分箱失败时返回 0.0。
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
        # qcut 分箱失败（稀疏特征），改用零/非零 + 非零部分再分箱
        zero_mask = feat == 0
        if zero_mask.all() or (~zero_mask).all():
            return 0.0
        nonzero_feat = feat[~zero_mask]
        try:
            nz_bins = pd.qcut(nonzero_feat, q=min(n_bins - 1, len(np.unique(nonzero_feat))),
                              duplicates="drop")
            nz_codes = nz_bins.codes + 1  # 非零部分的bin从1开始
        except (ValueError, TypeError):
            nz_codes = np.ones(len(nonzero_feat), dtype=int)
        bin_codes = np.zeros(len(feat), dtype=int)  # 0 = 零值bin
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
    """计算 KS 统计量（正负样本累积分布最大差异）。

    Parameters
    ----------
    feature : pd.Series
        特征值（数值型）。
    label : pd.Series
        二分类标签（0/1）。

    Returns
    -------
    float
        KS 值。
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
    """计算缺失率。"""
    if len(feature) == 0:
        return 1.0
    return float(feature.isna().sum() / len(feature))


def compute_redundancy(
    feature: pd.Series,
    existing_features: pd.DataFrame,
) -> float:
    """计算与已有特征的最大绝对相关系数。

    Parameters
    ----------
    feature : pd.Series
        待评估的新特征。
    existing_features : pd.DataFrame
        已接受的特征矩阵（每列一个特征）。

    Returns
    -------
    float
        最大绝对 Pearson 相关系数。无已有特征时返回 0.0。
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
    iv_fallback: float = 0.02,
    ks_fallback: float = 0.02,
    seed: int = 42,
) -> NoiseThresholds:
    """生成随机噪声特征，取 IV/KS 的高分位数作为动态阈值。

    Parameters
    ----------
    label : pd.Series
        二分类标签（0/1）。
    n_samples : int
        样本数（与特征长度一致）。
    n_random : int
        随机特征数量，默认 100。
    percentile : float
        取分位数，默认 95。
    iv_fallback : float
        IV 兜底最小阈值。
    ks_fallback : float
        KS 兜底最小阈值。
    seed : int
        随机种子。

    Returns
    -------
    NoiseThresholds
        包含 iv_threshold, ks_threshold 及完整分布。
    """
    rng = np.random.RandomState(seed)
    iv_list = []
    ks_list = []

    for _ in range(n_random):
        noise = pd.Series(rng.randn(n_samples))
        iv_list.append(compute_iv(noise, label))
        ks_list.append(compute_ks(noise, label))

    iv_arr = np.array(iv_list)
    ks_arr = np.array(ks_list)

    iv_thresh = max(float(np.percentile(iv_arr, percentile)), iv_fallback)
    ks_thresh = max(float(np.percentile(ks_arr, percentile)), ks_fallback)

    logger.info(
        "噪声校准完成: IV p%.0f=%.4f (threshold=%.4f), KS p%.0f=%.4f (threshold=%.4f)",
        percentile, float(np.percentile(iv_arr, percentile)), iv_thresh,
        percentile, float(np.percentile(ks_arr, percentile)), ks_thresh,
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
    """L2 统计验证：IV/KS/缺失率/冗余度综合判定。

    Parameters
    ----------
    feature : pd.Series
        待验证的特征列。
    label : pd.Series
        二分类标签（0/1），与 feature 等长。
    existing_features : pd.DataFrame, optional
        已接受的特征矩阵，用于冗余度计算。
    thresholds : NoiseThresholds, optional
        噪声校准阈值。未提供则使用配置中的 fallback 值。
    cfg : KnowFeatConfig, optional
        顶层配置。
    vcfg : VerificationConfig, optional
        验证配置。优先于 cfg.verification。

    Returns
    -------
    StatResult
        passed=True 表示通过 L2 验证。
    """
    if vcfg is None:
        vcfg = cfg.verification if cfg is not None else VerificationConfig()

    iv_thresh = thresholds.iv_threshold if thresholds else vcfg.iv_min_fallback
    ks_thresh = thresholds.ks_threshold if thresholds else vcfg.ks_min_fallback

    iv = compute_iv(feature, label)
    ks = compute_ks(feature, label)
    missing_rate = compute_missing_rate(feature)
    redundancy = compute_redundancy(feature, existing_features)

    reasons = []
    if iv <= iv_thresh:
        reasons.append(f"IV={iv:.4f} <= 阈值{iv_thresh:.4f}")
    if ks <= ks_thresh:
        reasons.append(f"KS={ks:.4f} <= 阈值{ks_thresh:.4f}")
    if missing_rate > vcfg.missing_rate_max:
        reasons.append(f"缺失率={missing_rate:.2%} > {vcfg.missing_rate_max:.0%}")
    if redundancy > vcfg.redundancy_max:
        reasons.append(f"冗余度={redundancy:.4f} > {vcfg.redundancy_max}")

    passed = len(reasons) == 0

    logger.info(
        "L2 统计验证 %s: IV=%.4f KS=%.4f 缺失率=%.2f%% 冗余度=%.4f%s",
        "通过" if passed else "未通过",
        iv, ks, missing_rate * 100, redundancy,
        "" if passed else f" | 原因: {'; '.join(reasons)}",
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
