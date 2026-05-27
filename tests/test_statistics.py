"""L2 统计验证 (verification/statistics.py) 的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from knowfeat.config import VerificationConfig
from knowfeat.verification.statistics import (
    NoiseThresholds,
    StatResult,
    calibrate_noise_thresholds,
    compute_iv,
    compute_ks,
    compute_missing_rate,
    compute_redundancy,
    validate_feature,
)


# ---- helpers ----

def _make_label(n: int = 1000, pos_rate: float = 0.1, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    labels = (rng.rand(n) < pos_rate).astype(int)
    return pd.Series(labels)


def _make_good_feature(label: pd.Series, noise: float = 0.3, seed: int = 42) -> pd.Series:
    """生成与 label 强相关的特征（高 IV/KS）。"""
    rng = np.random.RandomState(seed)
    feat = label.values.astype(float) + rng.randn(len(label)) * noise
    return pd.Series(feat)


def _make_random_feature(n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randn(n))


# ---- compute_iv ----

class TestComputeIV:
    def test_good_feature_has_high_iv(self):
        label = _make_label()
        feat = _make_good_feature(label)
        iv = compute_iv(feat, label)
        assert iv > 0.1, f"Good feature should have IV > 0.1, got {iv}"

    def test_random_feature_has_low_iv(self):
        label = _make_label()
        feat = _make_random_feature(len(label), seed=99)
        iv = compute_iv(feat, label)
        assert iv < 0.1, f"Random feature should have IV < 0.1, got {iv}"

    def test_constant_feature_returns_zero(self):
        label = _make_label()
        feat = pd.Series(np.ones(len(label)))
        iv = compute_iv(feat, label)
        assert iv == 0.0

    def test_empty_input(self):
        iv = compute_iv(pd.Series(dtype=float), pd.Series(dtype=int))
        assert iv == 0.0

    def test_all_missing(self):
        label = _make_label(100)
        feat = pd.Series([np.nan] * 100)
        iv = compute_iv(feat, label)
        assert iv == 0.0

    def test_no_positive_labels(self):
        feat = pd.Series(np.arange(100, dtype=float))
        label = pd.Series(np.zeros(100, dtype=int))
        iv = compute_iv(feat, label)
        assert iv == 0.0


# ---- compute_ks ----

class TestComputeKS:
    def test_good_feature_has_high_ks(self):
        label = _make_label()
        feat = _make_good_feature(label)
        ks = compute_ks(feat, label)
        assert ks > 0.3, f"Good feature should have KS > 0.3, got {ks}"

    def test_random_feature_has_low_ks(self):
        label = _make_label()
        feat = _make_random_feature(len(label), seed=99)
        ks = compute_ks(feat, label)
        assert ks < 0.15, f"Random feature should have KS < 0.15, got {ks}"

    def test_identical_distributions(self):
        label = _make_label()
        feat = pd.Series(np.ones(len(label)))
        ks = compute_ks(feat, label)
        assert ks == 0.0

    def test_empty_input(self):
        ks = compute_ks(pd.Series(dtype=float), pd.Series(dtype=int))
        assert ks == 0.0

    def test_ks_between_0_and_1(self):
        label = _make_label()
        feat = _make_good_feature(label)
        ks = compute_ks(feat, label)
        assert 0.0 <= ks <= 1.0


# ---- compute_missing_rate ----

class TestComputeMissingRate:
    def test_no_missing(self):
        feat = pd.Series([1.0, 2.0, 3.0])
        assert compute_missing_rate(feat) == 0.0

    def test_all_missing(self):
        feat = pd.Series([np.nan, np.nan, np.nan])
        assert compute_missing_rate(feat) == 1.0

    def test_partial_missing(self):
        feat = pd.Series([1.0, np.nan, 3.0, np.nan])
        assert compute_missing_rate(feat) == 0.5

    def test_empty_series(self):
        feat = pd.Series(dtype=float)
        assert compute_missing_rate(feat) == 1.0


# ---- compute_redundancy ----

class TestComputeRedundancy:
    def test_no_existing_features(self):
        feat = pd.Series(np.arange(100, dtype=float))
        assert compute_redundancy(feat, pd.DataFrame()) == 0.0
        assert compute_redundancy(feat, None) == 0.0

    def test_identical_feature_high_redundancy(self):
        feat = pd.Series(np.arange(100, dtype=float))
        existing = pd.DataFrame({"a": np.arange(100, dtype=float)})
        r = compute_redundancy(feat, existing)
        assert r > 0.99

    def test_uncorrelated_features_low_redundancy(self):
        rng = np.random.RandomState(42)
        feat = pd.Series(rng.randn(500))
        existing = pd.DataFrame({"a": rng.randn(500), "b": rng.randn(500)})
        r = compute_redundancy(feat, existing)
        assert r < 0.2

    def test_with_nan_values(self):
        rng = np.random.RandomState(42)
        n = 200
        feat = pd.Series(rng.randn(n))
        feat.iloc[:10] = np.nan
        existing = pd.DataFrame({"a": rng.randn(n)})
        r = compute_redundancy(feat, existing)
        assert 0.0 <= r <= 1.0


# ---- calibrate_noise_thresholds ----

class TestCalibrateNoiseThresholds:
    def test_returns_positive_thresholds(self):
        label = _make_label(2000, pos_rate=0.1)
        thresholds = calibrate_noise_thresholds(
            label, n_samples=2000, n_random=50, seed=42,
        )
        assert thresholds.iv_threshold > 0
        assert thresholds.ks_threshold > 0

    def test_threshold_respects_fallback(self):
        label = _make_label(2000)
        thresholds = calibrate_noise_thresholds(
            label, n_samples=2000, n_random=50,
            iv_fallback=0.05, ks_fallback=0.05, seed=42,
        )
        assert thresholds.iv_threshold >= 0.05
        assert thresholds.ks_threshold >= 0.05

    def test_distribution_length(self):
        label = _make_label(1000)
        n_random = 30
        thresholds = calibrate_noise_thresholds(
            label, n_samples=1000, n_random=n_random, seed=42,
        )
        assert len(thresholds.iv_distribution) == n_random
        assert len(thresholds.ks_distribution) == n_random

    def test_deterministic_with_seed(self):
        label = _make_label(1000)
        t1 = calibrate_noise_thresholds(label, 1000, n_random=20, seed=123)
        t2 = calibrate_noise_thresholds(label, 1000, n_random=20, seed=123)
        assert t1.iv_threshold == t2.iv_threshold
        assert t1.ks_threshold == t2.ks_threshold


# ---- validate_feature ----

class TestValidateFeature:
    def test_good_feature_passes(self):
        label = _make_label(2000, pos_rate=0.1)
        feat = _make_good_feature(label, noise=0.3)
        vcfg = VerificationConfig(
            iv_min_fallback=0.01,
            ks_min_fallback=0.01,
        )
        result = validate_feature(feat, label, vcfg=vcfg)
        assert result.passed, f"Good feature should pass, reasons: {result.reject_reasons}"
        assert result.iv > 0
        assert result.ks > 0
        assert result.missing_rate == 0.0
        assert result.redundancy == 0.0
        assert len(result.reject_reasons) == 0

    def test_random_feature_fails(self):
        label = _make_label(2000, pos_rate=0.1)
        thresholds = calibrate_noise_thresholds(
            label, n_samples=2000, n_random=50, seed=42,
        )
        feat = _make_random_feature(2000, seed=999)
        result = validate_feature(feat, label, thresholds=thresholds)
        assert not result.passed

    def test_high_missing_rate_fails(self):
        label = _make_label(1000)
        feat = pd.Series([np.nan] * 400 + list(range(600)), dtype=float)
        vcfg = VerificationConfig(
            iv_min_fallback=0.0,
            ks_min_fallback=0.0,
            missing_rate_max=0.3,
        )
        result = validate_feature(feat, label, vcfg=vcfg)
        assert not result.passed
        assert any("缺失率" in r for r in result.reject_reasons)

    def test_high_redundancy_fails(self):
        label = _make_label(1000)
        feat = _make_good_feature(label)
        existing = pd.DataFrame({"same": feat.values})
        vcfg = VerificationConfig(
            iv_min_fallback=0.0,
            ks_min_fallback=0.0,
            redundancy_max=0.7,
        )
        result = validate_feature(feat, label, existing_features=existing, vcfg=vcfg)
        assert not result.passed
        assert any("冗余度" in r for r in result.reject_reasons)

    def test_with_noise_thresholds(self):
        label = _make_label(2000)
        thresholds = calibrate_noise_thresholds(label, 2000, n_random=50, seed=42)
        feat = _make_good_feature(label, noise=0.3)
        result = validate_feature(feat, label, thresholds=thresholds)
        assert isinstance(result, StatResult)
        assert result.iv_threshold == thresholds.iv_threshold
        assert result.ks_threshold == thresholds.ks_threshold

    def test_uses_config_defaults(self):
        label = _make_label(1000)
        feat = _make_good_feature(label)
        result = validate_feature(feat, label)
        assert isinstance(result, StatResult)
