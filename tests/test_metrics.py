"""tests/test_metrics.py — 测试 MI、冗余度、Cramer's V、Bootstrap χ²、ΔPR-AUC、SHAP 排名、归一化。"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_bootstrap_chi2,
    compute_cramers_v,
    compute_delta_prauc,
    compute_mutual_info,
    compute_redundancy,
    compute_shap_rank,
    normalize_score,
)


# ---------- 测试数据 ----------

@pytest.fixture
def binary_label():
    rng = np.random.RandomState(42)
    return pd.Series(rng.choice([0, 1], size=200, p=[0.8, 0.2]))


@pytest.fixture
def informative_feature(binary_label):
    rng = np.random.RandomState(42)
    return pd.Series(binary_label.values * 5.0 + rng.normal(0, 1, size=len(binary_label)))


@pytest.fixture
def noise_feature(binary_label):
    rng = np.random.RandomState(99)
    return pd.Series(rng.normal(0, 1, size=len(binary_label)))


@pytest.fixture
def existing_features():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "f1": rng.normal(0, 1, 200),
        "f2": rng.normal(0, 1, 200),
    })


# ========== compute_mutual_info ==========

class TestMutualInfo:
    def test_informative_positive(self, informative_feature, binary_label):
        mi = compute_mutual_info(informative_feature, binary_label)
        assert mi > 0.01

    def test_noise_low(self, noise_feature, binary_label):
        mi = compute_mutual_info(noise_feature, binary_label)
        assert mi < 0.1

    def test_returns_float(self, informative_feature, binary_label):
        mi = compute_mutual_info(informative_feature, binary_label)
        assert isinstance(mi, float)

    def test_perfect_feature(self, binary_label):
        mi = compute_mutual_info(binary_label.astype(float), binary_label)
        assert mi > 0.3


# ========== compute_redundancy ==========

class TestRedundancy:
    def test_no_existing_features(self, informative_feature):
        assert compute_redundancy(informative_feature, pd.DataFrame()) == 0.0

    def test_none_existing_features(self, informative_feature):
        assert compute_redundancy(informative_feature, None) == 0.0

    def test_self_correlation(self, existing_features):
        feat = existing_features["f1"]
        redundancy = compute_redundancy(feat, existing_features)
        assert redundancy > 0.99

    def test_independent_low(self, noise_feature, existing_features):
        redundancy = compute_redundancy(noise_feature, existing_features)
        assert redundancy < 0.3

    def test_returns_float(self, noise_feature, existing_features):
        assert isinstance(compute_redundancy(noise_feature, existing_features), float)


# ========== compute_cramers_v ==========

class TestCramersV:
    def test_informative_positive(self, informative_feature, binary_label):
        v = compute_cramers_v(informative_feature, binary_label)
        assert v > 0.0

    def test_constant_feature_zero(self, binary_label):
        constant = pd.Series([1.0] * len(binary_label))
        v = compute_cramers_v(constant, binary_label)
        assert v == 0.0

    def test_categorical_feature(self, binary_label):
        cat = pd.Series(np.where(binary_label == 1, "risk", "normal"))
        v = compute_cramers_v(cat, binary_label)
        assert v > 0.5

    def test_returns_float(self, informative_feature, binary_label):
        assert isinstance(compute_cramers_v(informative_feature, binary_label), float)


# ========== compute_bootstrap_chi2 ==========

class TestBootstrapChi2:
    def test_informative_low_p(self, informative_feature, binary_label):
        p = compute_bootstrap_chi2(informative_feature, binary_label, n_resamples=100)
        assert p < 0.05

    def test_noise_higher_p(self, noise_feature, binary_label):
        p = compute_bootstrap_chi2(noise_feature, binary_label, n_resamples=100)
        assert p > 0.01

    def test_returns_float(self, informative_feature, binary_label):
        p = compute_bootstrap_chi2(informative_feature, binary_label, n_resamples=10)
        assert isinstance(p, float)

    def test_p_value_range(self, informative_feature, binary_label):
        p = compute_bootstrap_chi2(informative_feature, binary_label, n_resamples=50)
        assert 0.0 <= p <= 1.0


# ========== compute_delta_prauc ==========

class TestDeltaPrauc:
    def test_informative_positive_delta(self, existing_features, informative_feature, binary_label):
        delta = compute_delta_prauc(existing_features, informative_feature, binary_label)
        assert isinstance(delta, float)

    def test_noise_small_delta(self, existing_features, noise_feature, binary_label):
        delta = compute_delta_prauc(existing_features, noise_feature, binary_label)
        assert abs(delta) < 0.2

    def test_perfect_feature_positive(self, existing_features, binary_label):
        perfect = binary_label.astype(float)
        delta = compute_delta_prauc(existing_features, perfect, binary_label)
        assert delta > 0.0


# ========== compute_shap_rank ==========

class TestShapRank:
    def test_rank_range(self, existing_features, binary_label):
        df = existing_features.copy()
        df["label_copy"] = binary_label.astype(float)
        rank = compute_shap_rank(df, binary_label, "label_copy")
        assert 1 <= rank <= len(df.columns)

    def test_missing_feature_returns_max(self, existing_features, binary_label):
        rank = compute_shap_rank(existing_features, binary_label, "nonexistent")
        assert rank == len(existing_features.columns)

    def test_returns_int(self, existing_features, binary_label):
        rank = compute_shap_rank(existing_features, binary_label, "f1")
        assert isinstance(rank, int)


# ========== normalize_score ==========

class TestNormalizeScore:
    def test_min_value(self):
        assert normalize_score(0.0, [0.0, 5.0, 10.0]) == 0.0

    def test_max_value(self):
        assert normalize_score(10.0, [0.0, 5.0, 10.0]) == 1.0

    def test_mid_value(self):
        assert normalize_score(5.0, [0.0, 5.0, 10.0]) == 0.5

    def test_all_equal(self):
        assert normalize_score(3.0, [3.0, 3.0, 3.0]) == 0.5

    def test_negative_values(self):
        result = normalize_score(-5.0, [-10.0, -5.0, 0.0])
        assert result == 0.5
