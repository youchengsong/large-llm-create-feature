"""tests/test_stability.py — 测试 Nogueira 稳定性指数计算、bootstrap 逻辑。"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stability import compute_nogueira_stability, run_l4_stability


# ---------- 测试数据 ----------

@pytest.fixture
def binary_label():
    rng = np.random.RandomState(42)
    return pd.Series(rng.choice([0, 1], size=300, p=[0.8, 0.2]))


@pytest.fixture
def feature_matrix(binary_label):
    rng = np.random.RandomState(42)
    n = len(binary_label)
    return pd.DataFrame({
        "f_informative": binary_label.values * 3.0 + rng.normal(0, 1, n),
        "f_noise1": rng.normal(0, 1, n),
        "f_noise2": rng.normal(0, 1, n),
        "f_noise3": rng.normal(0, 1, n),
        "f_correlated": binary_label.values * 2.0 + rng.normal(0, 0.5, n),
    })


# ========== compute_nogueira_stability ==========

class TestNogueiraStability:
    def test_identical_sets_returns_one(self):
        sets = [{"a", "b", "c"}] * 5
        result = compute_nogueira_stability(sets, n_features=10)
        assert result == pytest.approx(1.0)

    def test_disjoint_sets_low_stability(self):
        sets = [
            {"a", "b"},
            {"c", "d"},
            {"e", "f"},
            {"g", "h"},
        ]
        result = compute_nogueira_stability(sets, n_features=10)
        assert result < 0.0

    def test_partial_overlap(self):
        sets = [
            {"a", "b", "c"},
            {"a", "b", "d"},
            {"a", "b", "e"},
        ]
        result = compute_nogueira_stability(sets, n_features=10)
        assert 0.0 < result < 1.0

    def test_single_set_returns_zero(self):
        result = compute_nogueira_stability([{"a", "b"}], n_features=5)
        assert result == 0.0

    def test_empty_input_returns_zero(self):
        result = compute_nogueira_stability([], n_features=5)
        assert result == 0.0

    def test_single_feature_returns_zero(self):
        sets = [{"a"}, {"a"}, {"a"}]
        result = compute_nogueira_stability(sets, n_features=1)
        assert result == 0.0

    def test_returns_float(self):
        sets = [{"a", "b"}, {"a", "c"}]
        result = compute_nogueira_stability(sets, n_features=5)
        assert isinstance(result, float)

    def test_range_bounded(self):
        sets = [{"a", "b"}, {"b", "c"}, {"a", "c"}]
        result = compute_nogueira_stability(sets, n_features=5)
        assert -1.0 <= result <= 1.0

    def test_more_overlap_higher_stability(self):
        high_overlap = [{"a", "b", "c"}, {"a", "b", "c"}, {"a", "b", "d"}]
        low_overlap = [{"a", "b", "c"}, {"d", "e", "f"}, {"g", "h", "i"}]
        s_high = compute_nogueira_stability(high_overlap, n_features=10)
        s_low = compute_nogueira_stability(low_overlap, n_features=10)
        assert s_high > s_low


# ========== run_l4_stability ==========

class TestRunL4Stability:
    def test_returns_required_keys(self, feature_matrix, binary_label):
        result = run_l4_stability(feature_matrix, binary_label, n_bootstrap=3, top_k=3)
        assert "stability_index" in result
        assert "per_round_rankings" in result
        assert "pass" in result

    def test_stability_index_is_float(self, feature_matrix, binary_label):
        result = run_l4_stability(feature_matrix, binary_label, n_bootstrap=3, top_k=3)
        assert isinstance(result["stability_index"], float)

    def test_per_round_rankings_count(self, feature_matrix, binary_label):
        n_bootstrap = 4
        result = run_l4_stability(feature_matrix, binary_label, n_bootstrap=n_bootstrap, top_k=3)
        assert len(result["per_round_rankings"]) == n_bootstrap

    def test_top_k_respected(self, feature_matrix, binary_label):
        top_k = 3
        result = run_l4_stability(feature_matrix, binary_label, n_bootstrap=3, top_k=top_k)
        for ranking in result["per_round_rankings"]:
            assert len(ranking) <= top_k

    def test_top_k_capped_at_num_features(self, feature_matrix, binary_label):
        result = run_l4_stability(feature_matrix, binary_label, n_bootstrap=3, top_k=100)
        for ranking in result["per_round_rankings"]:
            assert len(ranking) <= feature_matrix.shape[1]

    def test_pass_is_bool(self, feature_matrix, binary_label):
        result = run_l4_stability(feature_matrix, binary_label, n_bootstrap=3, top_k=3)
        assert isinstance(result["pass"], (bool, np.bool_))

    def test_rankings_contain_valid_features(self, feature_matrix, binary_label):
        result = run_l4_stability(feature_matrix, binary_label, n_bootstrap=3, top_k=3)
        valid_names = set(feature_matrix.columns)
        for ranking in result["per_round_rankings"]:
            assert all(f in valid_names for f in ranking)

    def test_stability_index_range(self, feature_matrix, binary_label):
        result = run_l4_stability(feature_matrix, binary_label, n_bootstrap=5, top_k=3)
        assert -1.0 <= result["stability_index"] <= 1.0
