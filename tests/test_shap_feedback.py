"""Tests for knowfeat.search.shap_feedback."""

import numpy as np
import pandas as pd
import pytest

from knowfeat.config import KnowFeatConfig, SearchConfig, VerificationConfig
from knowfeat.search.shap_feedback import (
    FeatureScore,
    ShapFeedback,
    aggregate_dimension_contributions,
    analyze_shap_feedback,
    compute_composite_score,
    compute_shap_importance,
    format_shap_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_data():
    """简单二分类数据（100 样本、3 特征），正样本 20%。"""
    rng = np.random.RandomState(42)
    n = 100
    label = np.array([1] * 20 + [0] * 80)
    X = pd.DataFrame({
        "feat_a": rng.randn(n) + label * 2,
        "feat_b": rng.randn(n) + label * 0.5,
        "feat_c": rng.randn(n),
    })
    return X, pd.Series(label, name="is_risk")


@pytest.fixture
def feature_dims():
    return {
        "feat_a": ["金额频次"],
        "feat_b": ["时间地域"],
        "feat_c": ["行为序列", "主体关联"],
    }


@pytest.fixture
def search_cfg():
    return SearchConfig(
        dimensions=["金额频次", "时间地域", "行为序列", "主体关联"],
        initial_budget_per_dim=5,
    )


# ---------------------------------------------------------------------------
# compute_shap_importance
# ---------------------------------------------------------------------------
class TestComputeShapImportance:

    def test_returns_dict_for_all_features(self, simple_data):
        X, label = simple_data
        result = compute_shap_importance(X, label, n_folds=2, seed=42)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(X.columns)

    def test_all_values_non_negative(self, simple_data):
        X, label = simple_data
        result = compute_shap_importance(X, label, n_folds=2, seed=42)
        for v in result.values():
            assert v >= 0.0

    def test_informative_feature_ranks_higher(self, simple_data):
        X, label = simple_data
        result = compute_shap_importance(X, label, n_folds=2, seed=42)
        assert result["feat_a"] > result["feat_c"]

    def test_empty_dataframe(self):
        X = pd.DataFrame()
        label = pd.Series([0, 1, 0], name="y")
        result = compute_shap_importance(X, label, n_folds=2)
        assert result == {}

    def test_single_feature(self):
        rng = np.random.RandomState(0)
        n = 50
        label = pd.Series([1] * 10 + [0] * 40)
        X = pd.DataFrame({"only": rng.randn(n) + label.values * 3})
        result = compute_shap_importance(X, label, n_folds=2, seed=0)
        assert "only" in result
        assert result["only"] > 0

    def test_reproducible_with_same_seed(self, simple_data):
        X, label = simple_data
        r1 = compute_shap_importance(X, label, n_folds=2, seed=99)
        r2 = compute_shap_importance(X, label, n_folds=2, seed=99)
        for k in r1:
            assert abs(r1[k] - r2[k]) < 1e-10

    def test_different_seeds_may_differ(self, simple_data):
        X, label = simple_data
        r1 = compute_shap_importance(X, label, n_folds=2, seed=1)
        r2 = compute_shap_importance(X, label, n_folds=2, seed=2)
        # They can still be equal for small data, so we just test no crash
        assert isinstance(r1, dict) and isinstance(r2, dict)


# ---------------------------------------------------------------------------
# aggregate_dimension_contributions
# ---------------------------------------------------------------------------
class TestAggregateDimensionContributions:

    def test_basic_single_dimension_features(self):
        shap = {"f1": 0.5, "f2": 0.3}
        dims = {"f1": ["金额频次"], "f2": ["时间地域"]}
        result = aggregate_dimension_contributions(
            shap, dims, ["金额频次", "时间地域", "行为序列", "主体关联"],
        )
        assert result["金额频次"] == pytest.approx(0.5)
        assert result["时间地域"] == pytest.approx(0.3)
        assert result["行为序列"] == pytest.approx(0.0)
        assert result["主体关联"] == pytest.approx(0.0)

    def test_multi_dimension_feature_splits_evenly(self):
        shap = {"f1": 1.0}
        dims = {"f1": ["金额频次", "时间地域"]}
        result = aggregate_dimension_contributions(
            shap, dims, ["金额频次", "时间地域"],
        )
        assert result["金额频次"] == pytest.approx(0.5)
        assert result["时间地域"] == pytest.approx(0.5)

    def test_feature_not_in_dimensions_ignored(self):
        shap = {"f1": 0.5, "f_unknown": 1.0}
        dims = {"f1": ["金额频次"]}
        result = aggregate_dimension_contributions(
            shap, dims, ["金额频次", "时间地域"],
        )
        assert result["金额频次"] == pytest.approx(0.5)
        assert result["时间地域"] == pytest.approx(0.0)

    def test_dimensions_inferred_when_none(self):
        shap = {"f1": 0.4}
        dims = {"f1": ["行为序列"]}
        result = aggregate_dimension_contributions(shap, dims)
        assert "行为序列" in result
        assert result["行为序列"] == pytest.approx(0.4)

    def test_empty_shap(self):
        result = aggregate_dimension_contributions(
            {}, {}, ["金额频次"],
        )
        assert result["金额频次"] == pytest.approx(0.0)

    def test_multiple_features_same_dimension(self):
        shap = {"f1": 0.3, "f2": 0.7}
        dims = {"f1": ["金额频次"], "f2": ["金额频次"]}
        result = aggregate_dimension_contributions(
            shap, dims, ["金额频次"],
        )
        assert result["金额频次"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------
class TestComputeCompositeScore:

    def test_default_weights(self):
        score = compute_composite_score(
            shap_rank=1, total_features=10,
            iv=0.5, redundancy=0.2, delta_recall=0.1,
        )
        expected = 0.30 * 1.0 + 0.20 * 0.5 + 0.20 * 0.8 + 0.30 * 0.1
        assert score == pytest.approx(expected)

    def test_rank_2(self):
        score = compute_composite_score(
            shap_rank=2, total_features=10,
            iv=0.5, redundancy=0.2, delta_recall=0.1,
        )
        expected = 0.30 * 0.5 + 0.20 * 0.5 + 0.20 * 0.8 + 0.30 * 0.1
        assert score == pytest.approx(expected)

    def test_custom_weights(self):
        scfg = SearchConfig(
            score_weight_shap=0.25,
            score_weight_iv=0.25,
            score_weight_novelty=0.25,
            score_weight_delta_recall=0.25,
        )
        score = compute_composite_score(
            shap_rank=1, total_features=5,
            iv=1.0, redundancy=0.0, delta_recall=1.0,
            search_cfg=scfg,
        )
        expected = 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0
        assert score == pytest.approx(expected)

    def test_zero_total_features(self):
        score = compute_composite_score(
            shap_rank=1, total_features=0,
            iv=0.0, redundancy=0.0, delta_recall=0.0,
        )
        # shap_component=0 (total_features=0), iv=0, novelty=1.0, dr=0
        expected = 0.20 * 1.0  # only novelty contributes
        assert score == pytest.approx(expected)

    def test_high_redundancy_lowers_score(self):
        s_low = compute_composite_score(
            shap_rank=1, total_features=10,
            iv=0.5, redundancy=0.1, delta_recall=0.1,
        )
        s_high = compute_composite_score(
            shap_rank=1, total_features=10,
            iv=0.5, redundancy=0.9, delta_recall=0.1,
        )
        assert s_low > s_high

    def test_negative_redundancy_clamped(self):
        score = compute_composite_score(
            shap_rank=1, total_features=5,
            iv=0.0, redundancy=1.5, delta_recall=0.0,
        )
        # max(0, 1 - 1.5) = 0
        assert score >= 0.0

    def test_rank_zero_treated_as_one(self):
        score = compute_composite_score(
            shap_rank=0, total_features=5,
            iv=0.0, redundancy=0.0, delta_recall=0.0,
        )
        expected = 0.30 * 1.0 + 0.20 * 0.0 + 0.20 * 1.0 + 0.30 * 0.0
        assert score == pytest.approx(expected)

    def test_via_cfg(self):
        cfg = KnowFeatConfig()
        score = compute_composite_score(
            shap_rank=1, total_features=5,
            iv=0.5, redundancy=0.5, delta_recall=0.5,
            cfg=cfg,
        )
        expected = 0.30 * 1.0 + 0.20 * 0.5 + 0.20 * 0.5 + 0.30 * 0.5
        assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# analyze_shap_feedback
# ---------------------------------------------------------------------------
class TestAnalyzeShapFeedback:

    def test_returns_shap_feedback(self, simple_data, feature_dims, search_cfg):
        X, label = simple_data
        result = analyze_shap_feedback(
            X, label, feature_dims,
            search_cfg=search_cfg,
            vcfg=VerificationConfig(cv_folds=2),
            seed=42,
        )
        assert isinstance(result, ShapFeedback)
        assert len(result.feature_scores) == 3
        assert len(result.dimension_contributions) == 4
        assert len(result.shap_values_mean) == 3

    def test_feature_scores_sorted_by_shap(self, simple_data, feature_dims, search_cfg):
        X, label = simple_data
        result = analyze_shap_feedback(
            X, label, feature_dims,
            search_cfg=search_cfg,
            vcfg=VerificationConfig(cv_folds=2),
        )
        shap_vals = [fs.shap_importance for fs in result.feature_scores]
        assert shap_vals == sorted(shap_vals, reverse=True)

    def test_ranks_are_consecutive(self, simple_data, feature_dims, search_cfg):
        X, label = simple_data
        result = analyze_shap_feedback(
            X, label, feature_dims,
            search_cfg=search_cfg,
            vcfg=VerificationConfig(cv_folds=2),
        )
        ranks = sorted(fs.shap_rank for fs in result.feature_scores)
        assert ranks == [1, 2, 3]

    def test_dimension_contributions_non_negative(self, simple_data, feature_dims, search_cfg):
        X, label = simple_data
        result = analyze_shap_feedback(
            X, label, feature_dims,
            search_cfg=search_cfg,
            vcfg=VerificationConfig(cv_folds=2),
        )
        for v in result.dimension_contributions.values():
            assert v >= 0.0

    def test_iv_and_delta_recall_included_in_score(self, simple_data, feature_dims, search_cfg):
        X, label = simple_data
        iv = {"feat_a": 0.5, "feat_b": 0.2, "feat_c": 0.01}
        dr = {"feat_a": 0.05, "feat_b": 0.02, "feat_c": 0.0}
        result = analyze_shap_feedback(
            X, label, feature_dims,
            feature_iv=iv,
            feature_delta_recall=dr,
            search_cfg=search_cfg,
            vcfg=VerificationConfig(cv_folds=2),
        )
        for fs in result.feature_scores:
            assert fs.composite_score >= 0.0
            if fs.name == "feat_a":
                assert fs.iv == 0.5
                assert fs.delta_recall == 0.05

    def test_with_full_config(self, simple_data, feature_dims):
        X, label = simple_data
        cfg = KnowFeatConfig()
        cfg.verification.cv_folds = 2
        result = analyze_shap_feedback(
            X, label, feature_dims, cfg=cfg,
        )
        assert isinstance(result, ShapFeedback)

    def test_missing_feature_dims_graceful(self, simple_data, search_cfg):
        X, label = simple_data
        # Only provide dims for feat_a, rest have empty dims
        partial_dims = {"feat_a": ["金额频次"]}
        result = analyze_shap_feedback(
            X, label, partial_dims,
            search_cfg=search_cfg,
            vcfg=VerificationConfig(cv_folds=2),
        )
        assert len(result.feature_scores) == 3
        for fs in result.feature_scores:
            if fs.name != "feat_a":
                assert fs.dimensions == []


# ---------------------------------------------------------------------------
# format_shap_summary
# ---------------------------------------------------------------------------
class TestFormatShapSummary:

    def _make_feedback(self) -> ShapFeedback:
        scores = [
            FeatureScore("f1", ["金额频次"], 0.8, 1, 0.5, 0.1, 0.05, 0.7),
            FeatureScore("f2", ["时间地域"], 0.3, 2, 0.2, 0.3, 0.02, 0.4),
            FeatureScore("f3", ["行为序列", "主体关联"], 0.1, 3, 0.01, 0.5, 0.0, 0.1),
        ]
        dim_contribs = {"金额频次": 0.8, "时间地域": 0.3, "行为序列": 0.05, "主体关联": 0.05}
        return ShapFeedback(
            feature_scores=scores,
            dimension_contributions=dim_contribs,
            shap_values_mean={"f1": 0.8, "f2": 0.3, "f3": 0.1},
        )

    def test_contains_header(self):
        text = format_shap_summary(self._make_feedback())
        assert "SHAP 重要性反馈" in text

    def test_contains_dimension_contributions(self):
        text = format_shap_summary(self._make_feedback())
        assert "金额频次" in text
        assert "时间地域" in text

    def test_contains_feature_table(self):
        text = format_shap_summary(self._make_feedback())
        assert "f1" in text
        assert "f2" in text
        assert "f3" in text

    def test_top_k_limits_output(self):
        text = format_shap_summary(self._make_feedback(), top_k=2)
        assert "f1" in text
        assert "f2" in text
        assert "f3" not in text

    def test_multi_dim_shown_with_slash(self):
        text = format_shap_summary(self._make_feedback())
        assert "行为序列/主体关联" in text

    def test_no_dims_shows_dash(self):
        fb = ShapFeedback(
            feature_scores=[
                FeatureScore("f_no_dim", [], 0.5, 1, 0.0, 0.0, 0.0, 0.0),
            ],
            dimension_contributions={},
            shap_values_mean={"f_no_dim": 0.5},
        )
        text = format_shap_summary(fb)
        assert "| - |" in text
