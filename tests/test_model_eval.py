"""L3 模型效果验证 (verification/model_eval.py) 的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from knowfeat.config import VerificationConfig
from knowfeat.verification.model_eval import (
    ModelEvalResult,
    _compute_metrics,
    _cv_evaluate,
    evaluate_feature,
)


# ---- helpers ----

def _make_dataset(
    n: int = 2000,
    n_base_features: int = 5,
    pos_rate: float = 0.05,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    label = pd.Series((rng.rand(n) < pos_rate).astype(int))
    baseline_X = pd.DataFrame(
        rng.randn(n, n_base_features),
        columns=[f"base_{i}" for i in range(n_base_features)],
    )
    return baseline_X, label


def _make_good_feature(label: pd.Series, noise: float = 0.3, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    feat = label.values.astype(float) + rng.randn(len(label)) * noise
    return pd.Series(feat, name="good_feat")


def _make_random_feature(n: int, seed: int = 99) -> pd.Series:
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randn(n), name="random_feat")


# ---- _compute_metrics ----

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        recall, precision, f1 = _compute_metrics(y, y)
        assert recall == 1.0
        assert precision == 1.0
        assert f1 == 1.0

    def test_all_wrong(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([1, 1, 0, 0])
        recall, precision, f1 = _compute_metrics(y, pred)
        assert recall == 0.0
        assert precision == 0.0
        assert f1 == 0.0

    def test_partial_recall(self):
        y = np.array([1, 1, 1, 0, 0])
        pred = np.array([1, 0, 0, 0, 0])
        recall, precision, f1 = _compute_metrics(y, pred)
        assert abs(recall - 1.0 / 3.0) < 1e-6
        assert precision == 1.0

    def test_no_positive_labels(self):
        y = np.array([0, 0, 0])
        pred = np.array([0, 0, 1])
        recall, precision, f1 = _compute_metrics(y, pred)
        assert recall == 0.0

    def test_no_positive_predictions(self):
        y = np.array([1, 1, 0])
        pred = np.array([0, 0, 0])
        recall, precision, f1 = _compute_metrics(y, pred)
        assert recall == 0.0
        assert precision == 0.0


# ---- _cv_evaluate ----

class TestCVEvaluate:
    def test_returns_three_floats(self):
        baseline_X, label = _make_dataset(n=500, pos_rate=0.1)
        r, p, f = _cv_evaluate(baseline_X, label.values, n_folds=3, seed=42)
        assert isinstance(r, float)
        assert isinstance(p, float)
        assert isinstance(f, float)

    def test_metrics_in_valid_range(self):
        baseline_X, label = _make_dataset(n=500, pos_rate=0.1)
        r, p, f = _cv_evaluate(baseline_X, label.values, n_folds=3, seed=42)
        for val in (r, p, f):
            assert 0.0 <= val <= 1.0

    def test_deterministic_with_seed(self):
        baseline_X, label = _make_dataset(n=500, pos_rate=0.1)
        r1, p1, f1 = _cv_evaluate(baseline_X, label.values, n_folds=3, seed=42)
        r2, p2, f2 = _cv_evaluate(baseline_X, label.values, n_folds=3, seed=42)
        assert r1 == r2
        assert p1 == p2
        assert f1 == f2

    def test_different_seeds_may_differ(self):
        baseline_X, label = _make_dataset(n=500, pos_rate=0.1)
        r1, _, _ = _cv_evaluate(baseline_X, label.values, n_folds=3, seed=42)
        r2, _, _ = _cv_evaluate(baseline_X, label.values, n_folds=3, seed=99)
        # They may or may not differ, just check they both run without error
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r2 <= 1.0


# ---- evaluate_feature ----

class TestEvaluateFeature:
    def test_good_feature_passes(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        feat = _make_good_feature(label, noise=0.2, seed=42)
        vcfg = VerificationConfig(
            cv_folds=3,
            precision_min=0.0,
            delta_recall_min=0.0,
        )
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        assert isinstance(result, ModelEvalResult)
        assert result.delta_recall > 0, (
            f"Strong feature should improve recall, got delta={result.delta_recall}"
        )
        assert result.passed

    def test_random_feature_low_delta(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        feat = _make_random_feature(2000, seed=99)
        vcfg = VerificationConfig(cv_folds=3, precision_min=0.0, delta_recall_min=0.0)
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        assert isinstance(result, ModelEvalResult)
        assert abs(result.delta_recall) < 0.1

    def test_precision_constraint_rejects(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        feat = _make_good_feature(label, noise=0.2, seed=42)
        vcfg = VerificationConfig(
            cv_folds=3,
            precision_min=0.99999,
            delta_recall_min=0.0,
        )
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        if result.new_precision < 0.99999:
            assert not result.passed
            assert any("precision" in r for r in result.reject_reasons)

    def test_delta_recall_constraint_rejects(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        feat = _make_random_feature(2000, seed=99)
        vcfg = VerificationConfig(
            cv_folds=3,
            precision_min=0.0,
            delta_recall_min=0.5,
        )
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        assert not result.passed
        assert any("delta_recall" in r for r in result.reject_reasons)

    def test_result_fields(self):
        baseline_X, label = _make_dataset(n=1000, pos_rate=0.1, seed=42)
        feat = _make_good_feature(label, noise=0.3, seed=42)
        vcfg = VerificationConfig(cv_folds=3, precision_min=0.0, delta_recall_min=0.0)
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        assert 0.0 <= result.baseline_recall <= 1.0
        assert 0.0 <= result.new_recall <= 1.0
        assert 0.0 <= result.new_precision <= 1.0
        assert 0.0 <= result.baseline_f1 <= 1.0
        assert 0.0 <= result.new_f1 <= 1.0
        assert result.delta_recall == pytest.approx(
            result.new_recall - result.baseline_recall, abs=1e-10,
        )
        assert result.delta_f1 == pytest.approx(
            result.new_f1 - result.baseline_f1, abs=1e-10,
        )

    def test_uses_default_config(self):
        baseline_X, label = _make_dataset(n=500, pos_rate=0.1, seed=42)
        feat = _make_good_feature(label, noise=0.3, seed=42)
        result = evaluate_feature(baseline_X, feat, label, seed=42)
        assert isinstance(result, ModelEvalResult)

    def test_handles_nan_in_new_feature(self):
        baseline_X, label = _make_dataset(n=1000, pos_rate=0.1, seed=42)
        feat = _make_good_feature(label, noise=0.3, seed=42)
        feat.iloc[:100] = np.nan
        vcfg = VerificationConfig(cv_folds=3, precision_min=0.0, delta_recall_min=0.0)
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        assert isinstance(result, ModelEvalResult)

    def test_handles_nan_in_baseline(self):
        baseline_X, label = _make_dataset(n=1000, pos_rate=0.1, seed=42)
        baseline_X.iloc[:50, 0] = np.nan
        feat = _make_good_feature(label, noise=0.3, seed=42)
        vcfg = VerificationConfig(cv_folds=3, precision_min=0.0, delta_recall_min=0.0)
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        assert isinstance(result, ModelEvalResult)

    def test_single_baseline_feature(self):
        baseline_X, label = _make_dataset(n=1000, n_base_features=1, pos_rate=0.1, seed=42)
        feat = _make_good_feature(label, noise=0.3, seed=42)
        vcfg = VerificationConfig(cv_folds=3, precision_min=0.0, delta_recall_min=0.0)
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        assert isinstance(result, ModelEvalResult)

    def test_empty_baseline(self):
        n = 1000
        label = pd.Series(
            np.concatenate([np.ones(100), np.zeros(900)]).astype(int)
        )
        baseline_X = pd.DataFrame(index=range(n))
        feat = _make_good_feature(label, noise=0.3, seed=42)
        vcfg = VerificationConfig(cv_folds=3, precision_min=0.0, delta_recall_min=0.0)
        result = evaluate_feature(baseline_X, feat, label, vcfg=vcfg, seed=42)
        assert isinstance(result, ModelEvalResult)
