"""tests/test_experiments.py — 测试 5 组增量贡献实验流程。"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.experiments import _cv_prauc, _compute_shap_ranks, run_ablation_experiments


# ---------- 测试数据 ----------

@pytest.fixture
def small_dataset():
    rng = np.random.RandomState(42)
    n = 200
    y = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    baseline = pd.DataFrame({
        "base_f1": y * 2.0 + rng.normal(0, 1, n),
        "base_f2": rng.normal(0, 1, n),
        "base_f3": rng.normal(0, 1, n),
    })
    llm = pd.DataFrame({
        "llm_f1": y * 3.0 + rng.normal(0, 0.5, n),
        "llm_f2": rng.normal(0, 1, n),
    })
    return baseline, llm, pd.Series(y)


# ========== _cv_prauc ==========

class TestCvPrauc:
    def test_returns_mean_and_std(self, small_dataset):
        baseline, _, y = small_dataset
        mean, std = _cv_prauc(baseline.values, y.values)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_prauc_range(self, small_dataset):
        baseline, _, y = small_dataset
        mean, std = _cv_prauc(baseline.values, y.values)
        assert 0.0 <= mean <= 1.0
        assert std >= 0.0

    def test_informative_beats_random(self):
        rng = np.random.RandomState(42)
        n = 300
        y = rng.choice([0, 1], size=n, p=[0.8, 0.2])
        X_good = np.column_stack([y * 5.0 + rng.normal(0, 0.3, n)])
        X_random = rng.randn(n, 1)
        mean_good, _ = _cv_prauc(X_good, y)
        mean_random, _ = _cv_prauc(X_random, y)
        assert mean_good > mean_random


# ========== _compute_shap_ranks ==========

class TestComputeShapRanks:
    def test_returns_all_features(self, small_dataset):
        baseline, llm, y = small_dataset
        X = pd.concat([baseline, llm], axis=1)
        ranks = _compute_shap_ranks(X, y.values)
        assert set(ranks.keys()) == set(X.columns)

    def test_ranks_are_sequential(self, small_dataset):
        baseline, llm, y = small_dataset
        X = pd.concat([baseline, llm], axis=1)
        ranks = _compute_shap_ranks(X, y.values)
        assert sorted(ranks.values()) == list(range(1, len(X.columns) + 1))

    def test_informative_feature_ranked_high(self, small_dataset):
        baseline, llm, y = small_dataset
        X = pd.concat([baseline, llm], axis=1)
        ranks = _compute_shap_ranks(X, y.values)
        assert ranks["llm_f1"] <= 2


# ========== run_ablation_experiments ==========

class TestRunAblationExperiments:
    def test_returns_all_experiment_keys(self, small_dataset):
        baseline, llm, y = small_dataset
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), list(llm.columns),
        )
        expected_keys = {
            "exp1_baseline", "exp2_augmented", "exp3_llm_only",
            "exp4_random_control", "exp5_ablation",
            "shap_ranks", "summary",
        }
        assert expected_keys == set(results.keys())

    def test_experiment_result_structure(self, small_dataset):
        baseline, llm, y = small_dataset
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), list(llm.columns),
        )
        for key in ["exp1_baseline", "exp2_augmented", "exp3_llm_only", "exp4_random_control"]:
            assert "prauc_mean" in results[key]
            assert "prauc_std" in results[key]
            assert 0.0 <= results[key]["prauc_mean"] <= 1.0

    def test_ablation_covers_all_llm_features(self, small_dataset):
        baseline, llm, y = small_dataset
        llm_names = list(llm.columns)
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), llm_names,
        )
        assert set(results["exp5_ablation"].keys()) == set(llm_names)

    def test_ablation_has_contribution(self, small_dataset):
        baseline, llm, y = small_dataset
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), list(llm.columns),
        )
        for fname, ablation_info in results["exp5_ablation"].items():
            assert "prauc_mean" in ablation_info
            assert "prauc_std" in ablation_info
            assert "contribution" in ablation_info

    def test_augmented_geq_baseline(self, small_dataset):
        baseline, llm, y = small_dataset
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), list(llm.columns),
        )
        assert results["exp2_augmented"]["prauc_mean"] >= results["exp1_baseline"]["prauc_mean"] - 0.05

    def test_summary_has_required_fields(self, small_dataset):
        baseline, llm, y = small_dataset
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), list(llm.columns),
        )
        summary = results["summary"]
        assert "delta_prauc" in summary
        assert "llm_independent_prauc" in summary
        assert "random_control_delta" in summary
        assert "most_critical_feature" in summary

    def test_delta_prauc_matches(self, small_dataset):
        baseline, llm, y = small_dataset
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), list(llm.columns),
        )
        expected_delta = results["exp2_augmented"]["prauc_mean"] - results["exp1_baseline"]["prauc_mean"]
        assert results["summary"]["delta_prauc"] == pytest.approx(expected_delta)

    def test_most_critical_feature_is_valid(self, small_dataset):
        baseline, llm, y = small_dataset
        llm_names = list(llm.columns)
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), llm_names,
        )
        assert results["summary"]["most_critical_feature"] in llm_names

    def test_shap_ranks_cover_all_features(self, small_dataset):
        baseline, llm, y = small_dataset
        all_names = list(baseline.columns) + list(llm.columns)
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), list(llm.columns),
        )
        assert set(results["shap_ranks"].keys()) == set(all_names)

    def test_random_control_weaker_than_llm(self, small_dataset):
        baseline, llm, y = small_dataset
        results = run_ablation_experiments(
            baseline, llm, y,
            list(baseline.columns), list(llm.columns),
        )
        llm_delta = results["summary"]["delta_prauc"]
        random_delta = results["summary"]["random_control_delta"]
        assert llm_delta >= random_delta - 0.05

    def test_single_llm_feature(self):
        rng = np.random.RandomState(42)
        n = 200
        y = rng.choice([0, 1], size=n, p=[0.8, 0.2])
        baseline = pd.DataFrame({"base": rng.normal(0, 1, n)})
        llm = pd.DataFrame({"llm_only": y * 3.0 + rng.normal(0, 0.5, n)})
        results = run_ablation_experiments(
            baseline, llm, pd.Series(y),
            ["base"], ["llm_only"],
        )
        assert "llm_only" in results["exp5_ablation"]
        ablation_prauc = results["exp5_ablation"]["llm_only"]["prauc_mean"]
        assert ablation_prauc == pytest.approx(results["exp1_baseline"]["prauc_mean"], abs=0.01)
