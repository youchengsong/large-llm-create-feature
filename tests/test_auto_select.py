"""model_select/auto_select.py 的单元测试。"""

import csv
import json
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from knowfeat.model_select.auto_select import (
    ALL_MODELS,
    DataProfile,
    ModelResult,
    ModelSelectionResult,
    analyze_data_profile,
    format_comparison_table,
    parse_results_csv,
    run_model_selection,
    run_modelset,
    save_selection_report,
    select_best_model,
    select_from_existing_results,
    _build_recommendation_reason,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """生成示例 global_results.csv。"""
    csv_path = tmp_path / "global_results.csv"
    rows = [
        ("CatBoost", "2026-03-15 14:49:47", "2026-03-15 14:49:59", 0.7475, 0.9751, 0.8116, 0.8859, 0.9966),
        ("LightGBM", "2026-03-15 14:50:00", "2026-03-15 14:50:10", 0.6500, 0.9500, 0.8500, 0.8900, 0.9900),
        ("GBDT_FM", "2026-03-15 14:51:00", "2026-03-15 14:51:55", 0.1212, 0.9787, 0.7598, 0.8555, 0.9935),
        ("DNN_Res", "2026-03-15 14:52:00", "2026-03-15 14:52:30", 0.5000, 0.8800, 0.7200, 0.7800, 0.9700),
        ("Scorecard", "2026-03-15 14:53:00", "2026-03-15 14:53:20", 0.4000, 0.9200, 0.6800, 0.7800, 0.9500),
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_name", "start_time", "end_time", "threshold", "precision", "recall", "f1_score", "auc"])
        for r in rows:
            w.writerow(r)
    return csv_path


@pytest.fixture
def sample_results():
    """生成 ModelResult 列表。"""
    return [
        ModelResult("CatBoost", precision=0.9751, recall=0.8116, f1_score=0.8859, auc=0.9966, threshold=0.7475),
        ModelResult("LightGBM", precision=0.9500, recall=0.8500, f1_score=0.8900, auc=0.9900, threshold=0.6500),
        ModelResult("GBDT_FM", precision=0.9787, recall=0.7598, f1_score=0.8555, auc=0.9935, threshold=0.1212),
        ModelResult("DNN_Res", precision=0.8800, recall=0.7200, f1_score=0.7800, auc=0.9700, threshold=0.5000),
        ModelResult("Scorecard", precision=0.9200, recall=0.6800, f1_score=0.7800, auc=0.9500, threshold=0.4000),
    ]


@pytest.fixture
def sample_dataframe():
    """生成测试用 DataFrame 和 label。"""
    rng = np.random.RandomState(42)
    n = 1000
    X = pd.DataFrame({
        "feat_a": rng.randn(n),
        "feat_b": rng.randn(n),
        "feat_c": rng.randint(0, 5, n),
        "feat_d": rng.randn(n),
        "feat_e": pd.Categorical(rng.choice(["A", "B", "C"], n)),
    })
    y = pd.Series((rng.rand(n) < 0.05).astype(int))
    return X, y


@pytest.fixture
def data_profile():
    return DataProfile(
        n_samples=10000, n_features=50, imbalance_ratio=19.0,
        pos_count=500, neg_count=9500, numeric_count=45,
        categorical_count=5, missing_rate=0.01,
    )


# ---------------------------------------------------------------------------
# parse_results_csv
# ---------------------------------------------------------------------------

class TestParseResultsCsv:
    def test_basic_parsing(self, sample_csv):
        results = parse_results_csv(sample_csv)
        assert len(results) == 5
        assert results[0].model_name == "CatBoost"
        assert results[0].precision == pytest.approx(0.9751)
        assert results[0].recall == pytest.approx(0.8116)
        assert results[0].f1_score == pytest.approx(0.8859)
        assert results[0].auc == pytest.approx(0.9966)
        assert results[0].threshold == pytest.approx(0.7475)

    def test_precision_flag(self, sample_csv):
        results = parse_results_csv(sample_csv, precision_min=0.90)
        meets = [r for r in results if r.meets_precision]
        not_meets = [r for r in results if not r.meets_precision]
        assert len(meets) == 4
        assert len(not_meets) == 1
        assert not_meets[0].model_name == "DNN_Res"

    def test_missing_file(self, tmp_path):
        results = parse_results_csv(tmp_path / "nonexistent.csv")
        assert results == []

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w") as f:
            f.write("model_name,start_time,end_time,threshold,precision,recall,f1_score,auc\n")
        results = parse_results_csv(csv_path)
        assert results == []

    def test_custom_precision_threshold(self, sample_csv):
        results = parse_results_csv(sample_csv, precision_min=0.95)
        meets = [r for r in results if r.meets_precision]
        assert len(meets) == 3

    def test_time_fields_parsed(self, sample_csv):
        results = parse_results_csv(sample_csv)
        assert results[0].start_time == "2026-03-15 14:49:47"
        assert results[0].end_time == "2026-03-15 14:49:59"

    def test_invalid_row_skipped(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model_name", "start_time", "end_time", "threshold", "precision", "recall", "f1_score", "auc"])
            w.writerow(["Good", "", "", "0.5", "0.95", "0.80", "0.87", "0.99"])
            w.writerow(["Bad", "", "", "NaN", "bad", "bad", "bad", "bad"])
        results = parse_results_csv(csv_path)
        assert len(results) == 1
        assert results[0].model_name == "Good"


# ---------------------------------------------------------------------------
# analyze_data_profile
# ---------------------------------------------------------------------------

class TestAnalyzeDataProfile:
    def test_basic_profile(self, sample_dataframe):
        X, y = sample_dataframe
        profile = analyze_data_profile(X, y)
        assert profile.n_samples == 1000
        assert profile.n_features == 5
        assert profile.pos_count > 0
        assert profile.neg_count > 0
        assert profile.imbalance_ratio > 1
        assert profile.numeric_count == 4
        assert profile.categorical_count == 1
        assert 0.0 <= profile.missing_rate <= 1.0

    def test_empty_dataframe(self):
        X = pd.DataFrame()
        y = pd.Series(dtype=int)
        profile = analyze_data_profile(X, y)
        assert profile.n_samples == 0
        assert profile.n_features == 0

    def test_all_numeric(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randint(0, 2, 100))
        profile = analyze_data_profile(X, y)
        assert profile.numeric_count == 3
        assert profile.categorical_count == 0

    def test_imbalance_ratio_no_positive(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 0, 0])
        profile = analyze_data_profile(X, y)
        assert profile.imbalance_ratio == float("inf")

    def test_feature_names_captured(self, sample_dataframe):
        X, y = sample_dataframe
        profile = analyze_data_profile(X, y)
        assert profile.feature_names == list(X.columns)

    def test_missing_rate_with_nans(self):
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, 6.0]})
        y = pd.Series([0, 1, 0])
        profile = analyze_data_profile(X, y)
        assert profile.missing_rate == pytest.approx(3 / 6, abs=0.01)


# ---------------------------------------------------------------------------
# select_best_model
# ---------------------------------------------------------------------------

class TestSelectBestModel:
    def test_selects_highest_recall_among_qualified(self, sample_results, data_profile):
        result = select_best_model(sample_results, precision_min=0.90, data_profile=data_profile)
        assert result.recommended_model == "LightGBM"
        assert len(result.qualified_results) == 4
        assert result.qualified_results[0].model_name == "LightGBM"

    def test_all_qualified_sorted_by_recall(self, sample_results, data_profile):
        result = select_best_model(sample_results, precision_min=0.90, data_profile=data_profile)
        recalls = [r.recall for r in result.qualified_results]
        assert recalls == sorted(recalls, reverse=True)

    def test_no_results(self, data_profile):
        result = select_best_model([], precision_min=0.90, data_profile=data_profile)
        assert result.recommended_model == ""
        assert "无模型结果" in result.recommendation_reason

    def test_none_qualified_fallback(self):
        results = [
            ModelResult("A", precision=0.80, recall=0.90, f1_score=0.85, auc=0.95, threshold=0.5),
            ModelResult("B", precision=0.85, recall=0.70, f1_score=0.77, auc=0.92, threshold=0.5),
        ]
        result = select_best_model(results, precision_min=0.90)
        assert result.recommended_model == "A"
        assert "无模型满足" in result.recommendation_reason
        assert len(result.qualified_results) == 0

    def test_strict_precision_threshold(self, sample_results, data_profile):
        result = select_best_model(sample_results, precision_min=0.98, data_profile=data_profile)
        assert len(result.qualified_results) == 0 or all(
            r.precision >= 0.98 for r in result.qualified_results
        )

    def test_single_model(self, data_profile):
        results = [ModelResult("Only", precision=0.95, recall=0.80, f1_score=0.87, auc=0.99, threshold=0.5)]
        result = select_best_model(results, precision_min=0.90, data_profile=data_profile)
        assert result.recommended_model == "Only"
        assert len(result.qualified_results) == 1

    def test_ties_in_recall(self, data_profile):
        results = [
            ModelResult("A", precision=0.95, recall=0.85, f1_score=0.90, auc=0.98, threshold=0.5),
            ModelResult("B", precision=0.92, recall=0.85, f1_score=0.88, auc=0.97, threshold=0.5),
        ]
        result = select_best_model(results, precision_min=0.90, data_profile=data_profile)
        assert result.recommended_model in ("A", "B")
        assert len(result.qualified_results) == 2

    def test_all_results_sorted_by_recall(self, sample_results, data_profile):
        result = select_best_model(sample_results, precision_min=0.90, data_profile=data_profile)
        recalls = [r.recall for r in result.all_results]
        assert recalls == sorted(recalls, reverse=True)

    def test_default_data_profile(self, sample_results):
        result = select_best_model(sample_results, precision_min=0.90)
        assert result.data_profile is not None
        assert result.data_profile.n_samples == 0


# ---------------------------------------------------------------------------
# _build_recommendation_reason
# ---------------------------------------------------------------------------

class TestBuildRecommendationReason:
    def test_basic_reason(self, data_profile):
        best = ModelResult("CatBoost", 0.97, 0.81, 0.88, 0.99, 0.5)
        reason = _build_recommendation_reason(best, data_profile, [best], [best])
        assert "CatBoost" in reason
        assert "10000" in reason
        assert "Recall" in reason

    def test_high_imbalance_note(self):
        profile = DataProfile(
            n_samples=5000, n_features=20, imbalance_ratio=25.0,
            pos_count=200, neg_count=4800, numeric_count=20,
            categorical_count=0, missing_rate=0.0,
        )
        best = ModelResult("CatBoost", 0.95, 0.80, 0.87, 0.99, 0.5)
        reason = _build_recommendation_reason(best, profile, [best], [best])
        assert "严重不平衡" in reason

    def test_moderate_imbalance_note(self):
        profile = DataProfile(
            n_samples=5000, n_features=20, imbalance_ratio=10.0,
            pos_count=500, neg_count=4500, numeric_count=20,
            categorical_count=0, missing_rate=0.0,
        )
        best = ModelResult("LightGBM", 0.95, 0.80, 0.87, 0.99, 0.5)
        reason = _build_recommendation_reason(best, profile, [best], [best])
        assert "中等不平衡" in reason

    def test_high_dim_note(self):
        profile = DataProfile(
            n_samples=5000, n_features=80, imbalance_ratio=3.0,
            pos_count=1250, neg_count=3750, numeric_count=80,
            categorical_count=0, missing_rate=0.0,
        )
        best = ModelResult("GA2M", 0.95, 0.80, 0.87, 0.99, 0.5)
        reason = _build_recommendation_reason(best, profile, [best], [best])
        assert "维度较高" in reason

    def test_categorical_dominant_note(self):
        profile = DataProfile(
            n_samples=5000, n_features=20, imbalance_ratio=3.0,
            pos_count=1250, neg_count=3750, numeric_count=5,
            categorical_count=15, missing_rate=0.0,
        )
        best = ModelResult("CatBoost", 0.95, 0.80, 0.87, 0.99, 0.5)
        reason = _build_recommendation_reason(best, profile, [best], [best])
        assert "类别特征" in reason


# ---------------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------------

class TestFormatComparisonTable:
    def test_contains_header(self, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        table = format_comparison_table(selection)
        assert "模型对比表" in table
        assert "推荐模型" in table
        assert "Recall" in table

    def test_all_models_listed(self, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        table = format_comparison_table(selection)
        for r in sample_results:
            assert r.model_name in table

    def test_recommended_bold(self, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        table = format_comparison_table(selection)
        assert f"**{selection.recommended_model}**" in table

    def test_precision_check_marks(self, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        table = format_comparison_table(selection)
        assert "✓" in table
        assert "✗" in table

    def test_reason_section(self, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        table = format_comparison_table(selection)
        assert "推荐理由" in table


# ---------------------------------------------------------------------------
# save_selection_report
# ---------------------------------------------------------------------------

class TestSaveSelectionReport:
    def test_creates_all_files(self, tmp_path, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        paths = save_selection_report(selection, tmp_path)
        assert paths["json"].exists()
        assert paths["markdown"].exists()
        assert paths["csv"].exists()

    def test_json_content(self, tmp_path, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        paths = save_selection_report(selection, tmp_path)
        with open(paths["json"]) as f:
            data = json.load(f)
        assert data["recommended_model"] == selection.recommended_model
        assert len(data["all_results"]) == len(sample_results)
        assert "data_profile" in data

    def test_csv_content(self, tmp_path, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        paths = save_selection_report(selection, tmp_path)
        df = pd.read_csv(paths["csv"])
        assert len(df) == len(sample_results)
        assert "rank" in df.columns
        assert "model_name" in df.columns

    def test_md_content(self, tmp_path, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        paths = save_selection_report(selection, tmp_path)
        md = paths["markdown"].read_text()
        assert "模型对比表" in md

    def test_creates_output_dir(self, tmp_path, sample_results, data_profile):
        selection = select_best_model(sample_results, data_profile=data_profile)
        nested = tmp_path / "a" / "b" / "c"
        paths = save_selection_report(selection, nested)
        assert nested.exists()
        assert paths["json"].exists()


# ---------------------------------------------------------------------------
# run_modelset
# ---------------------------------------------------------------------------

class TestRunModelset:
    @patch("knowfeat.model_select.auto_select.subprocess.run")
    def test_calls_runner(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        output = tmp_path / "results.csv"
        result = run_modelset(
            train_csv=tmp_path / "train.csv",
            test_csv=tmp_path / "test.csv",
            output_csv=output,
            model_dir=tmp_path / "models",
        )
        assert mock_run.called
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--model" in cmd
        assert "--mode" in cmd
        assert "both" in cmd

    @patch("knowfeat.model_select.auto_select.subprocess.run")
    def test_failure_raises(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")
        with pytest.raises(RuntimeError, match="runner"):
            run_modelset(
                train_csv=tmp_path / "train.csv",
                test_csv=tmp_path / "test.csv",
                output_csv=tmp_path / "results.csv",
            )

    @patch("knowfeat.model_select.auto_select.subprocess.run")
    def test_custom_models(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        run_modelset(
            train_csv=tmp_path / "train.csv",
            test_csv=tmp_path / "test.csv",
            output_csv=tmp_path / "results.csv",
            models=["catboost", "lgbm"],
        )
        cmd = mock_run.call_args[0][0]
        assert "catboost" in cmd
        assert "lgbm" in cmd
        assert "ensemble" not in cmd


# ---------------------------------------------------------------------------
# run_model_selection (integration, mocked runner)
# ---------------------------------------------------------------------------

class TestRunModelSelection:
    @patch("knowfeat.model_select.auto_select.run_modelset")
    def test_end_to_end(self, mock_runner, tmp_path, sample_csv, sample_dataframe):
        mock_runner.return_value = sample_csv
        X, y = sample_dataframe
        result = run_model_selection(
            train_csv=tmp_path / "train.csv",
            test_csv=tmp_path / "test.csv",
            output_dir=tmp_path / "output",
            data_X=X,
            data_y=y,
        )
        assert result.recommended_model != ""
        assert len(result.all_results) == 5
        assert (tmp_path / "output" / "model_selection.json").exists()
        assert (tmp_path / "output" / "model_selection.md").exists()

    @patch("knowfeat.model_select.auto_select.run_modelset")
    def test_auto_load_train_csv(self, mock_runner, tmp_path, sample_csv):
        mock_runner.return_value = sample_csv
        rng = np.random.RandomState(42)
        n = 100
        train_df = pd.DataFrame({
            "feat1": rng.randn(n),
            "feat2": rng.randn(n),
            "is_risk": (rng.rand(n) < 0.05).astype(int),
        })
        train_csv_path = tmp_path / "train.csv"
        train_df.to_csv(train_csv_path, index=False)
        result = run_model_selection(
            train_csv=train_csv_path,
            test_csv=tmp_path / "test.csv",
            output_dir=tmp_path / "output",
        )
        assert result.data_profile.n_samples == n
        assert result.data_profile.n_features == 2


# ---------------------------------------------------------------------------
# select_from_existing_results
# ---------------------------------------------------------------------------

class TestSelectFromExistingResults:
    def test_basic(self, sample_csv, sample_dataframe):
        X, y = sample_dataframe
        result = select_from_existing_results(sample_csv, data_X=X, data_y=y)
        assert result.recommended_model != ""
        assert len(result.all_results) == 5

    def test_without_data_profile(self, sample_csv):
        result = select_from_existing_results(sample_csv)
        assert result.recommended_model != ""
        assert result.data_profile.n_samples == 0

    def test_with_output_dir(self, sample_csv, tmp_path):
        result = select_from_existing_results(sample_csv, output_dir=tmp_path)
        assert (tmp_path / "model_selection.json").exists()
        assert (tmp_path / "model_selection.md").exists()

    def test_missing_csv(self, tmp_path):
        result = select_from_existing_results(tmp_path / "missing.csv")
        assert result.recommended_model == ""


# ---------------------------------------------------------------------------
# ALL_MODELS constant
# ---------------------------------------------------------------------------

class TestAllModels:
    def test_count(self):
        assert len(ALL_MODELS) == 11

    def test_known_models_present(self):
        expected = {"catboost", "ga2m", "scorecard", "acxgb", "wav_xgb",
                    "ensemble", "lgbm", "resample_boost", "gbdt_fm", "dnn_res", "mlp"}
        assert set(ALL_MODELS) == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_result_csv(self, tmp_path):
        csv_path = tmp_path / "single.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model_name", "start_time", "end_time", "threshold", "precision", "recall", "f1_score", "auc"])
            w.writerow(["OnlyModel", "", "", "0.5", "0.95", "0.80", "0.87", "0.99"])
        results = parse_results_csv(csv_path)
        selection = select_best_model(results)
        assert selection.recommended_model == "OnlyModel"

    def test_all_below_precision_uses_fallback(self, tmp_path):
        csv_path = tmp_path / "low_prec.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model_name", "start_time", "end_time", "threshold", "precision", "recall", "f1_score", "auc"])
            w.writerow(["A", "", "", "0.5", "0.80", "0.90", "0.85", "0.95"])
            w.writerow(["B", "", "", "0.5", "0.70", "0.85", "0.77", "0.92"])
        results = parse_results_csv(csv_path, precision_min=0.90)
        selection = select_best_model(results, precision_min=0.90)
        assert selection.recommended_model == "A"
        assert "无模型满足" in selection.recommendation_reason

    def test_precision_boundary(self):
        results = [
            ModelResult("Exact", precision=0.90, recall=0.80, f1_score=0.85, auc=0.95, threshold=0.5),
            ModelResult("Below", precision=0.8999, recall=0.85, f1_score=0.87, auc=0.96, threshold=0.5),
        ]
        selection = select_best_model(results, precision_min=0.90)
        assert len(selection.qualified_results) == 1
        assert selection.recommended_model == "Exact"
