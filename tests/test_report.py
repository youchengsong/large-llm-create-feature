"""Tests for knowfeat.output.report."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knowfeat.output.report import (
    AblationConfig,
    ExperimentReport,
    MetricsRow,
    compute_capture_rate,
    evaluate_ablation,
    evaluate_by_wallet_type,
    evaluate_overall,
    evaluate_with_cv,
    format_report_html,
    format_report_markdown,
    generate_report,
    load_report_json,
    save_report,
    _compute_binary_metrics,
    _dict_to_metrics_row,
    _metrics_rows_to_csv_lines,
    _report_to_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "feat_a": rng.randn(n),
        "feat_b": rng.randn(n),
        "feat_c": rng.randn(n),
    })
    y = pd.Series(np.concatenate([np.ones(30), np.zeros(170)]).astype(int))
    return X, y


@pytest.fixture
def sample_data_with_wallet():
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame({
        "feat_a": rng.randn(n),
        "feat_b": rng.randn(n),
    })
    y = pd.Series(np.concatenate([np.ones(40), np.zeros(260)]).astype(int))
    wallet_types = pd.Series(
        [1] * 100 + [2] * 100 + [3] * 100,
    )
    return X, y, wallet_types


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# MetricsRow
# ---------------------------------------------------------------------------

class TestMetricsRow:
    def test_to_dict(self):
        row = MetricsRow(
            name="test", recall=0.85, precision=0.90, f1=0.87,
            cr_at_k={1: 0.5, 3: 0.7, 5: 0.9}, n_samples=100, n_positives=10,
        )
        d = row.to_dict()
        assert d["name"] == "test"
        assert d["recall"] == 0.85
        assert d["CR@1%"] == 0.5
        assert d["CR@5%"] == 0.9

    def test_to_dict_empty_cr(self):
        row = MetricsRow(name="empty")
        d = row.to_dict()
        assert "CR@1%" not in d
        assert d["recall"] == 0.0

    def test_defaults(self):
        row = MetricsRow(name="defaults")
        assert row.recall == 0.0
        assert row.n_samples == 0
        assert row.cr_at_k == {}


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------

class TestComputeBinaryMetrics:
    def test_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        r, p, f = _compute_binary_metrics(y_true, y_pred)
        assert r == 1.0
        assert p == 1.0
        assert f == 1.0

    def test_all_wrong(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        r, p, f = _compute_binary_metrics(y_true, y_pred)
        assert r == 0.0
        assert p == 0.0

    def test_partial(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        r, p, f = _compute_binary_metrics(y_true, y_pred)
        assert r == 0.5
        assert p == 1.0

    def test_empty(self):
        y_true = np.array([])
        y_pred = np.array([])
        r, p, f = _compute_binary_metrics(y_true, y_pred)
        assert r == 0.0


# ---------------------------------------------------------------------------
# Capture Rate
# ---------------------------------------------------------------------------

class TestCaptureRate:
    def test_perfect_scores(self):
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        cr = compute_capture_rate(y_true, y_scores, k_percent=50)
        assert cr == 1.0

    def test_cr_at_10(self):
        y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        cr = compute_capture_rate(y_true, y_scores, k_percent=10)
        assert cr == 1.0

    def test_empty(self):
        cr = compute_capture_rate(np.array([]), np.array([]), k_percent=5)
        assert cr == 0.0

    def test_no_positives(self):
        y_true = np.zeros(10, dtype=int)
        y_scores = np.random.rand(10)
        cr = compute_capture_rate(y_true, y_scores, k_percent=10)
        assert cr == 0.0

    def test_all_positives(self):
        y_true = np.ones(10, dtype=int)
        y_scores = np.random.rand(10)
        cr = compute_capture_rate(y_true, y_scores, k_percent=50)
        assert cr > 0.0


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------

class TestEvaluateWithCV:
    def test_returns_valid_metrics(self, sample_data):
        X, y = sample_data
        r, p, f, cr = evaluate_with_cv(X, y, n_folds=3, seed=42)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= p <= 1.0
        assert 0.0 <= f <= 1.0
        assert 1 in cr
        assert 3 in cr
        assert 5 in cr

    def test_empty_features(self):
        X = pd.DataFrame()
        y = pd.Series([0, 1, 0, 1])
        r, p, f, cr = evaluate_with_cv(X, y)
        assert r == 0.0

    def test_empty_labels(self):
        X = pd.DataFrame({"a": []})
        y = pd.Series([], dtype=int)
        r, p, f, cr = evaluate_with_cv(X, y)
        assert r == 0.0

    def test_too_few_positives(self):
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        y = pd.Series([1, 0, 0, 0, 0])
        r, p, f, cr = evaluate_with_cv(X, y, n_folds=5)
        assert isinstance(r, float)


# ---------------------------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------------------------

class TestEvaluateOverall:
    def test_basic(self, sample_data):
        X, y = sample_data
        row = evaluate_overall(X, y, name="test", n_folds=3, seed=42)
        assert row.name == "test"
        assert row.n_samples == 200
        assert row.n_positives == 30
        assert 0.0 <= row.recall <= 1.0
        assert 1 in row.cr_at_k


# ---------------------------------------------------------------------------
# Wallet type evaluation
# ---------------------------------------------------------------------------

class TestEvaluateByWalletType:
    def test_basic(self, sample_data_with_wallet):
        X, y, wt = sample_data_with_wallet
        rows = evaluate_by_wallet_type(X, y, wt, n_folds=3, seed=42)
        assert len(rows) == 3
        names = [r.name for r in rows]
        assert "Corporate" in names
        assert "Personal" in names
        assert "Merchant" in names

    def test_empty_wallet_type(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        wt = pd.Series([1, 1, 1])
        rows = evaluate_by_wallet_type(X, y, wt, n_folds=2, seed=42)
        assert len(rows) == 1

    def test_no_positives_in_group(self):
        X = pd.DataFrame({"a": range(10)})
        y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        wt = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        rows = evaluate_by_wallet_type(X, y, wt, n_folds=2, seed=42)
        type2_row = [r for r in rows if r.name == "Personal"][0]
        assert type2_row.n_positives == 0
        assert type2_row.recall == 0.0


# ---------------------------------------------------------------------------
# Ablation evaluation
# ---------------------------------------------------------------------------

class TestEvaluateAblation:
    def test_basic(self, sample_data):
        X, y = sample_data
        configs = [
            AblationConfig(name="full", feature_names=["feat_a", "feat_b", "feat_c"]),
            AblationConfig(name="minus_c", feature_names=["feat_a", "feat_b"]),
        ]
        rows = evaluate_ablation(X, y, configs, n_folds=3, seed=42)
        assert len(rows) == 2
        assert rows[0].name == "full"
        assert rows[1].name == "minus_c"

    def test_no_valid_columns(self, sample_data):
        X, y = sample_data
        configs = [AblationConfig(name="empty", feature_names=["nonexistent"])]
        rows = evaluate_ablation(X, y, configs)
        assert len(rows) == 1
        assert rows[0].recall == 0.0

    def test_empty_configs(self, sample_data):
        X, y = sample_data
        rows = evaluate_ablation(X, y, [])
        assert rows == []


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_overall_only(self, sample_data):
        X, y = sample_data
        report = generate_report(X, y, stage="stage2", n_folds=3, seed=42)
        assert report.stage == "stage2"
        assert report.overall is not None
        assert report.overall.n_samples == 200
        assert report.wallet_type_rows == []
        assert report.ablation_rows == []

    def test_with_wallet(self, sample_data_with_wallet):
        X, y, wt = sample_data_with_wallet
        report = generate_report(
            X, y, stage="stage2", wallet_type_series=wt,
            n_folds=3, seed=42,
        )
        assert len(report.wallet_type_rows) == 3

    def test_with_ablation(self, sample_data):
        X, y = sample_data
        ablations = [
            AblationConfig(name="all", feature_names=list(X.columns)),
            AblationConfig(name="minus_a", feature_names=["feat_b", "feat_c"]),
        ]
        report = generate_report(
            X, y, ablation_configs=ablations, n_folds=3, seed=42,
        )
        assert len(report.ablation_rows) == 2


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

class TestFormatMarkdown:
    def test_overall_only(self):
        report = ExperimentReport(
            stage="stage2",
            overall=MetricsRow(
                name="overall", recall=0.8, precision=0.9, f1=0.85,
                cr_at_k={1: 0.3, 3: 0.6, 5: 0.8},
                n_samples=100, n_positives=10,
            ),
        )
        md = format_report_markdown(report)
        assert "# 实验报告 (stage2)" in md
        assert "总体结果" in md
        assert "0.8000" in md
        assert "CR@1%" in md

    def test_empty_report(self):
        report = ExperimentReport(stage="test")
        md = format_report_markdown(report)
        assert "test" in md

    def test_with_all_sections(self):
        report = ExperimentReport(
            stage="s2",
            overall=MetricsRow(name="all", recall=0.7, precision=0.8, f1=0.75,
                              cr_at_k={1: 0.2}, n_samples=50, n_positives=5),
            wallet_type_rows=[
                MetricsRow(name="Corporate", recall=0.6, precision=0.7, f1=0.65,
                          cr_at_k={1: 0.1}, n_samples=20, n_positives=2),
            ],
            ablation_rows=[
                MetricsRow(name="full", recall=0.7, precision=0.8, f1=0.75,
                          cr_at_k={1: 0.2}, n_samples=50, n_positives=5),
            ],
        )
        md = format_report_markdown(report)
        assert "Wallet Type 分场景结果" in md
        assert "消融实验对比" in md
        assert "Corporate" in md


# ---------------------------------------------------------------------------
# HTML formatting
# ---------------------------------------------------------------------------

class TestFormatHTML:
    def test_basic_structure(self):
        report = ExperimentReport(
            stage="stage2",
            overall=MetricsRow(
                name="all", recall=0.8, precision=0.9, f1=0.85,
                cr_at_k={1: 0.3}, n_samples=100, n_positives=10,
            ),
        )
        html_str = format_report_html(report)
        assert "<!DOCTYPE html>" in html_str
        assert "<table>" in html_str
        assert "0.8000" in html_str
        assert "stage2" in html_str

    def test_empty_report(self):
        report = ExperimentReport(stage="empty")
        html_str = format_report_html(report)
        assert "<h1>" in html_str

    def test_escapes_special_chars(self):
        report = ExperimentReport(
            stage="<script>alert(1)</script>",
            overall=MetricsRow(
                name="<b>bold</b>", recall=0.5, precision=0.5, f1=0.5,
                n_samples=10, n_positives=5,
            ),
        )
        html_str = format_report_html(report)
        assert "<script>" not in html_str
        assert "&lt;script&gt;" in html_str


# ---------------------------------------------------------------------------
# CSV lines
# ---------------------------------------------------------------------------

class TestCSVLines:
    def test_basic(self):
        rows = [MetricsRow(name="test", recall=0.8, precision=0.9, f1=0.85,
                           cr_at_k={1: 0.3, 5: 0.7}, n_samples=100, n_positives=10)]
        lines = _metrics_rows_to_csv_lines(rows)
        assert len(lines) == 2
        assert "name" in lines[0]
        assert "CR@1%" in lines[0]

    def test_empty(self):
        assert _metrics_rows_to_csv_lines([]) == []


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_save_creates_files(self, sample_data, tmp_dir):
        X, y = sample_data
        report = generate_report(X, y, stage="stage2", n_folds=3, seed=42)
        paths = save_report(report, tmp_dir)

        assert "csv" in paths
        assert "html" in paths
        assert "json" in paths
        assert paths["csv"].exists()
        assert paths["html"].exists()
        assert paths["json"].exists()

    def test_csv_content(self, tmp_dir):
        report = ExperimentReport(
            stage="stage2",
            overall=MetricsRow(
                name="overall", recall=0.8, precision=0.9, f1=0.85,
                cr_at_k={1: 0.3, 3: 0.6, 5: 0.8},
                n_samples=100, n_positives=10,
            ),
        )
        paths = save_report(report, tmp_dir)
        with open(paths["csv"], "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0][0] == "section"
        assert rows[1][0] == "overall"

    def test_json_roundtrip(self, tmp_dir):
        original = ExperimentReport(
            stage="stage2",
            overall=MetricsRow(
                name="overall", recall=0.8, precision=0.9, f1=0.85,
                cr_at_k={1: 0.3, 3: 0.6, 5: 0.8},
                n_samples=100, n_positives=10,
            ),
            wallet_type_rows=[
                MetricsRow(name="Corporate", recall=0.7, precision=0.8, f1=0.75,
                          cr_at_k={1: 0.2, 3: 0.5, 5: 0.7},
                          n_samples=50, n_positives=5),
            ],
            ablation_rows=[
                MetricsRow(name="full", recall=0.8, precision=0.9, f1=0.85,
                          cr_at_k={1: 0.3, 3: 0.6, 5: 0.8},
                          n_samples=100, n_positives=10),
            ],
        )
        paths = save_report(original, tmp_dir)
        loaded = load_report_json(paths["json"])

        assert loaded.stage == original.stage
        assert loaded.overall.recall == original.overall.recall
        assert loaded.overall.cr_at_k[1] == original.overall.cr_at_k[1]
        assert len(loaded.wallet_type_rows) == 1
        assert loaded.wallet_type_rows[0].name == "Corporate"
        assert len(loaded.ablation_rows) == 1
        assert loaded.ablation_rows[0].name == "full"


class TestLoadReportJSON:
    def test_load(self, tmp_dir):
        data = {
            "stage": "stage2",
            "overall": {
                "name": "test",
                "recall": 0.75,
                "precision": 0.85,
                "f1": 0.80,
                "CR@1%": 0.25,
                "CR@3%": 0.55,
                "CR@5%": 0.70,
                "n_samples": 200,
                "n_positives": 20,
            },
        }
        path = tmp_dir / "test.json"
        with open(path, "w") as f:
            json.dump(data, f)

        report = load_report_json(path)
        assert report.stage == "stage2"
        assert report.overall.recall == 0.75
        assert report.overall.cr_at_k[1] == 0.25
        assert report.overall.n_samples == 200


# ---------------------------------------------------------------------------
# _dict_to_metrics_row
# ---------------------------------------------------------------------------

class TestDictToMetricsRow:
    def test_full(self):
        d = {
            "name": "test",
            "recall": 0.8,
            "precision": 0.9,
            "f1": 0.85,
            "CR@1%": 0.3,
            "CR@3%": 0.6,
            "CR@5%": 0.8,
            "n_samples": 100,
            "n_positives": 10,
        }
        row = _dict_to_metrics_row(d)
        assert row.name == "test"
        assert row.recall == 0.8
        assert row.cr_at_k[1] == 0.3

    def test_missing_fields(self):
        row = _dict_to_metrics_row({})
        assert row.name == ""
        assert row.recall == 0.0
        assert row.cr_at_k == {}


# ---------------------------------------------------------------------------
# _report_to_dict
# ---------------------------------------------------------------------------

class TestReportToDict:
    def test_full(self):
        report = ExperimentReport(
            stage="s2",
            overall=MetricsRow(name="all", recall=0.8, precision=0.9, f1=0.85,
                              n_samples=100, n_positives=10),
            wallet_type_rows=[
                MetricsRow(name="Corp", recall=0.7),
            ],
            ablation_rows=[
                MetricsRow(name="full", recall=0.8),
            ],
        )
        d = _report_to_dict(report)
        assert d["stage"] == "s2"
        assert "overall" in d
        assert len(d["wallet_type"]) == 1
        assert len(d["ablation"]) == 1

    def test_empty_report(self):
        report = ExperimentReport(stage="s1")
        d = _report_to_dict(report)
        assert d["stage"] == "s1"
        assert "overall" not in d
        assert "wallet_type" not in d
