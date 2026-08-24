import json
from pathlib import Path

import pytest

from scripts.evaluate_cross_model import (
    DATASETS,
    MODEL_NAMES,
    load_comparison_data,
    summarize_model,
)


PROJECT_ROOT = Path(__file__).parent.parent
RESULT_PATH = PROJECT_ROOT / "paper" / "results" / "cross_model_analysis.json"


def test_comparison_data_excludes_identifiers_and_labels():
    baseline, augmented, labels = load_comparison_data("credit_g")

    assert len(baseline) == len(augmented) == len(labels) == 1000
    assert "row_id" not in baseline
    assert "class" not in baseline
    assert augmented.shape[1] > baseline.shape[1]


def test_model_summary_counts_wins_and_reports_paired_test():
    dataset_results = {
        "first": {"No-FE": {"mean": 0.70}, "KnowFeat": {"mean": 0.75}},
        "second": {"No-FE": {"mean": 0.80}, "KnowFeat": {"mean": 0.79}},
        "third": {"No-FE": {"mean": 0.60}, "KnowFeat": {"mean": 0.60}},
    }

    summary = summarize_model(dataset_results)

    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 1
    assert summary["mean_auc_delta"] == pytest.approx(0.0133333333)
    assert 0.0 <= summary["wilcoxon_p_value"] <= 1.0


def test_versioned_cross_model_result_is_complete():
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert payload["protocol"]["split_seed"] == 42
    assert payload["protocol"]["test_size"] == 0.2
    assert set(payload["datasets"]) == set(DATASETS)
    assert set(payload["models"]) == set(MODEL_NAMES)
    for dataset in DATASETS:
        assert len(payload["inputs"][dataset]["sha256"]) == 64
        assert set(payload["datasets"][dataset]) == set(MODEL_NAMES)
        for model in MODEL_NAMES:
            result = payload["datasets"][dataset][model]
            assert set(result) >= {"No-FE", "KnowFeat", "auc_delta"}
            assert result["No-FE"]["n_features"] < result["KnowFeat"]["n_features"]


def test_cross_model_xgboost_reproduces_versioned_holdout_inputs():
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    holdout_dir = PROJECT_ROOT / "outputs" / "eval_holdout"

    for dataset in DATASETS:
        expected = json.loads(
            (holdout_dir / f"{dataset}_KnowFeat.json").read_text(encoding="utf-8")
        )
        reproduced = payload["datasets"][dataset]["XGBoost"]["KnowFeat"]
        assert reproduced["n_features"] == expected["n_features"]
        assert reproduced["mean"] == pytest.approx(
            expected["metrics"]["auc"]["test_point"], abs=1e-12
        )
