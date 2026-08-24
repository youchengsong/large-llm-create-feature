import json

import pytest

from scripts.analyze_llm_stability import analyze_stability
from scripts.run_llm_stability import _require_api_key


def _write_run(root, dataset, run_id, features, auc):
    run_dir = root / dataset / f"run_{run_id}"
    run_dir.mkdir(parents=True)
    (run_dir / "generic_summary.json").write_text(json.dumps({
        "stage": "generic",
        "total_accepted": len(features),
        "accepted_features": features,
    }))
    (run_dir / "generic_features.json").write_text(json.dumps([
        {"name": feature, "dimensions": [], "code": ""}
        for feature in features
    ]))
    (run_dir / "holdout.json").write_text(json.dumps({
        "dataset": dataset,
        "protocol": "frozen_holdout_bootstrap",
        "split_seed": 42,
        "test_size": 0.2,
        "metrics": {"auc": {"test_point": auc}},
    }))


def test_analyze_stability_uses_three_independent_runs(tmp_path):
    _write_run(tmp_path, "heart", 1, ["a", "b"], 0.91)
    _write_run(tmp_path, "heart", 2, ["a", "c"], 0.92)
    _write_run(tmp_path, "heart", 3, ["a", "b", "c"], 0.90)

    result = analyze_stability(tmp_path, datasets=("heart",), expected_runs=3)
    heart = result["datasets"]["heart"]

    assert heart["n_runs"] == 3
    assert heart["accepted_count_mean"] == pytest.approx(7 / 3)
    assert heart["auc_mean"] == pytest.approx(0.91)
    assert heart["pairwise_jaccard_mean"] == pytest.approx(5 / 9)
    assert heart["common_features"] == ["a"]
    assert heart["union_feature_count"] == 3


def test_analyze_stability_rejects_incomplete_runs(tmp_path):
    _write_run(tmp_path, "heart", 1, ["a"], 0.91)
    _write_run(tmp_path, "heart", 2, ["a"], 0.92)

    with pytest.raises(ValueError, match="expected 3 runs"):
        analyze_stability(tmp_path, datasets=("heart",), expected_runs=3)


def test_stability_runner_requires_explicit_api_key(monkeypatch):
    monkeypatch.delenv("KNOWFEAT_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="KNOWFEAT_API_KEY"):
        _require_api_key()
