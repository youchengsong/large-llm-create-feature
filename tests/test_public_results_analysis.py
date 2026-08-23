from pathlib import Path

import pytest

from scripts.analyze_public_results import analyze, load_public_results


RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "eval_holdout"


def test_paper_results_are_complete_and_frozen():
    results = load_public_results(RESULTS_DIR)

    assert len(results) == 8
    assert len(results["credit_g"]) == 7
    assert results["adult"]["KnowFeat"]["n_test"] == 9769
    assert results["wine_quality"]["KnowFeat"]["bootstrap_ci95"][1] > 0.9


def test_analysis_reproduces_manuscript_ranks_and_non_significance():
    summary = analyze(RESULTS_DIR)
    ranks = summary["average_ranks"]

    assert ranks["KnowFeat"] == pytest.approx(2.5)
    assert ranks["OpenFE"] == pytest.approx(3.1875)
    assert ranks["No-FE"] == pytest.approx(3.3125)
    assert all(
        not comparison["significant_at_0.05"]
        for comparison in summary["knowfeat_comparisons"]
    )
    assert all(
        comparison["p_value_holm"] >= comparison["p_value"]
        for comparison in summary["knowfeat_comparisons"]
    )
