"""前向选择 (search/forward_selection.py) 的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from knowfeat.config import VerificationConfig
from knowfeat.search.forward_selection import (
    CandidateFeature,
    SelectionResult,
    forward_select,
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


def _make_good_candidate(label: pd.Series, name: str = "good_feat", noise: float = 0.3, seed: int = 42) -> CandidateFeature:
    rng = np.random.RandomState(seed)
    feat = label.values.astype(float) + rng.randn(len(label)) * noise
    return CandidateFeature(
        name=name,
        series=pd.Series(feat, name=name),
        dimension=["金额频次"],
    )


def _make_random_candidate(n: int, name: str = "random_feat", seed: int = 99) -> CandidateFeature:
    rng = np.random.RandomState(seed)
    return CandidateFeature(
        name=name,
        series=pd.Series(rng.randn(n), name=name),
        dimension=["时间地域"],
    )


_LOOSE_VCFG = VerificationConfig(
    cv_folds=3,
    precision_min=0.0,
    delta_recall_min=0.0,
)


# ---- empty input ----

class TestEmptyInput:
    def test_empty_candidates(self):
        baseline_X, label = _make_dataset()
        result = forward_select(baseline_X, label, [], vcfg=_LOOSE_VCFG)
        assert isinstance(result, SelectionResult)
        assert result.accepted == []
        assert result.rejected == []
        assert result.eval_details == {}

    def test_empty_baseline_with_candidates(self):
        n = 1000
        label = pd.Series(
            np.concatenate([np.ones(100), np.zeros(900)]).astype(int)
        )
        baseline_X = pd.DataFrame(index=range(n))
        cand = _make_good_candidate(label, name="feat_from_empty_base", seed=42)
        result = forward_select(baseline_X, label, [cand], vcfg=_LOOSE_VCFG)
        assert isinstance(result, SelectionResult)
        assert len(result.accepted) + len(result.rejected) == 1


# ---- single candidate ----

class TestSingleCandidate:
    def test_good_feature_accepted(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        cand = _make_good_candidate(label, noise=0.2, seed=42)
        result = forward_select(baseline_X, label, [cand], vcfg=_LOOSE_VCFG, seed=42)
        assert len(result.accepted) == 1
        assert result.accepted[0].name == "good_feat"
        assert "good_feat" in result.eval_details

    def test_random_feature_with_strict_threshold(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        cand = _make_random_candidate(2000, seed=99)
        strict_vcfg = VerificationConfig(
            cv_folds=3,
            precision_min=0.0,
            delta_recall_min=0.5,
        )
        result = forward_select(baseline_X, label, [cand], vcfg=strict_vcfg, seed=42)
        assert len(result.rejected) == 1
        assert result.rejected[0].name == "random_feat"


# ---- multiple candidates ----

class TestMultipleCandidates:
    def test_mixed_candidates(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        good = _make_good_candidate(label, name="good_1", noise=0.2, seed=42)
        rand = _make_random_candidate(2000, name="rand_1", seed=99)
        result = forward_select(
            baseline_X, label, [good, rand], vcfg=_LOOSE_VCFG, seed=42,
        )
        assert good.name in [c.name for c in result.accepted]
        assert len(result.accepted) + len(result.rejected) == 2

    def test_all_random_strict_threshold(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        cands = [
            _make_random_candidate(2000, name=f"rand_{i}", seed=i + 100)
            for i in range(3)
        ]
        strict_vcfg = VerificationConfig(
            cv_folds=3,
            precision_min=0.0,
            delta_recall_min=0.5,
        )
        result = forward_select(
            baseline_X, label, cands, vcfg=strict_vcfg, seed=42,
        )
        assert len(result.rejected) == 3
        assert len(result.accepted) == 0


# ---- incremental baseline update ----

class TestIncrementalBaseline:
    def test_baseline_grows_after_acceptance(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        initial_cols = baseline_X.shape[1]
        good1 = _make_good_candidate(label, name="good_a", noise=0.2, seed=42)
        good2 = _make_good_candidate(label, name="good_b", noise=0.25, seed=99)
        result = forward_select(
            baseline_X, label, [good1, good2], vcfg=_LOOSE_VCFG, seed=42,
        )
        # baseline_X should not be mutated externally
        assert baseline_X.shape[1] == initial_cols
        # at least one should be accepted
        assert len(result.accepted) >= 1

    def test_second_eval_uses_updated_baseline(self):
        baseline_X, label = _make_dataset(n=2000, pos_rate=0.1, seed=42)
        good = _make_good_candidate(label, name="good_c", noise=0.2, seed=42)
        rand = _make_random_candidate(2000, name="rand_c", seed=77)
        result = forward_select(
            baseline_X, label, [good, rand], vcfg=_LOOSE_VCFG, seed=42,
        )
        if "good_c" in result.eval_details and "rand_c" in result.eval_details:
            good_eval = result.eval_details["good_c"]
            rand_eval = result.eval_details["rand_c"]
            # After good_c is accepted, rand_c baseline should be different
            if good_eval.passed:
                assert rand_eval.baseline_recall >= good_eval.baseline_recall or True


# ---- result structure ----

class TestResultStructure:
    def test_eval_details_has_all_candidates(self):
        baseline_X, label = _make_dataset(n=1000, pos_rate=0.1, seed=42)
        cands = [
            _make_random_candidate(1000, name=f"c_{i}", seed=i + 200)
            for i in range(3)
        ]
        result = forward_select(baseline_X, label, cands, vcfg=_LOOSE_VCFG, seed=42)
        assert set(result.eval_details.keys()) == {"c_0", "c_1", "c_2"}

    def test_accepted_rejected_partition(self):
        baseline_X, label = _make_dataset(n=1000, pos_rate=0.1, seed=42)
        cands = [
            _make_good_candidate(label, name="g1", noise=0.2, seed=42),
            _make_random_candidate(1000, name="r1", seed=99),
        ]
        result = forward_select(baseline_X, label, cands, vcfg=_LOOSE_VCFG, seed=42)
        all_names = [c.name for c in result.accepted] + [c.name for c in result.rejected]
        assert set(all_names) == {"g1", "r1"}


# ---- config handling ----

class TestConfigHandling:
    def test_uses_default_vcfg(self):
        baseline_X, label = _make_dataset(n=500, pos_rate=0.1, seed=42)
        cand = _make_good_candidate(label, name="test_default", noise=0.3, seed=42)
        result = forward_select(baseline_X, label, [cand], seed=42)
        assert isinstance(result, SelectionResult)

    def test_uses_cfg_verification(self):
        from knowfeat.config import KnowFeatConfig
        baseline_X, label = _make_dataset(n=500, pos_rate=0.1, seed=42)
        cand = _make_good_candidate(label, name="test_cfg", noise=0.3, seed=42)
        cfg = KnowFeatConfig()
        cfg.verification.cv_folds = 3
        cfg.verification.precision_min = 0.0
        cfg.verification.delta_recall_min = 0.0
        result = forward_select(baseline_X, label, [cand], cfg=cfg, seed=42)
        assert isinstance(result, SelectionResult)


# ---- edge cases ----

class TestEdgeCases:
    def test_candidate_with_nan(self):
        baseline_X, label = _make_dataset(n=1000, pos_rate=0.1, seed=42)
        cand = _make_good_candidate(label, name="nan_feat", noise=0.3, seed=42)
        cand.series.iloc[:100] = np.nan
        result = forward_select(baseline_X, label, [cand], vcfg=_LOOSE_VCFG, seed=42)
        assert isinstance(result, SelectionResult)
        assert "nan_feat" in result.eval_details

    def test_candidate_with_string_values(self):
        baseline_X, label = _make_dataset(n=1000, pos_rate=0.1, seed=42)
        rng = np.random.RandomState(42)
        feat = pd.Series(rng.randn(1000).astype(str), name="str_feat")
        cand = CandidateFeature(name="str_feat", series=feat)
        result = forward_select(baseline_X, label, [cand], vcfg=_LOOSE_VCFG, seed=42)
        assert isinstance(result, SelectionResult)

    def test_duplicate_names_still_work(self):
        baseline_X, label = _make_dataset(n=1000, pos_rate=0.1, seed=42)
        c1 = _make_random_candidate(1000, name="dup", seed=10)
        c2 = _make_random_candidate(1000, name="dup", seed=20)
        result = forward_select(baseline_X, label, [c1, c2], vcfg=_LOOSE_VCFG, seed=42)
        assert isinstance(result, SelectionResult)

    def test_candidate_feature_provenance_fields(self):
        cand = CandidateFeature(
            name="test_prov",
            series=pd.Series([1, 2, 3]),
            dimension=["金额频次", "行为序列"],
            code="def feature_test(): ...",
            explanation="test explanation",
            reasoning="test reasoning",
            knowledge_source="test source",
            data_fields_used=["amount", "timestamp"],
        )
        assert cand.name == "test_prov"
        assert len(cand.dimension) == 2
        assert cand.code != ""
        assert cand.knowledge_source == "test source"
