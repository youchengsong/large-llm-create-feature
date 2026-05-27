"""Tests for knowfeat.search.cross_dim — 跨维度知识传播。"""

import pytest

from knowfeat.config import SearchConfig
from knowfeat.prompt.builder import CrossDimExample
from knowfeat.search.cross_dim import (
    AcceptedFeature,
    CrossDimPropagator,
    migrate_best,
)
from knowfeat.search.shap_feedback import FeatureScore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def search_cfg():
    return SearchConfig(cross_dim_top_k=3)


@pytest.fixture
def propagator(search_cfg):
    return CrossDimPropagator(search_cfg=search_cfg)


def _make_feature_score(
    name: str,
    dims: list[str],
    shap_rank: int = 1,
    composite_score: float = 0.5,
    iv: float = 0.1,
    delta_recall: float = 0.02,
) -> FeatureScore:
    return FeatureScore(
        name=name,
        dimensions=dims,
        shap_importance=1.0 / shap_rank,
        shap_rank=shap_rank,
        iv=iv,
        redundancy=0.0,
        delta_recall=delta_recall,
        composite_score=composite_score,
    )


# ── CrossDimPropagator 基础 ──────────────────────────────────────────────────

class TestPropagatorInit:
    def test_empty_on_creation(self, propagator):
        assert propagator.feature_count == 0
        assert propagator.get_all_features() == []

    def test_default_config(self):
        p = CrossDimPropagator()
        assert p._search_cfg.cross_dim_top_k == 5

    def test_from_knowfeat_config(self):
        from knowfeat.config import KnowFeatConfig
        cfg = KnowFeatConfig()
        cfg.search.cross_dim_top_k = 7
        p = CrossDimPropagator(cfg=cfg)
        assert p._search_cfg.cross_dim_top_k == 7


class TestRegisterFeature:
    def test_register_single(self, propagator):
        propagator.register_feature("feat_a", ["金额频次"], "code_a")
        assert propagator.feature_count == 1

    def test_register_multiple(self, propagator):
        propagator.register_feature("feat_a", ["金额频次"], "code_a")
        propagator.register_feature("feat_b", ["时间地域"], "code_b")
        assert propagator.feature_count == 2

    def test_register_overwrites_same_name(self, propagator):
        propagator.register_feature("feat_a", ["金额频次"], "code_v1")
        propagator.register_feature("feat_a", ["金额频次"], "code_v2")
        assert propagator.feature_count == 1
        assert propagator._features["feat_a"].code == "code_v2"

    def test_register_with_metrics(self, propagator):
        propagator.register_feature(
            "feat_a", ["金额频次"], "code_a", iv=0.15, delta_recall=0.03,
        )
        feat = propagator._features["feat_a"]
        assert feat.iv == 0.15
        assert feat.delta_recall == 0.03

    def test_register_multi_dim(self, propagator):
        propagator.register_feature("feat_a", ["金额频次", "行为序列"], "code_a")
        assert propagator._features["feat_a"].dimensions == ["金额频次", "行为序列"]


class TestRemoveFeature:
    def test_remove_existing(self, propagator):
        propagator.register_feature("feat_a", ["金额频次"], "code_a")
        assert propagator.remove_feature("feat_a") is True
        assert propagator.feature_count == 0

    def test_remove_nonexistent(self, propagator):
        assert propagator.remove_feature("nonexist") is False


class TestUpdateFromShap:
    def test_updates_rank_and_score(self, propagator):
        propagator.register_feature("feat_a", ["金额频次"], "code_a")
        propagator.register_feature("feat_b", ["时间地域"], "code_b")

        scores = [
            _make_feature_score("feat_a", ["金额频次"], shap_rank=1, composite_score=0.9),
            _make_feature_score("feat_b", ["时间地域"], shap_rank=3, composite_score=0.4),
        ]
        propagator.update_from_shap(scores)

        assert propagator._features["feat_a"].shap_rank == 1
        assert propagator._features["feat_a"].composite_score == 0.9
        assert propagator._features["feat_b"].shap_rank == 3
        assert propagator._features["feat_b"].composite_score == 0.4

    def test_unmatched_features_unchanged(self, propagator):
        propagator.register_feature("feat_a", ["金额频次"], "code_a")
        scores = [
            _make_feature_score("feat_b", ["时间地域"], shap_rank=1, composite_score=0.9),
        ]
        propagator.update_from_shap(scores)
        assert propagator._features["feat_a"].shap_rank == 0
        assert propagator._features["feat_a"].composite_score == 0.0

    def test_empty_scores(self, propagator):
        propagator.register_feature("feat_a", ["金额频次"], "code_a")
        propagator.update_from_shap([])
        assert propagator._features["feat_a"].shap_rank == 0


class TestGetExamplesForDimension:
    def _setup_features(self, propagator):
        propagator.register_feature("feat_amt1", ["金额频次"], "code_amt1")
        propagator.register_feature("feat_amt2", ["金额频次"], "code_amt2")
        propagator.register_feature("feat_time1", ["时间地域"], "code_time1")
        propagator.register_feature("feat_seq1", ["行为序列"], "code_seq1")
        propagator.register_feature("feat_entity1", ["主体关联"], "code_entity1")

        scores = [
            _make_feature_score("feat_amt1", ["金额频次"], shap_rank=1, composite_score=0.95),
            _make_feature_score("feat_time1", ["时间地域"], shap_rank=2, composite_score=0.80),
            _make_feature_score("feat_seq1", ["行为序列"], shap_rank=3, composite_score=0.70),
            _make_feature_score("feat_amt2", ["金额频次"], shap_rank=4, composite_score=0.60),
            _make_feature_score("feat_entity1", ["主体关联"], shap_rank=5, composite_score=0.50),
        ]
        propagator.update_from_shap(scores)

    def test_excludes_target_dimension(self, propagator):
        self._setup_features(propagator)
        examples = propagator.get_examples_for_dimension("金额频次")
        names = [e.feature_name for e in examples]
        assert "feat_amt1" not in names
        assert "feat_amt2" not in names

    def test_respects_top_k(self, propagator):
        self._setup_features(propagator)
        examples = propagator.get_examples_for_dimension("金额频次")
        assert len(examples) == 3  # cross_dim_top_k=3

    def test_sorted_by_composite_score(self, propagator):
        self._setup_features(propagator)
        examples = propagator.get_examples_for_dimension("金额频次")
        scores = [e.metrics["composite_score"] for e in examples]
        assert scores == sorted(scores, reverse=True)

    def test_returns_crossdimexample_type(self, propagator):
        self._setup_features(propagator)
        examples = propagator.get_examples_for_dimension("金额频次")
        for ex in examples:
            assert isinstance(ex, CrossDimExample)

    def test_example_has_correct_fields(self, propagator):
        self._setup_features(propagator)
        examples = propagator.get_examples_for_dimension("金额频次")
        ex = examples[0]  # feat_time1 (score=0.80)
        assert ex.feature_name == "feat_time1"
        assert ex.source_dimension == "时间地域"
        assert ex.code == "code_time1"
        assert ex.shap_rank == 2
        assert "composite_score" in ex.metrics
        assert "iv" in ex.metrics
        assert "delta_recall" in ex.metrics

    def test_custom_top_k(self, propagator):
        self._setup_features(propagator)
        examples = propagator.get_examples_for_dimension("金额频次", top_k=1)
        assert len(examples) == 1

    def test_no_features_returns_empty(self, propagator):
        examples = propagator.get_examples_for_dimension("金额频次")
        assert examples == []

    def test_all_same_dimension_returns_empty(self, propagator):
        propagator.register_feature("f1", ["金额频次"], "c1")
        propagator.register_feature("f2", ["金额频次"], "c2")
        examples = propagator.get_examples_for_dimension("金额频次")
        assert examples == []

    def test_multi_dim_feature_excluded_from_any_dim(self, propagator):
        propagator.register_feature("cross", ["金额频次", "时间地域"], "code_cross")
        scores = [
            _make_feature_score("cross", ["金额频次", "时间地域"], shap_rank=1, composite_score=0.99),
        ]
        propagator.update_from_shap(scores)

        ex_amt = propagator.get_examples_for_dimension("金额频次")
        assert len(ex_amt) == 0

        ex_time = propagator.get_examples_for_dimension("时间地域")
        assert len(ex_time) == 0

        ex_seq = propagator.get_examples_for_dimension("行为序列")
        assert len(ex_seq) == 1
        assert ex_seq[0].feature_name == "cross"

    def test_multi_dim_feature_source_dimension_is_first(self, propagator):
        propagator.register_feature("cross", ["行为序列", "主体关联"], "code_cross")
        scores = [
            _make_feature_score("cross", ["行为序列", "主体关联"], shap_rank=1, composite_score=0.8),
        ]
        propagator.update_from_shap(scores)
        examples = propagator.get_examples_for_dimension("金额频次")
        assert examples[0].source_dimension == "行为序列"


class TestGetAllFeatures:
    def test_sorted_by_composite_score(self, propagator):
        propagator.register_feature("f1", ["金额频次"], "c1")
        propagator.register_feature("f2", ["时间地域"], "c2")
        scores = [
            _make_feature_score("f1", ["金额频次"], composite_score=0.3),
            _make_feature_score("f2", ["时间地域"], composite_score=0.7),
        ]
        propagator.update_from_shap(scores)

        all_feats = propagator.get_all_features()
        assert all_feats[0].name == "f2"
        assert all_feats[1].name == "f1"


class TestSummary:
    def test_empty_summary(self, propagator):
        s = propagator.summary()
        assert "暂无" in s

    def test_summary_with_features(self, propagator):
        propagator.register_feature("f1", ["金额频次"], "c1")
        propagator.register_feature("f2", ["时间地域"], "c2")
        scores = [
            _make_feature_score("f1", ["金额频次"], composite_score=0.5),
            _make_feature_score("f2", ["时间地域"], composite_score=0.8),
        ]
        propagator.update_from_shap(scores)

        s = propagator.summary()
        assert "已注册特征: 2" in s
        assert "金额频次" in s
        assert "时间地域" in s
        assert "f2" in s  # Top feature


# ── migrate_best 无状态函数 ─────────────────────────────────────────────────

class TestMigrateBest:
    def _make_test_data(self):
        scores = [
            _make_feature_score("feat_a", ["金额频次"], shap_rank=1, composite_score=0.9),
            _make_feature_score("feat_b", ["时间地域"], shap_rank=2, composite_score=0.7),
            _make_feature_score("feat_c", ["行为序列"], shap_rank=3, composite_score=0.5),
            _make_feature_score("feat_d", ["主体关联"], shap_rank=4, composite_score=0.3),
        ]
        codes = {
            "feat_a": "def feature_a(): ...",
            "feat_b": "def feature_b(): ...",
            "feat_c": "def feature_c(): ...",
            "feat_d": "def feature_d(): ...",
        }
        dims = {
            "feat_a": ["金额频次"],
            "feat_b": ["时间地域"],
            "feat_c": ["行为序列"],
            "feat_d": ["主体关联"],
        }
        return scores, codes, dims

    def test_excludes_target_dimension(self):
        scores, codes, dims = self._make_test_data()
        examples = migrate_best(scores, codes, dims, "金额频次", top_k=5)
        names = [e.feature_name for e in examples]
        assert "feat_a" not in names
        assert len(examples) == 3

    def test_respects_top_k(self):
        scores, codes, dims = self._make_test_data()
        examples = migrate_best(scores, codes, dims, "金额频次", top_k=2)
        assert len(examples) == 2

    def test_sorted_by_score(self):
        scores, codes, dims = self._make_test_data()
        examples = migrate_best(scores, codes, dims, "金额频次", top_k=5)
        scores_list = [e.metrics["composite_score"] for e in examples]
        assert scores_list == sorted(scores_list, reverse=True)

    def test_returns_crossdimexample(self):
        scores, codes, dims = self._make_test_data()
        examples = migrate_best(scores, codes, dims, "金额频次", top_k=5)
        for ex in examples:
            assert isinstance(ex, CrossDimExample)

    def test_skips_missing_code(self):
        scores, codes, dims = self._make_test_data()
        del codes["feat_b"]
        examples = migrate_best(scores, codes, dims, "金额频次", top_k=5)
        names = [e.feature_name for e in examples]
        assert "feat_b" not in names
        assert len(examples) == 2

    def test_empty_input(self):
        examples = migrate_best([], {}, {}, "金额频次", top_k=5)
        assert examples == []

    def test_multi_dim_excluded_from_all(self):
        scores = [
            _make_feature_score("cross_ab", ["金额频次", "时间地域"], shap_rank=1, composite_score=0.95),
        ]
        codes = {"cross_ab": "def cross(): ..."}
        dims = {"cross_ab": ["金额频次", "时间地域"]}

        assert len(migrate_best(scores, codes, dims, "金额频次")) == 0
        assert len(migrate_best(scores, codes, dims, "时间地域")) == 0
        assert len(migrate_best(scores, codes, dims, "行为序列")) == 1

    def test_source_dimension_is_first(self):
        scores = [
            _make_feature_score("cross", ["行为序列", "主体关联"], shap_rank=1, composite_score=0.8),
        ]
        codes = {"cross": "code"}
        dims = {"cross": ["行为序列", "主体关联"]}
        examples = migrate_best(scores, codes, dims, "金额频次")
        assert examples[0].source_dimension == "行为序列"

    def test_metrics_populated(self):
        scores, codes, dims = self._make_test_data()
        examples = migrate_best(scores, codes, dims, "金额频次")
        ex = examples[0]
        assert ex.metrics["composite_score"] == 0.7
        assert ex.metrics["iv"] == 0.1
        assert ex.metrics["delta_recall"] == 0.02
        assert ex.shap_rank == 2
