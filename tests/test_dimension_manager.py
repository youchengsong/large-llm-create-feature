"""维度管理 (search/dimension_manager.py) 的单元测试。"""

import pytest

from knowfeat.config import KnowFeatConfig, SearchConfig
from knowfeat.search.dimension_manager import DimensionManager, DimensionState


# ---- helpers ----

DEFAULT_DIMS = ["金额频次", "时间地域", "行为序列", "主体关联"]


def _make_manager(**kwargs) -> DimensionManager:
    return DimensionManager(search_cfg=SearchConfig(**kwargs))


# ---- 初始化测试 ----

class TestInit:
    def test_default_four_dimensions(self):
        mgr = _make_manager()
        assert mgr.dimensions == DEFAULT_DIMS

    def test_initial_budget_uniform(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        for dim in DEFAULT_DIMS:
            assert mgr.get_budget(dim) == 5

    def test_total_budget(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        assert mgr.total_budget == 20

    def test_custom_dimensions(self):
        mgr = _make_manager(dimensions=["A", "B"])
        assert mgr.dimensions == ["A", "B"]
        assert mgr.total_budget == 10

    def test_custom_initial_budget(self):
        mgr = _make_manager(initial_budget_per_dim=8)
        for dim in DEFAULT_DIMS:
            assert mgr.get_budget(dim) == 8

    def test_from_knowfeat_config(self):
        cfg = KnowFeatConfig()
        mgr = DimensionManager(cfg=cfg)
        assert mgr.dimensions == DEFAULT_DIMS
        assert mgr.total_budget == 20

    def test_search_cfg_overrides_knowfeat_config(self):
        cfg = KnowFeatConfig()
        scfg = SearchConfig(initial_budget_per_dim=3)
        mgr = DimensionManager(cfg=cfg, search_cfg=scfg)
        assert mgr.total_budget == 12


# ---- get_state / all_states ----

class TestStateAccess:
    def test_get_state_returns_dimension_state(self):
        mgr = _make_manager()
        s = mgr.get_state("金额频次")
        assert isinstance(s, DimensionState)
        assert s.name == "金额频次"

    def test_get_state_unknown_raises(self):
        mgr = _make_manager()
        with pytest.raises(ValueError, match="未知维度"):
            mgr.get_state("不存在的维度")

    def test_all_states_ordered(self):
        mgr = _make_manager()
        states = mgr.all_states()
        assert len(states) == 4
        assert [s.name for s in states] == DEFAULT_DIMS


# ---- record_round ----

class TestRecordRound:
    def test_accumulates(self):
        mgr = _make_manager()
        mgr.record_round("金额频次", generated=3, accepted=1, candidates=1, rejected=1)
        mgr.record_round("金额频次", generated=2, accepted=1)
        s = mgr.get_state("金额频次")
        assert s.generated == 5
        assert s.accepted == 2
        assert s.candidates == 1
        assert s.rejected == 1

    def test_other_dims_unaffected(self):
        mgr = _make_manager()
        mgr.record_round("行为序列", generated=3, accepted=2)
        assert mgr.get_state("金额频次").generated == 0
        assert mgr.get_state("行为序列").generated == 3


# ---- record_feature (multi-dimension) ----

class TestRecordFeature:
    def test_single_dimension_accepted(self):
        mgr = _make_manager()
        mgr.record_feature(["金额频次"], accepted=True)
        assert mgr.get_state("金额频次").accepted == 1
        assert mgr.get_state("时间地域").accepted == 0

    def test_multi_dimension_accepted(self):
        mgr = _make_manager()
        mgr.record_feature(["金额频次", "时间地域"], accepted=True)
        assert mgr.get_state("金额频次").accepted == 1
        assert mgr.get_state("时间地域").accepted == 1

    def test_candidate_not_accepted(self):
        mgr = _make_manager()
        mgr.record_feature(["行为序列"], accepted=False)
        assert mgr.get_state("行为序列").candidates == 1
        assert mgr.get_state("行为序列").accepted == 0

    def test_multi_dim_candidate(self):
        mgr = _make_manager()
        mgr.record_feature(["行为序列", "主体关联"], accepted=False)
        assert mgr.get_state("行为序列").candidates == 1
        assert mgr.get_state("主体关联").candidates == 1

    def test_unknown_dimension_ignored(self):
        mgr = _make_manager()
        mgr.record_feature(["不存在"], accepted=True)
        assert mgr.total_accepted == 0

    def test_total_accepted_tracks(self):
        mgr = _make_manager()
        mgr.record_feature(["金额频次"], accepted=True)
        mgr.record_feature(["时间地域"], accepted=True)
        assert mgr.total_accepted == 2


# ---- update_budgets_from_shap ----

class TestShapBudgetUpdate:
    def test_inverse_proportion(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        contribs = {"金额频次": 0.4, "时间地域": 0.3, "行为序列": 0.2, "主体关联": 0.1}
        budgets = mgr.update_budgets_from_shap(contribs)
        # Weakest dimension (主体关联) should get the largest budget
        assert budgets["主体关联"] > budgets["金额频次"]
        assert sum(budgets.values()) == 20

    def test_total_budget_override(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        contribs = {"金额频次": 0.5, "时间地域": 0.3, "行为序列": 0.1, "主体关联": 0.1}
        budgets = mgr.update_budgets_from_shap(contribs, total_budget=12)
        assert sum(budgets.values()) == 12

    def test_min_budget_guaranteed(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        contribs = {"金额频次": 100.0, "时间地域": 0.001, "行为序列": 0.001, "主体关联": 0.001}
        budgets = mgr.update_budgets_from_shap(contribs, min_budget=2)
        for b in budgets.values():
            assert b >= 2

    def test_equal_contributions_equal_budgets(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        contribs = {"金额频次": 1.0, "时间地域": 1.0, "行为序列": 1.0, "主体关联": 1.0}
        budgets = mgr.update_budgets_from_shap(contribs)
        assert budgets["金额频次"] == budgets["主体关联"]
        assert sum(budgets.values()) == 20

    def test_updates_state_budget(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        contribs = {"金额频次": 0.5, "时间地域": 0.3, "行为序列": 0.1, "主体关联": 0.1}
        budgets = mgr.update_budgets_from_shap(contribs)
        for dim in DEFAULT_DIMS:
            assert mgr.get_budget(dim) == budgets[dim]

    def test_updates_shap_contribution_field(self):
        mgr = _make_manager()
        contribs = {"金额频次": 0.5, "时间地域": 0.3, "行为序列": 0.1, "主体关联": 0.1}
        mgr.update_budgets_from_shap(contribs)
        assert mgr.get_state("金额频次").shap_contribution == 0.5
        assert mgr.get_state("主体关联").shap_contribution == 0.1

    def test_empty_contributions_uniform(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        budgets = mgr.update_budgets_from_shap({})
        for b in budgets.values():
            assert b == 5

    def test_partial_contributions(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        contribs = {"金额频次": 0.5, "时间地域": 0.5}
        budgets = mgr.update_budgets_from_shap(contribs)
        assert sum(budgets.values()) == 20
        # Missing dims get fallback allocation
        assert budgets["行为序列"] >= 1
        assert budgets["主体关联"] >= 1

    def test_small_total_budget_respects_min(self):
        mgr = _make_manager(initial_budget_per_dim=1)
        contribs = {"金额频次": 0.5, "时间地域": 0.3, "行为序列": 0.1, "主体关联": 0.1}
        budgets = mgr.update_budgets_from_shap(contribs, total_budget=4, min_budget=1)
        assert all(b >= 1 for b in budgets.values())


# ---- reset_budgets ----

class TestResetBudgets:
    def test_reset_to_default(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        mgr.update_budgets_from_shap(
            {"金额频次": 0.9, "时间地域": 0.05, "行为序列": 0.03, "主体关联": 0.02}
        )
        mgr.reset_budgets()
        for dim in DEFAULT_DIMS:
            assert mgr.get_budget(dim) == 5

    def test_reset_to_custom(self):
        mgr = _make_manager()
        mgr.reset_budgets(budget_per_dim=10)
        for dim in DEFAULT_DIMS:
            assert mgr.get_budget(dim) == 10


# ---- summary ----

class TestSummary:
    def test_summary_contains_all_dims(self):
        mgr = _make_manager()
        text = mgr.summary()
        for dim in DEFAULT_DIMS:
            assert dim in text

    def test_summary_reflects_state(self):
        mgr = _make_manager()
        mgr.record_round("金额频次", generated=3, accepted=2)
        text = mgr.summary()
        assert "3" in text
        assert "2" in text


# ---- edge cases ----

class TestEdgeCases:
    def test_two_dimensions(self):
        mgr = _make_manager(dimensions=["A", "B"], initial_budget_per_dim=10)
        contribs = {"A": 0.8, "B": 0.2}
        budgets = mgr.update_budgets_from_shap(contribs)
        assert budgets["B"] > budgets["A"]
        assert sum(budgets.values()) == 20

    def test_single_dimension(self):
        mgr = _make_manager(dimensions=["only"], initial_budget_per_dim=10)
        contribs = {"only": 1.0}
        budgets = mgr.update_budgets_from_shap(contribs)
        assert budgets["only"] == 10

    def test_zero_contribution_handled(self):
        mgr = _make_manager(initial_budget_per_dim=5)
        contribs = {"金额频次": 0.0, "时间地域": 0.0, "行为序列": 0.0, "主体关联": 0.0}
        budgets = mgr.update_budgets_from_shap(contribs)
        assert sum(budgets.values()) == 20
        for b in budgets.values():
            assert b >= 1
