"""Tests for knowfeat/main.py — the main orchestration module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from knowfeat.agent.react_agent import RoundResult
from knowfeat.config import KnowFeatConfig, SearchConfig, VerificationConfig
from knowfeat.data.loader import Stage2Data
from knowfeat.prompt.builder import (
    FailureRecord as PromptFailureRecord,
    ShapFeedback as PromptShapFeedback,
)
from knowfeat.prompt.domain_knowledge import DomainKnowledge
from knowfeat.search.failure_memory import FailureRecord
from knowfeat.search.forward_selection import CandidateFeature, SelectionResult
from knowfeat.search.shap_feedback import FeatureScore, ShapFeedback
from knowfeat.verification.model_eval import ModelEvalResult
from knowfeat.verification.statistics import NoiseThresholds

from knowfeat.main import (
    StageResult,
    _failures_to_prompt,
    _shap_to_prompt,
    run_stage,
    save_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cfg(**overrides) -> KnowFeatConfig:
    cfg = KnowFeatConfig(
        search=SearchConfig(
            max_rounds=3,
            max_features=10,
            convergence_rounds=2,
            dimensions=["金额频次", "时间地域", "行为序列", "主体关联"],
            initial_budget_per_dim=2,
            failure_memory_inject_n=5,
            cross_dim_top_k=3,
        ),
        verification=VerificationConfig(
            noise_n_random_features=5,
            cv_folds=2,
        ),
        stage="stage2",
        seed=42,
        verbose=False,
    )
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        elif hasattr(cfg.search, k):
            setattr(cfg.search, k, v)
    return cfg


def _data(n: int = 200) -> Stage2Data:
    rng = np.random.RandomState(42)
    return Stage2Data(
        open_df=pd.DataFrame({
            "wallet_id": range(1, n + 1),
            "wallet_type": rng.choice([1, 2, 3], n),
        }),
        tx_df=pd.DataFrame({
            "tx_id": range(1, n + 1),
            "src": rng.randint(1, n + 1, n),
            "dst": rng.randint(1, n + 1, n),
            "amount": rng.exponential(1000, n),
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "is_risk": (rng.random(n) < 0.05).astype(int),
        }),
        close_df=pd.DataFrame({
            "wallet_id": list(range(1, 11)),
            "zs_timestamp": pd.date_range("2025-06-01", periods=10, freq="D"),
        }),
        baseline_features=pd.DataFrame(
            rng.randn(n, 3), columns=["bl_0", "bl_1", "bl_2"],
        ),
    )


def _noise() -> NoiseThresholds:
    return NoiseThresholds(
        iv_threshold=0.01, ks_threshold=0.01,
        iv_distribution=np.zeros(5), ks_distribution=np.zeros(5),
    )


def _dk() -> DomainKnowledge:
    return DomainKnowledge()


def _rr(round_num=1, dimension="金额频次", **kw) -> RoundResult:
    defaults = dict(
        round_num=round_num, dimension=dimension,
        n_generated=0, n_accepted=0, n_candidates=0, n_rejected=0,
        steps=1, terminated_reason="agent_completed",
    )
    defaults.update(kw)
    return RoundResult(**defaults)


# Shared mock patches used by run_stage tests.
# We patch at knowfeat.main.* because that's where the names are looked up.
_PATCHES = {
    "llm": "knowfeat.main.LLMClient",
    "schema": "knowfeat.main.generate_all_schema_text",
    "agent_cls": "knowfeat.main.ReactAgent",
    "ctx_cls": "knowfeat.main.ToolContext",
    "cal": "knowfeat.main.calibrate_noise_thresholds",
    "fs": "knowfeat.main.forward_select",
    "shap": "knowfeat.main.analyze_shap_feedback",
    "shap_fmt": "knowfeat.main.format_shap_summary",
}


class _MockToolCtx:
    """Lightweight stand-in for ToolContext that holds mutable state."""

    def __init__(self, data, cfg, noise_thresholds=None, baseline_X=None):
        self.data = data
        self.cfg = cfg
        self.vcfg = cfg.verification
        self.noise_thresholds = noise_thresholds
        self.label = data.tx_df["is_risk"].astype(int)
        if baseline_X is not None:
            self.baseline_X = baseline_X.copy()
        elif data.baseline_features is not None:
            self.baseline_X = data.baseline_features.copy()
        else:
            self.baseline_X = pd.DataFrame()
        self.registered = {}
        self.accepted_features: dict[str, CandidateFeature] = {}
        self.candidate_pool: list[CandidateFeature] = []
        self.rejected_features: list[dict] = []


def _run(
    cfg=None, data=None, dk_=None, noise=None,
    run_round_side_effect=None,
    extra_accepted=None,
    candidate_pool=None,
    rejected_features=None,
    forward_select_result=None,
    shap_result=None,
):
    """Helper: run run_stage with fully controlled mocks."""
    cfg = cfg or _cfg()
    data_ = data or _data()
    dk_val = dk_ or _dk()
    noise_val = noise or _noise()

    if run_round_side_effect is None:
        run_round_side_effect = (
            lambda round_num=1, dimension="", budget=0, **kw: _rr(round_num, dimension)
        )

    mock_tool_ctx = _MockToolCtx(data_, cfg, noise_thresholds=noise_val)
    if extra_accepted:
        mock_tool_ctx.accepted_features.update(extra_accepted)
        for n, c in extra_accepted.items():
            mock_tool_ctx.baseline_X[n] = c.series.values
    if candidate_pool:
        mock_tool_ctx.candidate_pool = list(candidate_pool)
    if rejected_features:
        mock_tool_ctx.rejected_features = list(rejected_features)

    with patch(_PATCHES["llm"]):
        with patch(_PATCHES["schema"], return_value="schema"):
            with patch(_PATCHES["ctx_cls"], return_value=mock_tool_ctx):
                with patch(_PATCHES["agent_cls"]) as MockAgent:
                    mock_agent = MagicMock()
                    mock_agent.run_round.side_effect = run_round_side_effect
                    MockAgent.return_value = mock_agent

                    with patch(_PATCHES["cal"], return_value=noise_val) as mock_cal:
                        fs_ret = forward_select_result or SelectionResult([], [], {})
                        with patch(_PATCHES["fs"], return_value=fs_ret):
                            shap_ret = shap_result or ShapFeedback([], {}, {})
                            with patch(_PATCHES["shap"], return_value=shap_ret):
                                with patch(_PATCHES["shap_fmt"], return_value=""):
                                    result = run_stage(
                                        cfg, data_, dk_val,
                                        noise_thresholds=noise_val,
                                    )

    return result, mock_agent, mock_tool_ctx, mock_cal


# ---------------------------------------------------------------------------
# Tests: type-conversion helpers
# ---------------------------------------------------------------------------

class TestShapToPrompt:
    def test_basic(self):
        fb = ShapFeedback(
            feature_scores=[
                FeatureScore("f1", ["金额频次"], 0.5, 1, 0.1, 0.2, 0.01, 0.6),
                FeatureScore("f2", ["时间地域", "行为序列"], 0.3, 2, 0.05, 0.1, 0.005, 0.4),
            ],
            dimension_contributions={"金额频次": 0.5, "时间地域": 0.3},
            shap_values_mean={"f1": 0.5, "f2": 0.3},
        )
        r = _shap_to_prompt(fb, 3)
        assert isinstance(r, PromptShapFeedback)
        assert r.round_num == 3
        assert r.dimension_contributions["金额频次"] == 0.5
        assert len(r.top_features) == 2
        assert r.top_features[0]["name"] == "f1"

    def test_empty(self):
        r = _shap_to_prompt(ShapFeedback([], {}, {}), 1)
        assert r.top_features == []

    def test_top10_cap(self):
        scores = [
            FeatureScore(f"f{i}", ["金额频次"], 1.0 / (i + 1), i + 1, 0.05, 0.1, 0.001, 0.3)
            for i in range(15)
        ]
        r = _shap_to_prompt(ShapFeedback(scores, {"金额频次": 1.0}, {}), 2)
        assert len(r.top_features) == 10


class TestFailuresToPrompt:
    def test_basic(self):
        recs = [FailureRecord("bad", "x=1", "IV low", 1, "金额频次")]
        [pr] = _failures_to_prompt(recs)
        assert isinstance(pr, PromptFailureRecord)
        assert pr.feature_name == "bad"

    def test_empty(self):
        assert _failures_to_prompt([]) == []

    def test_all_fields_preserved(self):
        rec = FailureRecord("f1", "def f(): pass", "KS fail", 3, "行为序列")
        [pr] = _failures_to_prompt([rec])
        assert pr.code == "def f(): pass"
        assert pr.round_num == 3
        assert pr.dimension == "行为序列"


# ---------------------------------------------------------------------------
# Tests: StageResult data class
# ---------------------------------------------------------------------------

class TestStageResult:
    def test_defaults(self):
        r = StageResult(stage="stage2")
        assert r.total_rounds == 0
        assert r.accepted_feature_names == []
        assert r.feature_codes == {}

    def test_with_values(self):
        r = StageResult(
            stage="stage1", total_rounds=5, total_accepted=12,
            accepted_feature_names=["f1", "f2"],
            feature_codes={"f1": "c1", "f2": "c2"},
            feature_dimensions={"f1": ["金额频次"]},
        )
        assert r.total_rounds == 5
        assert len(r.accepted_feature_names) == 2


# ---------------------------------------------------------------------------
# Tests: save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_files_created(self, tmp_path):
        r = StageResult(
            stage="stage2", total_rounds=2, total_accepted=1,
            total_candidates=0, total_rejected=3,
            convergence_reason="max rounds", elapsed_sec=12.345,
            accepted_feature_names=["fa"],
            feature_codes={"fa": "def fa(): pass"},
            feature_dimensions={"fa": ["金额频次"]},
        )
        paths = save_results(r, tmp_path)

        with open(paths["features"], encoding="utf-8") as f:
            feats = json.load(f)
        assert feats[0]["name"] == "fa"

        with open(paths["summary"], encoding="utf-8") as f:
            s = json.load(f)
        assert s["total_accepted"] == 1
        assert s["elapsed_sec"] == 12.3

    def test_empty(self, tmp_path):
        paths = save_results(StageResult(stage="s1"), tmp_path)
        with open(paths["features"], encoding="utf-8") as f:
            assert json.load(f) == []

    def test_creates_nested_dir(self, tmp_path):
        nested = tmp_path / "a" / "b"
        save_results(StageResult(stage="s2"), nested)
        assert nested.exists()

    def test_unicode(self, tmp_path):
        r = StageResult(
            stage="s2", accepted_feature_names=["中文"],
            feature_codes={"中文": "# 注释"}, feature_dimensions={"中文": ["金额频次"]},
        )
        paths = save_results(r, tmp_path)
        with open(paths["features"], encoding="utf-8") as f:
            assert json.load(f)[0]["name"] == "中文"


# ---------------------------------------------------------------------------
# Tests: run_stage convergence
# ---------------------------------------------------------------------------

class TestRunStageConvergence:
    def test_converges_on_no_new_features(self):
        c = _cfg(max_rounds=5, convergence_rounds=2)
        result, agent, *_ = _run(cfg=c)
        assert "连续 2 轮无新特征" in result.convergence_reason
        assert result.total_rounds == 2

    def test_max_rounds(self):
        c = _cfg(max_rounds=3, convergence_rounds=99)
        result, *_ = _run(cfg=c)
        assert "最大轮次" in result.convergence_reason
        assert result.total_rounds == 3

    def test_max_features_stops(self):
        c = _cfg(max_features=1, convergence_rounds=99)
        rng = np.random.RandomState(0)
        fake = pd.Series(rng.randn(200))

        call_idx = [0]
        def side_effect(round_num=1, dimension="", budget=0, **kw):
            call_idx[0] += 1
            rr = _rr(round_num, dimension)
            if call_idx[0] == 1:
                rr.n_accepted = 1
                rr.features_accepted = ["feat1"]
            return rr

        extra = {"feat1": CandidateFeature(
            name="feat1", series=fake, dimension=["金额频次"], code="c",
        )}
        shap = ShapFeedback(
            [FeatureScore("feat1", ["金额频次"], 0.5, 1, 0.1, 0.1, 0.01, 0.5)],
            {"金额频次": 0.5, "时间地域": 0.2, "行为序列": 0.2, "主体关联": 0.1},
            {"feat1": 0.5},
        )
        result, *_ = _run(
            cfg=c, run_round_side_effect=side_effect,
            extra_accepted=extra, shap_result=shap,
        )
        assert "特征上限" in result.convergence_reason
        assert result.total_accepted >= 1


# ---------------------------------------------------------------------------
# Tests: dimension iteration
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_all_visited(self):
        c = _cfg(max_rounds=1, convergence_rounds=99)
        visited = []
        def se(round_num=1, dimension="", budget=0, **kw):
            visited.append(dimension)
            return _rr(round_num, dimension)
        _run(cfg=c, run_round_side_effect=se)
        assert set(visited) == set(c.search.dimensions)

    def test_budget_forwarded(self):
        c = _cfg(max_rounds=1, convergence_rounds=99, initial_budget_per_dim=7)
        budgets = []
        def se(round_num=1, dimension="", budget=0, **kw):
            budgets.append(budget)
            return _rr(round_num, dimension)
        _run(cfg=c, run_round_side_effect=se)
        assert all(b == 7 for b in budgets)

    def test_zero_budget_skips(self):
        c = _cfg(max_rounds=1, convergence_rounds=99, initial_budget_per_dim=0)
        result, agent, *_ = _run(cfg=c)
        agent.run_round.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: forward selection
# ---------------------------------------------------------------------------

class TestForwardSelection:
    def test_candidates_passed_to_fs(self):
        c = _cfg(max_rounds=1, convergence_rounds=99)
        rng = np.random.RandomState(0)
        cand = CandidateFeature(
            name="cand1", series=pd.Series(rng.randn(200)),
            dimension=["金额频次"], code="def c(): pass",
        )
        fs_res = SelectionResult(
            accepted=[], rejected=[cand],
            eval_details={"cand1": ModelEvalResult(
                passed=False, baseline_recall=0.5, new_recall=0.5,
                delta_recall=0.0, baseline_f1=0.4, new_f1=0.4,
                delta_f1=0.0, new_precision=0.9,
                reject_reasons=["delta_recall<=0"],
            )},
        )
        result, *_ = _run(
            cfg=c, candidate_pool=[cand], forward_select_result=fs_res,
        )
        assert result.total_candidates == 0

    def test_accepted_candidates_promoted(self):
        c = _cfg(max_rounds=1, convergence_rounds=99)
        rng = np.random.RandomState(0)
        cand = CandidateFeature(
            name="cand_ok", series=pd.Series(rng.randn(200)),
            dimension=["时间地域"], code="def ok(): pass",
        )
        fs_res = SelectionResult(
            accepted=[cand], rejected=[], eval_details={},
        )
        shap = ShapFeedback(
            [FeatureScore("cand_ok", ["时间地域"], 0.3, 1, 0.05, 0.1, 0.01, 0.4)],
            {"金额频次": 0.2, "时间地域": 0.3, "行为序列": 0.3, "主体关联": 0.2},
            {"cand_ok": 0.3},
        )
        result, _, ctx, _ = _run(
            cfg=c, candidate_pool=[cand],
            forward_select_result=fs_res, shap_result=shap,
        )
        assert "cand_ok" in ctx.accepted_features


# ---------------------------------------------------------------------------
# Tests: noise calibration
# ---------------------------------------------------------------------------

class TestNoiseCalibration:
    def test_skips_when_provided(self):
        c = _cfg(max_rounds=1, convergence_rounds=99)
        _, _, _, mock_cal = _run(cfg=c, noise=_noise())
        mock_cal.assert_not_called()

    def test_calls_when_none(self):
        c = _cfg(max_rounds=1, convergence_rounds=99)
        data_ = _data()
        dk_val = _dk()

        with patch(_PATCHES["llm"]):
            with patch(_PATCHES["schema"], return_value="s"):
                with patch(_PATCHES["ctx_cls"], return_value=_MockToolCtx(data_, c, _noise())):
                    with patch(_PATCHES["agent_cls"]) as MA:
                        ma = MagicMock()
                        ma.run_round.side_effect = lambda round_num=1, dimension="", budget=0, **kw: _rr(round_num, dimension)
                        MA.return_value = ma
                        with patch(_PATCHES["cal"], return_value=_noise()) as mc:
                            with patch(_PATCHES["fs"], return_value=SelectionResult([], [], {})):
                                with patch(_PATCHES["shap"], return_value=ShapFeedback([], {}, {})):
                                    with patch(_PATCHES["shap_fmt"], return_value=""):
                                        run_stage(c, data_, dk_val, noise_thresholds=None)
                            mc.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: failure memory integration
# ---------------------------------------------------------------------------

class TestFailureMemory:
    def test_rejected_counted(self):
        c = _cfg(max_rounds=1, convergence_rounds=99)
        result, *_ = _run(
            cfg=c,
            rejected_features=[
                {"feature_name": "b1", "code": "x=1", "reason": "IV low", "stage": "L2"},
            ],
        )
        assert result.total_rejected == 1


# ---------------------------------------------------------------------------
# Tests: miscellaneous
# ---------------------------------------------------------------------------

class TestMisc:
    def test_elapsed_sec(self):
        result, *_ = _run()
        assert result.elapsed_sec >= 0

    def test_stage_flag(self):
        c = _cfg()
        c.stage = "stage1"
        result, *_ = _run(cfg=c)
        assert result.stage == "stage1"

    def test_round_results_collected(self):
        c = _cfg(convergence_rounds=2)
        result, *_ = _run(cfg=c)
        assert len(result.round_results) == 2 * len(c.search.dimensions)

    def test_single_dim(self):
        c = _cfg(max_rounds=1, convergence_rounds=99)
        c.search.dimensions = ["金额频次"]
        result, *_ = _run(cfg=c)
        assert result.total_rounds == 1
        assert len(result.round_results) == 1


# ---------------------------------------------------------------------------
# Tests: run() public API
# ---------------------------------------------------------------------------

class TestRunPublic:
    def test_calls_stages(self):
        with patch("knowfeat.main.load_config", return_value=_cfg()):
            with patch("knowfeat.main.load_stage2", return_value=_data()):
                with patch("knowfeat.main.load_domain_knowledge", return_value=_dk()):
                    with patch("knowfeat.main.run_stage", return_value=StageResult(stage="s2")):
                        with patch("knowfeat.main.save_results"):
                            from knowfeat.main import run
                            r = run(stage="stage2")
                            assert r.stage == "s2"
