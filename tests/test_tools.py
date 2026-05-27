"""agent/tools.py 测试 — explore_data / generate_feature / execute_sandbox。"""

import json
import textwrap

import numpy as np
import pandas as pd
import pytest

from knowfeat.agent.tools import (
    ALL_TOOL_SPECS,
    EXECUTE_SANDBOX_SPEC,
    EXPLORE_DATA_SPEC,
    EXPLORE_OUTPUT_MAX_CHARS,
    GENERATE_FEATURE_SPEC,
    PipelineResult,
    RegisteredFeature,
    ToolContext,
    _align_series,
    _extract_feature_column,
    _try_eval_last_line,
)
from knowfeat.config import KnowFeatConfig, VerificationConfig
from knowfeat.data.loader import Stage2Data
from knowfeat.llm.client import ToolSpec
from knowfeat.verification.statistics import NoiseThresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """创建最小测试数据集。"""
    rng = np.random.RandomState(42)
    n = 200

    open_df = pd.DataFrame({
        "wallet_id": [f"W{i:04d}" for i in range(n)],
        "wallet_type": rng.choice([1, 2, 3], n),
        "region_code": rng.choice(["BJ", "SH", "GZ"], n),
    })

    labels = np.zeros(n, dtype=int)
    labels[:20] = 1
    rng.shuffle(labels)

    tx_df = pd.DataFrame({
        "tx_id": [f"T{i:06d}" for i in range(n)],
        "src": open_df["wallet_id"].values,
        "dst": rng.choice(open_df["wallet_id"].values, n),
        "amount": rng.exponential(1000, n),
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "is_risk": labels,
    })

    close_df = pd.DataFrame({
        "wallet_id": rng.choice(open_df["wallet_id"].values, 10),
        "close_date": pd.date_range("2025-06-01", periods=10, freq="D"),
    })

    baseline = pd.DataFrame({
        "base_feat1": rng.randn(n),
        "base_feat2": rng.randn(n),
    })

    return Stage2Data(
        open_df=open_df,
        tx_df=tx_df,
        close_df=close_df,
        baseline_features=baseline,
    )


@pytest.fixture
def cfg():
    return KnowFeatConfig(
        verification=VerificationConfig(
            sandbox_timeout_sec=10,
            noise_n_random_features=10,
            iv_min_fallback=0.001,
            ks_min_fallback=0.001,
            redundancy_max=0.95,
            missing_rate_max=0.5,
            cv_folds=2,
            precision_min=0.0,
            delta_recall_min=-1.0,
        ),
        seed=42,
    )


@pytest.fixture
def ctx(sample_data, cfg):
    return ToolContext(data=sample_data, cfg=cfg)


@pytest.fixture
def good_feature_code():
    return textwrap.dedent("""\
        import pandas as pd
        def feature_tx_amount_mean(open_df, tx_df, close_df):
            agg = tx_df.groupby("src")["amount"].mean().reset_index()
            agg.columns = ["wallet_id", "tx_amount_mean"]
            return agg
    """)


@pytest.fixture
def bad_feature_code():
    return "def feature_broken(open_df, tx_df, close_df):\n    raise ValueError('boom')"


# ---------------------------------------------------------------------------
# ToolSpec 定义测试
# ---------------------------------------------------------------------------

class TestToolSpecs:
    def test_all_specs_are_toolspec(self):
        for spec in ALL_TOOL_SPECS:
            assert isinstance(spec, ToolSpec)

    def test_spec_names(self):
        names = {s.name for s in ALL_TOOL_SPECS}
        assert names == {"explore_data", "generate_feature", "execute_sandbox"}

    def test_to_openai_tool_format(self):
        tool = EXPLORE_DATA_SPEC.to_openai_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "explore_data"
        assert "parameters" in tool["function"]

    def test_generate_feature_required_params(self):
        params = GENERATE_FEATURE_SPEC.parameters
        assert "required" in params
        required = set(params["required"])
        assert "feature_name" in required
        assert "code" in required
        assert "dimension" in required

    def test_execute_sandbox_requires_feature_name(self):
        params = EXECUTE_SANDBOX_SPEC.parameters
        assert "feature_name" in params["required"]


# ---------------------------------------------------------------------------
# ToolContext 初始化测试
# ---------------------------------------------------------------------------

class TestToolContextInit:
    def test_init_with_baseline(self, sample_data, cfg):
        ctx = ToolContext(data=sample_data, cfg=cfg)
        assert ctx.baseline_X.shape[1] == 2
        assert len(ctx.label) == 200

    def test_init_without_baseline(self, cfg):
        data = Stage2Data(
            open_df=pd.DataFrame({"wallet_id": ["W1"]}),
            tx_df=pd.DataFrame({"tx_id": ["T1"], "is_risk": [0]}),
            close_df=pd.DataFrame(),
            baseline_features=None,
        )
        ctx = ToolContext(data=data, cfg=cfg)
        assert ctx.baseline_X.shape[1] == 0

    def test_init_custom_baseline_X(self, sample_data, cfg):
        custom = pd.DataFrame({"feat_a": [1] * 200})
        ctx = ToolContext(data=sample_data, cfg=cfg, baseline_X=custom)
        assert list(ctx.baseline_X.columns) == ["feat_a"]

    def test_get_tool_specs(self, ctx):
        specs = ctx.get_tool_specs()
        assert len(specs) == 3

    def test_empty_initial_state(self, ctx):
        assert len(ctx.registered) == 0
        assert len(ctx.accepted_features) == 0
        assert len(ctx.candidate_pool) == 0
        assert len(ctx.rejected_features) == 0


# ---------------------------------------------------------------------------
# explore_data 测试
# ---------------------------------------------------------------------------

class TestExploreData:
    def test_simple_expression(self, ctx):
        result = ctx.execute("explore_data", {"code": "len(tx_df)"})
        data = json.loads(result)
        assert "200" in data["output"]

    def test_print_output(self, ctx):
        result = ctx.execute("explore_data", {"code": "print('hello world')"})
        data = json.loads(result)
        assert "hello world" in data["output"]

    def test_pandas_operation(self, ctx):
        code = "tx_df['amount'].describe()"
        result = ctx.execute("explore_data", {"code": code})
        data = json.loads(result)
        assert "mean" in data["output"]

    def test_numpy_available(self, ctx):
        result = ctx.execute("explore_data", {"code": "np.mean([1, 2, 3])"})
        data = json.loads(result)
        assert "2.0" in data["output"]

    def test_empty_code_error(self, ctx):
        result = ctx.execute("explore_data", {"code": ""})
        data = json.loads(result)
        assert "error" in data

    def test_runtime_error(self, ctx):
        result = ctx.execute("explore_data", {"code": "1/0"})
        data = json.loads(result)
        assert "error" in data

    def test_output_truncation(self, ctx):
        code = f"print('x' * {EXPLORE_OUTPUT_MAX_CHARS + 100})"
        result = ctx.execute("explore_data", {"code": code})
        data = json.loads(result)
        assert "截断" in data["output"]

    def test_no_output(self, ctx):
        result = ctx.execute("explore_data", {"code": "x = 1"})
        data = json.loads(result)
        assert "无输出" in data["output"]

    def test_multiline_code(self, ctx):
        code = textwrap.dedent("""\
            counts = tx_df['is_risk'].value_counts()
            print(counts.to_dict())
        """)
        result = ctx.execute("explore_data", {"code": code})
        data = json.loads(result)
        assert "output" in data

    def test_baseline_features_accessible(self, ctx):
        code = "baseline_features.shape"
        result = ctx.execute("explore_data", {"code": code})
        data = json.loads(result)
        assert "200" in data["output"]


# ---------------------------------------------------------------------------
# generate_feature 测试
# ---------------------------------------------------------------------------

class TestGenerateFeature:
    def test_register_feature(self, ctx, good_feature_code):
        result = ctx.execute("generate_feature", {
            "feature_name": "tx_amount_mean",
            "dimension": ["金额频次"],
            "code": good_feature_code,
            "explanation": "交易金额均值",
            "reasoning": "异常交易金额通常偏离正常范围",
            "knowledge_source": "38风险指标",
            "data_fields_used": ["amount"],
        })
        data = json.loads(result)
        assert data["status"] == "registered"
        assert "tx_amount_mean" in ctx.registered

    def test_multi_dimension(self, ctx, good_feature_code):
        ctx.execute("generate_feature", {
            "feature_name": "feat_multi",
            "dimension": ["金额频次", "时间地域"],
            "code": good_feature_code,
            "explanation": "test",
            "reasoning": "test",
            "knowledge_source": "test",
            "data_fields_used": ["amount"],
        })
        assert ctx.registered["feat_multi"].dimension == ["金额频次", "时间地域"]

    def test_string_dimension_coerced_to_list(self, ctx, good_feature_code):
        ctx.execute("generate_feature", {
            "feature_name": "feat_single_dim",
            "dimension": "金额频次",
            "code": good_feature_code,
            "explanation": "test",
            "reasoning": "test",
            "knowledge_source": "test",
            "data_fields_used": [],
        })
        assert isinstance(ctx.registered["feat_single_dim"].dimension, list)

    def test_empty_name_error(self, ctx):
        result = ctx.execute("generate_feature", {
            "feature_name": "",
            "dimension": [],
            "code": "def f(): pass",
            "explanation": "",
            "reasoning": "",
            "knowledge_source": "",
            "data_fields_used": [],
        })
        data = json.loads(result)
        assert "error" in data

    def test_empty_code_error(self, ctx):
        result = ctx.execute("generate_feature", {
            "feature_name": "test",
            "dimension": [],
            "code": "",
            "explanation": "",
            "reasoning": "",
            "knowledge_source": "",
            "data_fields_used": [],
        })
        data = json.loads(result)
        assert "error" in data

    def test_overwrite_existing_registration(self, ctx, good_feature_code):
        ctx.execute("generate_feature", {
            "feature_name": "feat_x",
            "dimension": ["金额频次"],
            "code": "def feat_x(a, b, c): pass",
            "explanation": "v1",
            "reasoning": "",
            "knowledge_source": "",
            "data_fields_used": [],
        })
        ctx.execute("generate_feature", {
            "feature_name": "feat_x",
            "dimension": ["时间地域"],
            "code": good_feature_code,
            "explanation": "v2",
            "reasoning": "",
            "knowledge_source": "",
            "data_fields_used": [],
        })
        assert ctx.registered["feat_x"].explanation == "v2"
        assert ctx.registered["feat_x"].dimension == ["时间地域"]


# ---------------------------------------------------------------------------
# execute_sandbox 测试
# ---------------------------------------------------------------------------

class TestExecuteSandbox:
    def test_unregistered_feature_error(self, ctx):
        result = ctx.execute("execute_sandbox", {"feature_name": "nonexistent"})
        data = json.loads(result)
        assert "error" in data
        assert "未注册" in data["error"]

    def test_empty_name_error(self, ctx):
        result = ctx.execute("execute_sandbox", {"feature_name": ""})
        data = json.loads(result)
        assert "error" in data

    def test_l1_failure_bad_code(self, ctx, bad_feature_code):
        ctx.execute("generate_feature", {
            "feature_name": "bad_feat",
            "dimension": ["金额频次"],
            "code": bad_feature_code,
            "explanation": "test",
            "reasoning": "test",
            "knowledge_source": "test",
            "data_fields_used": [],
        })
        result = ctx.execute("execute_sandbox", {"feature_name": "bad_feat"})
        data = json.loads(result)
        assert data["stage_reached"] == "L1"
        assert data["accepted"] is False
        assert data["L1_sandbox"]["passed"] is False

    def test_l1_success_through_pipeline(self, ctx, good_feature_code):
        ctx.execute("generate_feature", {
            "feature_name": "tx_amount_mean",
            "dimension": ["金额频次"],
            "code": good_feature_code,
            "explanation": "交易金额均值",
            "reasoning": "异常交易金额偏离正常值",
            "knowledge_source": "38风险指标",
            "data_fields_used": ["amount"],
        })
        result = ctx.execute("execute_sandbox", {"feature_name": "tx_amount_mean"})
        data = json.loads(result)
        assert data["L1_sandbox"]["passed"] is True
        assert data["L1_sandbox"]["output_shape"] is not None
        assert data["stage_reached"] in ("L2", "L3")

    def test_full_pipeline_produces_result(self, ctx, good_feature_code):
        ctx.execute("generate_feature", {
            "feature_name": "tx_amount_mean",
            "dimension": ["金额频次"],
            "code": good_feature_code,
            "explanation": "交易金额均值",
            "reasoning": "test",
            "knowledge_source": "test",
            "data_fields_used": ["amount"],
        })
        result = ctx.execute("execute_sandbox", {"feature_name": "tx_amount_mean"})
        data = json.loads(result)
        assert "feature_name" in data
        assert "summary" in data
        # With very relaxed thresholds, should reach L3
        if data["stage_reached"] == "L3":
            assert "L3_model_eval" in data

    def test_accepted_feature_updates_state(self, ctx):
        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            def feature_risk_indicator(open_df, tx_df, close_df):
                rng = np.random.RandomState(42)
                risk = tx_df['is_risk'].values.astype(float) + rng.randn(len(tx_df)) * 0.1
                return pd.DataFrame({'tx_id': tx_df['tx_id'], 'risk_indicator': risk})
        """)
        ctx.execute("generate_feature", {
            "feature_name": "risk_indicator",
            "dimension": ["行为序列"],
            "code": code,
            "explanation": "test",
            "reasoning": "test",
            "knowledge_source": "test",
            "data_fields_used": ["is_risk"],
        })
        result = ctx.execute("execute_sandbox", {"feature_name": "risk_indicator"})
        data = json.loads(result)
        if data["accepted"]:
            assert "risk_indicator" in ctx.accepted_features
            assert "risk_indicator" in ctx.baseline_X.columns

    def test_l2_failure_records_rejection(self, ctx):
        code = textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            def feature_constant(open_df, tx_df, close_df):
                return pd.DataFrame({
                    'tx_id': tx_df['tx_id'],
                    'constant_val': [42.0] * len(tx_df),
                })
        """)
        cfg_strict = KnowFeatConfig(
            verification=VerificationConfig(
                sandbox_timeout_sec=10,
                iv_min_fallback=0.5,
                ks_min_fallback=0.5,
                cv_folds=2,
            ),
            seed=42,
        )
        strict_ctx = ToolContext(data=ctx.data, cfg=cfg_strict)
        strict_ctx.execute("generate_feature", {
            "feature_name": "constant_feat",
            "dimension": ["金额频次"],
            "code": code,
            "explanation": "test",
            "reasoning": "test",
            "knowledge_source": "test",
            "data_fields_used": [],
        })
        result = strict_ctx.execute("execute_sandbox", {"feature_name": "constant_feat"})
        data = json.loads(result)
        assert data["stage_reached"] == "L2"
        assert data["accepted"] is False
        assert len(strict_ctx.rejected_features) == 1


# ---------------------------------------------------------------------------
# dispatch 测试
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_unknown_tool(self, ctx):
        result = ctx.execute("unknown_tool", {})
        data = json.loads(result)
        assert "error" in data
        assert "未知工具" in data["error"]

    def test_exception_handling(self, ctx):
        result = ctx.execute("execute_sandbox", {"feature_name": "nonexistent"})
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# 内部辅助函数测试
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_try_eval_last_line_expression(self):
        ns: dict = {}
        exec("x = 42", ns)
        result = _try_eval_last_line("x = 42\nx", ns)
        assert result == 42

    def test_try_eval_last_line_comment(self):
        result = _try_eval_last_line("# comment", {})
        assert result is None

    def test_try_eval_last_line_import(self):
        result = _try_eval_last_line("import os", {})
        assert result is None

    def test_try_eval_last_line_empty(self):
        result = _try_eval_last_line("", {})
        assert result is None

    def test_extract_feature_column(self):
        df = pd.DataFrame({
            "wallet_id": ["W1", "W2"],
            "my_feat": [1.0, 2.0],
        })
        series = _extract_feature_column(df, "test_feat")
        assert series is not None
        assert series.name == "test_feat"
        assert list(series.values) == [1.0, 2.0]

    def test_extract_feature_column_only_id(self):
        df = pd.DataFrame({"wallet_id": ["W1", "W2"]})
        assert _extract_feature_column(df, "test") is None

    def test_extract_feature_column_tx_id(self):
        df = pd.DataFrame({
            "tx_id": ["T1", "T2"],
            "score": [0.5, 0.8],
        })
        series = _extract_feature_column(df, "test")
        assert series is not None

    def test_align_series_tx_id(self, sample_data):
        sandbox_df = pd.DataFrame({
            "tx_id": sample_data.tx_df["tx_id"],
            "feat": np.random.randn(200),
        })
        feature = pd.Series(sandbox_df["feat"].values, name="feat")
        label = sample_data.tx_df["is_risk"]
        aligned_feat, aligned_label = _align_series(
            feature, sandbox_df, label, sample_data,
        )
        assert len(aligned_feat) == len(aligned_label)

    def test_align_series_wallet_id(self, sample_data):
        sandbox_df = pd.DataFrame({
            "wallet_id": sample_data.open_df["wallet_id"],
            "feat": np.random.randn(200),
        })
        feature = pd.Series(sandbox_df["feat"].values, name="feat")
        label = sample_data.tx_df["is_risk"]
        aligned_feat, aligned_label = _align_series(
            feature, sandbox_df, label, sample_data,
        )
        assert len(aligned_feat) == len(aligned_label)


# ---------------------------------------------------------------------------
# PipelineResult / RegisteredFeature 数据类测试
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_pipeline_result_defaults(self):
        pr = PipelineResult(feature_name="test")
        assert pr.accepted is False
        assert pr.added_to_candidates is False
        assert pr.stage_reached == ""

    def test_registered_feature(self):
        rf = RegisteredFeature(
            feature_name="feat",
            dimension=["金额频次"],
            code="def feat(a,b,c): pass",
            explanation="test",
            reasoning="test",
            knowledge_source="test",
            data_fields_used=["amount"],
        )
        assert rf.feature_name == "feat"
        assert rf.dimension == ["金额频次"]

    def test_format_pipeline_l1_fail(self, ctx):
        from knowfeat.verification.sandbox import SandboxResult
        pr = PipelineResult(
            feature_name="test",
            l1_result=SandboxResult(success=False, error="timeout"),
            stage_reached="L1",
        )
        text = ToolContext._format_pipeline_result(pr)
        data = json.loads(text)
        assert data["L1_sandbox"]["passed"] is False
