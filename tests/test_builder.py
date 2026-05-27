"""Tests for knowfeat/prompt/builder.py — System Prompt 组装。"""

import pytest

from knowfeat.config import KnowFeatConfig, load_config
from knowfeat.prompt.builder import (
    CrossDimExample,
    FailureRecord,
    ShapFeedback,
    build_full_prompt,
    build_round_instruction,
    build_system_prompt,
    _format_shap_feedback,
    _format_failure_memory,
    _format_cross_dim_examples,
    _format_verification_rules,
)
from knowfeat.prompt.domain_knowledge import (
    DomainKnowledge,
    RiskIndicator,
    DetectionRule,
    TransactionPattern,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg_stage2():
    return load_config(stage="stage2")


@pytest.fixture
def cfg_stage1():
    return load_config(stage="stage1")


@pytest.fixture
def empty_dk():
    return DomainKnowledge()


@pytest.fixture
def sample_dk():
    return DomainKnowledge(
        risk_indicators=[
            RiskIndicator(1, "交易行为异常类风险", "金额与频次异常风险", "amount", "大额交易"),
            RiskIndicator(2, "主体与关联类风险", "账户关联风险", "wallet_id", "关联账户"),
        ],
        detection_rules=[
            DetectionRule("RULE_001", "匿名钱包ATM取现", "exists(x)", "高", "behavior", "线下跑分",
                          "# python hint"),
        ],
        transaction_patterns=[
            TransactionPattern("日常消费", "normal", "正常消费", ["白天", "小额"]),
            TransactionPattern("线下跑分", "abnormal", "ATM取现", ["凌晨", "大额"]),
        ],
        expert_opinion="专家认为数字人民币洗钱有四种模式。",
    )


@pytest.fixture
def sample_dk_stage1():
    return DomainKnowledge(
        stage1_risk_factors=[
            {
                "source": "湖南衡阳案",
                "risk_factors": [
                    {
                        "pattern": "多层级虚拟币交易",
                        "dimension": ["行为序列"],
                        "observable_signals": ["3层以上地址"],
                        "data_hint": "fan_out_depth",
                    }
                ],
            }
        ],
    )


@pytest.fixture
def sample_shap():
    return ShapFeedback(
        dimension_contributions={
            "金额频次": 0.35,
            "时间地域": 0.25,
            "行为序列": 0.10,
            "主体关联": 0.30,
        },
        top_features=[
            {"name": "feat_amount_std", "importance": 0.15, "dimension": "金额频次"},
            {"name": "feat_night_ratio", "importance": 0.10, "dimension": "时间地域"},
        ],
        round_num=3,
    )


@pytest.fixture
def sample_failures():
    return [
        FailureRecord(
            feature_name="feat_bad_1",
            code="def feat_bad_1(open_df, tx_df, close_df):\n    return None",
            reason="L1 执行超时",
            round_num=1,
            dimension="金额频次",
        ),
        FailureRecord(
            feature_name="feat_bad_2",
            code="def feat_bad_2(open_df, tx_df, close_df):\n    ...",
            reason="IV 低于噪声阈值",
            round_num=2,
            dimension="行为序列",
        ),
    ]


@pytest.fixture
def sample_cross_dim():
    return [
        CrossDimExample(
            feature_name="feat_amount_std",
            source_dimension="金额频次",
            code="def feat_amount_std(open_df, tx_df, close_df):\n    ...",
            metrics={"iv": 0.05, "ks": 0.12, "delta_recall": 0.008},
            shap_rank=1,
        ),
    ]


# ── build_system_prompt tests ─────────────────────────────────────────────────

class TestBuildSystemPrompt:

    def test_stage2_contains_role(self, cfg_stage2, sample_dk):
        prompt = build_system_prompt(cfg_stage2, sample_dk)
        assert "数字人民币反洗钱特征工程专家" in prompt

    def test_stage1_contains_role(self, cfg_stage1, sample_dk_stage1):
        prompt = build_system_prompt(cfg_stage1, sample_dk_stage1)
        assert "加密货币反洗钱特征工程专家" in prompt

    def test_contains_domain_knowledge(self, cfg_stage2, sample_dk):
        prompt = build_system_prompt(cfg_stage2, sample_dk)
        assert "领域知识" in prompt
        assert "风险指标" in prompt
        assert "检测规则" in prompt
        assert "交易模式" in prompt
        assert "专家意见" in prompt

    def test_contains_verification_rules(self, cfg_stage2, sample_dk):
        prompt = build_system_prompt(cfg_stage2, sample_dk)
        assert "L1 沙盒执行" in prompt
        assert "L2 统计质量" in prompt
        assert "L3 模型效果" in prompt

    def test_with_schema_text(self, cfg_stage2, sample_dk):
        schema = "## 数据 Schema\n| col | type |\n"
        prompt = build_system_prompt(cfg_stage2, sample_dk, schema_text=schema)
        assert "数据 Schema" in prompt

    def test_empty_dk_still_produces_prompt(self, cfg_stage2, empty_dk):
        prompt = build_system_prompt(cfg_stage2, empty_dk)
        assert "数字人民币反洗钱" in prompt
        assert len(prompt) > 100

    def test_with_shap_feedback(self, cfg_stage2, sample_dk, sample_shap):
        prompt = build_system_prompt(cfg_stage2, sample_dk, shap_feedback=sample_shap)
        assert "SHAP 维度贡献反馈" in prompt
        assert "金额频次" in prompt
        assert "35.0%" in prompt

    def test_with_failure_memory(self, cfg_stage2, sample_dk, sample_failures):
        prompt = build_system_prompt(cfg_stage2, sample_dk, failure_records=sample_failures)
        assert "失败记忆" in prompt
        assert "feat_bad_1" in prompt
        assert "L1 执行超时" in prompt

    def test_with_cross_dim(self, cfg_stage2, sample_dk, sample_cross_dim):
        prompt = build_system_prompt(cfg_stage2, sample_dk, cross_dim_examples=sample_cross_dim)
        assert "跨维度高分特征" in prompt
        assert "feat_amount_std" in prompt
        assert "金额频次" in prompt

    def test_all_dynamic_sections(self, cfg_stage2, sample_dk, sample_shap,
                                   sample_failures, sample_cross_dim):
        prompt = build_system_prompt(
            cfg_stage2, sample_dk,
            shap_feedback=sample_shap,
            failure_records=sample_failures,
            cross_dim_examples=sample_cross_dim,
        )
        assert "SHAP 维度贡献反馈" in prompt
        assert "失败记忆" in prompt
        assert "跨维度高分特征" in prompt

    def test_no_dynamic_sections_when_none(self, cfg_stage2, sample_dk):
        prompt = build_system_prompt(cfg_stage2, sample_dk)
        assert "SHAP 维度贡献反馈" not in prompt
        assert "失败记忆" not in prompt
        assert "跨维度高分特征" not in prompt

    def test_returns_string(self, cfg_stage2, sample_dk):
        prompt = build_system_prompt(cfg_stage2, sample_dk)
        assert isinstance(prompt, str)

    def test_stage1_with_risk_factors(self, cfg_stage1, sample_dk_stage1):
        prompt = build_system_prompt(cfg_stage1, sample_dk_stage1)
        assert "加密货币洗钱风险因子" in prompt
        assert "湖南衡阳案" in prompt


# ── build_round_instruction tests ─────────────────────────────────────────────

class TestBuildRoundInstruction:

    def test_basic_instruction(self):
        msg = build_round_instruction(
            round_num=1, dimension="金额频次", budget=5,
        )
        assert "第 1/10 轮" in msg
        assert "金额频次" in msg
        assert "5 个特征" in msg

    def test_with_progress(self):
        msg = build_round_instruction(
            round_num=5, dimension="时间地域", budget=3,
            n_accepted=12, n_candidates=8,
        )
        assert "12/30" in msg
        assert "8" in msg

    def test_max_features_warning(self):
        msg = build_round_instruction(
            round_num=8, dimension="行为序列", budget=2,
            n_accepted=30, max_features=30,
        )
        assert "已达到特征上限" in msg

    def test_custom_max_rounds(self):
        msg = build_round_instruction(
            round_num=3, dimension="主体关联", budget=4,
            max_rounds=15,
        )
        assert "第 3/15 轮" in msg

    def test_custom_max_features(self):
        msg = build_round_instruction(
            round_num=1, dimension="金额频次", budget=5,
            max_features=50,
        )
        assert "0/50" in msg

    def test_contains_guidance(self):
        msg = build_round_instruction(
            round_num=1, dimension="金额频次", budget=5,
        )
        assert "领域知识" in msg
        assert "冗余度" in msg
        assert "失败记忆" in msg


# ── build_full_prompt tests ───────────────────────────────────────────────────

class TestBuildFullPrompt:

    def test_returns_tuple(self, cfg_stage2, sample_dk):
        result = build_full_prompt(
            cfg_stage2, sample_dk,
            round_num=1, dimension="金额频次", budget=5,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_and_user(self, cfg_stage2, sample_dk):
        system, user = build_full_prompt(
            cfg_stage2, sample_dk,
            round_num=2, dimension="时间地域", budget=3,
            n_accepted=5, n_candidates=10,
        )
        assert "数字人民币反洗钱" in system
        assert "第 2/10 轮" in user
        assert "时间地域" in user

    def test_with_all_dynamic(self, cfg_stage2, sample_dk, sample_shap,
                               sample_failures, sample_cross_dim):
        system, user = build_full_prompt(
            cfg_stage2, sample_dk,
            shap_feedback=sample_shap,
            failure_records=sample_failures,
            cross_dim_examples=sample_cross_dim,
            round_num=4, dimension="行为序列", budget=2,
        )
        assert "SHAP" in system
        assert "失败记忆" in system
        assert "跨维度" in system
        assert "第 4/10 轮" in user


# ── _format helpers tests ─────────────────────────────────────────────────────

class TestFormatShapFeedback:

    def test_empty_contributions(self):
        fb = ShapFeedback()
        text = _format_shap_feedback(fb)
        assert "暂无 SHAP 反馈数据" in text

    def test_with_contributions(self, sample_shap):
        text = _format_shap_feedback(sample_shap)
        assert "金额频次" in text
        assert "35.0%" in text
        assert "✦ 强" in text

    def test_medium_status(self):
        fb = ShapFeedback(dimension_contributions={"测试维度": 0.20})
        text = _format_shap_feedback(fb)
        assert "● 中" in text

    def test_weak_status(self):
        fb = ShapFeedback(dimension_contributions={"测试维度": 0.05})
        text = _format_shap_feedback(fb)
        assert "○ 弱" in text

    def test_top_features_shown(self, sample_shap):
        text = _format_shap_feedback(sample_shap)
        assert "feat_amount_std" in text
        assert "第 3 轮" in text

    def test_top_features_limited_to_5(self):
        fb = ShapFeedback(
            dimension_contributions={"d": 0.5},
            top_features=[{"name": f"f{i}", "importance": 0.1, "dimension": "d"} for i in range(10)],
        )
        text = _format_shap_feedback(fb)
        assert "f0" in text
        assert "f4" in text
        assert "f5" not in text


class TestFormatFailureMemory:

    def test_empty_records(self):
        text = _format_failure_memory([])
        assert text == ""

    def test_with_records(self, sample_failures):
        text = _format_failure_memory(sample_failures)
        assert "失败记忆" in text
        assert "feat_bad_1" in text
        assert "feat_bad_2" in text
        assert "L1 执行超时" in text
        assert "IV 低于噪声阈值" in text

    def test_code_preview_truncation(self):
        long_code = "x" * 500
        rec = FailureRecord("f", long_code, "error")
        text = _format_failure_memory([rec])
        assert "..." in text

    def test_short_code_no_truncation(self):
        rec = FailureRecord("f", "short code", "error")
        text = _format_failure_memory([rec])
        assert "short code" in text
        assert "..." not in text

    def test_dimension_shown(self, sample_failures):
        text = _format_failure_memory(sample_failures)
        assert "金额频次" in text


class TestFormatCrossDimExamples:

    def test_empty_examples(self):
        text = _format_cross_dim_examples([])
        assert text == ""

    def test_with_examples(self, sample_cross_dim):
        text = _format_cross_dim_examples(sample_cross_dim)
        assert "跨维度高分特征" in text
        assert "feat_amount_std" in text
        assert "金额频次" in text
        assert "SHAP 排名: #1" in text

    def test_metrics_shown(self, sample_cross_dim):
        text = _format_cross_dim_examples(sample_cross_dim)
        assert "iv=" in text
        assert "delta_recall=" in text

    def test_no_shap_rank_zero(self):
        ex = CrossDimExample("f", "d", "code", {}, shap_rank=0)
        text = _format_cross_dim_examples([ex])
        assert "SHAP 排名" not in text


class TestFormatVerificationRules:

    def test_contains_all_levels(self, cfg_stage2):
        text = _format_verification_rules(cfg_stage2)
        assert "L1 沙盒执行" in text
        assert "L2 统计质量" in text
        assert "L3 模型效果" in text

    def test_contains_thresholds(self, cfg_stage2):
        text = _format_verification_rules(cfg_stage2)
        assert "30s" in text
        assert "0.7" in text
        assert "90%" in text
        assert "5-fold" in text


# ── Data structure tests ──────────────────────────────────────────────────────

class TestDataStructures:

    def test_shap_feedback_defaults(self):
        fb = ShapFeedback()
        assert fb.dimension_contributions == {}
        assert fb.top_features == []
        assert fb.round_num == 0

    def test_failure_record_defaults(self):
        rec = FailureRecord("name", "code", "reason")
        assert rec.round_num == 0
        assert rec.dimension == ""

    def test_cross_dim_example_defaults(self):
        ex = CrossDimExample("name", "dim", "code")
        assert ex.metrics == {}
        assert ex.shap_rank == 0
