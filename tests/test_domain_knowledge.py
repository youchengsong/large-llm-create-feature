"""Tests for knowfeat.prompt.domain_knowledge."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from knowfeat.prompt.domain_knowledge import (
    RiskIndicator,
    DetectionRule,
    TransactionPattern,
    DomainKnowledge,
    load_risk_indicators,
    format_risk_indicators,
    load_detection_rules,
    format_detection_rules,
    get_transaction_patterns,
    format_transaction_patterns,
    load_expert_opinion,
    format_expert_opinion,
    load_stage1_risk_factors,
    format_stage1_risk_factors,
    load_domain_knowledge,
    format_domain_knowledge,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_risk_indicators():
    return [
        RiskIndicator(1, "交易行为异常类风险", "金额与频次异常风险", "单笔交易金额", "单笔交易金额远超账户日常阈值"),
        RiskIndicator(2, "交易行为异常类风险", "金额与频次异常风险", "累计交易金额", ""),
        RiskIndicator(8, "交易行为异常类风险", "时间与地域异常风险", "单笔交易时间", "非日常交易时段"),
        RiskIndicator(24, "主体与关联类风险", "账户关联风险", "多账户交叉转账", "闭环转账、多层嵌套转账"),
    ]


@pytest.fixture
def sample_detection_rules():
    return [
        DetectionRule(
            rule_id="RULE_001",
            description="四类账户短时间内大额取现异常",
            logic_expr="∀x [...] → suspicious_activity(x)",
            confidence="高",
            rule_type="behavior",
            pattern_category="数字人民币相关异常模式",
            python_hint="# 四类匿名钱包短时间大额取现",
        ),
        DetectionRule(
            rule_id="RULE_002",
            description="休眠期异常",
            logic_expr="∀x [...] → money_laundering_risk(x)",
            confidence="高",
            rule_type="behavior",
            pattern_category="数字人民币相关异常模式",
            python_hint="",
        ),
    ]


@pytest.fixture
def risk_indicators_xlsx(tmp_path):
    """Create a minimal risk indicators Excel file."""
    data = {
        "数字人民币交易风险": ["序号", 1, 2, 3, 8],
        "Unnamed: 1": ["维度", "交易行为异常类风险", None, None, None],
        "Unnamed: 2": ["风险类", "金额与频次异常风险", None, None, "时间与地域异常风险"],
        "Unnamed: 3": ["数据类", "单笔交易金额", "累计交易金额", "交易频率", "单笔交易时间"],
        "Unnamed: 4": ["方向", "单笔交易金额远超阈值", None, "短期内大额高频转账", "非日常交易时段"],
    }
    df = pd.DataFrame(data)
    fp = tmp_path / "risk.xlsx"
    df.to_excel(fp, index=False)
    return fp


@pytest.fixture
def detection_rules_xlsx(tmp_path):
    """Create a minimal detection rules Excel file."""
    data = {
        "规则编号": ["RULE_001", "RULE_002", None],
        "自然语言描述": ["四类账户大额取现", "休眠账户激活", None],
        "一阶逻辑表达式": ["∀x [...]", "∀x [...]", None],
        "置信度": ["高", "高", None],
        "规则类型": ["behavior", "behavior", None],
        "异常模式类别": ["数字人民币相关异常模式", "数字人民币相关异常模式", None],
    }
    df = pd.DataFrame(data)
    fp = tmp_path / "rules.xlsx"
    df.to_excel(fp, index=False)
    return fp


@pytest.fixture
def stage1_factors_dir(tmp_path):
    """Create sample stage1 risk factor JSON files."""
    d = tmp_path / "risk_factors"
    d.mkdir()
    factor = {
        "source": "湖南衡阳9.15案",
        "risk_factors": [
            {
                "pattern": "多层级虚拟币交易网络",
                "dimension": ["主体关联", "行为序列"],
                "observable_signals": ["短时间资金经过3层以上地址"],
                "data_hint": "fan_out_depth",
            }
        ],
    }
    (d / "case1.json").write_text(json.dumps(factor, ensure_ascii=False), encoding="utf-8")
    return d


# ── 1. Risk Indicators ─────────────────────────────────────────────────────

class TestLoadRiskIndicators:
    def test_load_from_xlsx(self, risk_indicators_xlsx):
        indicators = load_risk_indicators(risk_indicators_xlsx)
        assert len(indicators) == 4
        assert indicators[0].index == 1
        assert indicators[0].dimension == "交易行为异常类风险"
        assert indicators[0].risk_class == "金额与频次异常风险"
        assert indicators[0].data_field == "单笔交易金额"

    def test_merged_cell_propagation(self, risk_indicators_xlsx):
        indicators = load_risk_indicators(risk_indicators_xlsx)
        assert indicators[1].dimension == "交易行为异常类风险"
        assert indicators[1].risk_class == "金额与频次异常风险"
        assert indicators[2].dimension == "交易行为异常类风险"

    def test_new_risk_class(self, risk_indicators_xlsx):
        indicators = load_risk_indicators(risk_indicators_xlsx)
        assert indicators[3].risk_class == "时间与地域异常风险"

    def test_missing_file(self, tmp_path):
        result = load_risk_indicators(tmp_path / "nonexist.xlsx")
        assert result == []

    def test_empty_direction(self, risk_indicators_xlsx):
        indicators = load_risk_indicators(risk_indicators_xlsx)
        assert indicators[1].direction == ""


class TestFormatRiskIndicators:
    def test_basic_formatting(self, sample_risk_indicators):
        text = format_risk_indicators(sample_risk_indicators)
        assert "4 项" in text
        assert "金额与频次异常风险" in text
        assert "时间与地域异常风险" in text
        assert "账户关联风险" in text
        assert "交易行为异常类风险" in text
        assert "主体与关联类风险" in text

    def test_empty_list(self):
        text = format_risk_indicators([])
        assert "未加载" in text

    def test_direction_arrow(self, sample_risk_indicators):
        text = format_risk_indicators(sample_risk_indicators)
        assert "→ 单笔交易金额远超账户日常阈值" in text

    def test_no_arrow_for_empty_direction(self, sample_risk_indicators):
        text = format_risk_indicators(sample_risk_indicators)
        lines = text.split("\n")
        line_2 = [l for l in lines if "累计交易金额" in l][0]
        assert "→" not in line_2


# ── 2. Detection Rules ──────────────────────────────────────────────────────

class TestLoadDetectionRules:
    def test_load_from_xlsx(self, detection_rules_xlsx):
        rules = load_detection_rules(detection_rules_xlsx)
        assert len(rules) == 2
        assert rules[0].rule_id == "RULE_001"
        assert rules[0].description == "四类账户大额取现"
        assert rules[0].rule_type == "behavior"

    def test_skips_nan_rows(self, detection_rules_xlsx):
        rules = load_detection_rules(detection_rules_xlsx)
        assert all(r.rule_id.startswith("RULE_") for r in rules)

    def test_python_hint_attached(self, detection_rules_xlsx):
        rules = load_detection_rules(detection_rules_xlsx)
        assert rules[0].python_hint != ""
        assert "四类匿名钱包" in rules[0].python_hint

    def test_missing_file(self, tmp_path):
        result = load_detection_rules(tmp_path / "nonexist.xlsx")
        assert result == []


class TestFormatDetectionRules:
    def test_basic_formatting(self, sample_detection_rules):
        text = format_detection_rules(sample_detection_rules)
        assert "2 条" in text
        assert "RULE_001" in text
        assert "RULE_002" in text
        assert "Python 示例" in text

    def test_no_python_hint_block_when_empty(self, sample_detection_rules):
        text = format_detection_rules(sample_detection_rules)
        parts = text.split("### RULE_002")
        assert "Python 示例" not in parts[1]

    def test_empty_list(self):
        text = format_detection_rules([])
        assert "未加载" in text


# ── 3. Transaction Patterns ─────────────────────────────────────────────────

class TestTransactionPatterns:
    def test_count(self):
        patterns = get_transaction_patterns()
        assert len(patterns) == 8

    def test_categories(self):
        patterns = get_transaction_patterns()
        normal = [p for p in patterns if p.category == "normal"]
        abnormal = [p for p in patterns if p.category == "abnormal"]
        assert len(normal) == 3
        assert len(abnormal) == 5

    def test_names(self):
        patterns = get_transaction_patterns()
        names = {p.name for p in patterns}
        assert "日常消费" in names
        assert "薅羊毛（批量开户套利）" in names
        assert "线下跑分（ATM 取现）" in names
        assert "线上跑分（钱包互转）" in names
        assert "商户收款码跑分" in names

    def test_signals_not_empty(self):
        for p in get_transaction_patterns():
            assert len(p.signals) > 0

    def test_returns_copy(self):
        p1 = get_transaction_patterns()
        p2 = get_transaction_patterns()
        p1.pop()
        assert len(p2) == 8

    def test_format(self):
        text = format_transaction_patterns(get_transaction_patterns())
        assert "8 种" in text
        assert "正常交易模式（3 种）" in text
        assert "异常交易模式（5 种）" in text
        assert "日常消费" in text
        assert "薅羊毛" in text


# ── 4. Expert Opinion ───────────────────────────────────────────────────────

class TestExpertOpinion:
    def test_load_missing_file(self, tmp_path):
        result = load_expert_opinion(tmp_path / "nonexist.docx")
        assert result == ""

    def test_load_without_docx_module(self, tmp_path):
        fp = tmp_path / "test.docx"
        fp.write_bytes(b"dummy")
        with patch.dict("sys.modules", {"docx": None}):
            result = load_expert_opinion(fp)
            assert result == ""

    def test_format_empty(self):
        text = format_expert_opinion("")
        assert "未加载" in text

    def test_format_with_content(self):
        text = format_expert_opinion("专家说了一些内容")
        assert "银行风控专家意见" in text
        assert "专家说了一些内容" in text

    def test_load_real_docx(self):
        real_path = Path(__file__).parent.parent / "inputs" / "domain_knowledge" / "邮储专家访谈_数币洗钱模式.docx"
        if not real_path.exists():
            pytest.skip("Real docx file not available")
        text = load_expert_opinion(real_path)
        assert len(text) > 100
        assert "数字人民币" in text or "薅羊毛" in text or "跑分" in text


# ── 5. Stage 1 Risk Factors ────────────────────────────────────────────────

class TestStage1RiskFactors:
    def test_load(self, stage1_factors_dir):
        factors = load_stage1_risk_factors(stage1_factors_dir)
        assert len(factors) == 1
        assert factors[0]["source"] == "湖南衡阳9.15案"
        assert len(factors[0]["risk_factors"]) == 1

    def test_missing_dir(self, tmp_path):
        result = load_stage1_risk_factors(tmp_path / "nonexist")
        assert result == []

    def test_bad_json(self, tmp_path):
        d = tmp_path / "factors"
        d.mkdir()
        (d / "bad.json").write_text("{invalid json", encoding="utf-8")
        result = load_stage1_risk_factors(d)
        assert result == []

    def test_format(self, stage1_factors_dir):
        factors = load_stage1_risk_factors(stage1_factors_dir)
        text = format_stage1_risk_factors(factors)
        assert "湖南衡阳9.15案" in text
        assert "多层级虚拟币交易网络" in text
        assert "主体关联" in text

    def test_format_empty(self):
        text = format_stage1_risk_factors([])
        assert "未加载" in text


# ── 6. Integration: load_domain_knowledge ───────────────────────────────────

class TestLoadDomainKnowledge:
    def test_stage2(self, tmp_path, risk_indicators_xlsx, detection_rules_xlsx):
        from knowfeat.config import KnowFeatConfig, PathConfig
        paths = PathConfig()
        paths.risk_indicators_xlsx = risk_indicators_xlsx
        paths.detection_rules_xlsx = detection_rules_xlsx
        paths.expert_interview_docx = tmp_path / "nonexist.docx"
        paths.stage1_risk_factors_dir = tmp_path / "nonexist"

        cfg = KnowFeatConfig(paths=paths, stage="stage2")
        dk = load_domain_knowledge(cfg)

        assert len(dk.risk_indicators) == 4
        assert len(dk.detection_rules) == 2
        assert len(dk.transaction_patterns) == 8
        assert dk.expert_opinion == ""
        assert dk.stage1_risk_factors == []

    def test_stage1_loads_risk_factors(self, tmp_path, risk_indicators_xlsx,
                                       detection_rules_xlsx, stage1_factors_dir):
        from knowfeat.config import KnowFeatConfig, PathConfig
        paths = PathConfig()
        paths.risk_indicators_xlsx = risk_indicators_xlsx
        paths.detection_rules_xlsx = detection_rules_xlsx
        paths.expert_interview_docx = tmp_path / "nonexist.docx"
        paths.stage1_risk_factors_dir = stage1_factors_dir

        cfg = KnowFeatConfig(paths=paths, stage="stage1")
        dk = load_domain_knowledge(cfg)

        assert len(dk.stage1_risk_factors) == 1


class TestFormatDomainKnowledge:
    def test_stage2_contains_all_sections(self, sample_risk_indicators, sample_detection_rules):
        dk = DomainKnowledge(
            risk_indicators=sample_risk_indicators,
            detection_rules=sample_detection_rules,
            transaction_patterns=get_transaction_patterns(),
            expert_opinion="专家访谈内容",
        )
        text = format_domain_knowledge(dk, stage="stage2")
        assert "领域知识" in text
        assert "风险指标" in text
        assert "检测规则" in text
        assert "交易模式" in text
        assert "专家意见" in text

    def test_stage1_contains_risk_factors(self):
        dk = DomainKnowledge(
            stage1_risk_factors=[{
                "source": "test",
                "risk_factors": [{"pattern": "测试模式", "dimension": [], "observable_signals": [], "data_hint": ""}],
            }],
        )
        text = format_domain_knowledge(dk, stage="stage1")
        assert "领域知识" in text
        assert "测试模式" in text
        assert "风险指标" not in text


# ── 7. Real Data Integration ───────────────────────────────────────────────

class TestRealDataIntegration:
    """Tests against real project files (skip if unavailable)."""

    @pytest.fixture
    def project_root(self):
        root = Path(__file__).parent.parent
        return root

    def test_real_risk_indicators(self, project_root):
        fp = project_root / "inputs" / "domain_knowledge" / "风险分类表_38指标.xlsx"
        if not fp.exists():
            pytest.skip("Real risk indicators xlsx not available")
        indicators = load_risk_indicators(fp)
        assert len(indicators) == 38
        dims = {i.dimension for i in indicators}
        assert "交易行为异常类风险" in dims
        assert "主体与关联类风险" in dims
        assert "合规与场景类风险" in dims

    def test_real_detection_rules(self, project_root):
        fp = project_root / "inputs" / "domain_knowledge" / "检测规则_15条.xlsx"
        if not fp.exists():
            pytest.skip("Real detection rules xlsx not available")
        rules = load_detection_rules(fp)
        assert len(rules) == 15
        assert all(r.python_hint for r in rules)

    def test_real_expert_opinion(self, project_root):
        fp = project_root / "inputs" / "domain_knowledge" / "邮储专家访谈_数币洗钱模式.docx"
        if not fp.exists():
            pytest.skip("Real expert interview docx not available")
        text = load_expert_opinion(fp)
        assert len(text) > 500
        assert "薅羊毛" in text
