"""Tests for knowfeat/extractor/key_clue_extractor.py"""

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowfeat.extractor.key_clue_extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    VALID_DIMENSIONS,
    ExtractionResult,
    RiskFactor,
    _extract_json_from_response,
    _parse_risk_factors,
    _validate_dimension,
    batch_extract,
    extract_from_file,
    extract_from_text,
    load_document,
    run_extraction,
    save_results,
)


# ── RiskFactor ────────────────────────────────────────────────────────────

class TestRiskFactor:
    def test_to_dict(self):
        rf = RiskFactor(
            pattern="多层转账",
            dimension=["行为序列", "主体关联"],
            observable_signals=["3层以上地址", "24h内完成"],
            data_hint="fan_out_depth",
        )
        d = rf.to_dict()
        assert d["pattern"] == "多层转账"
        assert d["dimension"] == ["行为序列", "主体关联"]
        assert len(d["observable_signals"]) == 2
        assert d["data_hint"] == "fan_out_depth"


# ── ExtractionResult ──────────────────────────────────────────────────────

class TestExtractionResult:
    def test_success_true(self):
        r = ExtractionResult(source="test")
        assert r.success is True

    def test_success_false_with_error(self):
        r = ExtractionResult(source="test", error="something went wrong")
        assert r.success is False

    def test_to_dict(self):
        rf = RiskFactor("p", ["金额频次"], ["s1"], "hint")
        r = ExtractionResult(source="src", risk_factors=[rf])
        d = r.to_dict()
        assert d["source"] == "src"
        assert len(d["risk_factors"]) == 1
        assert d["risk_factors"][0]["pattern"] == "p"

    def test_to_dict_empty(self):
        r = ExtractionResult(source="src")
        d = r.to_dict()
        assert d["risk_factors"] == []


# ── _extract_json_from_response ───────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        text = '{"source": "test", "risk_factors": []}'
        data = _extract_json_from_response(text)
        assert data["source"] == "test"

    def test_json_in_code_fence(self):
        text = '```json\n{"source": "test", "risk_factors": []}\n```'
        data = _extract_json_from_response(text)
        assert data["source"] == "test"

    def test_json_in_plain_fence(self):
        text = '```\n{"source": "test", "risk_factors": []}\n```'
        data = _extract_json_from_response(text)
        assert data["source"] == "test"

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"source": "x", "risk_factors": []}\nDone.'
        data = _extract_json_from_response(text)
        assert data["source"] == "x"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="未找到 JSON"):
            _extract_json_from_response("no json here")

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json_from_response("{invalid json}")


# ── _validate_dimension ───────────────────────────────────────────────────

class TestValidateDimension:
    def test_valid_single(self):
        assert _validate_dimension(["金额频次"]) == ["金额频次"]

    def test_valid_multi(self):
        result = _validate_dimension(["金额频次", "行为序列"])
        assert result == ["金额频次", "行为序列"]

    def test_filter_invalid(self):
        result = _validate_dimension(["金额频次", "无效维度"])
        assert result == ["金额频次"]

    def test_all_invalid_defaults(self):
        result = _validate_dimension(["无效1", "无效2"])
        assert result == ["行为序列"]

    def test_empty_defaults(self):
        result = _validate_dimension([])
        assert result == ["行为序列"]


# ── _parse_risk_factors ───────────────────────────────────────────────────

class TestParseRiskFactors:
    def test_normal_parse(self):
        data = {
            "risk_factors": [
                {
                    "pattern": "多层虚拟币转账",
                    "dimension": ["行为序列"],
                    "observable_signals": ["3层以上地址"],
                    "data_hint": "address_depth",
                }
            ]
        }
        factors = _parse_risk_factors(data)
        assert len(factors) == 1
        assert factors[0].pattern == "多层虚拟币转账"
        assert factors[0].dimension == ["行为序列"]

    def test_string_dimension_converted(self):
        data = {
            "risk_factors": [
                {
                    "pattern": "test",
                    "dimension": "金额频次",
                    "observable_signals": [],
                    "data_hint": "",
                }
            ]
        }
        factors = _parse_risk_factors(data)
        assert factors[0].dimension == ["金额频次"]

    def test_string_signals_converted(self):
        data = {
            "risk_factors": [
                {
                    "pattern": "test",
                    "dimension": ["时间地域"],
                    "observable_signals": "凌晨高频",
                    "data_hint": "",
                }
            ]
        }
        factors = _parse_risk_factors(data)
        assert factors[0].observable_signals == ["凌晨高频"]

    def test_skip_empty_pattern(self):
        data = {
            "risk_factors": [
                {"pattern": "", "dimension": ["金额频次"], "observable_signals": [], "data_hint": ""},
                {"pattern": "valid", "dimension": ["金额频次"], "observable_signals": [], "data_hint": ""},
            ]
        }
        factors = _parse_risk_factors(data)
        assert len(factors) == 1

    def test_skip_non_dict_items(self):
        data = {"risk_factors": ["not a dict", {"pattern": "ok", "dimension": ["金额频次"], "observable_signals": [], "data_hint": ""}]}
        factors = _parse_risk_factors(data)
        assert len(factors) == 1

    def test_missing_risk_factors_key(self):
        data = {"source": "test"}
        factors = _parse_risk_factors(data)
        assert factors == []

    def test_risk_factors_not_list(self):
        data = {"risk_factors": "not a list"}
        factors = _parse_risk_factors(data)
        assert factors == []

    def test_multiple_factors(self):
        data = {
            "risk_factors": [
                {"pattern": "p1", "dimension": ["金额频次"], "observable_signals": ["s1"], "data_hint": "h1"},
                {"pattern": "p2", "dimension": ["时间地域", "行为序列"], "observable_signals": ["s2", "s3"], "data_hint": "h2"},
            ]
        }
        factors = _parse_risk_factors(data)
        assert len(factors) == 2
        assert factors[1].dimension == ["时间地域", "行为序列"]

    def test_filter_blank_signals(self):
        data = {
            "risk_factors": [
                {"pattern": "p", "dimension": ["金额频次"], "observable_signals": ["ok", "", "  "], "data_hint": ""},
            ]
        }
        factors = _parse_risk_factors(data)
        assert factors[0].observable_signals == ["ok"]


# ── load_document ─────────────────────────────────────────────────────────

class TestLoadDocument:
    def test_load_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert load_document(f) == "hello world"

    def test_load_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# heading\ncontent", encoding="utf-8")
        assert "heading" in load_document(f)

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("data", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_document(f)


# ── extract_from_text ─────────────────────────────────────────────────────

class TestExtractFromText:
    def _make_mock_client(self, response_json: dict) -> MagicMock:
        client = MagicMock(spec=["chat"])
        resp = MagicMock()
        resp.content = json.dumps(response_json, ensure_ascii=False)
        client.chat.return_value = resp
        return client

    def test_basic_extraction(self):
        response_data = {
            "source": "测试案例",
            "risk_factors": [
                {
                    "pattern": "批量转账",
                    "dimension": ["行为序列"],
                    "observable_signals": ["5分钟内10笔以上转账"],
                    "data_hint": "tx_count, timestamp",
                }
            ],
        }
        client = self._make_mock_client(response_data)
        result = extract_from_text(client, "一些文档内容", "测试案例")

        assert result.success
        assert result.source == "测试案例"
        assert len(result.risk_factors) == 1
        assert result.risk_factors[0].pattern == "批量转账"

    def test_empty_text_returns_error(self):
        client = MagicMock()
        result = extract_from_text(client, "", "empty_doc")
        assert not result.success
        assert "文档内容为空" in result.error

    def test_llm_call_failure(self):
        client = MagicMock(spec=["chat"])
        client.chat.side_effect = RuntimeError("connection failed")
        result = extract_from_text(client, "some text", "fail_doc")
        assert not result.success
        assert "LLM 调用失败" in result.error

    def test_invalid_json_response(self):
        client = MagicMock(spec=["chat"])
        resp = MagicMock()
        resp.content = "This is not JSON at all."
        client.chat.return_value = resp
        result = extract_from_text(client, "some text", "bad_json")
        assert not result.success
        assert "JSON 解析失败" in result.error

    def test_source_fallback_to_name(self):
        response_data = {"risk_factors": [{"pattern": "p", "dimension": ["金额频次"], "observable_signals": [], "data_hint": ""}]}
        client = self._make_mock_client(response_data)
        result = extract_from_text(client, "text", "fallback_name")
        assert result.source == "fallback_name"

    def test_temperature_and_max_tokens_passed(self):
        response_data = {"source": "t", "risk_factors": []}
        client = self._make_mock_client(response_data)
        extract_from_text(client, "text", "test", temperature=0.1, max_tokens=2000)
        call_kwargs = client.chat.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.1
        assert call_kwargs.kwargs.get("max_tokens") == 2000

    def test_system_prompt_contains_dimensions(self):
        for dim in VALID_DIMENSIONS:
            assert dim in EXTRACTION_SYSTEM_PROMPT


# ── extract_from_file ─────────────────────────────────────────────────────

class TestExtractFromFile:
    def test_file_not_exists(self):
        client = MagicMock()
        result = extract_from_file(client, Path("/nonexistent/file.txt"))
        assert not result.success
        assert "文件不存在" in result.error

    def test_load_and_extract(self, tmp_path):
        f = tmp_path / "case.txt"
        f.write_text("犯罪团伙利用虚拟币洗钱", encoding="utf-8")

        response_data = {
            "source": "case",
            "risk_factors": [
                {"pattern": "虚拟币洗钱", "dimension": ["行为序列"], "observable_signals": ["链上转账"], "data_hint": "tx_hash"},
            ],
        }
        client = MagicMock(spec=["chat"])
        resp = MagicMock()
        resp.content = json.dumps(response_data, ensure_ascii=False)
        client.chat.return_value = resp

        result = extract_from_file(client, f)
        assert result.success
        assert len(result.risk_factors) == 1

    def test_custom_source_name(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")

        response_data = {"source": "custom", "risk_factors": []}
        client = MagicMock(spec=["chat"])
        resp = MagicMock()
        resp.content = json.dumps(response_data, ensure_ascii=False)
        client.chat.return_value = resp

        result = extract_from_file(client, f, source_name="自定义名称")
        assert result.source == "custom"

    def test_unsupported_file_format(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("data", encoding="utf-8")
        client = MagicMock()
        result = extract_from_file(client, f)
        assert not result.success
        assert "文件加载失败" in result.error


# ── batch_extract ─────────────────────────────────────────────────────────

class TestBatchExtract:
    def test_batch_processing(self, tmp_path):
        for i in range(3):
            (tmp_path / f"doc{i}.txt").write_text(f"文档内容{i}", encoding="utf-8")

        response_data = {
            "source": "batch",
            "risk_factors": [{"pattern": "p", "dimension": ["金额频次"], "observable_signals": [], "data_hint": ""}],
        }
        client = MagicMock(spec=["chat"])
        resp = MagicMock()
        resp.content = json.dumps(response_data, ensure_ascii=False)
        client.chat.return_value = resp

        paths = sorted(tmp_path.glob("*.txt"))
        results = batch_extract(client, paths)
        assert len(results) == 3
        assert all(r.success for r in results)
        assert client.chat.call_count == 3

    def test_batch_with_source_names(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")

        response_data = {"source": "named", "risk_factors": []}
        client = MagicMock(spec=["chat"])
        resp = MagicMock()
        resp.content = json.dumps(response_data, ensure_ascii=False)
        client.chat.return_value = resp

        results = batch_extract(client, [f], source_names=["自定义"])
        assert len(results) == 1

    def test_batch_mismatched_names_raises(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content", encoding="utf-8")
        client = MagicMock()
        with pytest.raises(ValueError, match="长度必须"):
            batch_extract(client, [f], source_names=["a", "b"])

    def test_batch_partial_failure(self, tmp_path):
        f1 = tmp_path / "good.txt"
        f1.write_text("content", encoding="utf-8")
        f2 = Path("/nonexistent.txt")

        response_data = {"source": "good", "risk_factors": []}
        client = MagicMock(spec=["chat"])
        resp = MagicMock()
        resp.content = json.dumps(response_data, ensure_ascii=False)
        client.chat.return_value = resp

        results = batch_extract(client, [f1, f2])
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success


# ── save_results ──────────────────────────────────────────────────────────

class TestSaveResults:
    def test_save_successful_results(self, tmp_path):
        rf = RiskFactor("模式A", ["金额频次"], ["信号1"], "hint")
        results = [
            ExtractionResult(source="案例A", risk_factors=[rf]),
            ExtractionResult(source="案例B", risk_factors=[rf, rf]),
        ]

        saved = save_results(results, tmp_path / "output")
        assert len(saved) == 2
        assert (tmp_path / "output").is_dir()

        data = json.loads(saved[0].read_text(encoding="utf-8"))
        assert data["source"] == "案例A"
        assert len(data["risk_factors"]) == 1

    def test_skip_failed_results(self, tmp_path):
        results = [
            ExtractionResult(source="fail", error="some error"),
        ]
        saved = save_results(results, tmp_path / "output")
        assert len(saved) == 0

    def test_skip_empty_factors(self, tmp_path):
        results = [
            ExtractionResult(source="empty", risk_factors=[]),
        ]
        saved = save_results(results, tmp_path / "output")
        assert len(saved) == 0

    def test_safe_filename(self, tmp_path):
        rf = RiskFactor("p", ["行为序列"], ["s"], "h")
        results = [
            ExtractionResult(source="湖南/衡阳 案例", risk_factors=[rf]),
        ]
        saved = save_results(results, tmp_path / "output")
        assert len(saved) == 1
        assert "/" not in saved[0].name
        assert " " not in saved[0].name

    def test_creates_output_dir(self, tmp_path):
        rf = RiskFactor("p", ["金额频次"], ["s"], "h")
        results = [ExtractionResult(source="test", risk_factors=[rf])]
        out_dir = tmp_path / "nested" / "dir"
        saved = save_results(results, out_dir)
        assert len(saved) == 1
        assert out_dir.is_dir()

    def test_json_is_valid(self, tmp_path):
        rf = RiskFactor("多层转账洗钱", ["行为序列", "主体关联"], ["3层+", "24h"], "depth")
        results = [ExtractionResult(source="测试", risk_factors=[rf])]
        saved = save_results(results, tmp_path)
        data = json.loads(saved[0].read_text(encoding="utf-8"))
        assert data["source"] == "测试"
        assert data["risk_factors"][0]["dimension"] == ["行为序列", "主体关联"]


# ── run_extraction ────────────────────────────────────────────────────────

class TestRunExtraction:
    def test_empty_dir(self, tmp_path):
        from knowfeat.config import KnowFeatConfig, PathConfig

        cfg = KnowFeatConfig()
        cfg.paths.stage1_knowledge_dir = tmp_path / "empty_input"
        cfg.paths.stage1_risk_factors_dir = tmp_path / "output"
        (tmp_path / "empty_input").mkdir()

        results = run_extraction(cfg, input_dir=tmp_path / "empty_input")
        assert results == []

    @patch("knowfeat.extractor.key_clue_extractor.LLMClient")
    def test_end_to_end(self, MockClient, tmp_path):
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        (in_dir / "case1.txt").write_text("案例1内容", encoding="utf-8")
        (in_dir / "case2.md").write_text("# 案例2\n内容", encoding="utf-8")

        call_count = [0]
        def make_response(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.content = json.dumps({
                "source": f"extracted_{call_count[0]}",
                "risk_factors": [
                    {"pattern": f"test_{call_count[0]}", "dimension": ["金额频次"], "observable_signals": ["s"], "data_hint": "h"},
                ],
            }, ensure_ascii=False)
            return resp

        mock_instance = MagicMock()
        mock_instance.chat.side_effect = make_response
        MockClient.return_value = mock_instance

        from knowfeat.config import KnowFeatConfig

        cfg = KnowFeatConfig()
        results = run_extraction(cfg, input_dir=in_dir, output_dir=out_dir)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert len(list(out_dir.glob("*.json"))) == 2

    @patch("knowfeat.extractor.key_clue_extractor.LLMClient")
    def test_custom_patterns(self, MockClient, tmp_path):
        in_dir = tmp_path / "input"
        in_dir.mkdir()

        (in_dir / "case.txt").write_text("content", encoding="utf-8")
        (in_dir / "skip.md").write_text("skip", encoding="utf-8")

        response_data = {"source": "t", "risk_factors": [{"pattern": "p", "dimension": ["金额频次"], "observable_signals": [], "data_hint": ""}]}
        mock_resp = MagicMock()
        mock_resp.content = json.dumps(response_data, ensure_ascii=False)
        mock_instance = MagicMock()
        mock_instance.chat.return_value = mock_resp
        MockClient.return_value = mock_instance

        from knowfeat.config import KnowFeatConfig

        cfg = KnowFeatConfig()
        results = run_extraction(cfg, input_dir=in_dir, output_dir=tmp_path / "out", file_patterns=["*.txt"])
        assert len(results) == 1


# ── Integration: output compatible with domain_knowledge loader ───────────

class TestOutputCompatibility:
    def test_output_matches_stage1_schema(self, tmp_path):
        """Verify saved JSON matches the schema expected by load_stage1_risk_factors."""
        rf = RiskFactor(
            pattern="多层虚拟币交易网络分散资金流向",
            dimension=["主体关联", "行为序列"],
            observable_signals=["短时间内资金经过3层以上地址", "每层地址存活时间<24h"],
            data_hint="可从 fan_out_depth, address_lifetime 构造",
        )
        result = ExtractionResult(source="湖南衡阳9.15案", risk_factors=[rf])
        saved = save_results([result], tmp_path)

        data = json.loads(saved[0].read_text(encoding="utf-8"))

        assert "source" in data
        assert "risk_factors" in data
        assert isinstance(data["risk_factors"], list)

        factor = data["risk_factors"][0]
        assert "pattern" in factor
        assert "dimension" in factor
        assert "observable_signals" in factor
        assert "data_hint" in factor
        assert isinstance(factor["dimension"], list)
        assert isinstance(factor["observable_signals"], list)

    def test_compatible_with_load_stage1_risk_factors(self, tmp_path):
        """End-to-end: save → load via domain_knowledge.load_stage1_risk_factors."""
        rf = RiskFactor("跑分模式", ["行为序列"], ["批量小额转账"], "amount, frequency")
        result = ExtractionResult(source="测试来源", risk_factors=[rf])
        save_results([result], tmp_path)

        from knowfeat.prompt.domain_knowledge import load_stage1_risk_factors

        factors = load_stage1_risk_factors(tmp_path)
        assert len(factors) == 1
        assert factors[0]["source"] == "测试来源"
        assert len(factors[0]["risk_factors"]) == 1
        assert factors[0]["risk_factors"][0]["pattern"] == "跑分模式"
