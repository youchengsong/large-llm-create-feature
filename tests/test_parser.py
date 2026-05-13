"""tests/test_parser.py — 测试 LLM 输出解析：JSON 提取、截断修复、特征验证。"""

import json

import pytest

from src.llm.parser import extract_json_from_text, parse_llm_output, _fix_truncated_json, _validate_feature


# ========== extract_json_from_text ==========

class TestExtractJsonFromText:

    def test_plain_json_array(self):
        text = '[{"feature_name": "f1", "code": "x"}]'
        assert extract_json_from_text(text) == text

    def test_markdown_code_block_json(self):
        text = '这是一段说明\n```json\n[{"feature_name": "f1", "code": "x"}]\n```\n其他文字'
        result = extract_json_from_text(text)
        assert result.startswith('[')
        parsed = json.loads(result)
        assert parsed[0]["feature_name"] == "f1"

    def test_markdown_code_block_no_lang(self):
        text = '```\n{"feature_name": "f1", "code": "x"}\n```'
        result = extract_json_from_text(text)
        assert result.startswith('{')
        parsed = json.loads(result)
        assert parsed["feature_name"] == "f1"

    def test_json_with_surrounding_text(self):
        text = '以下是生成的特征：\n[{"feature_name": "f1", "code": "pass"}]\n希望对你有帮助！'
        result = extract_json_from_text(text)
        parsed = json.loads(result)
        assert len(parsed) == 1

    def test_single_object_extraction(self):
        text = '结果如下 {"feature_name": "f1", "code": "x"} 完毕'
        result = extract_json_from_text(text)
        parsed = json.loads(result)
        assert parsed["feature_name"] == "f1"

    def test_no_json_returns_stripped(self):
        text = "  没有 JSON 内容  "
        assert extract_json_from_text(text) == "没有 JSON 内容"

    def test_multiple_code_blocks_picks_json(self):
        text = '```python\nprint("hi")\n```\n```json\n[{"feature_name": "a", "code": "b"}]\n```'
        result = extract_json_from_text(text)
        assert result.startswith('[')
        parsed = json.loads(result)
        assert parsed[0]["feature_name"] == "a"

    def test_nested_json_in_code(self):
        code_with_json = '{"feature_name": "f1", "code": "d = {\\\"a\\\": 1}"}'
        text = f"```json\n{code_with_json}\n```"
        result = extract_json_from_text(text)
        assert result.startswith('{')


# ========== _fix_truncated_json ==========

class TestFixTruncatedJson:

    def test_complete_json_unchanged(self):
        text = '[{"a": 1}]'
        assert _fix_truncated_json(text) == text

    def test_missing_closing_brace(self):
        text = '[{"feature_name": "f1", "code": "x"'
        fixed = _fix_truncated_json(text)
        assert fixed.endswith('}]')

    def test_missing_closing_bracket(self):
        text = '[{"a": 1}'
        fixed = _fix_truncated_json(text)
        assert fixed.endswith(']')

    def test_truncated_in_string(self):
        text = '[{"feature_name": "f1", "code": "some long code'
        fixed = _fix_truncated_json(text)
        assert fixed.count('"') % 2 == 0
        assert '}' in fixed and ']' in fixed

    def test_no_truncation_extra_close(self):
        text = '[{"a": 1}]]'
        assert _fix_truncated_json(text) == text

    def test_escaped_quotes_not_confused(self):
        text = '{"key": "val\\"ue"}'
        assert _fix_truncated_json(text) == text


# ========== _validate_feature ==========

class TestValidateFeature:

    def test_valid_feature(self):
        assert _validate_feature({"feature_name": "f1", "code": "x", "explanation": "desc"})

    def test_minimal_valid(self):
        assert _validate_feature({"feature_name": "f1", "code": "x"})

    def test_missing_feature_name(self):
        assert not _validate_feature({"code": "x"})

    def test_missing_code(self):
        assert not _validate_feature({"feature_name": "f1"})

    def test_empty_dict(self):
        assert not _validate_feature({})

    def test_not_a_dict(self):
        assert not _validate_feature("string")
        assert not _validate_feature([1, 2])
        assert not _validate_feature(None)


# ========== parse_llm_output ==========

class TestParseLlmOutput:

    def test_empty_input(self):
        assert parse_llm_output("") == []
        assert parse_llm_output("   ") == []
        assert parse_llm_output(None) == []

    def test_valid_json_array(self):
        text = json.dumps([
            {"feature_name": "f1", "code": "df['a'] + 1", "explanation": "加一"},
            {"feature_name": "f2", "code": "df['b'] * 2"},
        ])
        result = parse_llm_output(text)
        assert len(result) == 2
        assert result[0]["feature_name"] == "f1"
        assert result[1]["feature_name"] == "f2"

    def test_single_object_wrapped(self):
        text = json.dumps({"feature_name": "f1", "code": "x"})
        result = parse_llm_output(text)
        assert len(result) == 1
        assert result[0]["feature_name"] == "f1"

    def test_markdown_wrapped(self):
        features = [{"feature_name": "f1", "code": "pass"}]
        text = f"以下是特征：\n```json\n{json.dumps(features, ensure_ascii=False)}\n```\n请查收。"
        result = parse_llm_output(text)
        assert len(result) == 1

    def test_filters_invalid_features(self):
        text = json.dumps([
            {"feature_name": "f1", "code": "x"},
            {"name": "f2"},
            {"feature_name": "f3", "code": "y"},
        ])
        result = parse_llm_output(text)
        assert len(result) == 2
        names = [f["feature_name"] for f in result]
        assert "f1" in names
        assert "f3" in names

    def test_truncated_json_recovery(self):
        full = [{"feature_name": "f1", "code": "x"}, {"feature_name": "f2", "code": "y"}]
        text = json.dumps(full)
        truncated = text[:-3]  # remove ]
        result = parse_llm_output(truncated)
        assert len(result) >= 1

    def test_garbage_input(self):
        assert parse_llm_output("这不是 JSON，只是普通文字。") == []

    def test_non_list_non_dict(self):
        assert parse_llm_output("42") == []
        assert parse_llm_output('"just a string"') == []

    def test_realistic_llm_output(self):
        text = """好的，我为您生成了以下特征：

```json
[
  {
    "feature_name": "tx_amount_zscore",
    "code": "import numpy as np\\ndef extract(open_df, tx_df, close_df):\\n    mean = tx_df['amount'].mean()\\n    std = tx_df['amount'].std()\\n    tx_df['tx_amount_zscore'] = (tx_df['amount'] - mean) / std\\n    return tx_df[['tx_id', 'tx_amount_zscore']]",
    "explanation": "交易金额的Z-score标准化"
  },
  {
    "feature_name": "tx_hour_sin",
    "code": "import numpy as np\\ndef extract(open_df, tx_df, close_df):\\n    tx_df['tx_hour_sin'] = np.sin(2 * np.pi * tx_df['hour'] / 24)\\n    return tx_df[['tx_id', 'tx_hour_sin']]",
    "explanation": "交易小时的正弦编码"
  }
]
```

这些特征分别从金额和时间两个维度捕捉异常模式。"""

        result = parse_llm_output(text)
        assert len(result) == 2
        assert result[0]["feature_name"] == "tx_amount_zscore"
        assert result[1]["feature_name"] == "tx_hour_sin"
        assert "explanation" in result[0]

    def test_all_invalid_features_returns_empty(self):
        text = json.dumps([{"name": "bad1"}, {"name": "bad2"}])
        assert parse_llm_output(text) == []

    def test_extra_fields_preserved(self):
        text = json.dumps([{"feature_name": "f1", "code": "x", "custom": 42}])
        result = parse_llm_output(text)
        assert result[0]["custom"] == 42

    def test_multiline_code_in_feature(self):
        feature = {
            "feature_name": "complex_feat",
            "code": "def extract(open_df, tx_df, close_df):\n    result = tx_df.copy()\n    result['feat'] = result['amount'].rolling(3).mean()\n    return result[['tx_id', 'feat']]"
        }
        text = json.dumps([feature])
        result = parse_llm_output(text)
        assert len(result) == 1
        assert "\n" in result[0]["code"]
