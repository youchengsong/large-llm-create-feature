"""LLM 输出解析模块：从 LLM 原始文本中提取特征 JSON 数组。"""

import json
import re


def extract_json_from_text(text: str) -> str:
    """从文本中提取 JSON 内容。

    处理策略（按优先级）：
    1. markdown 代码块中的 JSON
    2. 裸 JSON 数组
    3. 裸 JSON 对象（包装为数组）
    """
    md_pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)
    matches = md_pattern.findall(text)

    if matches:
        for match in matches:
            stripped = match.strip()
            if stripped.startswith('[') or stripped.startswith('{'):
                return stripped

    array_pattern = re.compile(r'\[[\s\S]*\]')
    m = array_pattern.search(text)
    if m:
        return m.group(0)

    obj_pattern = re.compile(r'\{[\s\S]*\}')
    m = obj_pattern.search(text)
    if m:
        return m.group(0)

    return text.strip()


_REQUIRED_KEYS = {'feature_name', 'code'}


def _fix_truncated_json(text: str) -> str:
    """尝试修复截断的 JSON（补全缺失的括号/引号）。"""
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    if open_braces <= 0 and open_brackets <= 0:
        return text

    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        text += '"'

    text += '}' * max(0, open_braces)
    text += ']' * max(0, open_brackets)
    return text


def _validate_feature(item: dict) -> bool:
    """检查单个特征 dict 是否包含必需字段。"""
    if not isinstance(item, dict):
        return False
    return _REQUIRED_KEYS.issubset(item.keys())


def parse_llm_output(raw_text: str) -> list[dict]:
    """从 LLM 原始输出中解析特征列表。

    返回包含有效特征的 dict 列表。无法解析时返回空列表。
    """
    if not raw_text or not raw_text.strip():
        return []

    json_str = extract_json_from_text(raw_text)

    for attempt_text in [json_str, _fix_truncated_json(json_str)]:
        try:
            data = json.loads(attempt_text)
            break
        except json.JSONDecodeError:
            continue
    else:
        return []

    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        return []

    return [item for item in data if _validate_feature(item)]
