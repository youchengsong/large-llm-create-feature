"""Key-clue Extractor: 非结构化文档 → 结构化风险因子 JSON。

独立的 LLM 预处理步骤，将新闻案例、裁判文书、监管报告等
非结构化领域文档转化为结构化风险因子列表，供 Feature Agent prompt 注入。

输出 Schema:
{
  "source": "来源名称",
  "risk_factors": [
    {
      "pattern": "自然语言描述的洗钱模式",
      "dimension": ["维度1", "维度2"],
      "observable_signals": ["可观测信号1", "可观测信号2"],
      "data_hint": "可能用到的原始数据字段提示"
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knowfeat.config import KnowFeatConfig
from knowfeat.llm.client import LLMClient

logger = logging.getLogger(__name__)

VALID_DIMENSIONS = {"金额频次", "时间地域", "行为序列", "主体关联"}

EXTRACTION_SYSTEM_PROMPT = """\
你是一位反洗钱领域专家，擅长从非结构化文档中提取结构化的洗钱风险因子。

你的任务是阅读用户提供的文档，提取所有可用于特征工程的洗钱风险因子。

## 输出要求

请严格输出 JSON 格式（不要用 markdown 代码块包裹），结构如下：
{
  "source": "文档来源名称",
  "risk_factors": [
    {
      "pattern": "自然语言描述的洗钱模式（一句话）",
      "dimension": ["维度1"],
      "observable_signals": ["可量化的数据信号1", "可量化的数据信号2"],
      "data_hint": "可能用到的区块链/交易原始数据字段提示"
    }
  ]
}

## 维度选择（必须从以下四个中选，可多选）

- 金额频次: 交易金额分布、频率异常、金额阈值规避
- 时间地域: 时间模式（夜间、周期性）、地域分布异常
- 行为序列: 交易模式序列、行为链、资金流向路径
- 主体关联: 账户关系网络、批量控制、多层级结构

## 提取原则

1. 每个 risk_factor 描述一种独立的洗钱模式或行为特征
2. pattern 应简洁明确，概括核心洗钱手法
3. observable_signals 必须是可量化或可从数据中检测的信号（如"短时间内资金经过3层以上地址"）
4. data_hint 提示可能用到的原始数据字段（如 amount, timestamp, address 等）
5. 从文档中尽可能多地提取不同维度的风险因子
6. 如果文档内容与洗钱无关或无法提取有效风险因子，返回空的 risk_factors 列表
"""

EXTRACTION_USER_TEMPLATE = """\
请从以下文档中提取洗钱风险因子：

## 文档来源: {source_name}

{document_text}
"""


@dataclass
class RiskFactor:
    """单个结构化风险因子。"""
    pattern: str
    dimension: list[str]
    observable_signals: list[str]
    data_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "dimension": self.dimension,
            "observable_signals": self.observable_signals,
            "data_hint": self.data_hint,
        }


@dataclass
class ExtractionResult:
    """单个文档的提取结果。"""
    source: str
    risk_factors: list[RiskFactor] = field(default_factory=list)
    raw_response: str = ""
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "risk_factors": [rf.to_dict() for rf in self.risk_factors],
        }


def load_document(file_path: Path) -> str:
    """加载文档文本内容，支持 txt/md/docx。"""
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8")

    if suffix == ".docx":
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx 未安装，无法加载 .docx 文件")

        doc = docx.Document(str(file_path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    raise ValueError(f"不支持的文件格式: {suffix}（支持 .txt, .md, .docx）")


def _extract_json_from_response(text: str) -> dict[str, Any]:
    """从 LLM 响应中提取 JSON 对象。"""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start == -1 or brace_end == -1:
        raise ValueError("响应中未找到 JSON 对象")

    json_str = text[brace_start : brace_end + 1]
    return json.loads(json_str)


def _validate_dimension(dims: list[str]) -> list[str]:
    """验证并过滤维度标签，仅保留有效维度。"""
    valid = [d for d in dims if d in VALID_DIMENSIONS]
    if not valid:
        return ["行为序列"]
    return valid


def _parse_risk_factors(data: dict[str, Any]) -> list[RiskFactor]:
    """解析 JSON 中的 risk_factors 列表。"""
    factors: list[RiskFactor] = []

    raw_factors = data.get("risk_factors", [])
    if not isinstance(raw_factors, list):
        return factors

    for item in raw_factors:
        if not isinstance(item, dict):
            continue

        pattern = str(item.get("pattern", "")).strip()
        if not pattern:
            continue

        dimension = item.get("dimension", [])
        if isinstance(dimension, str):
            dimension = [dimension]
        dimension = _validate_dimension(dimension)

        signals = item.get("observable_signals", [])
        if isinstance(signals, str):
            signals = [signals]
        signals = [str(s).strip() for s in signals if str(s).strip()]

        data_hint = str(item.get("data_hint", "")).strip()

        factors.append(RiskFactor(
            pattern=pattern,
            dimension=dimension,
            observable_signals=signals,
            data_hint=data_hint,
        ))

    return factors


def extract_from_text(
    client: LLMClient,
    document_text: str,
    source_name: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ExtractionResult:
    """从单个文档文本中提取风险因子。

    Parameters
    ----------
    client : LLMClient
        LLM 客户端实例。
    document_text : str
        文档文本内容。
    source_name : str
        文档来源名称。
    temperature : float, optional
        LLM temperature 覆盖。
    max_tokens : int, optional
        LLM max_tokens 覆盖。

    Returns
    -------
    ExtractionResult
    """
    if not document_text.strip():
        return ExtractionResult(source=source_name, error="文档内容为空")

    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        source_name=source_name,
        document_text=document_text,
    )

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        response = client.chat(messages, **kwargs)
    except Exception as e:
        logger.error("LLM 调用失败 (%s): %s", source_name, e)
        return ExtractionResult(source=source_name, error=f"LLM 调用失败: {e}")

    raw = response.content or ""

    try:
        data = _extract_json_from_response(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("JSON 解析失败 (%s): %s", source_name, e)
        return ExtractionResult(
            source=source_name, raw_response=raw,
            error=f"JSON 解析失败: {e}",
        )

    source = str(data.get("source", source_name)).strip() or source_name
    factors = _parse_risk_factors(data)

    logger.info("提取完成: %s → %d 个风险因子", source_name, len(factors))
    return ExtractionResult(
        source=source,
        risk_factors=factors,
        raw_response=raw,
    )


def extract_from_file(
    client: LLMClient,
    file_path: Path,
    *,
    source_name: str | None = None,
    **kwargs,
) -> ExtractionResult:
    """从单个文件中提取风险因子。"""
    file_path = Path(file_path)
    name = source_name or file_path.stem

    if not file_path.exists():
        return ExtractionResult(source=name, error=f"文件不存在: {file_path}")

    try:
        text = load_document(file_path)
    except Exception as e:
        return ExtractionResult(source=name, error=f"文件加载失败: {e}")

    return extract_from_text(client, text, name, **kwargs)


def batch_extract(
    client: LLMClient,
    file_paths: list[Path],
    *,
    source_names: list[str] | None = None,
    **kwargs,
) -> list[ExtractionResult]:
    """批量处理多个文档。

    Parameters
    ----------
    client : LLMClient
        LLM 客户端实例。
    file_paths : list[Path]
        待处理的文件路径列表。
    source_names : list[str], optional
        文档来源名称列表（与 file_paths 一一对应）。

    Returns
    -------
    list[ExtractionResult]
    """
    names = source_names or [fp.stem for fp in file_paths]
    if len(names) != len(file_paths):
        raise ValueError("source_names 长度必须与 file_paths 一致")

    results: list[ExtractionResult] = []
    for i, fp in enumerate(file_paths):
        logger.info("处理 [%d/%d]: %s", i + 1, len(file_paths), fp.name)
        result = extract_from_file(client, fp, source_name=names[i], **kwargs)
        results.append(result)

    ok = sum(1 for r in results if r.success)
    logger.info("批量提取完成: %d/%d 成功", ok, len(results))
    return results


def save_results(
    results: list[ExtractionResult],
    output_dir: Path,
) -> list[Path]:
    """将提取结果保存为 JSON 文件。

    Parameters
    ----------
    results : list[ExtractionResult]
        提取结果列表。
    output_dir : Path
        输出目录。

    Returns
    -------
    list[Path]
        保存的文件路径列表（仅成功的）。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for result in results:
        if not result.success:
            logger.warning("跳过失败结果: %s — %s", result.source, result.error)
            continue
        if not result.risk_factors:
            logger.warning("跳过空结果: %s（无风险因子）", result.source)
            continue

        safe_name = re.sub(r'[^\w一-鿿]', '_', result.source)
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        out_path = output_dir / f"{safe_name}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info("已保存: %s (%d 因子)", out_path.name, len(result.risk_factors))
        saved.append(out_path)

    return saved


def run_extraction(
    cfg: KnowFeatConfig,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    file_patterns: list[str] | None = None,
) -> list[ExtractionResult]:
    """端到端运行 Key-clue Extraction。

    Parameters
    ----------
    cfg : KnowFeatConfig
        全局配置。
    input_dir : Path, optional
        输入目录，默认 stage1_knowledge_dir。
    output_dir : Path, optional
        输出目录，默认 stage1_risk_factors_dir。
    file_patterns : list[str], optional
        文件匹配模式，默认 ["*.txt", "*.md", "*.docx"]。

    Returns
    -------
    list[ExtractionResult]
    """
    input_dir = input_dir or cfg.paths.stage1_knowledge_dir
    output_dir = output_dir or cfg.paths.stage1_risk_factors_dir
    patterns = file_patterns or ["*.txt", "*.md", "*.docx"]

    file_paths: list[Path] = []
    for pat in patterns:
        file_paths.extend(sorted(input_dir.glob(pat)))

    if not file_paths:
        logger.warning("输入目录中无匹配文件: %s (patterns=%s)", input_dir, patterns)
        return []

    logger.info("Key-clue Extraction: %d 个文件待处理", len(file_paths))

    client = LLMClient(cfg.llm)
    results = batch_extract(client, file_paths)
    save_results(results, output_dir)

    total_factors = sum(len(r.risk_factors) for r in results if r.success)
    logger.info(
        "Key-clue Extraction 完成: %d 文件 → %d 个风险因子",
        len(results), total_factors,
    )
    return results
