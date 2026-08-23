"""Provenance Card 生成：从 generate_feature 参数自动生成特征溯源卡片。

每个特征的 Provenance Card 记录"从哪来(knowledge_source)、
为什么(reasoning + explanation)、用了什么(data_fields_used)"。
支持单张卡片生成、批量生成、JSON/Markdown 导出。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceCard:
    """单个特征的溯源卡片。"""

    feature_name: str
    dimensions: list[str]
    explanation: str
    reasoning: str
    knowledge_source: str
    data_fields_used: list[str]
    knowledge_ids: list[str] = field(default_factory=list)
    computation_steps: str = ""
    knowledge_resolved: list[dict[str, str]] = field(default_factory=list)
    code: str = ""
    code_hash: str = ""

    def __post_init__(self) -> None:
        if self.code and not self.code_hash:
            self.code_hash = hashlib.sha256(self.code.encode()).hexdigest()[:16]

    def resolve_knowledge(self, registry: dict[str, str]) -> None:
        """用知识注册表解析 knowledge_ids 为 [{id, content}]。"""
        self.knowledge_resolved = [
            {"id": kid, "content": registry.get(kid, "(未找到)")}
            for kid in self.knowledge_ids
        ]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("code", None)
        return d

    def to_markdown(self) -> str:
        lines = [
            f"## {self.feature_name}",
            "",
            f"**维度**: {', '.join(self.dimensions) if self.dimensions else '未指定'}",
            "",
            f"**含义**: {self.explanation}",
            "",
            f"**设计推理**: {self.reasoning}",
            "",
        ]
        if self.knowledge_resolved:
            lines.append("**知识溯源**:")
            lines.append("")
            for item in self.knowledge_resolved:
                lines.append(f"- `{item['id']}`: {item['content']}")
            lines.append("")
        elif self.knowledge_ids:
            lines.append(f"**知识溯源ID**: {', '.join(self.knowledge_ids)}")
            lines.append("")
        elif self.knowledge_source:
            lines.append(f"**知识来源（兼容字段）**: {self.knowledge_source}")
            lines.append("")
        else:
            lines.append("**知识溯源ID**: 未指定")
            lines.append("")

        lines.append(f"**计算步骤**: {self.computation_steps if self.computation_steps else '未指定'}")
        lines.append("")
        lines.append(f"**数据字段**: {', '.join(self.data_fields_used) if self.data_fields_used else '未指定'}")

        if self.code_hash:
            lines.extend(["", f"**代码哈希**: `{self.code_hash}`"])
        return "\n".join(lines)


def build_card(
    feature_name: str,
    dimensions: list[str],
    code: str,
    explanation: str,
    reasoning: str,
    knowledge_source: str,
    data_fields_used: list[str],
    knowledge_ids: Optional[list[str]] = None,
    computation_steps: str = "",
    registry: Optional[dict[str, str]] = None,
) -> ProvenanceCard:
    card = ProvenanceCard(
        feature_name=feature_name,
        dimensions=dimensions,
        explanation=explanation,
        reasoning=reasoning,
        knowledge_source=knowledge_source,
        data_fields_used=data_fields_used,
        knowledge_ids=knowledge_ids or [],
        computation_steps=computation_steps,
        code=code,
    )
    if registry:
        card.resolve_knowledge(registry)
    return card


def build_card_from_candidate(candidate: Any) -> ProvenanceCard:
    """从 CandidateFeature 或 RegisteredFeature 构建卡片。

    两种结构字段名略有不同：
    - CandidateFeature.name / RegisteredFeature.feature_name
    - 两者都有 dimension, code, explanation, reasoning, knowledge_source, data_fields_used
    """
    name = getattr(candidate, "feature_name", None) or getattr(candidate, "name", "")
    return ProvenanceCard(
        feature_name=name,
        dimensions=getattr(candidate, "dimension", []),
        explanation=getattr(candidate, "explanation", ""),
        reasoning=getattr(candidate, "reasoning", ""),
        knowledge_source=getattr(candidate, "knowledge_source", ""),
        data_fields_used=getattr(candidate, "data_fields_used", []),
        knowledge_ids=getattr(candidate, "knowledge_ids", []),
        computation_steps=getattr(candidate, "computation_steps", ""),
        code=getattr(candidate, "code", ""),
    )


def build_cards_from_accepted(
    accepted_features: dict[str, Any],
) -> list[ProvenanceCard]:
    cards: list[ProvenanceCard] = []
    for _name, candidate in accepted_features.items():
        cards.append(build_card_from_candidate(candidate))
    return cards


def save_provenance_json(
    cards: list[ProvenanceCard],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [card.to_dict() for card in cards]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Provenance Cards (JSON) → %s (%d 张)", output_path, len(cards))
    return output_path


def save_provenance_markdown(
    cards: list[ProvenanceCard],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections = ["# Provenance Cards", ""]
    for card in cards:
        sections.append(card.to_markdown())
        sections.append("")
        sections.append("---")
        sections.append("")
    content = "\n".join(sections).rstrip("\n") + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Provenance Cards (MD) → %s (%d 张)", output_path, len(cards))
    return output_path


def save_provenance(
    cards: list[ProvenanceCard],
    output_dir: Path,
    stage: str = "stage2",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_provenance_json(
        cards, output_dir / f"{stage}_provenance.json"
    )
    md_path = save_provenance_markdown(
        cards, output_dir / f"{stage}_provenance.md"
    )
    return {"json": json_path, "markdown": md_path}


def load_provenance_json(path: Path) -> list[ProvenanceCard]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cards: list[ProvenanceCard] = []
    for item in data:
        cards.append(ProvenanceCard(
            feature_name=item["feature_name"],
            dimensions=item.get("dimensions", []),
            explanation=item.get("explanation", ""),
            reasoning=item.get("reasoning", ""),
            knowledge_source=item.get("knowledge_source", ""),
            data_fields_used=item.get("data_fields_used", []),
            knowledge_ids=item.get("knowledge_ids", []),
            computation_steps=item.get("computation_steps", ""),
            knowledge_resolved=item.get("knowledge_resolved", []),
            code_hash=item.get("code_hash", ""),
        ))
    return cards
