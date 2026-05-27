"""Prompt 构建模块：领域知识加载 + System Prompt 组装。"""

from knowfeat.prompt.builder import (
    CrossDimExample,
    FailureRecord,
    ShapFeedback,
    build_full_prompt,
    build_round_instruction,
    build_system_prompt,
)
from knowfeat.prompt.domain_knowledge import (
    DomainKnowledge,
    DetectionRule,
    RiskIndicator,
    TransactionPattern,
    format_domain_knowledge,
    load_domain_knowledge,
)

__all__ = [
    "CrossDimExample",
    "DetectionRule",
    "DomainKnowledge",
    "FailureRecord",
    "RiskIndicator",
    "ShapFeedback",
    "TransactionPattern",
    "build_full_prompt",
    "build_round_instruction",
    "build_system_prompt",
    "format_domain_knowledge",
    "load_domain_knowledge",
]
