"""失败记忆管理 — 记录验证失败的特征，注入 prompt 避免重复错误。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """单条失败记录。"""
    feature_name: str
    code: str
    reason: str
    round_num: int = 0
    dimension: str = ""


class FailureMemory:
    """累积失败记录，按轮次倒序提供最近 N 条供 prompt 注入。"""

    def __init__(self) -> None:
        self._records: list[FailureRecord] = []
        self._seen: set[str] = set()

    def record_failure(
        self,
        feature_name: str,
        code: str,
        reason: str,
        round_num: int = 0,
        dimension: str = "",
    ) -> FailureRecord:
        """记录一条失败。同名特征只保留最后一次记录。"""
        if feature_name in self._seen:
            self._records = [
                r for r in self._records if r.feature_name != feature_name
            ]
        else:
            self._seen.add(feature_name)

        rec = FailureRecord(
            feature_name=feature_name,
            code=code,
            reason=reason,
            round_num=round_num,
            dimension=dimension,
        )
        self._records.append(rec)
        logger.debug(
            "记录失败: %s (round=%d, dim=%s, reason=%s)",
            feature_name, round_num, dimension, reason[:80],
        )
        return rec

    def get_recent(self, n: int) -> list[FailureRecord]:
        """返回最近 *n* 条失败记录（最新在前）。"""
        if n <= 0:
            return []
        return list(reversed(self._records[-n:]))

    def get_by_dimension(
        self, dimension: str, n: int | None = None,
    ) -> list[FailureRecord]:
        """返回指定维度的失败记录（最新在前）。"""
        matched = [r for r in self._records if r.dimension == dimension]
        matched.reverse()
        if n is not None and n > 0:
            matched = matched[:n]
        return matched

    def get_by_round(self, round_num: int) -> list[FailureRecord]:
        """返回指定轮次的失败记录。"""
        return [r for r in self._records if r.round_num == round_num]

    def get_all(self) -> list[FailureRecord]:
        """返回所有失败记录（插入顺序）。"""
        return list(self._records)

    def clear(self) -> None:
        """清空所有记录。"""
        self._records.clear()
        self._seen.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"FailureMemory(count={len(self._records)})"
