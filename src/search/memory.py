from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class FailureRecord:
    feature_name: str
    code: str
    failure_reason: str


class FailureMemory:
    def __init__(self, max_size: int = 50) -> None:
        self._records: deque[FailureRecord] = deque(maxlen=max_size)

    def add_failure(self, name: str, code: str, reason: str) -> None:
        self._records.append(FailureRecord(feature_name=name, code=code, failure_reason=reason))

    def get_recent(self, n: int = 5) -> list[dict]:
        items = list(self._records)[-n:]
        return [
            {
                "feature_name": r.feature_name,
                "code": r.code,
                "failure_reason": r.failure_reason,
            }
            for r in items
        ]

    def to_prompt_text(self, n: int = 5) -> str:
        recent = self.get_recent(n)
        if not recent:
            return "（暂无失败记录）"
        lines = []
        for r in recent:
            lines.append(f"- {r['feature_name']}: {r['failure_reason']}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._records)
