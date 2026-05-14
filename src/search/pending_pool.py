"""缓刑池：管理 L2 通过但 L3 未通过的特征，支持反向淘汰批量评估。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PendingFeature:
    name: str
    code: str
    explanation: str
    reasoning: str
    feature_values: np.ndarray
    l2_metrics: dict
    l3_metrics: dict
    island_name: str
    age: int = 0
    consecutive_rejects: int = 0


class PendingPool:
    def __init__(self, max_size: int = 20, max_age: int = 3, max_rejects: int = 2):
        self.max_size = max_size
        self.max_age = max_age
        self.max_rejects = max_rejects
        self._pool: list[PendingFeature] = []

    def add(self, feature: PendingFeature) -> None:
        if any(f.name == feature.name for f in self._pool):
            return
        self._pool.append(feature)
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        if len(self._pool) <= self.max_size:
            return
        self._pool.sort(key=lambda f: f.l2_metrics.get("mi", 0))
        while len(self._pool) > self.max_size:
            removed = self._pool.pop(0)
            logger.info("Pending pool capacity exceeded, removed %s (MI=%.4f)",
                        removed.name, removed.l2_metrics.get("mi", 0))

    def tick_age(self) -> None:
        for f in self._pool:
            f.age += 1

    def mark_rejected(self, names: set[str]) -> None:
        for f in self._pool:
            if f.name in names:
                f.consecutive_rejects += 1
            else:
                f.consecutive_rejects = 0

    def remove_accepted(self, names: set[str]) -> list[PendingFeature]:
        accepted = [f for f in self._pool if f.name in names]
        self._pool = [f for f in self._pool if f.name not in names]
        return accepted

    def purge_expired(self) -> list[str]:
        purged = []
        remaining = []
        for f in self._pool:
            if f.consecutive_rejects >= self.max_rejects:
                purged.append(f.name)
                logger.info("Purged %s: consecutive_rejects=%d", f.name, f.consecutive_rejects)
            elif f.age > self.max_age:
                purged.append(f.name)
                logger.info("Purged %s: age=%d exceeded max_age=%d", f.name, f.age, self.max_age)
            else:
                remaining.append(f)
        self._pool = remaining
        return purged

    def get_all(self) -> list[PendingFeature]:
        return list(self._pool)

    def is_empty(self) -> bool:
        return len(self._pool) == 0

    def __len__(self) -> int:
        return len(self._pool)
