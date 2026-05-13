from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from src.config import IslandConfig, SearchParams


@dataclass
class Island:
    name: str
    dimension: str
    fields: list[str]
    memory: list[dict] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: IslandConfig) -> Island:
        return cls(name=cfg.name, dimension=cfg.dimension, fields=list(cfg.fields))

    def add_feature(self, feature_dict: dict) -> None:
        self.memory.append(feature_dict)

    def has_feature(self, feature_name: str) -> bool:
        return any(f.get("name") == feature_name for f in self.memory)

    def boltzmann_sample(self, k: int = 2, tau: float = 1.0) -> list[dict]:
        if not self.memory:
            return []
        k = min(k, len(self.memory))
        scores = [f.get("score", 0.0) for f in self.memory]
        max_score = max(scores)
        weights = [math.exp((s - max_score) / tau) for s in scores]
        total = sum(weights)
        probs = [w / total for w in weights]
        indices = _weighted_sample_without_replacement(probs, k)
        return [self.memory[i] for i in indices]


class IslandManager:
    def __init__(self, island_configs: list[IslandConfig]) -> None:
        self.islands: list[Island] = [
            Island.from_config(cfg) for cfg in island_configs
        ]

    def get_island(self, name: str) -> Island | None:
        for island in self.islands:
            if island.name == name:
                return island
        return None

    def migrate_best(self, feature_dict: dict) -> None:
        name = feature_dict.get("name", "")
        for island in self.islands:
            if not island.has_feature(name):
                island.add_feature(feature_dict)


def _weighted_sample_without_replacement(
    probs: list[float], k: int
) -> list[int]:
    indices: list[int] = []
    remaining = list(range(len(probs)))
    remaining_probs = list(probs)
    for _ in range(k):
        total = sum(remaining_probs)
        if total <= 0:
            break
        normalized = [p / total for p in remaining_probs]
        r = random.random()
        cumulative = 0.0
        chosen_idx = len(remaining) - 1
        for j, p in enumerate(normalized):
            cumulative += p
            if r <= cumulative:
                chosen_idx = j
                break
        indices.append(remaining[chosen_idx])
        remaining.pop(chosen_idx)
        remaining_probs.pop(chosen_idx)
    return indices
