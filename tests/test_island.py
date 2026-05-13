"""tests/test_island.py — 测试 Boltzmann 采样、岛屿管理、迁移逻辑。"""

import math
import random

import pytest

from src.config import IslandConfig
from src.search.island import Island, IslandManager, _weighted_sample_without_replacement


# ---------- 测试数据 ----------

@pytest.fixture
def sample_config():
    return IslandConfig(
        name="金额频次",
        dimension="amount_frequency",
        description="金额分布、频率、拆分交易",
        fields=["amount", "interval", "hour"],
    )


@pytest.fixture
def sample_features():
    return [
        {"name": "f1", "score": 0.9, "code": "..."},
        {"name": "f2", "score": 0.5, "code": "..."},
        {"name": "f3", "score": 0.1, "code": "..."},
        {"name": "f4", "score": 0.7, "code": "..."},
    ]


@pytest.fixture
def all_configs():
    return [
        IslandConfig(name="金额频次", dimension="amount_frequency",
                     description="d1", fields=["amount"]),
        IslandConfig(name="时间地域", dimension="time_region",
                     description="d2", fields=["timestamp"]),
        IslandConfig(name="行为序列", dimension="behavior_sequence",
                     description="d3", fields=["transaction_mode"]),
        IslandConfig(name="主体关联", dimension="entity_association",
                     description="d4", fields=["wallet_open_cert"]),
    ]


# ---------- Island 基础操作 ----------

class TestIsland:
    def test_from_config(self, sample_config):
        island = Island.from_config(sample_config)
        assert island.name == "金额频次"
        assert island.dimension == "amount_frequency"
        assert island.fields == ["amount", "interval", "hour"]
        assert island.memory == []

    def test_add_feature(self, sample_config):
        island = Island.from_config(sample_config)
        island.add_feature({"name": "feat_a", "score": 0.5})
        assert len(island.memory) == 1
        assert island.memory[0]["name"] == "feat_a"

    def test_add_multiple_features(self, sample_config, sample_features):
        island = Island.from_config(sample_config)
        for f in sample_features:
            island.add_feature(f)
        assert len(island.memory) == 4

    def test_has_feature(self, sample_config):
        island = Island.from_config(sample_config)
        island.add_feature({"name": "feat_x", "score": 0.3})
        assert island.has_feature("feat_x") is True
        assert island.has_feature("nonexistent") is False

    def test_has_feature_empty(self, sample_config):
        island = Island.from_config(sample_config)
        assert island.has_feature("anything") is False


# ---------- Boltzmann 采样 ----------

class TestBoltzmannSample:
    def test_empty_memory_returns_empty(self, sample_config):
        island = Island.from_config(sample_config)
        result = island.boltzmann_sample(k=2, tau=1.0)
        assert result == []

    def test_k_exceeds_memory_size(self, sample_config):
        island = Island.from_config(sample_config)
        island.add_feature({"name": "f1", "score": 0.5})
        result = island.boltzmann_sample(k=10, tau=1.0)
        assert len(result) == 1

    def test_returns_correct_count(self, sample_config, sample_features):
        island = Island.from_config(sample_config)
        for f in sample_features:
            island.add_feature(f)
        result = island.boltzmann_sample(k=2, tau=1.0)
        assert len(result) == 2

    def test_no_duplicates(self, sample_config, sample_features):
        island = Island.from_config(sample_config)
        for f in sample_features:
            island.add_feature(f)
        random.seed(42)
        result = island.boltzmann_sample(k=3, tau=1.0)
        names = [r["name"] for r in result]
        assert len(names) == len(set(names))

    def test_high_score_preferred_low_tau(self, sample_config, sample_features):
        island = Island.from_config(sample_config)
        for f in sample_features:
            island.add_feature(f)
        counts = {"f1": 0, "f2": 0, "f3": 0, "f4": 0}
        for i in range(500):
            random.seed(i)
            result = island.boltzmann_sample(k=1, tau=0.1)
            counts[result[0]["name"]] += 1
        assert counts["f1"] > counts["f3"]

    def test_uniform_high_tau(self, sample_config, sample_features):
        island = Island.from_config(sample_config)
        for f in sample_features:
            island.add_feature(f)
        counts = {"f1": 0, "f2": 0, "f3": 0, "f4": 0}
        for i in range(2000):
            random.seed(i)
            result = island.boltzmann_sample(k=1, tau=100.0)
            counts[result[0]["name"]] += 1
        values = list(counts.values())
        assert max(values) - min(values) < 300

    def test_k_equals_memory_size(self, sample_config, sample_features):
        island = Island.from_config(sample_config)
        for f in sample_features:
            island.add_feature(f)
        result = island.boltzmann_sample(k=4, tau=1.0)
        assert len(result) == 4
        names = {r["name"] for r in result}
        assert names == {"f1", "f2", "f3", "f4"}

    def test_single_feature(self, sample_config):
        island = Island.from_config(sample_config)
        island.add_feature({"name": "only", "score": 1.0})
        result = island.boltzmann_sample(k=1, tau=1.0)
        assert len(result) == 1
        assert result[0]["name"] == "only"

    def test_zero_scores(self, sample_config):
        island = Island.from_config(sample_config)
        for i in range(3):
            island.add_feature({"name": f"f{i}", "score": 0.0})
        result = island.boltzmann_sample(k=2, tau=1.0)
        assert len(result) == 2

    def test_missing_score_defaults_to_zero(self, sample_config):
        island = Island.from_config(sample_config)
        island.add_feature({"name": "no_score", "code": "..."})
        island.add_feature({"name": "has_score", "score": 0.5})
        result = island.boltzmann_sample(k=2, tau=1.0)
        assert len(result) == 2

    def test_negative_scores(self, sample_config):
        island = Island.from_config(sample_config)
        island.add_feature({"name": "neg1", "score": -1.0})
        island.add_feature({"name": "neg2", "score": -0.5})
        island.add_feature({"name": "pos", "score": 0.5})
        result = island.boltzmann_sample(k=2, tau=1.0)
        assert len(result) == 2


# ---------- _weighted_sample_without_replacement ----------

class TestWeightedSample:
    def test_basic(self):
        random.seed(0)
        result = _weighted_sample_without_replacement([0.5, 0.3, 0.2], 2)
        assert len(result) == 2
        assert len(set(result)) == 2

    def test_k_zero(self):
        result = _weighted_sample_without_replacement([0.5, 0.5], 0)
        assert result == []

    def test_all_weight_on_one(self):
        random.seed(0)
        result = _weighted_sample_without_replacement([1.0, 0.0, 0.0], 1)
        assert result == [0]

    def test_full_sample(self):
        random.seed(42)
        result = _weighted_sample_without_replacement([0.25, 0.25, 0.25, 0.25], 4)
        assert sorted(result) == [0, 1, 2, 3]


# ---------- IslandManager ----------

class TestIslandManager:
    def test_init_creates_islands(self, all_configs):
        mgr = IslandManager(all_configs)
        assert len(mgr.islands) == 4

    def test_get_island_by_name(self, all_configs):
        mgr = IslandManager(all_configs)
        island = mgr.get_island("时间地域")
        assert island is not None
        assert island.dimension == "time_region"

    def test_get_island_not_found(self, all_configs):
        mgr = IslandManager(all_configs)
        assert mgr.get_island("不存在") is None

    def test_migrate_best_adds_to_all_other_islands(self, all_configs):
        mgr = IslandManager(all_configs)
        mgr.islands[0].add_feature({"name": "shared_feat", "score": 0.8})
        mgr.migrate_best({"name": "shared_feat", "score": 0.8})
        for island in mgr.islands:
            assert island.has_feature("shared_feat")

    def test_migrate_best_no_duplicate(self, all_configs):
        mgr = IslandManager(all_configs)
        feat = {"name": "dup_feat", "score": 0.6}
        mgr.islands[0].add_feature(feat)
        mgr.islands[1].add_feature(feat)
        mgr.migrate_best(feat)
        for island in mgr.islands:
            count = sum(1 for f in island.memory if f.get("name") == "dup_feat")
            assert count == 1

    def test_migrate_multiple_features(self, all_configs):
        mgr = IslandManager(all_configs)
        mgr.migrate_best({"name": "feat_a", "score": 0.9})
        mgr.migrate_best({"name": "feat_b", "score": 0.7})
        for island in mgr.islands:
            assert len(island.memory) == 2

    def test_island_fields_preserved(self, all_configs):
        mgr = IslandManager(all_configs)
        assert mgr.islands[0].fields == ["amount"]
        assert mgr.islands[3].fields == ["wallet_open_cert"]
