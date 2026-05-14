"""缓刑池 + 反向淘汰机制的单元测试。"""

import numpy as np
import pandas as pd
import pytest

from src.search.pending_pool import PendingPool, PendingFeature


def _make_pf(name: str, mi: float = 0.01) -> PendingFeature:
    return PendingFeature(
        name=name,
        code=f"result = tx_df[['tx_id']].assign(feature_value=0)",
        explanation="test",
        reasoning="test",
        feature_values=np.random.randn(100),
        l2_metrics={"mi": mi, "redundancy": 0.1, "cramers_v": 0.1, "bootstrap_p": 0.01},
        l3_metrics={"delta_prauc": 0.001, "shap_rank": 10},
        island_name="金额频次",
    )


class TestPendingPoolBasic:
    def test_add_and_len(self):
        pool = PendingPool()
        pool.add(_make_pf("A"))
        pool.add(_make_pf("B"))
        assert len(pool) == 2

    def test_no_duplicates(self):
        pool = PendingPool()
        pool.add(_make_pf("A"))
        pool.add(_make_pf("A"))
        assert len(pool) == 1

    def test_capacity_enforcement(self):
        pool = PendingPool(max_size=3)
        for i in range(5):
            pool.add(_make_pf(f"F{i}", mi=i * 0.01))
        assert len(pool) == 3
        names = {f.name for f in pool.get_all()}
        assert "F0" not in names
        assert "F1" not in names

    def test_tick_age(self):
        pool = PendingPool()
        pool.add(_make_pf("A"))
        pool.tick_age()
        pool.tick_age()
        assert pool.get_all()[0].age == 2

    def test_mark_rejected(self):
        pool = PendingPool()
        pool.add(_make_pf("A"))
        pool.add(_make_pf("B"))
        pool.mark_rejected({"A"})
        a = [f for f in pool.get_all() if f.name == "A"][0]
        b = [f for f in pool.get_all() if f.name == "B"][0]
        assert a.consecutive_rejects == 1
        assert b.consecutive_rejects == 0

    def test_consecutive_rejects_reset(self):
        pool = PendingPool()
        pool.add(_make_pf("A"))
        pool.mark_rejected({"A"})
        pool.mark_rejected(set())
        a = pool.get_all()[0]
        assert a.consecutive_rejects == 0

    def test_remove_accepted(self):
        pool = PendingPool()
        pool.add(_make_pf("A"))
        pool.add(_make_pf("B"))
        accepted = pool.remove_accepted({"A"})
        assert len(accepted) == 1
        assert accepted[0].name == "A"
        assert len(pool) == 1

    def test_purge_by_consecutive_rejects(self):
        pool = PendingPool(max_rejects=2)
        pool.add(_make_pf("A"))
        pool.mark_rejected({"A"})
        pool.mark_rejected({"A"})
        purged = pool.purge_expired()
        assert "A" in purged
        assert len(pool) == 0

    def test_purge_by_age(self):
        pool = PendingPool(max_age=2)
        pool.add(_make_pf("A"))
        pool.tick_age()
        pool.tick_age()
        pool.tick_age()
        purged = pool.purge_expired()
        assert "A" in purged
        assert len(pool) == 0

    def test_no_purge_within_limits(self):
        pool = PendingPool(max_age=3, max_rejects=2)
        pool.add(_make_pf("A"))
        pool.tick_age()
        pool.mark_rejected({"A"})
        purged = pool.purge_expired()
        assert len(purged) == 0
        assert len(pool) == 1


class TestBackwardElimination:
    def test_backward_eliminate_empty_pool(self):
        from src.evaluation.validator import backward_eliminate_pending
        X = pd.DataFrame({"f1": np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        accepted, rejected = backward_eliminate_pending(X, {}, y)
        assert len(accepted) == 0
        assert len(rejected) == 0

    def test_backward_eliminate_noise_feature(self):
        from src.evaluation.validator import backward_eliminate_pending
        np.random.seed(42)
        n = 200
        y = pd.Series(np.random.randint(0, 2, n))
        X = pd.DataFrame({"real_feat": y.values * 2.0 + np.random.randn(n) * 0.1})
        noise = {"noise_feat": np.random.randn(n)}
        accepted, rejected = backward_eliminate_pending(X, noise, y)
        assert "noise_feat" in rejected
