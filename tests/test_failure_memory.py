"""Tests for knowfeat.search.failure_memory."""

import pytest

from knowfeat.search.failure_memory import FailureMemory, FailureRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def memory():
    return FailureMemory()


@pytest.fixture
def populated_memory():
    """预填充 5 条不同轮次/维度的失败记录。"""
    mem = FailureMemory()
    mem.record_failure("feat_a", "code_a", "IV 过低", round_num=1, dimension="金额频次")
    mem.record_failure("feat_b", "code_b", "沙盒超时", round_num=1, dimension="时间地域")
    mem.record_failure("feat_c", "code_c", "冗余度过高", round_num=2, dimension="金额频次")
    mem.record_failure("feat_d", "code_d", "delta_recall <= 0", round_num=2, dimension="行为序列")
    mem.record_failure("feat_e", "code_e", "缺失率过高", round_num=3, dimension="主体关联")
    return mem


# ---------------------------------------------------------------------------
# FailureRecord dataclass
# ---------------------------------------------------------------------------
class TestFailureRecord:
    def test_required_fields(self):
        rec = FailureRecord(feature_name="f1", code="x=1", reason="error")
        assert rec.feature_name == "f1"
        assert rec.code == "x=1"
        assert rec.reason == "error"

    def test_defaults(self):
        rec = FailureRecord(feature_name="f", code="c", reason="r")
        assert rec.round_num == 0
        assert rec.dimension == ""

    def test_all_fields(self):
        rec = FailureRecord("f", "c", "r", round_num=3, dimension="金额频次")
        assert rec.round_num == 3
        assert rec.dimension == "金额频次"


# ---------------------------------------------------------------------------
# record_failure
# ---------------------------------------------------------------------------
class TestRecordFailure:
    def test_basic_record(self, memory):
        rec = memory.record_failure("feat_a", "code_a", "error_a")
        assert isinstance(rec, FailureRecord)
        assert rec.feature_name == "feat_a"
        assert len(memory) == 1

    def test_record_with_all_params(self, memory):
        rec = memory.record_failure(
            "feat_a", "code_a", "error_a", round_num=2, dimension="时间地域",
        )
        assert rec.round_num == 2
        assert rec.dimension == "时间地域"

    def test_multiple_records(self, memory):
        memory.record_failure("f1", "c1", "r1")
        memory.record_failure("f2", "c2", "r2")
        memory.record_failure("f3", "c3", "r3")
        assert len(memory) == 3

    def test_dedup_same_name_keeps_latest(self, memory):
        memory.record_failure("f1", "code_v1", "old error", round_num=1)
        memory.record_failure("f1", "code_v2", "new error", round_num=2)
        assert len(memory) == 1
        records = memory.get_all()
        assert records[0].code == "code_v2"
        assert records[0].reason == "new error"
        assert records[0].round_num == 2

    def test_dedup_preserves_order(self, memory):
        memory.record_failure("f1", "c1", "r1", round_num=1)
        memory.record_failure("f2", "c2", "r2", round_num=1)
        memory.record_failure("f1", "c1_new", "r1_new", round_num=2)
        all_recs = memory.get_all()
        assert len(all_recs) == 2
        assert all_recs[0].feature_name == "f2"
        assert all_recs[1].feature_name == "f1"
        assert all_recs[1].code == "c1_new"

    def test_returns_failure_record(self, memory):
        rec = memory.record_failure("f", "c", "r")
        assert isinstance(rec, FailureRecord)


# ---------------------------------------------------------------------------
# get_recent
# ---------------------------------------------------------------------------
class TestGetRecent:
    def test_empty(self, memory):
        assert memory.get_recent(5) == []

    def test_n_zero(self, populated_memory):
        assert populated_memory.get_recent(0) == []

    def test_n_negative(self, populated_memory):
        assert populated_memory.get_recent(-1) == []

    def test_returns_newest_first(self, populated_memory):
        recent = populated_memory.get_recent(3)
        assert len(recent) == 3
        assert recent[0].feature_name == "feat_e"
        assert recent[1].feature_name == "feat_d"
        assert recent[2].feature_name == "feat_c"

    def test_n_exceeds_total(self, populated_memory):
        recent = populated_memory.get_recent(100)
        assert len(recent) == 5
        assert recent[0].feature_name == "feat_e"

    def test_n_equals_total(self, populated_memory):
        recent = populated_memory.get_recent(5)
        assert len(recent) == 5

    def test_n_one(self, populated_memory):
        recent = populated_memory.get_recent(1)
        assert len(recent) == 1
        assert recent[0].feature_name == "feat_e"


# ---------------------------------------------------------------------------
# get_by_dimension
# ---------------------------------------------------------------------------
class TestGetByDimension:
    def test_empty_memory(self, memory):
        assert memory.get_by_dimension("金额频次") == []

    def test_no_match(self, populated_memory):
        assert populated_memory.get_by_dimension("不存在维度") == []

    def test_returns_matching_records(self, populated_memory):
        recs = populated_memory.get_by_dimension("金额频次")
        assert len(recs) == 2
        assert all(r.dimension == "金额频次" for r in recs)

    def test_newest_first(self, populated_memory):
        recs = populated_memory.get_by_dimension("金额频次")
        assert recs[0].feature_name == "feat_c"
        assert recs[1].feature_name == "feat_a"

    def test_with_n_limit(self, populated_memory):
        recs = populated_memory.get_by_dimension("金额频次", n=1)
        assert len(recs) == 1
        assert recs[0].feature_name == "feat_c"

    def test_n_none_returns_all(self, populated_memory):
        recs = populated_memory.get_by_dimension("金额频次", n=None)
        assert len(recs) == 2

    def test_single_match(self, populated_memory):
        recs = populated_memory.get_by_dimension("主体关联")
        assert len(recs) == 1
        assert recs[0].feature_name == "feat_e"


# ---------------------------------------------------------------------------
# get_by_round
# ---------------------------------------------------------------------------
class TestGetByRound:
    def test_empty_memory(self, memory):
        assert memory.get_by_round(1) == []

    def test_no_match(self, populated_memory):
        assert populated_memory.get_by_round(99) == []

    def test_returns_matching(self, populated_memory):
        recs = populated_memory.get_by_round(2)
        assert len(recs) == 2
        names = {r.feature_name for r in recs}
        assert names == {"feat_c", "feat_d"}

    def test_single_round(self, populated_memory):
        recs = populated_memory.get_by_round(3)
        assert len(recs) == 1
        assert recs[0].feature_name == "feat_e"


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------
class TestGetAll:
    def test_empty(self, memory):
        assert memory.get_all() == []

    def test_returns_insertion_order(self, populated_memory):
        recs = populated_memory.get_all()
        assert len(recs) == 5
        names = [r.feature_name for r in recs]
        assert names == ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]

    def test_returns_copy(self, populated_memory):
        recs = populated_memory.get_all()
        recs.clear()
        assert len(populated_memory) == 5


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------
class TestClear:
    def test_clear_empty(self, memory):
        memory.clear()
        assert len(memory) == 0

    def test_clear_populated(self, populated_memory):
        populated_memory.clear()
        assert len(populated_memory) == 0
        assert populated_memory.get_all() == []

    def test_can_add_after_clear(self, populated_memory):
        populated_memory.clear()
        populated_memory.record_failure("new", "code", "reason")
        assert len(populated_memory) == 1

    def test_dedup_reset_after_clear(self, populated_memory):
        populated_memory.clear()
        populated_memory.record_failure("feat_a", "code", "reason")
        assert len(populated_memory) == 1


# ---------------------------------------------------------------------------
# __len__ / __repr__
# ---------------------------------------------------------------------------
class TestDunderMethods:
    def test_len_empty(self, memory):
        assert len(memory) == 0

    def test_len_populated(self, populated_memory):
        assert len(populated_memory) == 5

    def test_repr_empty(self, memory):
        assert "count=0" in repr(memory)

    def test_repr_populated(self, populated_memory):
        assert "count=5" in repr(populated_memory)


# ---------------------------------------------------------------------------
# Integration: builder.py FailureRecord 兼容性
# ---------------------------------------------------------------------------
try:
    import pandas  # noqa: F401
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas not installed")
class TestBuilderCompat:
    """确保 failure_memory.FailureRecord 与 builder.py 的 FailureRecord 字段兼容。"""

    def test_fields_match_builder(self):
        from knowfeat.prompt.builder import FailureRecord as BuilderFR
        builder_fields = {f.name for f in BuilderFR.__dataclass_fields__.values()}
        local_fields = {f.name for f in FailureRecord.__dataclass_fields__.values()}
        assert builder_fields == local_fields

    def test_format_failure_memory_accepts_local_records(self):
        from knowfeat.prompt.builder import _format_failure_memory, FailureRecord as BuilderFR

        rec = FailureRecord("test_feat", "x = 1", "IV 过低", round_num=1, dimension="金额频次")
        builder_rec = BuilderFR(
            feature_name=rec.feature_name,
            code=rec.code,
            reason=rec.reason,
            round_num=rec.round_num,
            dimension=rec.dimension,
        )
        output = _format_failure_memory([builder_rec])
        assert "test_feat" in output
        assert "IV 过低" in output
