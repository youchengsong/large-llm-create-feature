"""Tests for knowfeat.output.provenance — Provenance Card 生成。"""

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from knowfeat.output.provenance import (
    ProvenanceCard,
    build_card,
    build_card_from_candidate,
    build_cards_from_accepted,
    save_provenance_json,
    save_provenance_markdown,
    save_provenance,
    load_provenance_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_card(**overrides) -> ProvenanceCard:
    defaults = dict(
        feature_name="tx_amount_std",
        dimensions=["金额频次"],
        explanation="每个钱包交易金额的标准差",
        reasoning="金额波动大可能表示洗钱行为",
        knowledge_source="检测规则_15条.xlsx - 规则3",
        data_fields_used=["tx_amount", "wallet_id"],
        code="def feature_tx_amount_std(open_df, tx_df, close_df):\n    return tx_df.groupby('wallet_id')['tx_amount'].std()",
    )
    defaults.update(overrides)
    return ProvenanceCard(**defaults)


def _make_cards(n: int = 3) -> list[ProvenanceCard]:
    return [_sample_card(feature_name=f"feat_{i}") for i in range(n)]


@dataclass
class FakeCandidateFeature:
    name: str = "cand_feat"
    dimension: list[str] = field(default_factory=lambda: ["行为序列"])
    code: str = "def feature_cand_feat(): pass"
    explanation: str = "candidate explanation"
    reasoning: str = "candidate reasoning"
    knowledge_source: str = "expert"
    data_fields_used: list[str] = field(default_factory=lambda: ["col_a"])


@dataclass
class FakeRegisteredFeature:
    feature_name: str = "reg_feat"
    dimension: list[str] = field(default_factory=lambda: ["主体关联"])
    code: str = "def feature_reg_feat(): pass"
    explanation: str = "registered explanation"
    reasoning: str = "registered reasoning"
    knowledge_source: str = "rule5"
    data_fields_used: list[str] = field(default_factory=lambda: ["col_b"])


# ===========================================================================
# ProvenanceCard dataclass
# ===========================================================================

class TestProvenanceCard:
    def test_basic_creation(self):
        card = _sample_card()
        assert card.feature_name == "tx_amount_std"
        assert card.dimensions == ["金额频次"]
        assert card.code_hash != ""

    def test_code_hash_computed(self):
        code = "def f(): pass"
        card = _sample_card(code=code)
        expected = hashlib.sha256(code.encode()).hexdigest()[:16]
        assert card.code_hash == expected

    def test_code_hash_empty_when_no_code(self):
        card = _sample_card(code="")
        assert card.code_hash == ""

    def test_code_hash_not_overridden_if_provided(self):
        card = ProvenanceCard(
            feature_name="f", dimensions=[], explanation="", reasoning="",
            knowledge_source="", data_fields_used=[], code="abc", code_hash="custom"
        )
        assert card.code_hash == "custom"

    def test_to_dict_excludes_code(self):
        card = _sample_card()
        d = card.to_dict()
        assert "code" not in d
        assert d["feature_name"] == "tx_amount_std"
        assert d["dimensions"] == ["金额频次"]
        assert "code_hash" in d

    def test_to_dict_fields(self):
        card = _sample_card()
        d = card.to_dict()
        expected_keys = {
            "feature_name", "dimensions", "explanation", "reasoning",
            "knowledge_source", "data_fields_used", "code_hash",
        }
        assert set(d.keys()) == expected_keys

    def test_to_markdown_contains_sections(self):
        card = _sample_card()
        md = card.to_markdown()
        assert "## tx_amount_std" in md
        assert "金额频次" in md
        assert "每个钱包交易金额的标准差" in md
        assert "金额波动大" in md
        assert "检测规则_15条" in md
        assert "tx_amount" in md
        assert "`" in md

    def test_to_markdown_no_hash_when_no_code(self):
        card = _sample_card(code="")
        md = card.to_markdown()
        assert "代码哈希" not in md

    def test_to_markdown_multi_dimensions(self):
        card = _sample_card(dimensions=["金额频次", "时间地域"])
        md = card.to_markdown()
        assert "金额频次, 时间地域" in md

    def test_to_markdown_empty_dimensions(self):
        card = _sample_card(dimensions=[])
        md = card.to_markdown()
        assert "未指定" in md

    def test_to_markdown_empty_data_fields(self):
        card = _sample_card(data_fields_used=[])
        md = card.to_markdown()
        assert "未指定" in md


# ===========================================================================
# build_card
# ===========================================================================

class TestBuildCard:
    def test_build_card(self):
        card = build_card(
            feature_name="f1",
            dimensions=["金额频次"],
            code="def f1(): pass",
            explanation="exp",
            reasoning="reason",
            knowledge_source="src",
            data_fields_used=["a", "b"],
        )
        assert card.feature_name == "f1"
        assert card.code_hash != ""
        assert card.dimensions == ["金额频次"]

    def test_build_card_empty_code(self):
        card = build_card(
            feature_name="f2", dimensions=[], code="",
            explanation="", reasoning="", knowledge_source="",
            data_fields_used=[],
        )
        assert card.code_hash == ""


# ===========================================================================
# build_card_from_candidate
# ===========================================================================

class TestBuildCardFromCandidate:
    def test_from_candidate_feature(self):
        cand = FakeCandidateFeature()
        card = build_card_from_candidate(cand)
        assert card.feature_name == "cand_feat"
        assert card.dimensions == ["行为序列"]
        assert card.explanation == "candidate explanation"

    def test_from_registered_feature(self):
        reg = FakeRegisteredFeature()
        card = build_card_from_candidate(reg)
        assert card.feature_name == "reg_feat"
        assert card.dimensions == ["主体关联"]
        assert card.knowledge_source == "rule5"

    def test_missing_attributes_default(self):
        class Minimal:
            name = "m"
        card = build_card_from_candidate(Minimal())
        assert card.feature_name == "m"
        assert card.dimensions == []
        assert card.explanation == ""


# ===========================================================================
# build_cards_from_accepted
# ===========================================================================

class TestBuildCardsFromAccepted:
    def test_empty(self):
        assert build_cards_from_accepted({}) == []

    def test_multiple(self):
        accepted = {
            "f1": FakeCandidateFeature(name="f1"),
            "f2": FakeCandidateFeature(name="f2"),
        }
        cards = build_cards_from_accepted(accepted)
        assert len(cards) == 2
        names = {c.feature_name for c in cards}
        assert names == {"f1", "f2"}

    def test_preserves_order(self):
        accepted = {
            "c": FakeCandidateFeature(name="c"),
            "a": FakeCandidateFeature(name="a"),
            "b": FakeCandidateFeature(name="b"),
        }
        cards = build_cards_from_accepted(accepted)
        assert [c.feature_name for c in cards] == ["c", "a", "b"]


# ===========================================================================
# save_provenance_json
# ===========================================================================

class TestSaveProvenanceJson:
    def test_creates_file(self, tmp_path):
        cards = _make_cards(2)
        path = save_provenance_json(cards, tmp_path / "prov.json")
        assert path.exists()

    def test_json_content(self, tmp_path):
        cards = _make_cards(2)
        path = save_provenance_json(cards, tmp_path / "prov.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["feature_name"] == "feat_0"
        assert "code" not in data[0]

    def test_empty_cards(self, tmp_path):
        path = save_provenance_json([], tmp_path / "empty.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == []

    def test_nested_dir_created(self, tmp_path):
        path = save_provenance_json(
            _make_cards(1),
            tmp_path / "nested" / "deep" / "prov.json",
        )
        assert path.exists()

    def test_chinese_preserved(self, tmp_path):
        cards = [_sample_card(explanation="中文说明")]
        path = save_provenance_json(cards, tmp_path / "cn.json")
        content = path.read_text(encoding="utf-8")
        assert "中文说明" in content


# ===========================================================================
# save_provenance_markdown
# ===========================================================================

class TestSaveProvenanceMarkdown:
    def test_creates_file(self, tmp_path):
        cards = _make_cards(2)
        path = save_provenance_markdown(cards, tmp_path / "prov.md")
        assert path.exists()

    def test_markdown_structure(self, tmp_path):
        cards = _make_cards(2)
        path = save_provenance_markdown(cards, tmp_path / "prov.md")
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# Provenance Cards")
        assert "## feat_0" in content
        assert "## feat_1" in content
        assert "---" in content

    def test_empty_cards(self, tmp_path):
        path = save_provenance_markdown([], tmp_path / "empty.md")
        content = path.read_text(encoding="utf-8")
        assert "# Provenance Cards" in content

    def test_ends_with_newline(self, tmp_path):
        path = save_provenance_markdown(_make_cards(1), tmp_path / "prov.md")
        content = path.read_text(encoding="utf-8")
        assert content.endswith("\n")


# ===========================================================================
# save_provenance (combined)
# ===========================================================================

class TestSaveProvenance:
    def test_saves_both_formats(self, tmp_path):
        cards = _make_cards(3)
        result = save_provenance(cards, tmp_path, stage="stage2")
        assert "json" in result
        assert "markdown" in result
        assert result["json"].exists()
        assert result["markdown"].exists()

    def test_file_naming(self, tmp_path):
        save_provenance(_make_cards(1), tmp_path, stage="stage1")
        assert (tmp_path / "stage1_provenance.json").exists()
        assert (tmp_path / "stage1_provenance.md").exists()

    def test_default_stage(self, tmp_path):
        save_provenance(_make_cards(1), tmp_path)
        assert (tmp_path / "stage2_provenance.json").exists()
        assert (tmp_path / "stage2_provenance.md").exists()


# ===========================================================================
# load_provenance_json
# ===========================================================================

class TestLoadProvenanceJson:
    def test_roundtrip(self, tmp_path):
        cards = _make_cards(3)
        path = save_provenance_json(cards, tmp_path / "rt.json")
        loaded = load_provenance_json(path)
        assert len(loaded) == 3
        for orig, loaded_card in zip(cards, loaded):
            assert loaded_card.feature_name == orig.feature_name
            assert loaded_card.dimensions == orig.dimensions
            assert loaded_card.explanation == orig.explanation
            assert loaded_card.code_hash == orig.code_hash

    def test_loaded_card_has_no_code(self, tmp_path):
        cards = [_sample_card()]
        path = save_provenance_json(cards, tmp_path / "nc.json")
        loaded = load_provenance_json(path)
        assert loaded[0].code == ""

    def test_missing_optional_fields(self, tmp_path):
        data = [{"feature_name": "minimal"}]
        path = tmp_path / "min.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_provenance_json(path)
        assert loaded[0].feature_name == "minimal"
        assert loaded[0].dimensions == []
        assert loaded[0].explanation == ""

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        loaded = load_provenance_json(path)
        assert loaded == []


# ===========================================================================
# Integration: real CandidateFeature-like flow
# ===========================================================================

class TestIntegrationFlow:
    def test_full_pipeline(self, tmp_path):
        accepted = {
            "feat_a": FakeCandidateFeature(name="feat_a"),
            "feat_b": FakeRegisteredFeature(feature_name="feat_b"),
        }
        cards = build_cards_from_accepted(accepted)
        assert len(cards) == 2
        result = save_provenance(cards, tmp_path, stage="stage2")
        loaded = load_provenance_json(result["json"])
        assert len(loaded) == 2
        assert loaded[0].feature_name == "feat_a"
        assert loaded[1].feature_name == "feat_b"

    def test_markdown_readability(self, tmp_path):
        card = build_card(
            feature_name="night_tx_ratio",
            dimensions=["时间地域", "行为序列"],
            code="def feature_night_tx_ratio(open_df, tx_df, close_df):\n    pass",
            explanation="深夜（0-6点）交易占比",
            reasoning="洗钱行为常在深夜进行以规避监控",
            knowledge_source="上海警方《严打跑分洗钱》- 夜间集中转账特征",
            data_fields_used=["tx_time", "wallet_id"],
        )
        path = save_provenance_markdown([card], tmp_path / "readable.md")
        content = path.read_text(encoding="utf-8")
        assert "night_tx_ratio" in content
        assert "时间地域, 行为序列" in content
        assert "深夜" in content
        assert "上海警方" in content
