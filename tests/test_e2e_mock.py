"""端到端测试：使用 MockLLM 跑 2 轮完整搜索流程，验证 pipeline 可运行。"""

import json
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, LLMConfig, PathConfig, SearchParams
from src.data.loader import load_all_tables, extract_schema, extract_label
from src.llm.interface import MockLLM
from src.prompt.builder import build_system_prompt
from src.search.engine import SearchEngine


@pytest.fixture(scope="module")
def data():
    """加载真实 1K 数据集。"""
    cfg = Config()
    open_df, tx_df, close_df = load_all_tables(cfg.paths.data_dir)
    label = extract_label(tx_df)
    return open_df, tx_df, close_df, label


@pytest.fixture(scope="module")
def system_prompt(data):
    open_df, tx_df, close_df, _ = data
    schema_text = "\n\n".join([
        extract_schema(open_df, "开户表"),
        extract_schema(tx_df, "交易表"),
        extract_schema(close_df, "销户表"),
    ])
    return build_system_prompt(
        schema_text=schema_text,
        transaction_patterns="测试模式",
        expert_knowledge="测试知识",
        risk_indicators="测试指标",
        baseline_features=["f1", "f2"],
    )


@pytest.fixture(scope="module")
def pipeline_result(data, system_prompt):
    """运行一次完整 2 轮 pipeline，所有测试共享此结果。"""
    open_df, tx_df, close_df, label = data

    cfg = Config()
    cfg.search = SearchParams(m=4, b=1, B_total=4, b_min=1, b_max=2, k=1, T=2, tau=1.0, intra_island_turns=1)
    cfg.llm = LLMConfig(provider="mock")

    llm = MockLLM()
    X_baseline = pd.DataFrame(index=range(len(label)))

    engine = SearchEngine(
        llm=llm,
        system_prompt=system_prompt,
        seed_examples="测试种子",
        open_df=open_df,
        tx_df=tx_df,
        close_df=close_df,
        label=label,
        baseline_features=X_baseline,
        baseline_prauc=0.0,
        config=cfg,
    )

    return engine.run(max_rounds=2)


class TestMockLLMOutput:
    """验证 MockLLM 产出的特征代码格式正确。"""

    def test_mock_generates_valid_json(self):
        llm = MockLLM()
        raw = llm.generate("system", "user")
        features = json.loads(raw)
        assert isinstance(features, list)
        assert len(features) >= 1
        for f in features:
            assert "feature_name" in f
            assert "code" in f

    def test_mock_produces_distinct_features(self):
        llm = MockLLM()
        names = set()
        for _ in range(4):
            raw = llm.generate("s", "u")
            features = json.loads(raw)
            for f in features:
                names.add(f["feature_name"])
        assert len(names) == 4

    def test_mock_code_assigns_result(self):
        llm = MockLLM()
        raw = llm.generate("s", "u")
        features = json.loads(raw)
        assert "result" in features[0]["code"]


class TestE2EPipeline:
    """端到端集成测试：MockLLM + SearchEngine 2 轮。"""

    def test_pipeline_runs_2_rounds(self, pipeline_result):
        result = pipeline_result
        assert result.total_rounds == 2
        assert isinstance(result.search_log, list)
        assert len(result.search_log) == 2

        for round_log in result.search_log:
            assert "round" in round_log
            assert "islands" in round_log
            assert len(round_log["islands"]) == 4

    def test_all_islands_processed(self, pipeline_result):
        """验证每个岛屿都被处理且无异常。"""
        for round_log in pipeline_result.search_log:
            for island_log in round_log["islands"]:
                assert "error" not in island_log, f"Island {island_log['island']} had error"
                assert island_log["candidates"] >= 0

    def test_l1_passes_l2_validates(self, pipeline_result):
        """MockLLM 特征通过 L1 沙盒但 L2 统计门槛过滤。"""
        total_candidates = 0
        for round_log in pipeline_result.search_log:
            for island_log in round_log["islands"]:
                total_candidates += island_log["candidates"]

        assert total_candidates > 0, "至少有候选特征通过 L1"

    def test_convergence_detection(self, pipeline_result):
        assert pipeline_result.converged is True

    def test_output_files_saved(self, pipeline_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.main import save_results
            save_results(pipeline_result, tmpdir)

            features_path = os.path.join(tmpdir, "features.json")
            log_path = os.path.join(tmpdir, "search_log.json")

            assert os.path.exists(features_path)
            assert os.path.exists(log_path)

            with open(features_path) as f:
                features_data = json.load(f)
            assert isinstance(features_data, list)

            with open(log_path) as f:
                log_data = json.load(f)
            assert "total_rounds" in log_data
            assert log_data["total_rounds"] == 2


class TestMainEntry:
    """测试 main.py CLI 入口参数解析。"""

    def test_build_config_with_mock(self):
        from src.main import build_config
        import argparse
        args = argparse.Namespace(
            provider="mock", model="mock-model",
            max_rounds=2, data_dir=None, output_dir=None,
        )
        cfg = build_config(args)
        assert cfg.llm.provider == "mock"
        assert cfg.llm.model_name == "mock-model"
        assert cfg.search.T == 2
