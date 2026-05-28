"""agent/react_agent.py 测试 — ReAct 主循环。"""

import json
import textwrap
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from knowfeat.agent.react_agent import (
    MAX_L1_REPAIRS,
    MAX_STEPS_PER_ROUND,
    ReactAgent,
    RoundResult,
    _build_repair_message,
    _parse_observation,
)
from knowfeat.agent.tools import ToolContext
from knowfeat.config import KnowFeatConfig, VerificationConfig
from knowfeat.data.loader import Stage2Data
from knowfeat.llm.client import LLMClient, LLMResponse, ToolCall, ToolSpec
from knowfeat.prompt.builder import (
    CrossDimExample,
    DomainKnowledge,
    FailureRecord,
    ShapFeedback,
)
from knowfeat.prompt.domain_knowledge import DomainKnowledge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    n = 200

    open_df = pd.DataFrame({
        "wallet_id": [f"W{i:04d}" for i in range(n)],
        "wallet_type": rng.choice([1, 2, 3], n),
        "region_code": rng.choice(["BJ", "SH", "GZ"], n),
    })

    labels = np.zeros(n, dtype=int)
    labels[:20] = 1
    rng.shuffle(labels)

    tx_df = pd.DataFrame({
        "tx_id": [f"T{i:06d}" for i in range(n)],
        "src": open_df["wallet_id"].values,
        "dst": rng.choice(open_df["wallet_id"].values, n),
        "amount": rng.exponential(1000, n),
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "is_risk": labels,
    })

    close_df = pd.DataFrame({
        "wallet_id": rng.choice(open_df["wallet_id"].values, 10),
        "close_date": pd.date_range("2025-06-01", periods=10, freq="D"),
    })

    baseline = pd.DataFrame({
        "base_feat1": rng.randn(n),
        "base_feat2": rng.randn(n),
    })

    return Stage2Data(
        open_df=open_df,
        tx_df=tx_df,
        close_df=close_df,
        baseline_features=baseline,
    )


@pytest.fixture
def cfg():
    return KnowFeatConfig(
        verification=VerificationConfig(
            sandbox_timeout_sec=10,
            noise_n_random_features=10,
            iv_min_fallback=0.001,
            ks_min_fallback=0.001,
            redundancy_max=0.95,
            missing_rate_max=0.5,
            cv_folds=2,
            precision_min=0.0,
            delta_recall_min=-1.0,
        ),
        seed=42,
    )


@pytest.fixture
def tool_ctx(sample_data, cfg):
    return ToolContext(data=sample_data, cfg=cfg)


@pytest.fixture
def domain_knowledge():
    return DomainKnowledge()


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMClient)


@pytest.fixture
def agent(mock_llm, tool_ctx, cfg, domain_knowledge):
    return ReactAgent(
        llm=mock_llm,
        tool_ctx=tool_ctx,
        cfg=cfg,
        domain_knowledge=domain_knowledge,
        schema_text="## 测试 Schema\n测试用",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _text_response(content: str) -> LLMResponse:
    """创建纯文本响应（无 tool call）。"""
    return LLMResponse(content=content, tool_calls=[], finish_reason="stop")


def _tool_response(
    tool_calls: list[tuple[str, dict]],
    content: str | None = None,
) -> LLMResponse:
    """创建带 tool call 的响应。"""
    tcs = [
        ToolCall(id=f"call_{i}", name=name, arguments=args)
        for i, (name, args) in enumerate(tool_calls)
    ]
    return LLMResponse(content=content, tool_calls=tcs, finish_reason="tool_calls")


# ---------------------------------------------------------------------------
# RoundResult 测试
# ---------------------------------------------------------------------------

class TestRoundResult:
    def test_default_values(self):
        r = RoundResult(round_num=1, dimension="金额频次")
        assert r.round_num == 1
        assert r.dimension == "金额频次"
        assert r.n_generated == 0
        assert r.n_accepted == 0
        assert r.n_candidates == 0
        assert r.n_rejected == 0
        assert r.steps == 0
        assert r.features_generated == []
        assert r.features_accepted == []
        assert r.terminated_reason == ""


# ---------------------------------------------------------------------------
# _parse_observation 测试
# ---------------------------------------------------------------------------

class TestParseObservation:
    def test_valid_json(self):
        obs = json.dumps({"accepted": True, "summary": "ok"})
        parsed = _parse_observation(obs)
        assert parsed["accepted"] is True
        assert parsed["summary"] == "ok"

    def test_invalid_json(self):
        assert _parse_observation("not json") == {}

    def test_none_input(self):
        assert _parse_observation(None) == {}

    def test_empty_string(self):
        assert _parse_observation("") == {}


# ---------------------------------------------------------------------------
# _build_repair_message 测试
# ---------------------------------------------------------------------------

class TestBuildRepairMessage:
    def test_basic_repair_message(self):
        obs = {
            "L1_sandbox": {"error": "NameError: name 'x' is not defined"},
            "summary": "L1 失败: NameError",
        }
        msg = _build_repair_message("feat_test", obs)
        assert "feat_test" in msg
        assert "NameError" in msg
        assert "generate_feature" in msg
        assert "execute_sandbox" in msg

    def test_repair_message_fallback_to_summary(self):
        obs = {"summary": "L1 失败: SyntaxError"}
        msg = _build_repair_message("feat_test", obs)
        assert "L1 失败: SyntaxError" in msg

    def test_repair_message_no_info(self):
        msg = _build_repair_message("feat_test", {})
        assert "feat_test" in msg
        assert "未知错误" in msg


# ---------------------------------------------------------------------------
# ReactAgent 核心测试
# ---------------------------------------------------------------------------

class TestReactAgentTermination:
    def test_agent_completes_with_text_response(self, agent, mock_llm):
        """Agent 返回纯文本时终止。"""
        mock_llm.chat.return_value = _text_response("本轮探索完成。")

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        assert result.terminated_reason == "agent_completed"
        assert result.steps == 1
        assert result.n_generated == 0
        mock_llm.chat.assert_called_once()

    def test_budget_exhaustion(self, agent, mock_llm, tool_ctx):
        """验证次数达到预算时终止。"""
        generate_resp = _tool_response([
            ("generate_feature", {
                "feature_name": "feat_1",
                "dimension": ["金额频次"],
                "code": "def feature_feat_1(open_df, tx_df, close_df):\n    return open_df[['wallet_id']]",
                "explanation": "test",
                "reasoning": "test",
                "knowledge_source": "test",
                "data_fields_used": ["amount"],
            }),
        ])
        sandbox_resp = _tool_response([
            ("execute_sandbox", {"feature_name": "feat_1"}),
        ])
        done_resp = _text_response("完成。")

        mock_llm.chat.side_effect = [generate_resp, sandbox_resp, done_resp]

        def mock_execute(name, args):
            if name == "generate_feature":
                return json.dumps({"status": "registered", "feature_name": "feat_1"})
            if name == "execute_sandbox":
                return json.dumps({
                    "feature_name": "feat_1",
                    "stage_reached": "L2",
                    "accepted": False,
                    "summary": "L2 失败: IV too low",
                })
            return json.dumps({"error": "unknown"})

        tool_ctx.execute = MagicMock(side_effect=mock_execute)

        result = agent.run_round(round_num=1, dimension="金额频次", budget=1)

        assert result.terminated_reason == "budget_exhausted"
        assert result.n_generated == 1
        assert "feat_1" in result.features_generated

    def test_max_steps_reached(self, agent, mock_llm, tool_ctx):
        """达到最大步数时终止。"""
        explore_resp = _tool_response([
            ("explore_data", {"code": "print('hello')"})
        ])
        tool_ctx.execute = MagicMock(
            return_value=json.dumps({"output": "hello"})
        )
        mock_llm.chat.return_value = explore_resp

        result = agent.run_round(round_num=1, dimension="金额频次", budget=100)

        assert result.terminated_reason == "max_steps_reached"
        assert result.steps == MAX_STEPS_PER_ROUND


class TestReactAgentToolExecution:
    def test_explore_data_tool_call(self, agent, mock_llm, tool_ctx):
        """explore_data 正确执行。"""
        explore_resp = _tool_response([
            ("explore_data", {"code": "tx_df.shape"})
        ])
        done_resp = _text_response("探索完成。")

        mock_llm.chat.side_effect = [explore_resp, done_resp]
        tool_ctx.execute = MagicMock(
            return_value=json.dumps({"output": "(200, 6)"})
        )

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        assert result.terminated_reason == "agent_completed"
        assert result.n_generated == 0
        tool_ctx.execute.assert_called_once_with("explore_data", {"code": "tx_df.shape"})

    def test_generate_feature_registers(self, agent, mock_llm, tool_ctx):
        """generate_feature 正确计数。"""
        gen_args = {
            "feature_name": "feat_amount",
            "dimension": ["金额频次"],
            "code": "def feature_x(open_df, tx_df, close_df):\n  return open_df[['wallet_id']]",
            "explanation": "t",
            "reasoning": "t",
            "knowledge_source": "t",
            "data_fields_used": ["amount"],
        }
        gen_resp = _tool_response([("generate_feature", gen_args)])
        done_resp = _text_response("完成。")

        mock_llm.chat.side_effect = [gen_resp, done_resp]
        tool_ctx.execute = MagicMock(
            return_value=json.dumps({"status": "registered", "feature_name": "feat_amount"})
        )

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        assert result.n_generated == 1
        assert "feat_amount" in result.features_generated

    def test_execute_sandbox_accepted(self, agent, mock_llm, tool_ctx):
        """execute_sandbox 返回 accepted 时正确计数。"""
        gen_resp = _tool_response([("generate_feature", {
            "feature_name": "feat_good",
            "dimension": ["金额频次"],
            "code": "def f(o, t, c):\n  return o[['wallet_id']]",
            "explanation": "t", "reasoning": "t",
            "knowledge_source": "t", "data_fields_used": [],
        })])
        exec_resp = _tool_response([("execute_sandbox", {"feature_name": "feat_good"})])
        done_resp = _text_response("完成。")

        mock_llm.chat.side_effect = [gen_resp, exec_resp, done_resp]

        call_count = [0]
        def mock_execute(name, args):
            call_count[0] += 1
            if name == "generate_feature":
                return json.dumps({"status": "registered", "feature_name": "feat_good"})
            if name == "execute_sandbox":
                return json.dumps({
                    "feature_name": "feat_good",
                    "stage_reached": "L3",
                    "accepted": True,
                    "summary": "L3 通过 — 已接受!",
                })
            return "{}"

        tool_ctx.execute = mock_execute

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        assert result.n_accepted == 1
        assert "feat_good" in result.features_accepted

    def test_execute_sandbox_candidate_pool(self, agent, mock_llm, tool_ctx):
        """execute_sandbox L3 未通过但加入候选池。"""
        gen_resp = _tool_response([("generate_feature", {
            "feature_name": "feat_pool",
            "dimension": ["金额频次"],
            "code": "def f(o, t, c):\n  return o[['wallet_id']]",
            "explanation": "t", "reasoning": "t",
            "knowledge_source": "t", "data_fields_used": [],
        })])
        exec_resp = _tool_response([("execute_sandbox", {"feature_name": "feat_pool"})])
        done_resp = _text_response("完成。")

        mock_llm.chat.side_effect = [gen_resp, exec_resp, done_resp]

        def mock_execute(name, args):
            if name == "generate_feature":
                return json.dumps({"status": "registered"})
            if name == "execute_sandbox":
                return json.dumps({
                    "feature_name": "feat_pool",
                    "stage_reached": "L3",
                    "accepted": False,
                    "note": "特征已加入候选池",
                    "summary": "L2 通过但 L3 未通过",
                })
            return "{}"

        tool_ctx.execute = mock_execute

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        assert result.n_candidates == 1


class TestReactAgentL1Repair:
    def test_l1_failure_triggers_repair_message(self, agent, mock_llm, tool_ctx):
        """L1 失败时构造修复 prompt。"""
        gen_resp = _tool_response([("generate_feature", {
            "feature_name": "feat_broken",
            "dimension": ["金额频次"],
            "code": "broken code",
            "explanation": "t", "reasoning": "t",
            "knowledge_source": "t", "data_fields_used": [],
        })])
        exec_resp = _tool_response([("execute_sandbox", {"feature_name": "feat_broken"})])
        done_resp = _text_response("无法修复。")

        mock_llm.chat.side_effect = [gen_resp, exec_resp, done_resp]

        def mock_execute(name, args):
            if name == "generate_feature":
                return json.dumps({"status": "registered"})
            if name == "execute_sandbox":
                return json.dumps({
                    "feature_name": "feat_broken",
                    "stage_reached": "L1",
                    "accepted": False,
                    "summary": "L1 失败: SyntaxError",
                    "L1_sandbox": {"passed": False, "error": "SyntaxError: invalid syntax"},
                })
            return "{}"

        tool_ctx.execute = mock_execute

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        messages_sent = [c.args[0] for c in mock_llm.chat.call_args_list]
        last_messages = messages_sent[-1]
        repair_msgs = [
            m for m in last_messages
            if m.get("role") == "user" and "修复" in m.get("content", "")
        ]
        assert len(repair_msgs) >= 1

    def test_l1_repair_limited_to_max(self, agent, mock_llm, tool_ctx):
        """L1 修复最多 MAX_L1_REPAIRS 次。"""
        gen_resp = _tool_response([("generate_feature", {
            "feature_name": "feat_repeat",
            "dimension": ["金额频次"],
            "code": "bad",
            "explanation": "t", "reasoning": "t",
            "knowledge_source": "t", "data_fields_used": [],
        })])
        exec_resp = _tool_response([("execute_sandbox", {"feature_name": "feat_repeat"})])

        responses = [gen_resp]
        for _ in range(MAX_L1_REPAIRS + 2):
            responses.append(exec_resp)
        responses.append(_text_response("放弃。"))

        mock_llm.chat.side_effect = responses

        def mock_execute(name, args):
            if name == "generate_feature":
                return json.dumps({"status": "registered"})
            if name == "execute_sandbox":
                return json.dumps({
                    "feature_name": "feat_repeat",
                    "stage_reached": "L1",
                    "accepted": False,
                    "summary": "L1 失败",
                    "L1_sandbox": {"passed": False, "error": "Error"},
                })
            return "{}"

        tool_ctx.execute = mock_execute

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        last_messages = mock_llm.chat.call_args_list[-1].args[0]
        repair_msgs = [
            m for m in last_messages
            if m.get("role") == "user" and "修复" in m.get("content", "")
        ]
        assert len(repair_msgs) <= MAX_L1_REPAIRS


class TestReactAgentMultipleToolCalls:
    def test_multiple_tool_calls_in_single_response(self, agent, mock_llm, tool_ctx):
        """单次响应中多个 tool call。"""
        multi_resp = _tool_response([
            ("explore_data", {"code": "tx_df.shape"}),
            ("explore_data", {"code": "open_df.shape"}),
        ])
        done_resp = _text_response("完成探索。")

        mock_llm.chat.side_effect = [multi_resp, done_resp]

        call_log = []
        def mock_execute(name, args):
            call_log.append((name, args))
            return json.dumps({"output": "ok"})

        tool_ctx.execute = mock_execute

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        assert len(call_log) == 2
        assert all(name == "explore_data" for name, _ in call_log)


class TestReactAgentMessageConstruction:
    def test_system_and_user_messages_present(self, agent, mock_llm):
        """消息列表包含 system 和 user 消息。"""
        mock_llm.chat.return_value = _text_response("完成。")

        agent.run_round(round_num=1, dimension="金额频次", budget=5)

        messages = mock_llm.chat.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "金额频次" in messages[1]["content"]

    def test_tool_result_appended_to_messages(self, agent, mock_llm, tool_ctx):
        """工具结果正确追加到消息列表。"""
        explore_resp = _tool_response([
            ("explore_data", {"code": "42"})
        ])
        done_resp = _text_response("完成。")

        mock_llm.chat.side_effect = [explore_resp, done_resp]
        tool_ctx.execute = MagicMock(
            return_value=json.dumps({"output": "42"})
        )

        agent.run_round(round_num=1, dimension="金额频次", budget=5)

        second_call_messages = mock_llm.chat.call_args_list[1].args[0]
        assert any(m.get("role") == "tool" for m in second_call_messages)
        assert any(m.get("role") == "assistant" for m in second_call_messages)

    def test_assistant_tool_calls_message_format(self, agent, mock_llm, tool_ctx):
        """assistant 消息包含 tool_calls 字段。"""
        explore_resp = _tool_response(
            [("explore_data", {"code": "1+1"})],
            content="让我探索一下",
        )
        done_resp = _text_response("完成。")

        mock_llm.chat.side_effect = [explore_resp, done_resp]
        tool_ctx.execute = MagicMock(
            return_value=json.dumps({"output": "2"})
        )

        agent.run_round(round_num=1, dimension="金额频次", budget=5)

        second_call_messages = mock_llm.chat.call_args_list[1].args[0]
        assistant_msgs = [m for m in second_call_messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        assert "tool_calls" in assistant_msgs[0]


class TestReactAgentDynamicInjection:
    def test_shap_feedback_passed_to_builder(self, agent, mock_llm):
        """SHAP 反馈传递给 prompt builder。"""
        mock_llm.chat.return_value = _text_response("完成。")
        shap = ShapFeedback(
            dimension_contributions={"金额频次": 0.4, "时间地域": 0.3},
            round_num=2,
        )

        with patch("knowfeat.agent.react_agent.build_full_prompt") as mock_build:
            mock_build.return_value = ("system", "user")
            agent.run_round(
                round_num=2, dimension="金额频次", budget=5,
                shap_feedback=shap,
            )
            mock_build.assert_called_once()
            _, kwargs = mock_build.call_args
            assert kwargs["shap_feedback"] is shap

    def test_failure_records_passed_to_builder(self, agent, mock_llm):
        """失败记忆传递给 prompt builder。"""
        mock_llm.chat.return_value = _text_response("完成。")
        records = [
            FailureRecord(feature_name="f1", code="x", reason="L2 失败"),
        ]

        with patch("knowfeat.agent.react_agent.build_full_prompt") as mock_build:
            mock_build.return_value = ("system", "user")
            agent.run_round(
                round_num=1, dimension="金额频次", budget=5,
                failure_records=records,
            )
            _, kwargs = mock_build.call_args
            assert kwargs["failure_records"] is records

    def test_cross_dim_examples_passed_to_builder(self, agent, mock_llm):
        """跨维度示例传递给 prompt builder。"""
        mock_llm.chat.return_value = _text_response("完成。")
        examples = [
            CrossDimExample(
                feature_name="f_cross",
                source_dimension="时间地域",
                code="...",
                metrics={"iv": 0.1},
                shap_rank=1,
            ),
        ]

        with patch("knowfeat.agent.react_agent.build_full_prompt") as mock_build:
            mock_build.return_value = ("system", "user")
            agent.run_round(
                round_num=1, dimension="金额频次", budget=5,
                cross_dim_examples=examples,
            )
            _, kwargs = mock_build.call_args
            assert kwargs["cross_dim_examples"] is examples


class TestReactAgentFullFlow:
    def test_explore_then_generate_then_execute(self, agent, mock_llm, tool_ctx):
        """完整流程：探索 → 生成(自动验证) → 完成。"""
        explore_resp = _tool_response([
            ("explore_data", {"code": "tx_df['amount'].describe()"})
        ])
        gen_resp = _tool_response([("generate_feature", {
            "feature_name": "feat_flow",
            "dimension": ["金额频次"],
            "code": "def feature_flow(o, t, c):\n  return o[['wallet_id']]",
            "explanation": "t", "reasoning": "t",
            "knowledge_source": "t", "data_fields_used": ["amount"],
        })])
        done_resp = _text_response("本轮特征探索完成。")

        # generate_feature now auto-triggers execute_sandbox, so Agent
        # no longer needs a separate execute_sandbox call
        mock_llm.chat.side_effect = [explore_resp, gen_resp, done_resp]

        call_order = []
        def mock_execute(name, args):
            call_order.append(name)
            if name == "explore_data":
                return json.dumps({"output": "count 200\nmean 1000"})
            if name == "generate_feature":
                return json.dumps({"status": "registered", "feature_name": "feat_flow"})
            if name == "execute_sandbox":
                return json.dumps({
                    "feature_name": "feat_flow",
                    "stage_reached": "L3",
                    "accepted": True,
                    "summary": "L3 通过 — 已接受!",
                })
            return "{}"

        tool_ctx.execute = mock_execute

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        # generate_feature triggers auto execute_sandbox
        assert call_order == ["explore_data", "generate_feature", "execute_sandbox"]
        assert result.n_generated == 1
        assert result.n_accepted == 1
        assert result.terminated_reason == "agent_completed"
        assert result.steps == 3

    def test_multiple_features_in_round(self, agent, mock_llm, tool_ctx):
        """单轮生成多个特征（generate_feature 自动触发验证）。"""
        responses = []
        for i in range(3):
            responses.append(_tool_response([("generate_feature", {
                "feature_name": f"feat_{i}",
                "dimension": ["金额频次"],
                "code": f"def feature_{i}(o, t, c):\n  return o[['wallet_id']]",
                "explanation": "t", "reasoning": "t",
                "knowledge_source": "t", "data_fields_used": [],
            })]))
        responses.append(_text_response("完成。"))

        mock_llm.chat.side_effect = responses

        def mock_execute(name, args):
            if name == "generate_feature":
                return json.dumps({"status": "registered"})
            if name == "execute_sandbox":
                return json.dumps({
                    "feature_name": args.get("feature_name"),
                    "stage_reached": "L3",
                    "accepted": True,
                    "summary": "通过",
                })
            return "{}"

        tool_ctx.execute = mock_execute

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)

        assert result.n_generated == 3
        assert result.n_accepted == 3
        assert len(result.features_generated) == 3
        assert len(result.features_accepted) == 3


class TestReactAgentEdgeCases:
    def test_tool_execution_error(self, agent, mock_llm, tool_ctx):
        """工具执行抛出异常时不崩溃。"""
        explore_resp = _tool_response([
            ("explore_data", {"code": "crash"})
        ])
        done_resp = _text_response("完成。")

        mock_llm.chat.side_effect = [explore_resp, done_resp]
        tool_ctx.execute = MagicMock(
            return_value=json.dumps({"error": "工具执行异常: RuntimeError"})
        )

        result = agent.run_round(round_num=1, dimension="金额频次", budget=5)
        assert result.terminated_reason == "agent_completed"

    def test_empty_budget(self, agent, mock_llm):
        """预算为 0 时第一次 generate 就触发终止。"""
        mock_llm.chat.return_value = _text_response("预算为 0。")

        result = agent.run_round(round_num=1, dimension="金额频次", budget=0)
        assert result.terminated_reason == "agent_completed"

    def test_round_result_includes_dimension(self, agent, mock_llm):
        """结果包含正确的维度信息。"""
        mock_llm.chat.return_value = _text_response("完成。")

        for dim in ["金额频次", "时间地域", "行为序列", "主体关联"]:
            result = agent.run_round(round_num=1, dimension=dim, budget=5)
            assert result.dimension == dim

    def test_round_num_passed_correctly(self, agent, mock_llm):
        """轮次编号正确传递。"""
        mock_llm.chat.return_value = _text_response("完成。")

        for r in [1, 5, 10]:
            result = agent.run_round(round_num=r, dimension="金额频次", budget=5)
            assert result.round_num == r

    def test_tool_specs_passed_to_llm(self, agent, mock_llm, tool_ctx):
        """工具定义传递给 LLM。"""
        mock_llm.chat.return_value = _text_response("完成。")

        agent.run_round(round_num=1, dimension="金额频次", budget=5)

        _, kwargs = mock_llm.chat.call_args
        tools = kwargs.get("tools") or mock_llm.chat.call_args.args[1]
        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert tool_names == {"explore_data", "generate_feature", "execute_sandbox"}
