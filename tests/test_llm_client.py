"""Tests for knowfeat.llm.client."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from knowfeat.config import LLMConfig
from knowfeat.llm.client import (
    LLMClient,
    LLMResponse,
    ToolCall,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------

class TestToolSpec:
    def test_to_openai_tool_format(self):
        spec = ToolSpec(
            name="explore_data",
            description="Execute Python code on DataFrames",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code"},
                },
                "required": ["code"],
            },
        )
        result = spec.to_openai_tool()
        assert result["type"] == "function"
        assert result["function"]["name"] == "explore_data"
        assert result["function"]["description"] == "Execute Python code on DataFrames"
        assert "properties" in result["function"]["parameters"]

    def test_multiple_tools(self):
        specs = [
            ToolSpec(name="tool_a", description="A", parameters={"type": "object"}),
            ToolSpec(name="tool_b", description="B", parameters={"type": "object"}),
        ]
        tools = [s.to_openai_tool() for s in specs]
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "tool_a"
        assert tools[1]["function"]["name"] == "tool_b"


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_no_tool_calls(self):
        resp = LLMResponse(content="Hello", finish_reason="stop")
        assert not resp.has_tool_calls
        assert resp.content == "Hello"

    def test_with_tool_calls(self):
        resp = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="tc_1", name="explore_data", arguments={"code": "print(1)"})],
            finish_reason="tool_calls",
        )
        assert resp.has_tool_calls
        assert resp.tool_calls[0].name == "explore_data"

    def test_empty_content(self):
        resp = LLMResponse(content=None)
        assert resp.content is None
        assert not resp.has_tool_calls


# ---------------------------------------------------------------------------
# Helper: mock OpenAI response
# ---------------------------------------------------------------------------

def _make_mock_response(content=None, tool_calls=None, finish_reason="stop"):
    """Build a mock that mimics OpenAI ChatCompletion response structure."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _make_mock_tool_call(tc_id, name, arguments_dict):
    func = SimpleNamespace(name=name, arguments=json.dumps(arguments_dict, ensure_ascii=False))
    return SimpleNamespace(id=tc_id, function=func)


# ---------------------------------------------------------------------------
# LLMClient — text responses
# ---------------------------------------------------------------------------

class TestLLMClientTextResponse:
    @patch("knowfeat.llm.client.OpenAI")
    def test_chat_text_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="Feature idea: tx_count_per_hour"
        )

        client = LLMClient(LLMConfig(api_key="test-key"))
        messages = [{"role": "user", "content": "Suggest a feature"}]
        resp = client.chat(messages)

        assert resp.content == "Feature idea: tx_count_per_hour"
        assert not resp.has_tool_calls
        assert resp.finish_reason == "stop"

    @patch("knowfeat.llm.client.OpenAI")
    def test_ask_convenience(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="42"
        )

        client = LLMClient(LLMConfig(api_key="test-key"))
        result = client.ask("What is 6*7?", system="You are a calculator")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert result == "42"


# ---------------------------------------------------------------------------
# LLMClient — Function Calling
# ---------------------------------------------------------------------------

class TestLLMClientFunctionCalling:
    @patch("knowfeat.llm.client.OpenAI")
    def test_chat_with_tool_calls(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        tc = _make_mock_tool_call("call_1", "explore_data", {"code": "df.shape"})
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        )

        client = LLMClient(LLMConfig(api_key="test-key"))
        tools = [ToolSpec(
            name="explore_data",
            description="Run code",
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )]
        resp = client.chat(
            [{"role": "user", "content": "Explore data"}],
            tools=tools,
        )

        assert resp.has_tool_calls
        assert resp.tool_calls[0].id == "call_1"
        assert resp.tool_calls[0].name == "explore_data"
        assert resp.tool_calls[0].arguments == {"code": "df.shape"}

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["function"]["name"] == "explore_data"

    @patch("knowfeat.llm.client.OpenAI")
    def test_multiple_tool_calls(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        tc1 = _make_mock_tool_call("call_1", "explore_data", {"code": "df.head()"})
        tc2 = _make_mock_tool_call("call_2", "generate_feature", {"name": "feat_a"})
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="Let me explore and generate", tool_calls=[tc1, tc2], finish_reason="tool_calls"
        )

        client = LLMClient(LLMConfig(api_key="test-key"))
        resp = client.chat([{"role": "user", "content": "Do things"}])

        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "explore_data"
        assert resp.tool_calls[1].name == "generate_feature"

    @patch("knowfeat.llm.client.OpenAI")
    def test_malformed_tool_arguments(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        func = SimpleNamespace(name="explore_data", arguments="not valid json{")
        tc = SimpleNamespace(id="call_bad", function=func)
        mock_client.chat.completions.create.return_value = _make_mock_response(
            tool_calls=[tc], finish_reason="tool_calls"
        )

        client = LLMClient(LLMConfig(api_key="test-key"))
        resp = client.chat([{"role": "user", "content": "test"}])

        assert resp.has_tool_calls
        assert resp.tool_calls[0].arguments == {"_raw": "not valid json{"}


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

class TestMessageHelpers:
    def test_make_tool_result_message(self):
        msg = LLMClient.make_tool_result_message("call_1", "shape: (100, 5)")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert msg["content"] == "shape: (100, 5)"

    def test_make_assistant_tool_calls_message(self):
        tcs = [
            ToolCall(id="call_1", name="explore_data", arguments={"code": "x"}),
            ToolCall(id="call_2", name="generate_feature", arguments={"name": "f"}),
        ]
        msg = LLMClient.make_assistant_tool_calls_message("Thinking...", tcs)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Thinking..."
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["id"] == "call_1"
        assert msg["tool_calls"][0]["function"]["name"] == "explore_data"
        parsed = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        assert parsed == {"code": "x"}

    def test_assistant_message_no_content(self):
        tcs = [ToolCall(id="call_1", name="test", arguments={})]
        msg = LLMClient.make_assistant_tool_calls_message(None, tcs)
        assert "content" not in msg
        assert len(msg["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    @patch("knowfeat.llm.client.time.sleep")
    @patch("knowfeat.llm.client.OpenAI")
    def test_retries_on_connection_error(self, mock_openai_cls, mock_sleep):
        from openai import APIConnectionError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            _make_mock_response(content="Success"),
        ]

        cfg = LLMConfig(api_key="test-key", max_retries=2)
        client = LLMClient(cfg)
        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "Success"
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()

    @patch("knowfeat.llm.client.time.sleep")
    @patch("knowfeat.llm.client.OpenAI")
    def test_retries_on_rate_limit(self, mock_openai_cls, mock_sleep):
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limited",
                response=mock_response,
                body=None,
            ),
            _make_mock_response(content="OK"),
        ]

        cfg = LLMConfig(api_key="test-key", max_retries=2)
        client = LLMClient(cfg)
        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "OK"

    @patch("knowfeat.llm.client.time.sleep")
    @patch("knowfeat.llm.client.OpenAI")
    def test_raises_after_max_retries(self, mock_openai_cls, mock_sleep):
        from openai import APIConnectionError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )

        cfg = LLMConfig(api_key="test-key", max_retries=3)
        client = LLMClient(cfg)

        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            client.chat([{"role": "user", "content": "hi"}])

        assert mock_client.chat.completions.create.call_count == 3

    @patch("knowfeat.llm.client.time.sleep")
    @patch("knowfeat.llm.client.OpenAI")
    def test_exponential_backoff(self, mock_openai_cls, mock_sleep):
        from openai import APIConnectionError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            APIConnectionError(request=MagicMock()),
            _make_mock_response(content="OK"),
        ]

        cfg = LLMConfig(api_key="test-key", max_retries=3)
        client = LLMClient(cfg)
        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "OK"
        assert mock_sleep.call_count == 2
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls[0] == 2   # 2^1
        assert calls[1] == 4   # 2^2


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

class TestConfigWiring:
    @patch("knowfeat.llm.client.OpenAI")
    def test_passes_config_to_openai(self, mock_openai_cls):
        cfg = LLMConfig(
            api_key="my-key",
            base_url="https://custom.api/v1",
            timeout=60,
        )
        LLMClient(cfg)
        mock_openai_cls.assert_called_once_with(
            api_key="my-key",
            base_url="https://custom.api/v1",
            timeout=60,
        )

    @patch("knowfeat.llm.client.OpenAI")
    def test_default_config(self, mock_openai_cls):
        client = LLMClient()
        assert client.config.model == "deepseek-chat"

    @patch("knowfeat.llm.client.OpenAI")
    def test_temperature_override(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(content="ok")

        client = LLMClient(LLMConfig(api_key="k"))
        client.chat([{"role": "user", "content": "hi"}], temperature=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9

    @patch("knowfeat.llm.client.OpenAI")
    def test_max_tokens_override(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(content="ok")

        client = LLMClient(LLMConfig(api_key="k"))
        client.chat([{"role": "user", "content": "hi"}], max_tokens=1024)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1024


# ---------------------------------------------------------------------------
# Multi-turn conversation flow
# ---------------------------------------------------------------------------

class TestMultiTurnConversation:
    """Verify the full ReAct-style message assembly works."""

    @patch("knowfeat.llm.client.OpenAI")
    def test_full_tool_call_round_trip(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        tc = _make_mock_tool_call("call_1", "explore_data", {"code": "df.shape"})
        step1_resp = _make_mock_response(
            content="Let me explore.", tool_calls=[tc], finish_reason="tool_calls"
        )
        step2_resp = _make_mock_response(
            content="The data has 100 rows.", finish_reason="stop"
        )
        mock_client.chat.completions.create.side_effect = [step1_resp, step2_resp]

        client = LLMClient(LLMConfig(api_key="k"))

        messages = [{"role": "user", "content": "Analyze the data"}]
        resp1 = client.chat(messages)

        assert resp1.has_tool_calls
        assert resp1.tool_calls[0].name == "explore_data"

        messages.append(
            LLMClient.make_assistant_tool_calls_message(resp1.content, resp1.tool_calls)
        )
        messages.append(
            LLMClient.make_tool_result_message("call_1", "Shape: (100, 5)")
        )
        resp2 = client.chat(messages)

        assert not resp2.has_tool_calls
        assert resp2.content == "The data has 100 rows."
        assert mock_client.chat.completions.create.call_count == 2
