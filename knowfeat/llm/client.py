"""OpenAI 兼容 LLM 客户端 — 支持 Function Calling + 重试"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from knowfeat.config import LLMConfig

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


# ---------------------------------------------------------------------------
# Tool / Message 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """Function Calling 工具定义。"""
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """LLM 返回的一次工具调用。"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """单次 LLM 响应。"""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class LLMClient:
    """OpenAI 兼容 LLM 客户端（DeepSeek / GPT / 其他）。"""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    # ------------------------------------------------------------------
    # 核心调用
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """发送 chat completion 请求，支持 Function Calling。

        Parameters
        ----------
        messages : OpenAI 格式的消息列表
        tools : 可选的工具定义列表
        temperature / max_tokens : 覆盖默认值

        Returns
        -------
        LLMResponse
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]

        response = self._request_with_retry(**kwargs)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # 便捷方法：单轮文本问答
    # ------------------------------------------------------------------
    def ask(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        """单轮文本问答，返回纯文本。"""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.chat(messages, **kwargs)
        return resp.content or ""

    # ------------------------------------------------------------------
    # 会话管理辅助
    # ------------------------------------------------------------------
    @staticmethod
    def make_tool_result_message(tool_call_id: str, content: str) -> dict[str, Any]:
        """构造 tool result 消息，用于拼接回 messages 列表。"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    @staticmethod
    def make_assistant_tool_calls_message(
        content: str | None,
        tool_calls: list[ToolCall],
    ) -> dict[str, Any]:
        """将 LLMResponse 中的 assistant 消息还原为 OpenAI 格式，用于多轮对话。"""
        msg: dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ]
        return msg

    # ------------------------------------------------------------------
    # 重试逻辑
    # ------------------------------------------------------------------
    def _request_with_retry(self, **kwargs) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except RETRYABLE_ERRORS as exc:
                last_exc = exc
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, self.config.max_retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"LLM request failed after {self.config.max_retries} retries"
        ) from last_exc

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(response) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            raw=response,
        )
