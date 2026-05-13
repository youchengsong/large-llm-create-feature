"""LLM 可插拔接口：支持 OpenAI / Gemini / DeepSeek / Local / Mock。"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from src.config import LLMConfig


class LLMInterface(ABC):
    """LLM 基类，所有 provider 继承此类。"""

    def __init__(self, model_name: str, api_key: str = "", base_url: str | None = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.timeout = kwargs.get("timeout", 120)

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 返回文本。"""

    @classmethod
    def from_config(cls, config: LLMConfig) -> "LLMInterface":
        """根据 LLMConfig 创建对应 provider 实例。"""
        provider_map = {
            "openai": OpenAILLM,
            "gemini": GeminiLLM,
            "deepseek": DeepSeekLLM,
            "local": LocalLLM,
            "mock": MockLLM,
        }
        llm_cls = provider_map.get(config.provider)
        if llm_cls is None:
            raise ValueError(f"Unknown provider: {config.provider}")
        return llm_cls(
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )


class OpenAILLM(LLMInterface):
    """GPT 系列，使用 openai SDK。"""

    def __init__(self, model_name: str, api_key: str = "", base_url: str | None = None, **kwargs):
        super().__init__(model_name, api_key, base_url, **kwargs)
        from openai import OpenAI
        client_kwargs = {"api_key": api_key, "timeout": self.timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class GeminiLLM(LLMInterface):
    """Google Gemini，使用 google-genai SDK。"""

    def __init__(self, model_name: str, api_key: str = "", base_url: str | None = None, **kwargs):
        super().__init__(model_name, api_key, base_url, **kwargs)
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import types
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text or ""


class DeepSeekLLM(LLMInterface):
    """DeepSeek，通过 OpenAI 兼容接口。"""

    def __init__(self, model_name: str, api_key: str = "", base_url: str | None = None, **kwargs):
        base_url = base_url or "https://api.deepseek.com"
        super().__init__(model_name, api_key, base_url, **kwargs)
        from openai import OpenAI
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class LocalLLM(LLMInterface):
    """开源模型，通过 OpenAI 兼容接口（如 vLLM / Ollama）。"""

    def __init__(self, model_name: str, api_key: str = "", base_url: str | None = None, **kwargs):
        base_url = base_url or "http://localhost:8000/v1"
        api_key = api_key or "not-needed"
        super().__init__(model_name, api_key, base_url, **kwargs)
        from openai import OpenAI
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


_MOCK_FEATURE_TEMPLATES = [
    {
        "feature_name": "mock_amount_std",
        "explanation": "计算钱包历史交易金额的标准差，高标准差可能表示异常交易模式",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "stats = tx_df.groupby('src')['amount'].std().reset_index()\n"
            "stats.columns = ['wallet_id', 'mock_amount_std']\n"
            "stats['mock_amount_std'] = stats['mock_amount_std'].fillna(0)\n"
            "result = stats\n"
        ),
    },
    {
        "feature_name": "mock_tx_count",
        "explanation": "计算每个钱包的交易次数，高频交易可能暗示刷单或洗钱行为",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "counts = tx_df.groupby('src').size().reset_index(name='mock_tx_count')\n"
            "counts.columns = ['wallet_id', 'mock_tx_count']\n"
            "result = counts\n"
        ),
    },
    {
        "feature_name": "mock_amount_mean",
        "explanation": "计算每个钱包的平均交易金额",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "means = tx_df.groupby('src')['amount'].mean().reset_index()\n"
            "means.columns = ['wallet_id', 'mock_amount_mean']\n"
            "result = means\n"
        ),
    },
    {
        "feature_name": "mock_unique_dst",
        "explanation": "计算每个钱包的不同交易对手数量",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "dst_counts = tx_df.groupby('src')['dst'].nunique().reset_index()\n"
            "dst_counts.columns = ['wallet_id', 'mock_unique_dst']\n"
            "result = dst_counts\n"
        ),
    },
]


class MockLLM(LLMInterface):
    """测试用 mock，返回预定义特征 JSON。"""

    def __init__(self, model_name: str = "mock", api_key: str = "", base_url: str | None = None, **kwargs):
        super().__init__(model_name or "mock", api_key, base_url, **kwargs)
        self._call_count = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._call_count += 1
        idx = (self._call_count - 1) % len(_MOCK_FEATURE_TEMPLATES)
        tmpl = _MOCK_FEATURE_TEMPLATES[idx]
        feat = dict(tmpl)
        feat["feature_name"] = f"{tmpl['feature_name']}_{self._call_count}"
        feat["code"] = tmpl["code"].replace(tmpl["feature_name"], feat["feature_name"])
        return json.dumps([feat], ensure_ascii=False, indent=2)
