"""多Agent特征工程系统：Generator + Critic 对抗式协作。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from openai import OpenAI

from src.data.merge import merge_feature_to_transactions
from src.evaluation.validator import run_l1_validation, run_l2_validation, run_l3_validation
from src.prompt.domain_knowledge import (
    load_transaction_patterns,
    load_expert_interview,
)

logger = logging.getLogger(__name__)


# ============================================================
# Agent Base
# ============================================================

class BaseAgent:
    """Agent基类，封装LLM对话能力。"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        client: OpenAI,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.temperature = temperature
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.call_count: int = 0

    def chat(self, user_message: str) -> str:
        """发送消息并获取回复。"""
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
        )

        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        self.call_count += 1
        return reply

    def reset_context(self):
        """重置对话上下文（保留system prompt）。"""
        self.messages = [{"role": "system", "content": self.system_prompt}]


# ============================================================
# Generator Agent
# ============================================================

GENERATOR_SYSTEM_PROMPT = """你是一个专业的反洗钱特征工程专家（Generator Agent）。你的任务是为数字人民币交易数据设计和生成高质量的检测特征。

## 你的职责
1. 基于领域知识设计有区分力的特征
2. 编写完整的Python特征代码
3. 根据Critic的反馈修改和优化特征

## 代码要求
- 函数签名: def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame
- 返回DataFrame必须包含 wallet_id 或 tx_id 列作为合并键
- 不能使用标签列(is_risk, is_laundering, is_src_victim等)
- 需要处理空值，避免除零错误
- import语句放在函数内部

## 输出格式
每次生成一个特征时，请严格按以下JSON格式输出：
```json
{
    "feature_name": "特征名(英文snake_case)",
    "dimension": "所属维度",
    "code": "完整Python函数代码",
    "explanation": "业务含义说明",
    "reasoning": "设计推理过程"
}
```

## 设计原则
- 特征应该有明确的业务含义，可向监管人员解释
- 优先考虑洗钱者难以规避的结构性特征
- 避免与已有特征高度相关的冗余设计"""


CRITIC_SYSTEM_PROMPT = """你是一个反洗钱对抗性审查专家（Critic Agent）。你同时具备两个视角：
1. 作为洗钱者：思考如何规避检测特征
2. 作为审计员：评估特征的实际部署可行性

## 你的职责
对Generator生成的每个特征进行严格审查，从以下维度评估：

### 1. 鲁棒性（洗钱者能否轻易规避？）
- 该特征是否依赖了容易伪造的信号（如交易时间可以人为调整）？
- 洗钱者是否可以通过简单改变行为来绕过该特征？
- 该特征是否对正常用户行为变化过度敏感（高误报率）？

### 2. 可部署性（实际系统中能否使用？）
- 该特征是否需要全局统计信息（无法实时计算）？
- 计算复杂度是否合理？
- 是否依赖了线上环境可能获取不到的字段？

### 3. 冗余性（是否与已有特征重复？）
- 是否本质上测量了与已有特征相同的信号？
- 能否通过已有特征的简单组合得到类似效果？

### 4. 可解释性（监管审计能否理解？）
- 该特征的业务含义是否清晰？
- 触发高风险值时能否给出明确的解释？

## 输出格式
请严格按以下JSON格式输出审查意见：
```json
{
    "verdict": "pass" 或 "revise" 或 "reject",
    "robustness_score": 1-5分,
    "deployability_score": 1-5分,
    "redundancy_risk": "low/medium/high",
    "explainability_score": 1-5分,
    "issues": ["问题1", "问题2"],
    "suggestions": ["改进建议1", "改进建议2"],
    "evasion_scenario": "洗钱者可能如何规避该特征的具体场景描述"
}
```

## 审查原则
- 不要过于苛刻，完美的特征不存在
- 重点关注"致命缺陷"：容易规避、数据泄露、无法部署
- 如果特征整体合理只是有小改进空间，给"pass"并附上建议
- 只有存在实质性问题时才给"revise"或"reject"
"""


# ============================================================
# Validator (Tool-based, not LLM)
# ============================================================

class Validator:
    """验证器：沙箱执行 + 统计评估（非LLM，确定性逻辑）。"""

    def __init__(
        self,
        open_df: pd.DataFrame,
        tx_df: pd.DataFrame,
        close_df: pd.DataFrame,
        label: pd.Series,
        existing_features: pd.DataFrame,
    ):
        self.open_df = open_df
        self.tx_df = tx_df
        self.close_df = close_df
        self.label = label
        self.existing_features = existing_features

    def validate_code(self, code: str) -> tuple[pd.DataFrame | None, str]:
        """L1沙箱验证。"""
        return run_l1_validation(code, self.open_df, self.tx_df, self.close_df)

    def evaluate_quality(self, feature_df: pd.DataFrame, name: str) -> dict:
        """L2+L3评估。"""
        try:
            merged = merge_feature_to_transactions(feature_df, self.tx_df)
        except Exception as e:
            return {"passed": False, "error": f"合并失败: {e}"}

        if len(merged) != len(self.tx_df):
            return {"passed": False, "error": f"行数不匹配: {len(merged)} vs {len(self.tx_df)}"}

        feat_cols = [c for c in merged.columns if c not in self.tx_df.columns]
        if not feat_cols:
            return {"passed": False, "error": "无新列产生"}

        feature_series = merged[feat_cols[0]].fillna(0)
        feature_series.name = name

        l2_passed, l2_metrics = run_l2_validation(
            feature_series, self.label, self.existing_features
        )
        if not l2_passed:
            return {"passed": False, "stage": "L2", "metrics": l2_metrics}

        l3_passed, l3_metrics = run_l3_validation(
            self.existing_features, feature_series, self.label
        )

        return {
            "passed": l3_passed,
            "stage": "L3" if not l3_passed else "accepted",
            "l2_metrics": l2_metrics,
            "l3_metrics": l3_metrics,
            "feature_series": feature_series,
        }


# ============================================================
# Multi-Agent Orchestrator
# ============================================================

@dataclass
class MultiAgentResult:
    """多Agent系统运行结果。"""
    accepted_features: list[dict] = field(default_factory=list)
    rejected_features: list[dict] = field(default_factory=list)
    interaction_log: list[dict] = field(default_factory=list)
    total_rounds: int = 0
    generator_calls: int = 0
    critic_calls: int = 0


class MultiAgentFeatureEngine:
    """
    多Agent特征工程系统。

    协作流程：
    Generator → Validator(沙箱) → Critic(对抗审查) → [修改循环] → 最终评估

    相比单Agent的优势：
    1. Generator专注生成，Critic专注审查，各司其职
    2. 对抗性审查确保特征的鲁棒性（洗钱者难以规避）
    3. 特征通过双重把关：统计有效性 + 业务合理性
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        open_df: pd.DataFrame = None,
        tx_df: pd.DataFrame = None,
        close_df: pd.DataFrame = None,
        label: pd.Series = None,
        baseline_features: pd.DataFrame = None,
        max_features: int = 20,
        max_revisions: int = 2,
        temperature: float = 0.7,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.open_df = open_df
        self.tx_df = tx_df
        self.close_df = close_df
        self.label = label
        self.existing_features = baseline_features.copy() if baseline_features is not None else pd.DataFrame()
        self.max_features = max_features
        self.max_revisions = max_revisions

        # 初始化三个组件
        self.generator = BaseAgent(
            name="Generator",
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            client=self.client,
            model=model,
            temperature=temperature,
        )
        self.critic = BaseAgent(
            name="Critic",
            system_prompt=CRITIC_SYSTEM_PROMPT,
            client=self.client,
            model=model,
            temperature=0.3,  # Critic用较低温度，更确定性
        )
        self.validator = Validator(
            open_df=open_df,
            tx_df=tx_df,
            close_df=close_df,
            label=label,
            existing_features=self.existing_features,
        )

        self.result = MultiAgentResult()
        self._knowledge_cache: str = ""

    def _load_knowledge(self) -> str:
        """加载领域知识（缓存）。"""
        if not self._knowledge_cache:
            self._knowledge_cache = (
                load_transaction_patterns() + "\n\n" + load_expert_interview()
            )
        return self._knowledge_cache

    def _get_schema_info(self) -> str:
        """获取数据schema信息。"""
        lines = []
        for name, df in [("open_df", self.open_df), ("tx_df", self.tx_df), ("close_df", self.close_df)]:
            cols = [f"{c}({df[c].dtype})" for c in df.columns]
            lines.append(f"{name} [{df.shape[0]}行]: {', '.join(cols)}")
        return "\n".join(lines)

    def _build_generation_prompt(self, round_num: int, dimension: str = "") -> str:
        """构建特征生成请求。"""
        knowledge = self._load_knowledge()
        schema = self._get_schema_info()

        accepted_names = [f["name"] for f in self.result.accepted_features]
        rejected_info = [
            f"- {f['name']}: {f.get('reject_reason', '')}"
            for f in self.result.rejected_features[-5:]
        ]

        prompt = f"""## 任务：生成第 {round_num} 个反洗钱特征

### 数据Schema
{schema}

### 领域知识
{knowledge[:3000]}

### 已接受的特征（避免重复）
{', '.join(accepted_names) if accepted_names else '暂无'}

### 最近被拒绝的特征（避免类似方向）
{chr(10).join(rejected_info) if rejected_info else '暂无'}
"""
        if dimension:
            prompt += f"\n### 本轮指定维度: {dimension}\n请围绕该维度设计特征。"
        else:
            prompt += "\n请自主选择一个有潜力的维度进行特征设计。"

        prompt += "\n\n请生成一个特征（JSON格式）。"
        return prompt

    def _build_critic_prompt(self, feature: dict, accepted_names: list[str]) -> str:
        """构建Critic审查请求。"""
        return f"""请审查以下特征：

## 特征信息
- 名称: {feature['feature_name']}
- 维度: {feature.get('dimension', '未指定')}
- 解释: {feature.get('explanation', '')}
- 推理: {feature.get('reasoning', '')}

## 代码
```python
{feature.get('code', '')}
```

## 上下文
- 已有特征: {', '.join(accepted_names) if accepted_names else '暂无'}
- 该特征已通过沙箱执行验证（代码无错误，输出格式正确）

请从鲁棒性、可部署性、冗余性、可解释性四个维度进行审查，输出JSON格式的审查意见。"""

    def _build_revision_prompt(self, feature: dict, critique: dict) -> str:
        """构建修改请求。"""
        return f"""你之前生成的特征被Critic审查后需要修改。

## 原特征
- 名称: {feature['feature_name']}
- 代码:
```python
{feature.get('code', '')}
```

## Critic的审查意见
- 判定: {critique.get('verdict', '')}
- 鲁棒性: {critique.get('robustness_score', '')}/5
- 可部署性: {critique.get('deployability_score', '')}/5
- 问题: {json.dumps(critique.get('issues', []), ensure_ascii=False)}
- 建议: {json.dumps(critique.get('suggestions', []), ensure_ascii=False)}
- 规避场景: {critique.get('evasion_scenario', '')}

请根据以上反馈修改特征，使其更加鲁棒和实用。输出完整的JSON格式（包含修改后的代码）。"""

    def _parse_feature_json(self, text: str) -> dict | None:
        """从LLM输出中解析特征JSON。"""
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
        return None

    def _parse_critique_json(self, text: str) -> dict:
        """从Critic输出中解析审查JSON。"""
        parsed = self._parse_feature_json(text)
        if parsed and "verdict" in parsed:
            return parsed
        return {"verdict": "pass", "issues": [], "suggestions": []}

    def generate_one_feature(self, round_num: int, dimension: str = "") -> dict | None:
        """完整的单特征生成流程：生成→验证→审查→[修改]→评估。"""

        interaction = {
            "round": round_num,
            "dimension": dimension,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": [],
        }

        # === Step 1: Generator 生成 ===
        gen_prompt = self._build_generation_prompt(round_num, dimension)
        gen_response = self.generator.chat(gen_prompt)
        feature = self._parse_feature_json(gen_response)

        if not feature or "code" not in feature:
            interaction["steps"].append({"agent": "Generator", "action": "generate", "result": "解析失败"})
            self.result.interaction_log.append(interaction)
            return None

        interaction["steps"].append({
            "agent": "Generator",
            "action": "generate",
            "feature_name": feature.get("feature_name", ""),
            "result": "成功生成",
        })

        # === Step 2: Validator 沙箱验证 ===
        code = feature.get("code", "")
        feature_df, error = self.validator.validate_code(code)

        if feature_df is None:
            # 让Generator修复
            fix_prompt = f"代码执行失败，错误: {error}\n请修复代码并重新输出完整JSON。"
            fix_response = self.generator.chat(fix_prompt)
            feature = self._parse_feature_json(fix_response) or feature
            code = feature.get("code", code)
            feature_df, error = self.validator.validate_code(code)

            if feature_df is None:
                interaction["steps"].append({"agent": "Validator", "action": "sandbox", "result": f"失败: {error}"})
                self.result.rejected_features.append({
                    "name": feature.get("feature_name", ""),
                    "reject_reason": f"沙箱执行失败: {error}",
                })
                self.result.interaction_log.append(interaction)
                return None

        interaction["steps"].append({"agent": "Validator", "action": "sandbox", "result": "通过"})

        # === Step 3: Critic 对抗审查 ===
        accepted_names = [f["name"] for f in self.result.accepted_features]
        critic_prompt = self._build_critic_prompt(feature, accepted_names)
        critic_response = self.critic.chat(critic_prompt)
        critique = self._parse_critique_json(critic_response)

        interaction["steps"].append({
            "agent": "Critic",
            "action": "review",
            "verdict": critique.get("verdict", ""),
            "robustness": critique.get("robustness_score", ""),
            "issues": critique.get("issues", []),
        })

        # === Step 4: 如果Critic要求修改，进入修改循环 ===
        revision_count = 0
        while critique.get("verdict") == "revise" and revision_count < self.max_revisions:
            revision_count += 1
            rev_prompt = self._build_revision_prompt(feature, critique)
            rev_response = self.generator.chat(rev_prompt)
            revised = self._parse_feature_json(rev_response)

            if not revised or "code" not in revised:
                break

            feature = revised
            code = feature.get("code", "")

            # 重新沙箱验证
            feature_df, error = self.validator.validate_code(code)
            if feature_df is None:
                interaction["steps"].append({
                    "agent": "Validator", "action": f"re-sandbox(rev{revision_count})", "result": f"失败: {error}"
                })
                break

            # 重新Critic审查
            critic_prompt = self._build_critic_prompt(feature, accepted_names)
            critic_response = self.critic.chat(critic_prompt)
            critique = self._parse_critique_json(critic_response)

            interaction["steps"].append({
                "agent": "Critic",
                "action": f"re-review(rev{revision_count})",
                "verdict": critique.get("verdict", ""),
                "robustness": critique.get("robustness_score", ""),
            })

        # === Step 5: 如果Critic最终reject ===
        if critique.get("verdict") == "reject":
            self.result.rejected_features.append({
                "name": feature.get("feature_name", ""),
                "reject_reason": f"Critic拒绝: {critique.get('issues', [])}",
                "critique": critique,
            })
            interaction["steps"].append({"agent": "System", "action": "reject", "result": "Critic最终拒绝"})
            self.result.interaction_log.append(interaction)
            return None

        # === Step 6: 最终统计+模型评估 ===
        if feature_df is None:
            self.result.interaction_log.append(interaction)
            return None

        name = feature.get("feature_name", f"feature_{round_num}")
        eval_result = self.validator.evaluate_quality(feature_df, name)

        if not eval_result.get("passed"):
            interaction["steps"].append({
                "agent": "Validator",
                "action": "evaluate",
                "result": f"未通过({eval_result.get('stage', '')}: {eval_result.get('error', '')})",
            })
            self.result.rejected_features.append({
                "name": name,
                "reject_reason": f"评估未通过: {eval_result.get('stage', '')}",
                "metrics": eval_result.get("l2_metrics", {}),
            })
            self.result.interaction_log.append(interaction)
            return None

        # === 接受特征 ===
        feature_series = eval_result["feature_series"]
        self.existing_features[name] = feature_series.values
        self.validator.existing_features = self.existing_features

        feature_record = {
            "name": name,
            "code": feature.get("code", ""),
            "explanation": feature.get("explanation", ""),
            "reasoning": feature.get("reasoning", ""),
            "dimension": feature.get("dimension", ""),
            "l2_metrics": eval_result.get("l2_metrics", {}),
            "l3_metrics": eval_result.get("l3_metrics", {}),
            "critique": critique,
            "revisions": revision_count,
        }
        self.result.accepted_features.append(feature_record)

        interaction["steps"].append({
            "agent": "System",
            "action": "accept",
            "result": f"通过 (IV={eval_result.get('l2_metrics', {}).get('iv', 0):.4f})",
        })
        self.result.interaction_log.append(interaction)

        logger.info(
            "Round %d: 接受特征 '%s' (维度=%s, 修改次数=%d, 鲁棒性=%s/5)",
            round_num, name, feature.get("dimension", ""),
            revision_count, critique.get("robustness_score", "?")
        )

        return feature_record

    def run(self, dimensions: list[str] | None = None) -> MultiAgentResult:
        """
        运行多Agent系统。

        Args:
            dimensions: 指定探索的维度列表。如果为None，由Generator自主决定。
        """
        if dimensions is None:
            dimensions = ["时间模式", "金额模式", "网络结构", "账户行为", "设备/环境", ""]

        logger.info("=== MultiAgent特征工程系统启动 ===")
        logger.info("目标: %d 个特征, 探索维度: %s", self.max_features, dimensions)

        round_num = 0
        dim_index = 0
        consecutive_failures = 0

        while len(self.result.accepted_features) < self.max_features:
            round_num += 1
            if round_num > self.max_features * 3:
                logger.warning("达到最大尝试次数，停止")
                break

            # 轮流指定维度
            dim = dimensions[dim_index % len(dimensions)] if dimensions else ""
            dim_index += 1

            logger.info("--- Round %d (维度: %s) ---", round_num, dim or "自由")

            try:
                record = self.generate_one_feature(round_num, dim)
                if record:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                # 连续失败太多次，重置Generator上下文避免陷入循环
                if consecutive_failures >= 3:
                    logger.info("连续失败 %d 次，重置Generator上下文", consecutive_failures)
                    self.generator.reset_context()
                    consecutive_failures = 0

            except Exception as e:
                logger.error("Round %d error: %s", round_num, e)
                consecutive_failures += 1

        self.result.total_rounds = round_num
        self.result.generator_calls = self.generator.call_count
        self.result.critic_calls = self.critic.call_count

        logger.info("=== MultiAgent系统完成 ===")
        logger.info("接受: %d, 拒绝: %d, 总轮次: %d",
                   len(self.result.accepted_features),
                   len(self.result.rejected_features),
                   round_num)
        logger.info("Generator调用: %d, Critic调用: %d",
                   self.generator.call_count, self.critic.call_count)

        return self.result

    def save_results(self, output_path: str = "outputs/multi_agent_features.json"):
        """保存结果。"""
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        results = {
            "agent_type": "multi",
            "agents": ["Generator", "Critic", "Validator"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "model": self.model,
                "max_features": self.max_features,
                "max_revisions": self.max_revisions,
            },
            "features": [
                {
                    "name": f["name"],
                    "code": f["code"],
                    "explanation": f["explanation"],
                    "reasoning": f.get("reasoning", ""),
                    "dimension": f.get("dimension", ""),
                    "l2_metrics": f.get("l2_metrics", {}),
                    "l3_metrics": f.get("l3_metrics", {}),
                    "critique": f.get("critique", {}),
                    "revisions": f.get("revisions", 0),
                }
                for f in self.result.accepted_features
            ],
            "rejected": [
                {
                    "name": f["name"],
                    "reason": f.get("reject_reason", ""),
                }
                for f in self.result.rejected_features
            ],
            "interaction_log": self.result.interaction_log,
            "summary": {
                "total_accepted": len(self.result.accepted_features),
                "total_rejected": len(self.result.rejected_features),
                "total_rounds": self.result.total_rounds,
                "generator_calls": self.result.generator_calls,
                "critic_calls": self.result.critic_calls,
                "avg_revisions": (
                    sum(f.get("revisions", 0) for f in self.result.accepted_features)
                    / max(len(self.result.accepted_features), 1)
                ),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("Results saved to %s", output_path)
