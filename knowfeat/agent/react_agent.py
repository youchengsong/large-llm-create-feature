"""ReAct 主循环：Thought → Action → Observation 迭代式特征生成。

Agent 在每一步先思考 → 调用工具 → 根据观测决定下一步。
单轮终止条件：预算用完 / LLM 声明完成 / 达到最大交互步数。
L1 失败后自动构造修复 prompt（最多 2 次）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from knowfeat.agent.tools import ToolContext, PipelineResult
from knowfeat.config import KnowFeatConfig
from knowfeat.llm.client import LLMClient, LLMResponse, ToolCall
from knowfeat.prompt.builder import (
    CrossDimExample,
    DomainKnowledge,
    FailureRecord,
    ShapFeedback,
    build_full_prompt,
)

logger = logging.getLogger(__name__)

MAX_STEPS_PER_ROUND = 100
MAX_L1_REPAIRS = 2


@dataclass
class RoundResult:
    """单轮 Agent 探索的结果摘要。"""
    round_num: int
    dimension: str
    n_generated: int = 0
    n_accepted: int = 0
    n_candidates: int = 0
    n_rejected: int = 0
    steps: int = 0
    features_generated: list[str] = field(default_factory=list)
    features_accepted: list[str] = field(default_factory=list)
    terminated_reason: str = ""


class ReactAgent:
    """ReAct 主循环 Agent。

    Parameters
    ----------
    llm : LLMClient
        LLM 客户端。
    tool_ctx : ToolContext
        工具执行上下文（持有数据、配置和运行时状态）。
    cfg : KnowFeatConfig
        全局配置。
    domain_knowledge : DomainKnowledge
        领域知识容器。
    schema_text : str, optional
        预生成的 Schema 文本。
    """

    def __init__(
        self,
        llm: LLMClient,
        tool_ctx: ToolContext,
        cfg: KnowFeatConfig,
        domain_knowledge: DomainKnowledge,
        schema_text: Optional[str] = None,
        knowledge_filter: Optional[str] = None,
    ):
        self.llm = llm
        self.tool_ctx = tool_ctx
        self.cfg = cfg
        self.domain_knowledge = domain_knowledge
        self.schema_text = schema_text
        self.knowledge_filter = knowledge_filter

    def run_round(
        self,
        round_num: int,
        dimension: str,
        budget: int,
        shap_feedback: Optional[ShapFeedback] = None,
        failure_records: Optional[list[FailureRecord]] = None,
        cross_dim_examples: Optional[list[CrossDimExample]] = None,
        event_callback: Optional[Any] = None,
    ) -> RoundResult:
        """执行单轮 Agent 探索。

        Parameters
        ----------
        round_num : 当前轮次编号。
        dimension : 当前维度名称。
        budget : 本轮该维度的特征预算。
        shap_feedback : SHAP 维度贡献反馈（动态）。
        failure_records : 失败记忆列表（动态）。
        cross_dim_examples : 跨维度高分特征示例（动态）。

        Returns
        -------
        RoundResult
        """
        result = RoundResult(round_num=round_num, dimension=dimension)

        system_prompt, user_message = build_full_prompt(
            cfg=self.cfg,
            domain_knowledge=self.domain_knowledge,
            schema_text=self.schema_text,
            shap_feedback=shap_feedback,
            failure_records=failure_records,
            cross_dim_examples=cross_dim_examples,
            round_num=round_num,
            dimension=dimension,
            budget=budget,
            n_accepted=len(self.tool_ctx.accepted_features),
            n_candidates=len(self.tool_ctx.candidate_pool),
            knowledge_filter=self.knowledge_filter,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        def emit(event_type: str, **data):
            if event_callback:
                try:
                    event_callback(event_type, data)
                except Exception:
                    pass

        tool_specs = self.tool_ctx.get_tool_specs()
        l1_repair_counts: dict[str, int] = {}
        n_validated = 0

        for step in range(1, MAX_STEPS_PER_ROUND + 1):
            result.steps = step

            logger.info(
                "Round %d / %s — step %d (generated=%d, budget=%d)",
                round_num, dimension, step,
                result.n_generated, budget,
            )

            emit("agent_step", round_num=round_num, dimension=dimension,
                 step=step, generated=result.n_generated, budget=budget)

            resp = self.llm.chat(messages, tools=tool_specs)

            if resp.content:
                emit("agent_thought", round_num=round_num, dimension=dimension,
                     step=step, thought=resp.content[:500])

            if not resp.has_tool_calls:
                result.terminated_reason = "agent_completed"
                logger.info(
                    "Agent 声明完成 (round=%d, dim=%s): %s",
                    round_num, dimension,
                    (resp.content or "")[:200],
                )
                break

            assistant_msg = LLMClient.make_assistant_tool_calls_message(
                resp.content, resp.tool_calls,
            )
            messages.append(assistant_msg)

            for tc in resp.tool_calls:
                observation = self.tool_ctx.execute(tc.name, tc.arguments)
                messages.append(
                    LLMClient.make_tool_result_message(tc.id, observation)
                )

                if tc.name == "generate_feature":
                    fname = tc.arguments.get("feature_name", "")
                    if fname:
                        result.n_generated += 1
                        result.features_generated.append(fname)
                        logger.info("特征已注册: %s", fname)

                        # Auto-trigger execute_sandbox after registration.
                        # Append the verification result to the existing
                        # tool result so the LLM sees it without breaking
                        # the message format (tool results must follow
                        # a tool_calls message).
                        logger.info("自动触发验证: %s", fname)
                        sandbox_obs = self.tool_ctx.execute(
                            "execute_sandbox", {"feature_name": fname},
                        )
                        # Replace the last tool-result message with a
                        # combined registration + verification result.
                        combined = (
                            observation
                            + "\n\n--- 自动验证结果 ---\n"
                            + sandbox_obs
                        )
                        messages[-1] = LLMClient.make_tool_result_message(
                            tc.id, combined,
                        )

                        obs = _parse_observation(sandbox_obs)
                        n_validated += 1

                        emit("verification_result",
                             feature_name=fname,
                             round_num=round_num,
                             dimension=dimension,
                             accepted=obs.get("accepted", False),
                             stage_reached=obs.get("stage_reached", ""),
                             summary=obs.get("summary", ""),
                             l1=obs.get("L1_sandbox"),
                             l2=obs.get("L2_statistics"),
                             l3=obs.get("L3_model_eval"))

                        if obs.get("accepted"):
                            result.n_accepted += 1
                            result.features_accepted.append(fname)
                        elif obs.get("stage_reached") == "L1" and not obs.get("accepted"):
                            repair_count = l1_repair_counts.get(fname, 0)
                            if repair_count < MAX_L1_REPAIRS:
                                l1_repair_counts[fname] = repair_count + 1
                                repair_msg = _build_repair_message(fname, obs)
                                messages.append(
                                    {"role": "user", "content": repair_msg}
                                )
                                logger.info(
                                    "L1 失败 → 自动修复 (%d/%d): %s",
                                    repair_count + 1, MAX_L1_REPAIRS, fname,
                                )
                        elif not obs.get("accepted") and obs.get("note"):
                            result.n_candidates += 1

                if tc.name == "execute_sandbox":
                    fname = tc.arguments.get("feature_name", "")
                    if fname in result.features_generated:
                        logger.info("跳过重复验证: %s (已自动验证)", fname)
                        continue
                    obs = _parse_observation(observation)
                    n_validated += 1

                    if obs.get("accepted"):
                        result.n_accepted += 1
                        result.features_accepted.append(fname)
                    elif obs.get("stage_reached") == "L1" and not obs.get("accepted"):
                        repair_count = l1_repair_counts.get(fname, 0)
                        if repair_count < MAX_L1_REPAIRS:
                            l1_repair_counts[fname] = repair_count + 1
                            repair_msg = _build_repair_message(fname, obs)
                            messages.append(
                                {"role": "user", "content": repair_msg}
                            )
                            logger.info(
                                "L1 失败 → 自动修复 (%d/%d): %s",
                                repair_count + 1, MAX_L1_REPAIRS, fname,
                            )
                    elif not obs.get("accepted") and obs.get("note"):
                        result.n_candidates += 1

            if n_validated >= budget:
                result.terminated_reason = "budget_exhausted"
                logger.info(
                    "预算用完 (generated=%d, validated=%d, budget=%d)",
                    result.n_generated, n_validated, budget,
                )
                break
        else:
            result.terminated_reason = "max_steps_reached"
            logger.warning(
                "达到最大步数 %d (round=%d, dim=%s)",
                MAX_STEPS_PER_ROUND, round_num, dimension,
            )

        result.n_rejected = len([
            r for r in self.tool_ctx.rejected_features
            if r.get("feature_name") in result.features_generated
        ])

        logger.info(
            "Round %d / %s 完成: generated=%d, accepted=%d, "
            "candidates=%d, rejected=%d, steps=%d, reason=%s",
            round_num, dimension,
            result.n_generated, result.n_accepted,
            result.n_candidates, result.n_rejected,
            result.steps, result.terminated_reason,
        )

        return result


def _parse_observation(observation: str) -> dict[str, Any]:
    """安全解析工具返回的 JSON observation。"""
    try:
        return json.loads(observation)
    except (json.JSONDecodeError, TypeError):
        return {}


def _build_repair_message(feature_name: str, obs: dict[str, Any]) -> str:
    """构造 L1 失败后的自动修复提示。"""
    l1_info = obs.get("L1_sandbox", {})
    error = l1_info.get("error", obs.get("summary", "未知错误"))

    return (
        f"特征 `{feature_name}` 的 L1 沙盒执行失败:\n"
        f"```\n{error}\n```\n\n"
        f"请修复代码中的问题，然后重新调用 generate_feature 提交修复后的代码，"
        f"再调用 execute_sandbox 验证。注意:\n"
        f"1. 检查函数签名是否为 def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame\n"
        f"2. 确保输出 DataFrame 包含 wallet_id 或 tx_id 列\n"
        f"3. 检查是否有 import 缺失或变量未定义的问题"
    )
