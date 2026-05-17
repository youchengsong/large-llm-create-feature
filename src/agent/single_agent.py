"""单Agent特征工程系统：基于ReAct范式，完整复刻原系统逻辑。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from src.config import DEFAULT_CONFIG, ISLANDS, ValidationThresholds
from src.data.merge import merge_feature_to_transactions
from src.data.stats_query import execute_stats_query
from src.evaluation.validator import (
    run_l1_validation,
    run_l2_validation,
    run_l3_validation,
    compute_composite_score,
    backward_eliminate_pending,
)
from src.llm.parser import extract_json_from_text
from src.prompt.domain_knowledge import (
    load_transaction_patterns,
    load_expert_interview,
    load_risk_indicators,
    load_detection_rules,
    rules_to_python_examples,
)
from src.search.island import Island, IslandManager
from src.search.memory import FailureMemory
from src.search.pending_pool import PendingPool, PendingFeature
from src.search.shap_feedback import generate_shap_feedback, shap_feedback_to_prompt, compute_island_budgets

logger = logging.getLogger(__name__)


# ============================================================
# Tool Definitions (OpenAI function calling format)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "检索领域知识。可获取交易模式说明、专家访谈知识、风险指标或检测规则示例。",
            "parameters": {
                "type": "object",
                "properties": {
                    "knowledge_type": {
                        "type": "string",
                        "enum": ["transaction_patterns", "expert_interview", "risk_indicators", "detection_rules"],
                        "description": "要检索的知识类型"
                    }
                },
                "required": ["knowledge_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": "查看数据表结构，了解可用字段和数据类型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "enum": ["open_df", "tx_df", "close_df"],
                        "description": "要查看的数据表"
                    }
                },
                "required": ["table"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explore_data",
            "description": "执行数据探索查询，查看数据分布或统计信息。代码中将结果赋值给query_result变量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "要探索的问题"
                    },
                    "code": {
                        "type": "string",
                        "description": "Python代码，使用open_df/tx_df/close_df，将结果赋值给query_result"
                    }
                },
                "required": ["question", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_feature",
            "description": "生成一个特征的Python代码。需要指定特征名称、所属维度、代码实现和解释。",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_name": {
                        "type": "string",
                        "description": "特征名称（英文，snake_case）"
                    },
                    "dimension": {
                        "type": "string",
                        "description": "特征所属维度（如时间维度、金额频次、行为序列、主体关联等）"
                    },
                    "code": {
                        "type": "string",
                        "description": "完整的Python函数代码，签名为 def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "特征的业务含义和检测逻辑说明"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "设计该特征的推理过程，基于哪条领域知识"
                    }
                },
                "required": ["feature_name", "dimension", "code", "explanation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sandbox",
            "description": "在沙箱中执行特征代码，验证代码正确性和输出格式。通过后自动进行L2/L3质量评估。",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_name": {
                        "type": "string",
                        "description": "要执行的特征名称"
                    },
                    "code": {
                        "type": "string",
                        "description": "要执行的Python代码（如不提供则使用generate_feature中保存的代码）"
                    }
                },
                "required": ["feature_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "结束特征生成过程。当认为已生成足够多的特征或所有维度已覆盖时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "结束的原因"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]


# ============================================================
# Agent State
# ============================================================

@dataclass
class AgentState:
    accepted_features: list[dict] = field(default_factory=list)
    failed_features: list[dict] = field(default_factory=list)
    pending_sandbox: dict[str, dict] = field(default_factory=dict)
    sandbox_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    total_attempts: int = 0
    dimensions_explored: set = field(default_factory=set)
    finished: bool = False
    finish_reason: str = ""


# ============================================================
# Single Agent
# ============================================================

class FeatureEngineerAgent:
    """
    单Agent特征工程系统 — 完整复刻原系统(src/search/engine.py)逻辑。

    与原系统一致的机制：
    1. L1重试：沙箱失败后主动让LLM修复代码（最多2次）
    2. L2/L3验证：完全一致的阈值和判定逻辑
    3. Pending Pool：L2通过但L3失败→缓刑池，每轮backward_eliminate_pending评估
    4. SHAP反馈：generate_shap_feedback驱动维度探索方向
    5. 岛屿预算：compute_island_budgets反比分配候选数
    6. Composite Score：30%SHAP + 20%IV + 20%冗余度 + 30%delta_f1
    7. 岛屿管理：island.add_feature + migrate_best（跨岛屿传播好特征）
    8. 可解释性报告：每个通过L3的特征调LLM生成explainability
    9. 失败记忆：FailureMemory记录失败原因，避免重复
    10. 数据探索：execute_stats_query支持LLM查询数据分布
    11. 收敛检测：连续2轮无新特征终止
    12. 多轮反馈：每轮注入接受/失败反馈引导下一轮生成
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
        max_steps: int = 250,
        max_features: int = 30,
        temperature: float = 0.7,
        validation_thresholds=None,
        baseline_metrics: dict = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.open_df = open_df
        self.tx_df = tx_df
        self.close_df = close_df
        self.label = label
        self.baseline_features = baseline_features.copy() if baseline_features is not None else pd.DataFrame()
        self.existing_features = self.baseline_features.copy()
        self.max_steps = max_steps
        self.max_features = max_features
        self.temperature = temperature
        self.validation_thresholds = validation_thresholds or DEFAULT_CONFIG.validation
        self.baseline_metrics = baseline_metrics or {}

        # 原系统组件（与engine.py完全对应）
        self.island_manager = IslandManager(DEFAULT_CONFIG.islands)
        self.failure_memory = FailureMemory()
        self.pending_pool = PendingPool(
            max_size=DEFAULT_CONFIG.pending_pool.max_size,
            max_age=DEFAULT_CONFIG.pending_pool.max_age,
            max_rejects=DEFAULT_CONFIG.pending_pool.max_rejects,
        )
        self.all_scores: dict[str, list[float]] = {
            "shap_rank": [],
            "iv": [],
            "redundancy": [],
            "delta_f1": [],
        }
        self._rounds_without_new = 0
        self._current_round_new_count = 0
        self._round_num = 0
        self._steps_in_round = 0
        self._steps_per_round = 25

        self.state = AgentState()
        self.messages: list[dict] = []
        self.step_log: list[dict] = []

    @staticmethod
    def _clean_code(code: str) -> str:
        """清理LLM返回的代码中的转义字符。"""
        code = code.replace('\\"', '"')
        code = code.replace('\\n', '\n') if '\\n' in code and '\n' not in code else code
        return code

    # ----------------------------------------------------------
    # SHAP反馈 & 岛屿预算（对应engine.py _compute_budgets）
    # ----------------------------------------------------------

    def _get_shap_feedback(self) -> dict:
        """生成SHAP反馈（与engine.py generate_shap_feedback一致）。"""
        if self.existing_features.shape[1] == 0:
            return {}
        try:
            return generate_shap_feedback(self.existing_features, self.label)
        except Exception as e:
            logger.warning("SHAP feedback failed: %s", e)
            return {}

    def _get_island_budgets(self) -> dict[str, int]:
        """计算岛屿预算（与engine.py _compute_budgets完全一致）。"""
        island_dims = [ic.dimension for ic in ISLANDS]

        if self._round_num <= 1 or self.existing_features.shape[1] == 0:
            return {dim: DEFAULT_CONFIG.search.b for dim in island_dims}

        try:
            shap_fb = self._get_shap_feedback()
            if not shap_fb:
                return {dim: DEFAULT_CONFIG.search.b for dim in island_dims}

            dim_importance = {}
            for ic in ISLANDS:
                dim_vals = [
                    shap_fb["feature_importance"].get(f, 0.0)
                    for f in self.existing_features.columns
                    if any(field.lower() in f.lower() for field in ic.fields)
                ]
                dim_importance[ic.dimension] = sum(dim_vals) if dim_vals else 0.0

            return compute_island_budgets(
                dim_importance, island_dims,
                B_total=DEFAULT_CONFIG.search.B_total,
                b_min=DEFAULT_CONFIG.search.b_min,
                b_max=DEFAULT_CONFIG.search.b_max,
            )
        except Exception as e:
            logger.warning("Budget computation failed: %s, using uniform", e)
            return {dim: DEFAULT_CONFIG.search.b for dim in island_dims}

    # ----------------------------------------------------------
    # Pending Pool评估（对应engine.py run_pending_pool_evaluation）
    # ----------------------------------------------------------

    def _run_pending_pool_evaluation(self) -> list[dict]:
        """每轮结束时评估缓刑池（与engine.py完全一致）。"""
        if self.pending_pool.is_empty():
            return []
        if self.existing_features.shape[1] == 0:
            return []

        pending_features = {
            pf.name: pf.feature_values
            for pf in self.pending_pool.get_all()
        }

        accepted_names, rejected_names = backward_eliminate_pending(
            self.existing_features, pending_features, self.label,
            baseline_metrics=self.baseline_metrics or None,
        )

        logger.info("Pending pool evaluation: %d accepted, %d rejected out of %d",
                    len(accepted_names), len(rejected_names), len(pending_features))

        self.pending_pool.mark_rejected(rejected_names)
        accepted_pfs = self.pending_pool.remove_accepted(accepted_names)

        new_records = []
        for pf in accepted_pfs:
            record = self._accept_pending_feature(pf)
            new_records.append(record)
            logger.info("Accepted from pending pool: %s (score=%.4f)", pf.name, record["score"])

        self.pending_pool.tick_age()
        purged = self.pending_pool.purge_expired()
        if purged:
            logger.info("Purged %d expired features from pending pool: %s", len(purged), purged)

        return new_records

    def _accept_pending_feature(self, pf: PendingFeature) -> dict:
        """接受缓刑池特征（与engine.py _accept_pending_feature完全一致）。"""
        feature_series = pd.Series(pf.feature_values, name=pf.name)

        self.all_scores["shap_rank"].append(pf.l3_metrics.get("shap_rank", 999))
        self.all_scores["iv"].append(pf.l2_metrics.get("iv", 0))
        self.all_scores["redundancy"].append(pf.l2_metrics.get("redundancy", 0))
        self.all_scores["delta_f1"].append(pf.l3_metrics.get("delta_f1", 0))

        score = compute_composite_score(
            shap_rank=pf.l3_metrics.get("shap_rank", 999),
            iv=pf.l2_metrics.get("iv", 0),
            redundancy=pf.l2_metrics.get("redundancy", 0),
            delta_f1=pf.l3_metrics.get("delta_f1", 0),
            all_scores=self.all_scores,
        )

        self.existing_features[pf.name] = pf.feature_values

        explainability = self._generate_explainability(
            name=pf.name,
            code=pf.code,
            explanation=pf.explanation,
            l2_metrics=pf.l2_metrics,
            l3_metrics=pf.l3_metrics,
            feature_series=feature_series,
        )

        feature_record = {
            "name": pf.name,
            "code": pf.code,
            "explanation": pf.explanation,
            "reasoning": pf.reasoning,
            "dimension": pf.island_name,
            "l2_metrics": pf.l2_metrics,
            "l3_metrics": pf.l3_metrics,
            "score": score,
            "explainability": explainability,
            "accepted_via": "pending_pool",
        }

        island = self.island_manager.get_island(pf.island_name)
        if island:
            island.add_feature(feature_record)
        self.island_manager.migrate_best(feature_record)
        self.state.accepted_features.append(feature_record)
        self._current_round_new_count += 1

        return feature_record

    # ----------------------------------------------------------
    # 可解释性报告（对应engine.py _generate_explainability）
    # ----------------------------------------------------------

    def _generate_explainability(
        self,
        name: str,
        code: str,
        explanation: str,
        l2_metrics: dict,
        l3_metrics: dict,
        feature_series: pd.Series,
    ) -> dict:
        """通过L3后调用LLM生成特征可解释性报告（与engine.py完全一致）。"""
        normal_vals = feature_series[self.label == 0]
        abnormal_vals = feature_series[self.label == 1]

        prompt = f"""特征 {name} 已通过三级验证，请为它生成可解释性报告。

## 特征信息
- 名称: {name}
- 说明: {explanation}
- 代码: ```python
{code}
```

## 验证指标
- SHAP Rank: #{l3_metrics.get('shap_rank', 'N/A')}
- IV: {l2_metrics.get('iv', 0):.4f}
- KS: {l2_metrics.get('ks', 0):.4f}
- Delta F1: {l3_metrics.get('delta_f1', 0):.6f}
- Delta Recall: {l3_metrics.get('delta_recall', 0):.6f}

## 该特征在正常 vs 异常交易上的分布
- 正常交易: 中位数={normal_vals.median():.4f}, 均值={normal_vals.mean():.4f}, std={normal_vals.std():.4f}
- 异常交易: 中位数={abnormal_vals.median():.4f}, 均值={abnormal_vals.mean():.4f}, std={abnormal_vals.std():.4f}

请输出一个 JSON 对象，包含以下字段：
- value_meaning: 不同取值区间的业务含义
- risk_threshold: 高风险阈值及判断依据
- synergy: 建议与哪些已有特征配合使用
- false_positive_cases: 可能的误判场景
- regulatory_explanation: 面向监管审计人员的一句话解释"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是反洗钱特征可解释性分析专家。请输出JSON格式。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            raw = response.choices[0].message.content or ""
            json_str = extract_json_from_text(raw)
            if json_str:
                report = json.loads(json_str)
                if isinstance(report, dict):
                    logger.info("Explainability report generated for %s", name)
                    return report
            if len(raw) > 20:
                return {"raw_explanation": raw}
        except Exception as e:
            logger.warning("Explainability generation failed for %s: %s", name, e)

        return {}

    # ----------------------------------------------------------
    # L1重试（对应engine.py _try_l1_with_retry）
    # ----------------------------------------------------------

    def _try_l1_with_retry(self, code: str, name: str, max_retries: int = 2) -> tuple:
        """L1验证+失败重试（与engine.py _try_l1_with_retry完全一致）。"""
        feature_df, error = run_l1_validation(code, self.open_df, self.tx_df, self.close_df)
        if feature_df is not None:
            return feature_df, "", code

        for attempt in range(max_retries):
            fix_prompt = (
                f"特征 '{name}' 的代码执行失败。\n"
                f"错误信息: {error}\n\n"
                f"原始代码:\n```python\n{code}\n```\n\n"
                f"请修复代码，输出完整的修复后的Python函数代码（def feature_xxx(open_df, tx_df, close_df)格式）。"
                f"代码最后要有 result = feature_xxx(open_df, tx_df, close_df)"
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是Python代码修复专家。直接输出修复后的完整代码，不要其他内容。"},
                        {"role": "user", "content": fix_prompt},
                    ],
                    temperature=0.3,
                )
                raw = response.choices[0].message.content or ""
                # 提取代码块
                fixed_code = self._extract_code_from_response(raw)
                if fixed_code:
                    fixed_code = self._clean_code(fixed_code)
                    feature_df, error = run_l1_validation(fixed_code, self.open_df, self.tx_df, self.close_df)
                    if feature_df is not None:
                        logger.info("L1 retry %d succeeded for %s", attempt + 1, name)
                        return feature_df, "", fixed_code
            except Exception as e:
                error = str(e)
                logger.warning("L1 retry %d failed for %s: %s", attempt + 1, name, e)

        return None, error, code

    @staticmethod
    def _extract_code_from_response(raw: str) -> str:
        """从LLM响应中提取代码块。"""
        if "```python" in raw:
            parts = raw.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) > 1:
                return parts[1].strip()
        if "def feature_" in raw:
            return raw.strip()
        return raw.strip()

    # ----------------------------------------------------------
    # System Prompt
    # ----------------------------------------------------------

    def _build_system_prompt(self) -> str:
        island_desc = "\n".join(
            f"  - {ic.name}({ic.dimension}): {ic.description}, 字段: {', '.join(ic.fields)}"
            for ic in ISLANDS
        )

        return f"""你是一个专业的反洗钱特征工程Agent。你的任务是为数字人民币交易数据自动生成有效的检测特征。

## 你的能力
你可以使用以下工具：
1. retrieve_knowledge - 检索领域知识（交易模式、专家访谈、风险指标、检测规则）
2. inspect_schema - 查看数据表结构
3. explore_data - 执行数据探索查询（查看分布、统计信息）
4. generate_feature - 生成特征代码
5. execute_sandbox - 在沙箱中验证代码（通过后自动评估L2/L3质量）
6. finish - 结束生成过程

## 工作流程
1. 首先检索领域知识，了解洗钱模式
2. 查看数据schema，了解可用字段
3. 可选：用explore_data查看数据分布，发现异常模式
4. 基于知识和数据，围绕指定维度设计并生成特征（generate_feature）
5. 在沙箱中验证代码（execute_sandbox）— 通过后系统自动评估质量
6. 根据评估反馈和SHAP分析调整策略，重点探索薄弱维度
7. 每轮生成约12个候选特征（4个维度各3个），覆盖多维度后用finish结束

## 维度岛屿（4个探索方向）
{island_desc}

## 特征代码要求
- 写一个Python函数，签名: def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame
- 代码最后必须调用该函数并将返回值赋给 result 变量:
  result = feature_xxx(open_df, tx_df, close_df)
- 返回的DataFrame必须包含 wallet_id 或 tx_id 列作为索引
- 不能使用标签列(is_risk, is_laundering等)
- 需要处理空值情况(fillna)
- 只能使用 pandas, numpy, scipy, sklearn, datetime, collections, math

## 策略建议
- 重点关注SHAP分析中的薄弱维度（预算更高的维度）
- 从领域知识中寻找灵感
- 如果某个方向连续失败，换方向探索
- L3未通过的特征不会立即丢弃，会进入缓刑池等待后续评估
- 优先探索与已有特征低相关的方向"""

    # ----------------------------------------------------------
    # 轮次引导Prompt（含SHAP反馈 + 预算 + 反馈）
    # ----------------------------------------------------------

    def _build_round_prompt(self) -> str:
        """构建每轮开始时的引导prompt（与engine.py run_round对应）。"""
        self._round_num += 1

        # SHAP反馈文本
        shap_fb = self._get_shap_feedback()
        island_dims = [ic.dimension for ic in ISLANDS]
        shap_text = shap_feedback_to_prompt(shap_fb, island_dims) if shap_fb else "暂无SHAP反馈，所有维度均需探索。"

        # 岛屿预算
        budgets = self._get_island_budgets()
        budget_text = "\n".join(
            f"  - {ic.name}({ic.dimension}): {budgets.get(ic.dimension, 3)} 个候选"
            for ic in ISLANDS
        )

        # 失败记忆（与engine.py failure_memory.get_recent(5)一致）
        recent_failures = self.failure_memory.get_recent(5)
        fail_text = "\n".join(
            f"  - {f.get('feature_name', '?')}: {f.get('failure_reason', '?')}"
            for f in recent_failures
        ) if recent_failures else "（暂无）"

        # 上轮最佳特征（与engine.py island.boltzmann_sample对应）
        top_features = []
        for island in self.island_manager.islands:
            sampled = island.boltzmann_sample(k=DEFAULT_CONFIG.search.k, tau=DEFAULT_CONFIG.search.tau)
            for f in sampled:
                if f not in top_features:
                    top_features.append(f)
        top_text = "\n".join(
            f"  - {f.get('name', '?')}: score={f.get('score', 0):.4f}, "
            f"SHAP_rank={f.get('shap_rank', f.get('l3_metrics', {}).get('shap_rank', '?'))}"
            for f in top_features[:10]
        ) if top_features else "（暂无）"

        # Pending pool状态
        pool_text = f"\n缓刑池: {len(self.pending_pool)} 个特征待评估" if not self.pending_pool.is_empty() else ""

        total_budget = sum(budgets.values())

        return f"""## 第 {self._round_num} 轮迭代

{shap_text}

## 本轮预算分配（共 {total_budget} 个候选特征）
{budget_text}

## 上轮最佳特征（记忆库中的高分示例）
{top_text}

## 近期失败特征（避免重复类似方向）
{fail_text}
{pool_text}

## 任务
请按照预算分配，为每个维度生成指定数量的候选特征。
- 薄弱维度（预算更多的）应重点探索
- 对每个候选特征：先 generate_feature，然后立即 execute_sandbox 验证
- 如果需要了解数据分布，可先用 explore_data 查询"""

    # ----------------------------------------------------------
    # 轮内反馈Prompt（对应engine.py _build_turn_feedback_prompt）
    # ----------------------------------------------------------

    def _build_turn_feedback_prompt(self, turn_accepted: list[dict], turn_failed: list[dict]) -> str:
        """构建轮内多轮对话的反馈prompt（与engine.py完全一致）。"""
        lines = [f"## 上一轮生成结果反馈\n"]

        if turn_accepted:
            lines.append("通过验证的特征：")
            for f in turn_accepted:
                lines.append(
                    f"  - {f['name']}: score={f.get('score', 0):.4f}, "
                    f"delta_f1={f.get('l3_metrics', {}).get('delta_f1', 0):.6f}, "
                    f"delta_recall={f.get('l3_metrics', {}).get('delta_recall', 0):.6f}"
                )
            lines.append("")

        if turn_failed:
            lines.append("未通过的特征（请避免类似方向）：")
            for f in turn_failed:
                lines.append(f"  - {f.get('name', 'unknown')}: {f.get('reason', '')}")
            lines.append("")

        lines.append(
            "请基于以上反馈，继续生成不同角度的新特征。"
            "尝试与上轮不同的思路，探索还未覆盖的维度。"
        )
        return "\n".join(lines)

    # ----------------------------------------------------------
    # Tool执行
    # ----------------------------------------------------------

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "retrieve_knowledge":
            return self._tool_retrieve_knowledge(arguments)
        elif tool_name == "inspect_schema":
            return self._tool_inspect_schema(arguments)
        elif tool_name == "explore_data":
            return self._tool_explore_data(arguments)
        elif tool_name == "generate_feature":
            return self._tool_generate_feature(arguments)
        elif tool_name == "execute_sandbox":
            return self._tool_execute_sandbox(arguments)
        elif tool_name == "finish":
            return self._tool_finish(arguments)
        else:
            return f"未知工具: {tool_name}"

    def _tool_retrieve_knowledge(self, args: dict) -> str:
        knowledge_type = args.get("knowledge_type", "")
        if knowledge_type == "transaction_patterns":
            return load_transaction_patterns()
        elif knowledge_type == "expert_interview":
            return load_expert_interview()
        elif knowledge_type == "risk_indicators":
            try:
                return load_risk_indicators()
            except Exception as e:
                return f"风险指标加载失败: {e}"
        elif knowledge_type == "detection_rules":
            try:
                rules = load_detection_rules()
                return rules_to_python_examples(rules, n=2)
            except Exception as e:
                return f"检测规则加载失败: {e}"
        return "未知的知识类型"

    def _tool_inspect_schema(self, args: dict) -> str:
        table = args.get("table", "")
        df_map = {"open_df": self.open_df, "tx_df": self.tx_df, "close_df": self.close_df}
        df = df_map.get(table)
        if df is None:
            return f"未知数据表: {table}"
        lines = [f"## {table} (shape: {df.shape})"]
        lines.append("| 列名 | 类型 | 非空数 | 示例值 |")
        lines.append("|------|------|--------|--------|")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else "N/A"
            if len(sample) > 30:
                sample = sample[:30] + "..."
            lines.append(f"| {col} | {dtype} | {non_null} | {sample} |")
        return "\n".join(lines)

    def _tool_explore_data(self, args: dict) -> str:
        """执行数据探索查询（与engine.py _execute_queries一致）。"""
        question = args.get("question", "")
        code = self._clean_code(args.get("code", ""))

        result_text, error = execute_stats_query(code, self.open_df, self.tx_df, self.close_df)
        if error:
            return f"查询失败: {error}"
        return f"查询: {question}\n结果:\n{result_text}"

    def _tool_generate_feature(self, args: dict) -> str:
        name = args.get("feature_name", "")
        code = self._clean_code(args.get("code", ""))
        explanation = args.get("explanation", "")
        dimension = args.get("dimension", "")
        reasoning = args.get("reasoning", "")

        self.state.pending_sandbox[name] = {
            "code": code,
            "explanation": explanation,
            "dimension": dimension,
            "reasoning": reasoning,
        }
        self.state.dimensions_explored.add(dimension)
        self.state.total_attempts += 1

        return f"特征 '{name}' 已生成，请使用 execute_sandbox 工具验证代码。"

    def _tool_execute_sandbox(self, args: dict) -> str:
        name = args.get("feature_name", "")
        code = self._clean_code(args.get("code", ""))

        if not code and name in self.state.pending_sandbox:
            code = self.state.pending_sandbox[name]["code"]

        # L1验证 + 重试（与engine.py _try_l1_with_retry一致）
        feature_df, error, final_code = self._try_l1_with_retry(code, name, max_retries=2)

        if feature_df is None:
            self.failure_memory.add_failure(name, code, f"L1: {error}")
            self.state.failed_features.append({"name": name, "error": f"L1: {error}"})
            return (
                f"✗ 沙箱执行失败（已尝试自动修复2次）\n"
                f"错误信息: {error}\n"
                f"请换一个特征方向，或用不同的实现方式。"
            )

        # 更新代码（可能是修复后的版本）
        if name in self.state.pending_sandbox:
            self.state.pending_sandbox[name]["code"] = final_code

        self.state.sandbox_results[name] = feature_df
        shape = feature_df.shape
        cols = list(feature_df.columns)

        # 自动触发评估
        eval_result = self._evaluate_feature(name)
        return (
            f"✓ 沙箱执行成功！\n"
            f"输出shape: {shape}\n"
            f"输出列: {cols}\n\n"
            f"--- 自动评估结果 ---\n{eval_result}"
        )

    def _evaluate_feature(self, name: str) -> str:
        """完整的L2+L3评估流程（与engine.py evaluate_candidates完全一致）。"""
        feature_df = self.state.sandbox_results.get(name)
        if feature_df is None:
            return f"特征 '{name}' 尚未通过沙箱验证。"

        # 检查重复
        if any(f["name"] == name for f in self.state.accepted_features):
            return f"特征 '{name}' 已被接受，请生成不同的特征。"

        # Merge
        try:
            merged = merge_feature_to_transactions(feature_df, self.tx_df)
        except Exception as e:
            self.failure_memory.add_failure(name, "", f"merge失败: {e}")
            self.state.failed_features.append({"name": name, "error": f"merge失败: {e}"})
            return f"特征合并失败: {e}"

        if len(merged) != len(self.tx_df):
            reason = f"行数不匹配: {len(merged)} vs {len(self.tx_df)}"
            self.failure_memory.add_failure(name, "", reason)
            self.state.failed_features.append({"name": name, "error": reason})
            return f"合并后{reason}"

        feat_cols = [c for c in merged.columns if c not in self.tx_df.columns]
        if not feat_cols:
            self.failure_memory.add_failure(name, "", "合并后无新列")
            self.state.failed_features.append({"name": name, "error": "合并后无新列"})
            return "合并后无新列产生"

        feature_series = merged[feat_cols[0]].fillna(0)
        feature_series.name = name

        # === L2验证 ===
        l2_passed, l2_metrics = run_l2_validation(
            feature_series, self.label, self.existing_features,
            thresholds=self.validation_thresholds,
        )

        if not l2_passed:
            th = self.validation_thresholds
            reasons = []
            if l2_metrics.get("iv", 0) < th.iv_min:
                reasons.append(f"IV={l2_metrics['iv']:.6f}<{th.iv_min:.6f}")
            if l2_metrics.get("ks", 0) < th.ks_min:
                reasons.append(f"KS={l2_metrics['ks']:.6f}<{th.ks_min:.6f}")
            if l2_metrics.get("redundancy", 0) >= th.corr_max:
                reasons.append(f"redundancy={l2_metrics['redundancy']:.4f}>={th.corr_max}")
            if l2_metrics.get("missing_rate", 0) > th.missing_rate_max:
                reasons.append(f"missing={l2_metrics['missing_rate']:.2%}>{th.missing_rate_max:.0%}")
            reason_str = f"L2未通过: {'; '.join(reasons)}"
            self.failure_memory.add_failure(name, self.state.pending_sandbox.get(name, {}).get("code", ""), reason_str)
            self.state.failed_features.append({"name": name, "error": reason_str})
            return (
                f"✗ L2统计质量未通过\n"
                f"IV: {l2_metrics.get('iv', 0):.4f}\n"
                f"KS: {l2_metrics.get('ks', 0):.4f}\n"
                f"冗余度: {l2_metrics.get('redundancy', 0):.4f}\n"
                f"缺失率: {l2_metrics.get('missing_rate', 0):.2%}\n"
                f"原因: {'; '.join(reasons)}\n"
                f"建议: 尝试不同的计算逻辑或探索其他维度。"
            )

        # === L3验证 ===
        l3_passed, l3_metrics = run_l3_validation(
            self.existing_features, feature_series, self.label,
            baseline_metrics=self.baseline_metrics,
        )

        if not l3_passed:
            # 进入Pending Pool（与engine.py完全一致）
            info = self.state.pending_sandbox.get(name, {})
            pending_feat = PendingFeature(
                name=name,
                code=info.get("code", ""),
                explanation=info.get("explanation", ""),
                reasoning=info.get("reasoning", ""),
                feature_values=feature_series.values,
                l2_metrics=l2_metrics,
                l3_metrics=l3_metrics,
                island_name=info.get("dimension", ""),
            )
            self.pending_pool.add(pending_feat)
            logger.info("L3 failed for %s (delta_f1=%.6f, delta_recall=%.6f, delta_precision=%.6f), "
                        "added to pending pool (%d in pool)",
                        name, l3_metrics.get("delta_f1", 0), l3_metrics.get("delta_recall", 0),
                        l3_metrics.get("delta_precision", 0), len(self.pending_pool))
            return (
                f"△ L3模型增量不足，已加入缓刑池（{len(self.pending_pool)}个待评估）\n"
                f"IV: {l2_metrics.get('iv', 0):.4f}, KS: {l2_metrics.get('ks', 0):.4f}\n"
                f"Delta F1: {l3_metrics.get('delta_f1', 0):.6f}\n"
                f"Delta Recall: {l3_metrics.get('delta_recall', 0):.6f}\n"
                f"Delta Precision: {l3_metrics.get('delta_precision', 0):.6f}\n"
                f"注: 该特征统计上有区分力，将在轮末与其他特征组合重新评估。"
            )

        # === 接受特征（与engine.py完全一致） ===
        self.all_scores["shap_rank"].append(l3_metrics["shap_rank"])
        self.all_scores["iv"].append(l2_metrics["iv"])
        self.all_scores["redundancy"].append(l2_metrics["redundancy"])
        self.all_scores["delta_f1"].append(l3_metrics.get("delta_f1", 0))

        score = compute_composite_score(
            shap_rank=l3_metrics["shap_rank"],
            iv=l2_metrics["iv"],
            redundancy=l2_metrics["redundancy"],
            delta_f1=l3_metrics.get("delta_f1", 0),
            all_scores=self.all_scores,
        )

        self.existing_features[name] = feature_series.values
        self._current_round_new_count += 1

        # 可解释性报告
        info = self.state.pending_sandbox.get(name, {})
        explainability = self._generate_explainability(
            name=name,
            code=info.get("code", ""),
            explanation=info.get("explanation", ""),
            l2_metrics=l2_metrics,
            l3_metrics=l3_metrics,
            feature_series=feature_series,
        )

        feature_record = {
            "name": name,
            "code": info.get("code", ""),
            "explanation": info.get("explanation", ""),
            "reasoning": info.get("reasoning", ""),
            "dimension": info.get("dimension", ""),
            "l2_metrics": l2_metrics,
            "l3_metrics": l3_metrics,
            "score": score,
            "explainability": explainability,
        }

        # 岛屿管理（与engine.py一致：add_feature + migrate_best）
        dimension = info.get("dimension", "")
        island = self.island_manager.get_island(dimension)
        if island:
            island.add_feature(feature_record)
        self.island_manager.migrate_best(feature_record)
        self.state.accepted_features.append(feature_record)

        logger.info("Accepted feature %s: score=%.4f, delta_f1=%.6f, delta_recall=%.6f",
                    name, score, l3_metrics.get("delta_f1", 0), l3_metrics.get("delta_recall", 0))

        return (
            f"✓ 特征评估通过！已接受。(score={score:.4f})\n"
            f"IV: {l2_metrics.get('iv', 0):.4f}\n"
            f"KS: {l2_metrics.get('ks', 0):.4f}\n"
            f"冗余度: {l2_metrics.get('redundancy', 0):.4f}\n"
            f"SHAP Rank: #{l3_metrics.get('shap_rank', '?')}\n"
            f"Delta F1: {l3_metrics.get('delta_f1', 0):.6f}\n"
            f"Delta Recall: {l3_metrics.get('delta_recall', 0):.6f}\n"
            f"当前已接受特征数: {len(self.state.accepted_features)}/{self.max_features}\n"
            f"缓刑池: {len(self.pending_pool)} 个待评估"
        )

    def _tool_finish(self, args: dict) -> str:
        reason = args.get("reason", "")
        self.state.finished = True
        self.state.finish_reason = reason
        n = len(self.state.accepted_features)
        dims = list(self.state.dimensions_explored)
        return (
            f"Agent结束运行。\n"
            f"原因: {reason}\n"
            f"共生成 {n} 个有效特征\n"
            f"覆盖维度: {dims}\n"
            f"总尝试次数: {self.state.total_attempts}\n"
            f"缓刑池剩余: {len(self.pending_pool)} 个"
        )

    # ----------------------------------------------------------
    # 轮次边界检查
    # ----------------------------------------------------------

    def _check_round_boundary(self) -> bool:
        """检查是否到达轮次边界，触发Pending Pool评估和新轮引导。"""
        self._steps_in_round += 1
        if self._steps_in_round < self._steps_per_round:
            return False

        self._steps_in_round = 0

        # 轮末：评估Pending Pool（与engine.py run_round末尾一致）
        pool_accepted = self._run_pending_pool_evaluation()
        if pool_accepted:
            self._current_round_new_count += len(pool_accepted)

        # 收敛检测（与engine.py check_convergence一致）
        if self._current_round_new_count == 0:
            self._rounds_without_new += 1
        else:
            self._rounds_without_new = 0

        logger.info("Round %d finished: %d new features, rounds_without_new=%d",
                    self._round_num, self._current_round_new_count, self._rounds_without_new)

        if self._rounds_without_new >= 2:
            self.state.finished = True
            self.state.finish_reason = "连续2轮无新特征，收敛终止"
            logger.info("Converged: 2 consecutive rounds without new features")
            return True

        # 重置轮计数
        self._current_round_new_count = 0

        # 注入新轮引导prompt（含SHAP反馈 + 预算 + 轮内反馈）
        round_prompt = self._build_round_prompt()
        self.messages.append({"role": "user", "content": round_prompt})
        logger.info("=== Round %d started (SHAP feedback + budget injected) ===", self._round_num)
        return True

    # ----------------------------------------------------------
    # Step & Run
    # ----------------------------------------------------------

    def step(self) -> dict:
        """执行一步Agent推理。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=self.temperature,
        )

        message = response.choices[0].message
        self.messages.append(message.model_dump())

        step_info = {
            "step": len(self.step_log) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "thought": message.content or "",
            "tool_calls": [],
        }

        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                try:
                    observation = self._execute_tool(func_name, arguments)
                except Exception as e:
                    observation = f"工具执行异常: {e}"

                tool_info = {
                    "tool": func_name,
                    "arguments": arguments,
                    "observation": observation[:500],
                }
                step_info["tool_calls"].append(tool_info)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                })

                logger.info("Step %d: %s(%s) → %s",
                           step_info["step"], func_name,
                           json.dumps(arguments, ensure_ascii=False)[:100],
                           observation[:200])
        else:
            logger.info("Step %d: Agent思考: %s", step_info["step"], (message.content or "")[:200])

        self.step_log.append(step_info)
        return step_info

    def run(self) -> dict:
        """运行Agent直到完成或达到最大步数。"""
        self.messages = [{"role": "system", "content": self._build_system_prompt()}]

        initial_prompt = (
            f"开始自动特征工程任务。\n"
            f"目标: 生成最多 {self.max_features} 个高质量反洗钱特征\n"
            f"数据规模: {len(self.tx_df)} 条交易, {self.open_df.shape[0]} 个钱包\n"
            f"已有基线特征: {self.baseline_features.shape[1]} 个\n"
            f"正样本率: {self.label.mean()*100:.4f}%\n\n"
            f"请开始工作。建议先检索领域知识和查看数据结构，然后按维度逐步生成特征。"
        )
        self.messages.append({"role": "user", "content": initial_prompt})

        logger.info("=== FeatureEngineerAgent 启动 ===")
        logger.info("目标: %d 个特征, 最大步数: %d", self.max_features, self.max_steps)

        for step_num in range(self.max_steps):
            if self.state.finished:
                break
            if len(self.state.accepted_features) >= self.max_features:
                self.state.finished = True
                self.state.finish_reason = f"已达到目标特征数 {self.max_features}"
                break

            try:
                self.step()
            except Exception as e:
                logger.error("Step %d error: %s", step_num + 1, e)
                if (self.messages and self.messages[-1].get("role") == "assistant"
                        and self.messages[-1].get("tool_calls")):
                    self.messages.pop()
                self.messages.append({
                    "role": "user",
                    "content": f"上一步出现系统错误: {e}。请继续工作，尝试生成新的特征。"
                })

            # 轮次边界检查
            self._check_round_boundary()

        # 最终Pending Pool评估
        final_pool = self._run_pending_pool_evaluation()
        if final_pool:
            logger.info("Final pending pool evaluation: %d accepted", len(final_pool))

        result = {
            "accepted_features": self.state.accepted_features,
            "failed_features": self.state.failed_features,
            "total_steps": len(self.step_log),
            "total_attempts": self.state.total_attempts,
            "dimensions_explored": list(self.state.dimensions_explored),
            "finish_reason": self.state.finish_reason,
            "step_log": self.step_log,
            "pending_pool_remaining": len(self.pending_pool),
            "rounds_completed": self._round_num,
        }

        logger.info("=== Agent 完成 ===")
        logger.info("接受: %d, 失败: %d, 步数: %d, 轮数: %d, 缓刑池剩余: %d",
                   len(self.state.accepted_features),
                   len(self.state.failed_features),
                   len(self.step_log),
                   self._round_num,
                   len(self.pending_pool))

        return result
