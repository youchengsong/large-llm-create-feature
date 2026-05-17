"""单Agent特征工程系统：基于ReAct范式，完整复刻原系统逻辑（含SHAP反馈、Pending Pool、岛屿预算）。"""

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
from src.evaluation.validator import (
    run_l1_validation,
    run_l2_validation,
    run_l3_validation,
    compute_composite_score,
    backward_eliminate_pending,
)
from src.prompt.domain_knowledge import (
    load_transaction_patterns,
    load_expert_interview,
    load_risk_indicators,
    load_detection_rules,
    rules_to_python_examples,
)
from src.search.shap_feedback import generate_shap_feedback, shap_feedback_to_prompt, compute_island_budgets
from src.search.pending_pool import PendingPool, PendingFeature

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
                        "description": "特征所属维度（如时间、金额、网络、账户等）"
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
                        "description": "要执行的Python代码"
                    }
                },
                "required": ["feature_name", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_feature",
            "description": "评估已通过沙箱验证的特征的统计质量（IV、KS、冗余度）和模型增量效果（delta_f1、delta_recall）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_name": {
                        "type": "string",
                        "description": "要评估的特征名称"
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
                    },
                    "summary": {
                        "type": "string",
                        "description": "本次生成的总结"
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
    """Agent的运行状态。"""
    accepted_features: list[dict] = field(default_factory=list)
    failed_features: list[dict] = field(default_factory=list)
    pending_sandbox: dict[str, dict] = field(default_factory=dict)
    sandbox_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    total_attempts: int = 0
    dimensions_explored: set = field(default_factory=set)
    finished: bool = False
    finish_reason: str = ""


# ============================================================
# Single Agent (with full original system logic)
# ============================================================

class FeatureEngineerAgent:
    """
    单Agent特征工程系统。

    完整复刻原系统（src/search/engine.py）逻辑：
    - SHAP反馈驱动维度探索（弱维度获得更多关注）
    - Pending Pool（L2通过但L3失败的特征进入缓刑池，每轮重新评估）
    - 岛屿预算分配（基于SHAP重要性反比分配候选数）
    - Composite Score（30%SHAP + 20%IV + 20%冗余度 + 30%delta_f1）
    - 失败记忆（避免重复类似失败方向）
    - 收敛检测（连续2轮无新特征则终止）
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

        # 原系统组件
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
        self._steps_per_round = 25  # ~12 candidates per round (generate+sandbox each)

        self.state = AgentState()
        self.messages: list[dict] = []
        self.step_log: list[dict] = []

    @staticmethod
    def _clean_code(code: str) -> str:
        """清理LLM返回的代码中的转义字符。"""
        code = code.replace('\\"', '"')
        code = code.replace('\\n', '\n') if '\\n' in code and '\n' not in code else code
        return code

    def _get_shap_feedback(self) -> str:
        """生成SHAP反馈文本（与原系统 generate_shap_feedback + shap_feedback_to_prompt 一致）。"""
        if self.existing_features.shape[1] == 0:
            return "暂无已接受特征，所有维度均需探索。"

        try:
            shap_fb = generate_shap_feedback(self.existing_features, self.label)
            island_dims = [ic.dimension for ic in ISLANDS]
            return shap_feedback_to_prompt(shap_fb, island_dims)
        except Exception as e:
            logger.warning("SHAP feedback failed: %s", e)
            return "SHAP分析暂时不可用。"

    def _get_island_budgets(self) -> dict[str, int]:
        """计算每个维度的探索预算（与原系统 _compute_budgets 一致）。"""
        island_dims = [ic.dimension for ic in ISLANDS]

        if self._round_num <= 1 or self.existing_features.shape[1] == 0:
            return {dim: DEFAULT_CONFIG.search.b for dim in island_dims}

        try:
            shap_fb = generate_shap_feedback(self.existing_features, self.label)
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

    def _run_pending_pool_evaluation(self) -> list[dict]:
        """每轮结束时评估缓刑池（与原系统 run_pending_pool_evaluation 一致）。"""
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

            feature_record = {
                "name": pf.name,
                "code": pf.code,
                "explanation": pf.explanation,
                "reasoning": pf.reasoning,
                "dimension": pf.island_name,
                "l2_metrics": pf.l2_metrics,
                "l3_metrics": pf.l3_metrics,
                "score": score,
                "accepted_via": "pending_pool",
            }
            self.state.accepted_features.append(feature_record)
            new_records.append(feature_record)
            logger.info("Accepted from pending pool: %s (score=%.4f)", pf.name, score)

        self.pending_pool.tick_age()
        purged = self.pending_pool.purge_expired()
        if purged:
            logger.info("Purged %d expired features from pending pool", len(purged))

        return new_records

    def _build_system_prompt(self) -> str:
        """构建Agent的system prompt。"""
        island_desc = "\n".join(
            f"  - {ic.name}({ic.dimension}): {ic.description}, 字段: {', '.join(ic.fields)}"
            for ic in ISLANDS
        )

        return f"""你是一个专业的反洗钱特征工程Agent。你的任务是为数字人民币交易数据自动生成有效的检测特征。

## 你的能力
你可以使用以下工具：
1. retrieve_knowledge - 检索领域知识（交易模式、专家访谈、风险指标、检测规则）
2. inspect_schema - 查看数据表结构
3. generate_feature - 生成特征代码
4. execute_sandbox - 在沙箱中验证代码（通过后自动评估L2/L3质量）
5. evaluate_feature - 手动评估特征质量
6. finish - 结束生成过程

## 工作流程
1. 首先检索领域知识，了解洗钱模式
2. 查看数据schema，了解可用字段
3. 基于知识和数据，围绕指定维度设计并生成特征（generate_feature）
4. 在沙箱中验证代码（execute_sandbox）— 通过后系统自动评估质量
5. 根据评估反馈和SHAP分析调整策略，重点探索薄弱维度
6. 每轮生成约12个候选特征（4个维度各3个），覆盖多维度后用finish结束

## 维度岛屿
你需要覆盖以下4个维度，每轮根据SHAP反馈动态分配各维度的探索量：
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
- 重点关注SHAP分析中的薄弱维度（获得更多预算的维度）
- 从领域知识中寻找灵感，特别是裁判文书中的真实模式
- 如果某个方向连续失败，换方向探索（参考失败记忆）
- L3未通过的特征不会立即丢弃，会进入缓刑池等待后续评估"""

    def _build_round_prompt(self) -> str:
        """构建每轮开始时的引导prompt（含SHAP反馈和预算分配）。"""
        self._round_num += 1

        # SHAP反馈
        shap_text = self._get_shap_feedback()

        # 岛屿预算
        budgets = self._get_island_budgets()
        budget_text = "\n".join(
            f"  - {ic.name}({ic.dimension}): {budgets.get(ic.dimension, 3)} 个候选"
            for ic in ISLANDS
        )

        # 失败记忆（最近5个）
        recent_failures = self.state.failed_features[-5:] if self.state.failed_features else []
        fail_text = "\n".join(
            f"  - {f.get('name', '?')}: {f.get('error', '?')}"
            for f in recent_failures
        ) if recent_failures else "（暂无）"

        # 已接受特征摘要
        accepted_text = "\n".join(
            f"  - {f['name']}: score={f.get('score', 0):.4f}, "
            f"delta_f1={f.get('l3_metrics', {}).get('delta_f1', 0):.6f}, "
            f"SHAP_rank={f.get('l3_metrics', {}).get('shap_rank', '?')}"
            for f in self.state.accepted_features[-10:]
        ) if self.state.accepted_features else "（暂无）"

        # Pending pool状态
        pool_text = f"缓刑池中有 {len(self.pending_pool)} 个待评估特征" if not self.pending_pool.is_empty() else ""

        total_budget = sum(budgets.values())

        prompt = f"""## 第 {self._round_num} 轮迭代

{shap_text}

## 本轮预算分配（共 {total_budget} 个候选）
{budget_text}

## 近期失败特征（避免重复类似方向）
{fail_text}

## 已接受的高分特征
{accepted_text}

{pool_text}

## 任务
请按照预算分配，为每个维度生成指定数量的候选特征。
重点探索薄弱维度（预算更多的维度），尝试与已有特征不冗余的角度。
对每个候选特征：先 generate_feature，然后立即 execute_sandbox 验证。"""

        return prompt

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用，返回观察结果。"""
        if tool_name == "retrieve_knowledge":
            return self._tool_retrieve_knowledge(arguments)
        elif tool_name == "inspect_schema":
            return self._tool_inspect_schema(arguments)
        elif tool_name == "generate_feature":
            return self._tool_generate_feature(arguments)
        elif tool_name == "execute_sandbox":
            return self._tool_execute_sandbox(arguments)
        elif tool_name == "evaluate_feature":
            return self._tool_evaluate_feature(arguments)
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

        feature_df, error = run_l1_validation(code, self.open_df, self.tx_df, self.close_df)

        if feature_df is None:
            self.state.failed_features.append({"name": name, "error": f"L1: {error}"})
            return (
                f"✗ 沙箱执行失败\n"
                f"错误信息: {error}\n"
                f"请修复代码后重新生成，或换一个特征方向。"
            )

        self.state.sandbox_results[name] = feature_df
        shape = feature_df.shape
        cols = list(feature_df.columns)

        eval_result = self._tool_evaluate_feature({"feature_name": name})
        return (
            f"✓ 沙箱执行成功！\n"
            f"输出shape: {shape}\n"
            f"输出列: {cols}\n\n"
            f"--- 自动评估结果 ---\n{eval_result}"
        )

    def _tool_evaluate_feature(self, args: dict) -> str:
        name = args.get("feature_name", "")
        feature_df = self.state.sandbox_results.get(name)

        if feature_df is None:
            return f"特征 '{name}' 尚未通过沙箱验证，请先调用 execute_sandbox。"

        try:
            merged = merge_feature_to_transactions(feature_df, self.tx_df)
        except Exception as e:
            self.state.failed_features.append({"name": name, "error": f"merge失败: {e}"})
            return f"特征合并失败: {e}"

        if len(merged) != len(self.tx_df):
            self.state.failed_features.append({"name": name, "error": f"行数不匹配: {len(merged)} vs {len(self.tx_df)}"})
            return f"合并后行数不匹配: {len(merged)} vs {len(self.tx_df)}"

        feat_cols = [c for c in merged.columns if c not in self.tx_df.columns]
        if not feat_cols:
            self.state.failed_features.append({"name": name, "error": "合并后无新列"})
            return "合并后无新列产生"

        feature_series = merged[feat_cols[0]].fillna(0)
        feature_series.name = name

        # === L2验证（与原系统 evaluate_candidates 完全一致） ===
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

        # === L3验证（与原系统完全一致） ===
        l3_passed, l3_metrics = run_l3_validation(
            self.existing_features, feature_series, self.label,
            baseline_metrics=self.baseline_metrics,
        )

        if not l3_passed:
            # === 进入Pending Pool（与原系统一致：L2通过但L3失败不直接丢弃） ===
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
            logger.info("L3 failed for %s (delta_f1=%.6f), added to pending pool (%d in pool)",
                        name, l3_metrics.get("delta_f1", 0), len(self.pending_pool))
            return (
                f"△ L3模型增量不足，已加入缓刑池（{len(self.pending_pool)}个待评估）\n"
                f"IV: {l2_metrics.get('iv', 0):.4f}, KS: {l2_metrics.get('ks', 0):.4f}\n"
                f"Delta F1: {l3_metrics.get('delta_f1', 0):.6f}\n"
                f"Delta Recall: {l3_metrics.get('delta_recall', 0):.6f}\n"
                f"注: 该特征统计上有区分力，将在后续轮次中与其他特征组合重新评估。\n"
                f"建议: 继续探索其他方向，缓刑池特征可能在后续被接受。"
            )

        # === 接受特征（与原系统完全一致：记分 + 加入existing_features） ===
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

        info = self.state.pending_sandbox.get(name, {})
        feature_record = {
            "name": name,
            "code": info.get("code", ""),
            "explanation": info.get("explanation", ""),
            "reasoning": info.get("reasoning", ""),
            "dimension": info.get("dimension", ""),
            "l2_metrics": l2_metrics,
            "l3_metrics": l3_metrics,
            "score": score,
        }
        self.state.accepted_features.append(feature_record)

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

    def _check_round_boundary(self, step_num: int) -> bool:
        """检查是否到达轮次边界，如果是则触发Pending Pool评估和SHAP反馈。"""
        self._steps_in_round += 1
        if self._steps_in_round >= self._steps_per_round:
            self._steps_in_round = 0

            # 轮末：评估Pending Pool（与原系统 run_round 末尾一致）
            pool_accepted = self._run_pending_pool_evaluation()
            if pool_accepted:
                self._current_round_new_count += len(pool_accepted)

            # 收敛检测（与原系统一致：连续2轮无新特征则终止）
            if self._current_round_new_count == 0:
                self._rounds_without_new += 1
            else:
                self._rounds_without_new = 0

            if self._rounds_without_new >= 2:
                self.state.finished = True
                self.state.finish_reason = "连续2轮无新特征，收敛终止"
                logger.info("Converged: 2 rounds without new features")
                return True

            # 重置轮计数
            self._current_round_new_count = 0

            # 注入新轮引导prompt（含SHAP反馈和预算分配）
            round_prompt = self._build_round_prompt()
            self.messages.append({"role": "user", "content": round_prompt})
            logger.info("=== Round %d started (budget injected) ===", self._round_num)
            return True

        return False

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

            # 轮次边界检查：触发Pending Pool评估和SHAP反馈
            self._check_round_boundary(step_num)

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
