"""单Agent特征工程系统：基于ReAct范式的自主特征生成Agent。"""

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
    load_risk_indicators,
    load_detection_rules,
    rules_to_python_examples,
)

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
            "description": "在沙箱中执行特征代码，验证代码正确性和输出格式。返回执行结果或错误信息。",
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
    pending_sandbox: dict[str, dict] = field(default_factory=dict)  # name -> {code, explanation, ...}
    sandbox_results: dict[str, pd.DataFrame] = field(default_factory=dict)  # name -> feature_df
    total_attempts: int = 0
    dimensions_explored: set = field(default_factory=set)
    finished: bool = False
    finish_reason: str = ""


# ============================================================
# Single Agent
# ============================================================

class FeatureEngineerAgent:
    """
    单Agent特征工程系统。

    使用ReAct范式，通过Tool Use自主决定：
    1. 检索哪些领域知识
    2. 探索哪个特征维度
    3. 生成什么特征代码
    4. 如何修复失败的代码
    5. 何时停止生成
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
        max_steps: int = 50,
        max_features: int = 20,
        temperature: float = 0.7,
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

        self.state = AgentState()
        self.messages: list[dict] = []
        self.step_log: list[dict] = []

    def _build_system_prompt(self) -> str:
        """构建Agent的system prompt。"""
        return """你是一个专业的反洗钱特征工程Agent。你的任务是为数字人民币交易数据自动生成有效的检测特征。

## 你的能力
你可以使用以下工具：
1. retrieve_knowledge - 检索领域知识（交易模式、专家访谈、风险指标、检测规则）
2. inspect_schema - 查看数据表结构
3. generate_feature - 生成特征代码
4. execute_sandbox - 在沙箱中验证代码
5. evaluate_feature - 评估特征质量和增量效果
6. finish - 结束生成过程

## 工作流程
1. 首先检索领域知识，了解洗钱模式
2. 查看数据schema，了解可用字段
3. 基于知识和数据，设计并生成特征
4. 在沙箱中验证代码正确性
5. 评估特征的统计质量和模型增量
6. 根据反馈调整策略，生成更多特征
7. 覆盖多个维度后结束

## 特征代码要求
- 函数签名: def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame
- 返回DataFrame必须包含 wallet_id 或 tx_id 列
- 不能使用标签列(is_risk, is_laundering等)
- 需要处理空值情况

## 策略建议
- 优先探索与已有特征低相关的方向
- 从领域知识中寻找灵感，特别是裁判文书中的真实模式
- 考虑时间、金额、网络、账户、设备等多个维度
- 如果某个维度连续失败，换方向探索"""

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
        code = args.get("code", "")
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
        code = args.get("code", "")

        if not code and name in self.state.pending_sandbox:
            code = self.state.pending_sandbox[name]["code"]

        feature_df, error = run_l1_validation(code, self.open_df, self.tx_df, self.close_df)

        if feature_df is not None:
            self.state.sandbox_results[name] = feature_df
            shape = feature_df.shape
            cols = list(feature_df.columns)
            return (
                f"✓ 沙箱执行成功！\n"
                f"输出shape: {shape}\n"
                f"输出列: {cols}\n"
                f"请使用 evaluate_feature 工具评估该特征的质量。"
            )
        else:
            self.state.failed_features.append({"name": name, "error": error})
            return (
                f"✗ 沙箱执行失败\n"
                f"错误信息: {error}\n"
                f"请修复代码后重新生成，或换一个特征方向。"
            )

    def _tool_evaluate_feature(self, args: dict) -> str:
        name = args.get("feature_name", "")
        feature_df = self.state.sandbox_results.get(name)

        if feature_df is None:
            return f"特征 '{name}' 尚未通过沙箱验证，请先调用 execute_sandbox。"

        try:
            merged = merge_feature_to_transactions(feature_df, self.tx_df)
        except Exception as e:
            return f"特征合并失败: {e}"

        if len(merged) != len(self.tx_df):
            return f"合并后行数不匹配: {len(merged)} vs {len(self.tx_df)}"

        feat_cols = [c for c in merged.columns if c not in self.tx_df.columns]
        if not feat_cols:
            return "合并后无新列产生"

        feature_series = merged[feat_cols[0]].fillna(0)
        feature_series.name = name

        l2_passed, l2_metrics = run_l2_validation(
            feature_series, self.label, self.existing_features
        )

        if not l2_passed:
            reasons = []
            if l2_metrics.get("iv", 0) < 0.02:
                reasons.append(f"IV过低({l2_metrics['iv']:.4f}<0.02)")
            if l2_metrics.get("redundancy", 0) >= 0.7:
                reasons.append(f"冗余度过高({l2_metrics['redundancy']:.4f}>=0.7)")
            return (
                f"✗ 统计质量评估未通过\n"
                f"IV: {l2_metrics.get('iv', 0):.4f}\n"
                f"KS: {l2_metrics.get('ks', 0):.4f}\n"
                f"冗余度: {l2_metrics.get('redundancy', 0):.4f}\n"
                f"原因: {'; '.join(reasons)}\n"
                f"建议: 尝试不同的计算逻辑或探索其他维度。"
            )

        l3_passed, l3_metrics = run_l3_validation(
            self.existing_features, feature_series, self.label
        )

        if l3_passed:
            self.existing_features[name] = feature_series.values
            info = self.state.pending_sandbox.get(name, {})
            feature_record = {
                "name": name,
                "code": info.get("code", ""),
                "explanation": info.get("explanation", ""),
                "reasoning": info.get("reasoning", ""),
                "dimension": info.get("dimension", ""),
                "l2_metrics": l2_metrics,
                "l3_metrics": l3_metrics,
            }
            self.state.accepted_features.append(feature_record)

            return (
                f"✓ 特征评估通过！已接受。\n"
                f"IV: {l2_metrics.get('iv', 0):.4f}\n"
                f"KS: {l2_metrics.get('ks', 0):.4f}\n"
                f"冗余度: {l2_metrics.get('redundancy', 0):.4f}\n"
                f"Delta F1: {l3_metrics.get('delta_f1', 0):.6f}\n"
                f"Delta Recall: {l3_metrics.get('delta_recall', 0):.6f}\n"
                f"当前已接受特征数: {len(self.state.accepted_features)}/{self.max_features}"
            )
        else:
            return (
                f"△ 统计质量合格但模型增量不足\n"
                f"IV: {l2_metrics.get('iv', 0):.4f}, KS: {l2_metrics.get('ks', 0):.4f}\n"
                f"Delta F1: {l3_metrics.get('delta_f1', 0):.6f}\n"
                f"Delta Recall: {l3_metrics.get('delta_recall', 0):.6f}\n"
                f"建议: 该特征统计上有区分力但对模型增量贡献不大，可能与已有特征信息重叠。尝试探索不同方向。"
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
            f"总尝试次数: {self.state.total_attempts}"
        )

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

                observation = self._execute_tool(func_name, arguments)

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
            f"请开始工作。建议先检索领域知识和查看数据结构。"
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
                self.messages.append({
                    "role": "user",
                    "content": f"系统错误: {e}。请继续工作。"
                })

        result = {
            "accepted_features": self.state.accepted_features,
            "failed_features": self.state.failed_features,
            "total_steps": len(self.step_log),
            "total_attempts": self.state.total_attempts,
            "dimensions_explored": list(self.state.dimensions_explored),
            "finish_reason": self.state.finish_reason,
            "step_log": self.step_log,
        }

        logger.info("=== Agent 完成 ===")
        logger.info("接受: %d, 失败: %d, 步数: %d",
                   len(self.state.accepted_features),
                   len(self.state.failed_features),
                   len(self.step_log))

        return result

    def save_results(self, output_path: str = "outputs/agent_features.json"):
        """保存生成结果。"""
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        results = {
            "agent_type": "single",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "model": self.model,
                "max_steps": self.max_steps,
                "max_features": self.max_features,
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
                }
                for f in self.state.accepted_features
            ],
            "summary": {
                "total_accepted": len(self.state.accepted_features),
                "total_failed": len(self.state.failed_features),
                "total_steps": len(self.step_log),
                "dimensions": list(self.state.dimensions_explored),
                "finish_reason": self.state.finish_reason,
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("Results saved to %s", output_path)
