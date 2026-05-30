"""3 个 Function Calling 工具：explore_data / generate_feature / execute_sandbox。

提供 ToolSpec 定义（供 LLM Function Calling）和执行逻辑（ToolContext）。
"""

from __future__ import annotations

import io
import json
import logging
import sys
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from knowfeat.config import KnowFeatConfig, VerificationConfig
from knowfeat.data.loader import (
    BASELINE_EXCLUDE_FEATURES, Stage1Data, Stage2Data,
    get_label_series, get_label_series_stage1,
)
from knowfeat.llm.client import ToolSpec
from knowfeat.search.forward_selection import CandidateFeature
from knowfeat.verification.model_eval import ModelEvalResult, evaluate_feature
from knowfeat.verification.sandbox import SandboxResult, run_sandbox, run_sandbox_stage1
from knowfeat.verification.statistics import (
    NoiseThresholds,
    StatResult,
    validate_feature,
)

StageData = Stage1Data | Stage2Data

logger = logging.getLogger(__name__)

EXPLORE_TIMEOUT = 30
EXPLORE_OUTPUT_MAX_CHARS = 4000

# ---------------------------------------------------------------------------
# ToolSpec 定义（供 LLM Function Calling）
# ---------------------------------------------------------------------------

EXPLORE_DATA_SPEC = ToolSpec(
    name="explore_data",
    description=(
        "在 DataFrame 上执行 Python 代码探索数据分布。"
        "可用变量: open_df, tx_df, close_df, baseline_features。"
        "代码最后一行的值（或 print 输出）将作为结果返回。超时 15 秒。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码（可使用 pandas/numpy）",
            },
        },
        "required": ["code"],
    },
)

GENERATE_FEATURE_SPEC = ToolSpec(
    name="generate_feature",
    description=(
        "提交特征代码和溯源信息。代码必须定义函数 "
        "def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame，"
        "输出必须含 wallet_id 或 tx_id 列。"
        "提交后需调用 execute_sandbox 执行验证。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_name": {
                "type": "string",
                "description": "特征名称（英文 snake_case）",
            },
            "dimension": {
                "type": "array",
                "items": {"type": "string"},
                "description": "所属维度列表，如 [\"金额频次\", \"时间地域\"]",
            },
            "code": {
                "type": "string",
                "description": (
                    "Python 代码，签名 def feature_xxx(open_df, tx_df, close_df) "
                    "-> pd.DataFrame"
                ),
            },
            "explanation": {
                "type": "string",
                "description": "特征含义解释",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "设计推理过程：按计算步骤逐步解释，每一步先说明知识依据（因为），再说明计算操作（所以）。"
                    "格式示例: '步骤1: 因为RULE_002指出休眠账户(开户后≥3个月无交易)突然激活是洗钱预警信号，"
                    "所以计算dormant_days=(first_tx_time-wallet_open_date).days来量化休眠时长; "
                    "步骤2: 因为IND_04(高金额交易)和IND_05(金额偏离历史均值)表明激活后突然出现大额交易是异常模式，"
                    "所以筛选amount>10000的交易计算大额占比来捕捉这一信号; "
                    "步骤3: 因为RULE_002描述的完整模式是\"休眠+激活后大额频繁\"的复合特征，"
                    "所以将休眠天数与大额占比相乘得到dormant_large_tx_ratio，综合捕捉复合风险'"
                ),
            },
            "knowledge_source": {
                "type": "string",
                "description": "知识溯源的自然语言描述",
            },
            "knowledge_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "引用的知识资产ID列表，必须从领域知识中精确引用。"
                    "格式: RULE_001~015(检测规则), IND_01~38(风险指标), "
                    "EXP_01~N(专家意见段落), PAT_A01~05/PAT_N01~03(交易模式), "
                    "RF_S1_FF_NNN(Stage1风险因子), RF_S2_FFF_NNN(裁判文书风险因子)"
                ),
            },
            "computation_steps": {
                "type": "string",
                "description": (
                    "特征的计算步骤说明：逐步描述从原始字段/基线特征到最终特征值的每一步变换。"
                    "示例: '步骤1: 按src分组统计tx_df中amount>10000的笔数; "
                    "步骤2: 除以该src的总交易笔数得到大额占比; "
                    "步骤3: 取log1p变换稳定分布'"
                ),
            },
            "data_fields_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "使用的原始数据字段/基线特征列名列表",
            },
        },
        "required": [
            "feature_name",
            "dimension",
            "code",
            "explanation",
            "reasoning",
            "knowledge_source",
            "knowledge_ids",
            "computation_steps",
            "data_fields_used",
        ],
    },
)

EXECUTE_SANDBOX_SPEC = ToolSpec(
    name="execute_sandbox",
    description=(
        "对已提交的特征执行 L1→L2→L3 三级验证流水线。"
        "L1 沙盒执行 → L2 IV/KS/冗余度/缺失率 → L3 XGBoost delta-recall。"
        "返回完整验证结果。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_name": {
                "type": "string",
                "description": "要验证的特征名称（必须先通过 generate_feature 提交）",
            },
        },
        "required": ["feature_name"],
    },
)

# Stage 1 工具定义（单表输入）
EXPLORE_DATA_SPEC_STAGE1 = ToolSpec(
    name="explore_data",
    description=(
        "在 BABD-13 DataFrame 上执行 Python 代码探索数据分布。"
        "可用变量: df (完整数据表，含 account, 149 个特征列, label, is_illicit)。"
        "代码最后一行的值（或 print 输出）将作为结果返回。超时 15 秒。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码（可使用 pandas/numpy）",
            },
        },
        "required": ["code"],
    },
)

GENERATE_FEATURE_SPEC_STAGE1 = ToolSpec(
    name="generate_feature",
    description=(
        "提交特征代码和溯源信息。代码必须定义函数 "
        "def feature_xxx(df) -> pd.DataFrame，"
        "输入 df 是 BABD-13 完整数据表，"
        "输出必须含 account 列作为主键。"
        "提交后需调用 execute_sandbox 执行验证。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "feature_name": {
                "type": "string",
                "description": "特征名称（英文 snake_case）",
            },
            "dimension": {
                "type": "array",
                "items": {"type": "string"},
                "description": "所属维度列表，如 [\"金额频次\", \"时间地域\"]",
            },
            "code": {
                "type": "string",
                "description": (
                    "Python 代码，签名 def feature_xxx(df) -> pd.DataFrame"
                ),
            },
            "explanation": {
                "type": "string",
                "description": "特征含义解释",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "设计推理过程：按计算步骤逐步解释，每一步先说明知识依据（因为），再说明计算操作（所以）。"
                    "格式示例: '步骤1: 因为RF_S1_01_002指出洗钱地址资金集中在少数大额交易中，"
                    "所以计算PAIa14-R3/PAIa22-1得到集中分散比来量化这一模式; "
                    "步骤2: 因为原始比值存在极端长尾分布，"
                    "所以取log1p变换稳定分布，便于后续建模'"
                ),
            },
            "knowledge_source": {
                "type": "string",
                "description": "知识溯源的自然语言描述",
            },
            "knowledge_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "引用的知识资产ID列表，必须从领域知识中精确引用。"
                    "格式: RF_S1_FF_NNN(风险因子，FF=文件编号，NNN=因子序号)"
                ),
            },
            "computation_steps": {
                "type": "string",
                "description": (
                    "特征的计算步骤说明：逐步描述从原始字段/基线特征到最终特征值的每一步变换。"
                    "示例: '步骤1: 取PAIa14-R3(集中度)和PAIa22-1(分散度); "
                    "步骤2: 计算比值ratio=PAIa14-R3/PAIa22-1; "
                    "步骤3: log1p变换稳定分布'"
                ),
            },
            "data_fields_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "使用的原始数据字段/基线特征列名列表",
            },
        },
        "required": [
            "feature_name",
            "dimension",
            "code",
            "explanation",
            "reasoning",
            "knowledge_source",
            "knowledge_ids",
            "computation_steps",
            "data_fields_used",
        ],
    },
)

ALL_TOOL_SPECS = [EXPLORE_DATA_SPEC, GENERATE_FEATURE_SPEC, EXECUTE_SANDBOX_SPEC]
ALL_TOOL_SPECS_STAGE1 = [EXPLORE_DATA_SPEC_STAGE1, GENERATE_FEATURE_SPEC_STAGE1, EXECUTE_SANDBOX_SPEC]


def build_generic_tool_specs(
    dataset_name: str, id_col: str, target_col: str,
) -> list[ToolSpec]:
    """构建公开数据集的通用工具定义。"""
    explore = ToolSpec(
        name="explore_data",
        description=(
            f"在 {dataset_name} DataFrame 上执行 Python 代码探索数据分布。"
            f"可用变量: df (完整数据表，含 {id_col}, 原始特征列, {target_col})。"
            "代码最后一行的值（或 print 输出）将作为结果返回。超时 15 秒。"
        ),
        parameters=EXPLORE_DATA_SPEC_STAGE1.parameters,
    )
    generate = ToolSpec(
        name="generate_feature",
        description=(
            "提交特征代码和溯源信息。代码必须定义函数 "
            "def feature_xxx(df) -> pd.DataFrame，"
            f"输入 df 是 {dataset_name} 完整数据表，"
            f"输出必须含 {id_col} 列作为主键。"
            "提交后需调用 execute_sandbox 执行验证。"
        ),
        parameters=GENERATE_FEATURE_SPEC_STAGE1.parameters,
    )
    return [explore, generate, EXECUTE_SANDBOX_SPEC]


# ---------------------------------------------------------------------------
# 特征注册信息
# ---------------------------------------------------------------------------

@dataclass
class RegisteredFeature:
    """已通过 generate_feature 注册的特征。"""
    feature_name: str
    dimension: list[str]
    code: str
    explanation: str
    reasoning: str
    knowledge_source: str
    data_fields_used: list[str]
    knowledge_ids: list[str] = field(default_factory=list)
    computation_steps: str = ""


# ---------------------------------------------------------------------------
# 验证流水线结果
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """L1→L2→L3 三级验证完整结果。"""
    feature_name: str
    l1_result: Optional[SandboxResult] = None
    l2_result: Optional[StatResult] = None
    l3_result: Optional[ModelEvalResult] = None
    stage_reached: str = ""
    accepted: bool = False
    added_to_candidates: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------
# ToolContext: 持有共享状态，执行工具逻辑
# ---------------------------------------------------------------------------

class ToolContext:
    """工具执行上下文，持有数据、配置和运行时状态。

    Parameters
    ----------
    data : Stage1Data | Stage2Data
        Stage 1 单表或 Stage 2 三表数据 + 基线特征。
    cfg : KnowFeatConfig
        全局配置。
    noise_thresholds : NoiseThresholds, optional
        噪声校准阈值（如未提供，L2 使用 fallback 值）。
    baseline_X : pd.DataFrame, optional
        当前基线特征矩阵（已接受的特征）。未提供则从 data 中构造。
    """

    def __init__(
        self,
        data: StageData,
        cfg: KnowFeatConfig,
        noise_thresholds: Optional[NoiseThresholds] = None,
        baseline_X: Optional[pd.DataFrame] = None,
    ):
        self.data = data
        self.cfg = cfg
        self.vcfg = cfg.verification
        self.noise_thresholds = noise_thresholds
        self.is_stage1 = isinstance(data, Stage1Data)
        self.is_generic = cfg.stage == "generic"

        if self.is_stage1:
            self.label = get_label_series_stage1(data)
        else:
            self.label = get_label_series(data)

        if baseline_X is not None:
            self.baseline_X = baseline_X.copy()
        elif data.baseline_features is not None:
            self.baseline_X = data.baseline_features.copy()
        else:
            self.baseline_X = pd.DataFrame(index=self.label.index)

        if not self.is_stage1:
            drop_cols = [c for c in BASELINE_EXCLUDE_FEATURES if c in self.baseline_X.columns]
            if drop_cols:
                self.baseline_X = self.baseline_X.drop(columns=drop_cols)

        self.registered: dict[str, RegisteredFeature] = {}
        self.accepted_features: dict[str, CandidateFeature] = {}
        self.candidate_pool: list[CandidateFeature] = []
        self.rejected_features: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 调度：按 tool name 执行
    # ------------------------------------------------------------------
    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """执行工具调用，返回结果文本。"""
        handlers = {
            "explore_data": self._explore_data,
            "generate_feature": self._generate_feature,
            "execute_sandbox": self._execute_sandbox,
        }
        handler = handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            return handler(arguments)
        except Exception as exc:
            logger.exception("工具 %s 执行异常", name)
            return json.dumps(
                {"error": f"工具执行异常: {exc}"}, ensure_ascii=False
            )

    # ------------------------------------------------------------------
    # explore_data
    # ------------------------------------------------------------------
    def _explore_data(self, args: dict[str, Any]) -> str:
        code = args.get("code", "")
        if not code.strip():
            return json.dumps({"error": "代码不能为空"}, ensure_ascii=False)

        if self.is_stage1:
            result = _run_explore_in_process_stage1(
                code, self.data.df, timeout=EXPLORE_TIMEOUT,
            )
        else:
            result = _run_explore_in_process(
                code, self.data.open_df, self.data.tx_df,
                self.data.close_df, self.data.baseline_features,
                timeout=EXPLORE_TIMEOUT,
            )
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # generate_feature
    # ------------------------------------------------------------------
    def _generate_feature(self, args: dict[str, Any]) -> str:
        feature_name = args.get("feature_name", "")
        if not feature_name:
            return json.dumps(
                {"error": "feature_name 不能为空"}, ensure_ascii=False
            )

        code = args.get("code", "")
        if not code.strip():
            return json.dumps({"error": "code 不能为空"}, ensure_ascii=False)

        dimension = args.get("dimension", [])
        if isinstance(dimension, str):
            dimension = [dimension]

        knowledge_ids = args.get("knowledge_ids", [])
        if isinstance(knowledge_ids, str):
            knowledge_ids = [knowledge_ids]

        reg = RegisteredFeature(
            feature_name=feature_name,
            dimension=dimension,
            code=code,
            explanation=args.get("explanation", ""),
            reasoning=args.get("reasoning", ""),
            knowledge_source=args.get("knowledge_source", ""),
            data_fields_used=args.get("data_fields_used", []),
            knowledge_ids=knowledge_ids,
            computation_steps=args.get("computation_steps", ""),
        )
        self.registered[feature_name] = reg

        logger.info("特征已注册: %s (维度=%s)", feature_name, dimension)

        return json.dumps(
            {
                "status": "registered",
                "feature_name": feature_name,
                "dimension": dimension,
                "message": f"特征 '{feature_name}' 已注册，请调用 execute_sandbox 执行验证。",
            },
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # execute_sandbox → L1 → L2 → L3 流水线
    # ------------------------------------------------------------------
    def _execute_sandbox(self, args: dict[str, Any]) -> str:
        feature_name = args.get("feature_name", "")
        if not feature_name:
            return json.dumps(
                {"error": "feature_name 不能为空"}, ensure_ascii=False
            )

        reg = self.registered.get(feature_name)
        if reg is None:
            return json.dumps(
                {"error": f"特征 '{feature_name}' 未注册，请先调用 generate_feature"},
                ensure_ascii=False,
            )

        result = self._run_pipeline(reg)
        return self._format_pipeline_result(result)

    def _run_pipeline(self, reg: RegisteredFeature) -> PipelineResult:
        pr = PipelineResult(feature_name=reg.feature_name)

        # --- L1: 沙盒执行 ---
        if self.is_stage1:
            l1 = run_sandbox_stage1(
                code=reg.code,
                df=self.data.df,
                timeout=self.vcfg.sandbox_timeout_sec,
            )
        else:
            l1 = run_sandbox(
                code=reg.code,
                open_df=self.data.open_df,
                tx_df=self.data.tx_df,
                close_df=self.data.close_df,
                timeout=self.vcfg.sandbox_timeout_sec,
            )
        pr.l1_result = l1
        pr.stage_reached = "L1"

        if not l1.success:
            pr.summary = f"L1 失败: {l1.error}"
            logger.info("L1 沙盒失败: %s — %s", reg.feature_name, l1.error)
            return pr

        # 提取特征列（除 ID 列外的第一个数值列）
        feature_series = _extract_feature_column(l1.df, reg.feature_name)
        if feature_series is None:
            pr.summary = "L1 通过但无法提取特征列（输出仅含 ID 列）"
            logger.warning("无法提取特征列: %s", reg.feature_name)
            return pr

        # 对齐 label 索引
        if self.is_stage1:
            feature_aligned, label_aligned = _align_series_stage1(
                feature_series, l1.df, self.label, self.data,
            )
        else:
            feature_aligned, label_aligned = _align_series(
                feature_series, l1.df, self.label, self.data,
            )

        # --- L2: 统计验证 ---
        existing_features = self._get_existing_features_df()
        l2 = validate_feature(
            feature=feature_aligned,
            label=label_aligned,
            existing_features=existing_features if not existing_features.empty else None,
            thresholds=self.noise_thresholds,
            vcfg=self.vcfg,
        )
        pr.l2_result = l2
        pr.stage_reached = "L2"

        if not l2.passed:
            pr.summary = f"L2 失败: {'; '.join(l2.reject_reasons)}"
            self.rejected_features.append({
                "feature_name": reg.feature_name,
                "code": reg.code,
                "reason": pr.summary,
                "stage": "L2",
            })
            logger.info("L2 统计失败: %s", reg.feature_name)
            return pr

        # --- L3: 模型效果验证 ---
        l3 = evaluate_feature(
            baseline_X=self.baseline_X,
            new_feature=feature_aligned.rename(reg.feature_name),
            label=label_aligned,
            vcfg=self.vcfg,
            seed=self.cfg.seed,
        )
        pr.l3_result = l3
        pr.stage_reached = "L3"

        if l3.passed:
            # 直接接受：更新 baseline_X
            pr.accepted = True
            cand = self._make_candidate(reg, feature_aligned)
            self.accepted_features[reg.feature_name] = cand
            self.baseline_X[reg.feature_name] = feature_aligned.values
            pr.summary = (
                f"L3 通过 — 已接受! delta_recall={l3.delta_recall:+.4f}, "
                f"precision={l3.new_precision:.4f}"
            )
            logger.info(
                "L3 通过 → 已接受: %s (delta_recall=%+.4f)",
                reg.feature_name, l3.delta_recall,
            )
        else:
            # L3 未通过 → 加入候选池等待前向选择
            pr.added_to_candidates = True
            cand = self._make_candidate(reg, feature_aligned)
            self.candidate_pool.append(cand)
            pr.summary = (
                f"L2 通过但 L3 未通过 ({'; '.join(l3.reject_reasons)}) "
                f"— 已加入候选池等待前向选择"
            )
            logger.info(
                "L3 未通过 → 候选池: %s (%s)",
                reg.feature_name, "; ".join(l3.reject_reasons),
            )

        return pr

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _get_existing_features_df(self) -> pd.DataFrame:
        """返回已接受的特征 DataFrame（用于冗余度计算）。"""
        if not self.accepted_features:
            return pd.DataFrame()
        return pd.DataFrame({
            name: cand.series
            for name, cand in self.accepted_features.items()
        })

    def _make_candidate(
        self,
        reg: RegisteredFeature,
        feature_series: pd.Series,
    ) -> CandidateFeature:
        return CandidateFeature(
            name=reg.feature_name,
            series=feature_series,
            dimension=reg.dimension,
            code=reg.code,
            explanation=reg.explanation,
            reasoning=reg.reasoning,
            knowledge_source=reg.knowledge_source,
            data_fields_used=reg.data_fields_used,
            knowledge_ids=reg.knowledge_ids,
            computation_steps=reg.computation_steps,
        )

    @staticmethod
    def _format_pipeline_result(pr: PipelineResult) -> str:
        result: dict[str, Any] = {
            "feature_name": pr.feature_name,
            "stage_reached": pr.stage_reached,
            "accepted": pr.accepted,
            "summary": pr.summary,
        }

        if pr.l1_result is not None:
            l1 = pr.l1_result
            result["L1_sandbox"] = {
                "passed": l1.success,
                "error": l1.error or None,
                "output_shape": list(l1.df.shape) if l1.df is not None else None,
            }

        if pr.l2_result is not None:
            l2 = pr.l2_result
            result["L2_statistics"] = {
                "passed": l2.passed,
                "IV": round(l2.iv, 4),
                "KS": round(l2.ks, 4),
                "missing_rate": round(l2.missing_rate, 4),
                "redundancy": round(l2.redundancy, 4),
                "iv_threshold": round(l2.iv_threshold, 4),
                "ks_threshold": round(l2.ks_threshold, 4),
                "reject_reasons": l2.reject_reasons or None,
            }

        if pr.l3_result is not None:
            l3 = pr.l3_result
            result["L3_model_eval"] = {
                "passed": l3.passed,
                "baseline_recall": round(l3.baseline_recall, 4),
                "new_recall": round(l3.new_recall, 4),
                "delta_recall": round(l3.delta_recall, 4),
                "new_precision": round(l3.new_precision, 4),
                "delta_f1": round(l3.delta_f1, 4),
                "reject_reasons": l3.reject_reasons or None,
            }

        if pr.added_to_candidates:
            result["note"] = "特征已加入候选池，将在前向选择阶段评估组合效应"

        return json.dumps(result, ensure_ascii=False, indent=2)

    def get_tool_specs(self) -> list[ToolSpec]:
        """返回所有工具的 ToolSpec 列表。"""
        if self.is_generic:
            pub = self.cfg.public_dataset
            return build_generic_tool_specs(
                pub.dataset_name, pub.id_col, pub.target_col,
            )
        if self.is_stage1:
            return list(ALL_TOOL_SPECS_STAGE1)
        return list(ALL_TOOL_SPECS)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _explore_worker(code, open_df, tx_df, close_df, baseline_features, result_queue):
    """Worker function for explore_data, runs in a subprocess via multiprocessing."""
    import numpy as np

    namespace = {
        "pd": pd,
        "np": np,
        "open_df": open_df,
        "tx_df": tx_df,
        "close_df": close_df,
        "baseline_features": baseline_features,
    }
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        compiled = compile(code, "<explore>", "exec")
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compiled, namespace)
        last_expr = _try_eval_last_line(code, namespace)

        output_parts = []
        stdout_text = stdout_buf.getvalue().strip()
        if stdout_text:
            output_parts.append(stdout_text)
        if last_expr is not None:
            output_parts.append(str(last_expr))

        output = "\n".join(output_parts) if output_parts else "(无输出)"
        if len(output) > EXPLORE_OUTPUT_MAX_CHARS:
            output = output[:EXPLORE_OUTPUT_MAX_CHARS] + "\n... [输出截断]"
        result_queue.put({"output": output})
    except Exception as exc:
        result_queue.put({"error": f"代码执行错误: {exc}"})


def _run_explore_in_process(code, open_df, tx_df, close_df, baseline_features, timeout=30):
    """Run explore_data code in a child process with hard timeout."""
    import multiprocessing
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_explore_worker,
        args=(code, open_df, tx_df, close_df, baseline_features, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return {"error": f"explore_data 执行超时（>{timeout}s），请简化代码或减少数据量"}

    if not result_queue.empty():
        return result_queue.get_nowait()
    return {"error": f"explore_data 执行异常（进程退出码={proc.exitcode}）"}


def _try_eval_last_line(code: str, namespace: dict[str, Any]) -> Any:
    """尝试对代码最后一行求值，如果是表达式则返回其值。"""
    lines = code.strip().splitlines()
    if not lines:
        return None
    last = lines[-1].strip()
    if not last or last.startswith(("#", "import", "from", "def", "class")):
        return None
    try:
        return eval(last, namespace)
    except Exception:
        return None


def _extract_feature_column(
    df: pd.DataFrame,
    feature_name: str,
) -> Optional[pd.Series]:
    """从沙盒输出中提取特征列（排除 ID 列后的第一个列）。"""
    id_cols = {"wallet_id", "tx_id", "account", "row_id"}
    feature_cols = [c for c in df.columns if c not in id_cols]
    if not feature_cols:
        return None
    col = feature_cols[0]
    return pd.to_numeric(df[col], errors="coerce").rename(feature_name)


def _align_series(
    feature_series: pd.Series,
    sandbox_df: pd.DataFrame,
    label: pd.Series,
    data: Stage2Data,
) -> tuple[pd.Series, pd.Series]:
    """将沙盒输出的特征列与标签对齐。

    沙盒输出可能是 wallet_id 级别或 tx_id 级别。
    根据输出 ID 列类型与 tx_df 进行 JOIN 对齐。
    """
    if "tx_id" in sandbox_df.columns:
        tx_index = data.tx_df["tx_id"].values if "tx_id" in data.tx_df.columns else None
        if tx_index is not None and len(sandbox_df) == len(data.tx_df):
            aligned_feat = feature_series.values[:len(label)]
            return (
                pd.Series(aligned_feat, index=label.index, name=feature_series.name),
                label,
            )

    if "wallet_id" in sandbox_df.columns:
        wallet_map = dict(zip(sandbox_df["wallet_id"], feature_series.values))
        if "src" in data.tx_df.columns:
            aligned = data.tx_df["src"].map(wallet_map)
        elif "wallet_id" in data.tx_df.columns:
            aligned = data.tx_df["wallet_id"].map(wallet_map)
        else:
            aligned = pd.Series(feature_series.values[:len(label)], index=label.index)
        return (
            aligned.fillna(0).rename(feature_series.name).reset_index(drop=True),
            label.reset_index(drop=True),
        )

    aligned_feat = feature_series.values[:len(label)]
    return (
        pd.Series(aligned_feat, index=label.index, name=feature_series.name),
        label,
    )


# ---------------------------------------------------------------------------
# Stage 1 辅助函数
# ---------------------------------------------------------------------------

def _align_series_stage1(
    feature_series: pd.Series,
    sandbox_df: pd.DataFrame,
    label: pd.Series,
    data: Stage1Data,
) -> tuple[pd.Series, pd.Series]:
    """Stage 1 / Generic: 将沙盒输出与标签对齐。

    通过 id_col（account / row_id 等）做 JOIN 对齐。
    """
    id_col = data.id_col
    if id_col in sandbox_df.columns and id_col in data.df.columns:
        id_map = dict(zip(sandbox_df[id_col], feature_series.values))
        aligned = data.df[id_col].map(id_map)
        return (
            aligned.fillna(0).rename(feature_series.name).reset_index(drop=True),
            label.reset_index(drop=True),
        )

    if len(sandbox_df) == len(data.df):
        return (
            pd.Series(feature_series.values, index=label.index, name=feature_series.name),
            label,
        )

    aligned_feat = feature_series.values[:len(label)]
    return (
        pd.Series(aligned_feat, index=label.index, name=feature_series.name),
        label,
    )


def _explore_worker_stage1(code, df, result_queue):
    """Stage 1 explore worker: 单表输入。"""
    import numpy as np

    namespace = {
        "pd": pd,
        "np": np,
        "df": df,
    }
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        compiled = compile(code, "<explore>", "exec")
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compiled, namespace)
        last_expr = _try_eval_last_line(code, namespace)

        output_parts = []
        stdout_text = stdout_buf.getvalue().strip()
        if stdout_text:
            output_parts.append(stdout_text)
        if last_expr is not None:
            output_parts.append(str(last_expr))

        output = "\n".join(output_parts) if output_parts else "(无输出)"
        if len(output) > EXPLORE_OUTPUT_MAX_CHARS:
            output = output[:EXPLORE_OUTPUT_MAX_CHARS] + "\n... [输出截断]"
        result_queue.put({"output": output})
    except Exception as exc:
        result_queue.put({"error": f"代码执行错误: {exc}"})


def _run_explore_in_process_stage1(code, df, timeout=30):
    """Stage 1: run explore_data in child process."""
    import multiprocessing
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_explore_worker_stage1,
        args=(code, df, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return {"error": f"explore_data 执行超时（>{timeout}s），请简化代码或减少数据量"}

    if not result_queue.empty():
        return result_queue.get_nowait()
    return {"error": f"explore_data 执行异常（进程退出码={proc.exitcode}）"}
