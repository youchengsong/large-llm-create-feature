"""System Prompt 组装：静态知识 + 动态反馈 → Feature Agent 的完整上下文。

将领域知识、数据 Schema、SHAP 反馈、失败记忆、跨维度高分示例
组装为结构化 System Prompt，供 ReAct Agent 每轮使用。
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from knowfeat.config import KnowFeatConfig
from knowfeat.data.loader import Stage1Data, Stage2Data
from knowfeat.prompt.domain_knowledge import PublicDatasetKnowledge
from knowfeat.data.schema_stats import generate_all_schema_text, generate_all_schema_text_stage1
from knowfeat.prompt.domain_knowledge import (
    DomainKnowledge,
    format_domain_knowledge,
)

logger = logging.getLogger(__name__)


# ── 动态注入数据结构 ──────────────────────────────────────────────────────────

@dataclass
class ShapFeedback:
    """SHAP 维度贡献摘要，由 search/shap_feedback.py 生成。"""
    dimension_contributions: dict[str, float] = field(default_factory=dict)
    top_features: list[dict] = field(default_factory=list)
    round_num: int = 0


@dataclass
class FailureRecord:
    """单条失败记录，由 search/failure_memory.py 管理。"""
    feature_name: str
    code: str
    reason: str
    round_num: int = 0
    dimension: str = ""


@dataclass
class CrossDimExample:
    """跨维度传播的高分特征摘要，由 search/cross_dim.py 生成。"""
    feature_name: str
    source_dimension: str
    code: str
    metrics: dict[str, float] = field(default_factory=dict)
    shap_rank: int = 0


# ── 格式化辅助函数 ────────────────────────────────────────────────────────────

def _format_shap_feedback(feedback: ShapFeedback) -> str:
    lines = ["## SHAP 维度贡献反馈\n"]
    if not feedback.dimension_contributions:
        lines.append("暂无 SHAP 反馈数据。\n")
        return "\n".join(lines)

    sorted_dims = sorted(
        feedback.dimension_contributions.items(),
        key=lambda x: x[1], reverse=True,
    )
    lines.append("| 维度 | SHAP 贡献占比 | 状态 |")
    lines.append("|------|--------------|------|")
    for dim, score in sorted_dims:
        status = "✦ 强" if score >= 0.3 else ("● 中" if score >= 0.15 else "○ 弱")
        lines.append(f"| {dim} | {score:.1%} | {status} |")

    if feedback.top_features:
        lines.append(f"\n**当前 Top 特征（截至第 {feedback.round_num} 轮）**:")
        for feat in feedback.top_features[:5]:
            name = feat.get("name", "")
            importance = feat.get("importance", 0)
            dim = feat.get("dimension", "")
            lines.append(f"- {name} (SHAP={importance:.4f}, 维度={dim})")

    lines.append("")
    return "\n".join(lines)


def _format_failure_memory(records: list[FailureRecord]) -> str:
    if not records:
        return ""

    lines = [f"## 失败记忆（最近 {len(records)} 条，避免重复相同错误）\n"]
    for i, rec in enumerate(records, 1):
        lines.append(f"### 失败 #{i}: {rec.feature_name}")
        if rec.dimension:
            lines.append(f"- 维度: {rec.dimension}")
        lines.append(f"- 失败原因: {rec.reason}")
        lines.append(f"- 代码片段:")
        code_preview = rec.code[:300] + "..." if len(rec.code) > 300 else rec.code
        lines.append(f"```python\n{code_preview}\n```")
        lines.append("")

    return "\n".join(lines)


def _format_cross_dim_examples(examples: list[CrossDimExample]) -> str:
    if not examples:
        return ""

    lines = [f"## 跨维度高分特征参考（{len(examples)} 个）\n"]
    lines.append("以下是其他维度中表现优秀的特征，可作为灵感来源：\n")
    for ex in examples:
        lines.append(f"### {ex.feature_name} (来自「{ex.source_dimension}」维度)")
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in ex.metrics.items()
        )
        if metrics_str:
            lines.append(f"- 指标: {metrics_str}")
        if ex.shap_rank > 0:
            lines.append(f"- SHAP 排名: #{ex.shap_rank}")
        lines.append(f"```python\n{ex.code}\n```")
        lines.append("")

    return "\n".join(lines)


# ── 任务角色定义 ──────────────────────────────────────────────────────────────

_ROLE_STAGE2 = """\
# 角色

你是一个数字人民币反洗钱特征工程专家 Agent。你的任务是基于领域知识和数据分析，\
设计并生成能有效识别洗钱行为的统计特征。

## 目标

- 生成能提升洗钱检测 Recall 的特征（Precision 需 ≥ 90%）
- 每个特征必须有明确的领域知识依据和可追溯的设计逻辑
- **溯源要求**：必须在 knowledge_ids 中填写具体的知识资产ID（如 RULE_002, IND_05, EXP_03, PAT_A02, RF_S2_100_001），不要用笼统描述
- **计算步骤**：必须在 computation_steps 中逐步描述从原始字段到最终特征值的每一步变换
- **设计推理**：必须在 reasoning 中按计算步骤逐步解释，每一步先说明知识依据（因为），再说明计算操作（所以）。格式: 步骤N: 因为ID(含义)指出...，所以...
- 特征代码必须可执行，签名为 `def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame`
- 输出 DataFrame 必须包含 `wallet_id` 或 `tx_id` 列作为主键

## 可用工具

1. **explore_data**: 在 DataFrame 上执行查询代码，探索数据分布和特征
2. **generate_feature**: 提交特征代码 + 名称 + 维度 + 溯源信息
3. **execute_sandbox**: 在隔离沙盒中执行代码，触发 L1→L2→L3 三级验证

## 工作流程（严格遵守）

1. 阅读数据 Schema 和领域知识，理解哪些维度/信号可能有区分度
2. 使用 explore_data 探索数据分布，验证假设
3. 设计特征，使用 generate_feature 提交
4. **立即**调用 execute_sandbox 验证该特征
5. 根据验证结果迭代改进，然后设计下一个特征

**重要：必须逐个生成和验证！每次 generate_feature 后必须立即调用 execute_sandbox，不要批量注册多个特征后再验证。只有通过验证的特征才计入预算。**
"""

_ROLE_STAGE1 = """\
# 角色

你是一个加密货币反洗钱特征工程专家 Agent。你的任务是基于加密货币洗钱案例的\
结构化风险因子，在 BABD-13 数据集上设计识别洗钱地址的统计特征。

## 数据集说明

BABD-13 是比特币地址行为数据集，每行代表一个比特币地址，包含 149 个预计算的\
链上行为特征（PAI/PDI/PTI/CI/S 五类指标）和 13 类标签。\
二分类标签 is_illicit: Illicit={Blackmail, Darknet Market, Gambling, \
Gov Criminal Blacklist, Money Laundering, Ponzi Scheme, Tumbler}。

## 目标

- 生成能提升洗钱地址检测 Recall 的特征（Precision 需 ≥ 90%）
- 每个特征必须有明确的知识溯源（来自哪个案例/风险因子）
- **溯源要求**：必须在 knowledge_ids 中填写具体的知识资产ID（如 RF_S1_01_002），不要用笼统描述
- **计算步骤**：必须在 computation_steps 中逐步描述从原始字段到最终特征值的每一步变换
- **设计推理**：必须在 reasoning 中按计算步骤逐步解释，每一步先说明知识依据（因为），再说明计算操作（所以）。格式: 步骤N: 因为ID(含义)指出...，所以...
- 特征代码必须可执行，签名为 `def feature_xxx(df) -> pd.DataFrame`
- 输入 df 是 BABD-13 完整数据表（含 account, 149 个特征列, label, is_illicit）
- 输出 DataFrame 必须包含 `account` 列作为主键 + 一个特征值列

## 可用工具

1. **explore_data**: 在 DataFrame 上执行查询代码，探索数据分布（可用变量: df）
2. **generate_feature**: 提交特征代码 + 名称 + 维度 + 溯源信息
3. **execute_sandbox**: 在隔离沙盒中执行代码，触发三级验证

## 工作流程（严格遵守）

1. 阅读 BABD-13 Schema 和风险因子，理解可用字段和洗钱模式
2. 使用 explore_data 探索数据分布，验证假设
3. 设计特征（可以是现有特征的交叉、组合、变换），使用 generate_feature 提交
4. **立即**调用 execute_sandbox 验证该特征
5. 根据验证结果迭代改进，然后设计下一个特征

**重要：必须逐个生成和验证！每次 generate_feature 后必须立即调用 execute_sandbox，不要批量注册多个特征后再验证。只有通过验证的特征才计入预算。**
"""


def _build_role_generic(pk: "PublicDatasetKnowledge", id_col: str = "row_id") -> str:
    """构建公开数据集的通用角色模板。"""
    return f"""\
# 角色

你是一个{pk.domain}特征工程专家 Agent。你的任务是基于领域知识和数据分析，\
在 {pk.dataset_name} 数据集上设计能有效提升 {pk.task} 性能的统计特征。

## 数据集说明

{pk.dataset_name} 数据集，每行代表一个样本，目标变量为 `{pk.target}`。\
这是一个单表数据集，所有特征和标签在同一个 DataFrame 中。

## 目标

- 生成能提升分类性能（AUC / Accuracy）的特征
- 每个特征必须有明确的领域知识依据和可追溯的设计逻辑
- **溯源要求**：必须在 knowledge_ids 中填写具体的知识资产ID（如 RULE_001, IND_01, EXP_01），不要用笼统描述
- **计算步骤**：必须在 computation_steps 中逐步描述从原始字段到最终特征值的每一步变换
- **设计推理**：必须在 reasoning 中按计算步骤逐步解释，每一步先说明知识依据（因为），再说明计算操作（所以）。格式: 步骤N: 因为ID(含义)指出...，所以...
- 特征代码必须可执行，签名为 `def feature_xxx(df) -> pd.DataFrame`
- 输入 df 是完整数据表（含 {id_col}, 原始特征列, {pk.target}）
- 输出 DataFrame 必须包含 `{id_col}` 列作为主键 + 一个或多个特征值列

## 可用工具

1. **explore_data**: 在 DataFrame 上执行查询代码，探索数据分布（可用变量: df）
2. **generate_feature**: 提交特征代码 + 名称 + 维度 + 溯源信息
3. **execute_sandbox**: 在隔离沙盒中执行代码，触发三级验证

## 工作流程（严格遵守）

1. 阅读数据 Schema 和领域知识，理解可用字段和领域模式
2. 使用 explore_data 探索数据分布，验证假设
3. 设计特征（可以是现有特征的交叉、组合、变换），使用 generate_feature 提交
4. **立即**调用 execute_sandbox 验证该特征
5. 根据验证结果迭代改进，然后设计下一个特征

**重要：必须逐个生成和验证！每次 generate_feature 后必须立即调用 execute_sandbox，不要批量注册多个特征后再验证。只有通过验证的特征才计入预算。**
"""


# ── 验证规则说明 ──────────────────────────────────────────────────────────────

def _format_verification_rules(cfg: KnowFeatConfig) -> str:
    v = cfg.verification
    if cfg.stage == "generic":
        id_col = cfg.public_dataset.id_col
        id_text = f"输出 DataFrame 必须包含 `{id_col}` 列"
    else:
        id_text = "输出 DataFrame 必须包含 wallet_id 或 tx_id 列"
    return f"""\
## 验证规则（你的特征需通过以下三级验证）

### L1 沙盒执行
- 代码必须在 {v.sandbox_timeout_sec}s 内执行完毕
- {id_text}
- 执行失败可自动修复（最多 {v.sandbox_max_retries} 次）

### L2 统计质量
- IV > 动态阈值（噪声校准 p{v.noise_percentile:.0f}）
- KS > 动态阈值
- 冗余度 < {v.redundancy_max}（与已有特征的最大相关系数）
- 缺失率 < {v.missing_rate_max:.0%}

### L3 模型效果（XGBoost {v.cv_folds}-fold CV）
- Delta Recall > {v.delta_recall_min}（在已有特征基础上提升）
- Precision ≥ {v.precision_min:.0%}
"""


# ── 每轮任务指令 ──────────────────────────────────────────────────────────────

def build_round_instruction(
    round_num: int,
    dimension: str,
    budget: int,
    n_accepted: int = 0,
    n_candidates: int = 0,
    max_rounds: int = 10,
    max_features: int = 30,
) -> str:
    """构建每轮的任务指令，注入当前轮次上下文。

    Parameters
    ----------
    round_num : int
        当前轮次编号（从 1 开始）。
    dimension : str
        当前维度名称（如 "金额频次"）。
    budget : int
        本轮该维度的特征预算（可提交的特征数）。
    n_accepted : int
        已接受的特征总数。
    n_candidates : int
        候选池中的特征数。
    max_rounds : int
        最大轮次。
    max_features : int
        最大特征数。

    Returns
    -------
    str
        格式化的任务指令文本。
    """
    lines = [
        f"## 当前任务（第 {round_num}/{max_rounds} 轮）\n",
        f"- **当前维度**: {dimension}",
        f"- **本轮预算**: {budget} 个特征",
        f"- **已接受特征**: {n_accepted}/{max_features}",
        f"- **候选池特征**: {n_candidates}",
        "",
        f"请针对「{dimension}」维度，设计 {budget} 个有区分度的特征。",
        "优先关注以下方面：",
        f"1. 领域知识中与「{dimension}」相关的风险指标和检测规则",
        "2. 分布差异表中差异倍数最大的字段",
        "3. 避免与已有特征高度相关（冗余度 < 0.7）",
        "4. 参考失败记忆，不要重复相同的错误",
    ]

    if n_accepted >= max_features:
        lines.append(f"\n**注意**: 已达到特征上限 {max_features}，本轮仅用于替换低分特征。")

    return "\n".join(lines)


# ── 主组装函数 ────────────────────────────────────────────────────────────────

def build_system_prompt(
    cfg: KnowFeatConfig,
    domain_knowledge: DomainKnowledge,
    data: Optional[Stage2Data | Stage1Data] = None,
    schema_text: Optional[str] = None,
    shap_feedback: Optional[ShapFeedback] = None,
    failure_records: Optional[list[FailureRecord]] = None,
    cross_dim_examples: Optional[list[CrossDimExample]] = None,
    knowledge_filter: Optional[str] = None,
) -> str:
    """组装完整的 System Prompt。

    Parameters
    ----------
    cfg : KnowFeatConfig
        全局配置。
    domain_knowledge : DomainKnowledge
        领域知识容器（由 load_domain_knowledge 加载）。
    data : Stage2Data | Stage1Data, optional
        数据容器。如提供则自动生成 schema_text。
    schema_text : str, optional
        预生成的 Schema 文本。如不提供且 data 不为 None，则自动生成。
    shap_feedback : ShapFeedback, optional
        SHAP 维度贡献反馈（动态，每轮更新）。
    failure_records : list[FailureRecord], optional
        失败记忆列表（动态，累积注入）。
    cross_dim_examples : list[CrossDimExample], optional
        跨维度高分特征示例（动态，每轮更新）。

    Returns
    -------
    str
        完整的 System Prompt 文本。
    """
    stage = cfg.stage
    sections: list[str] = []

    # 1. 角色定义
    if stage == "stage2":
        sections.append(_ROLE_STAGE2)
    elif stage == "generic":
        pk = domain_knowledge.public_knowledge
        if pk is not None:
            sections.append(_build_role_generic(pk, cfg.public_dataset.id_col))
        else:
            sections.append(_ROLE_STAGE1)
    else:
        sections.append(_ROLE_STAGE1)

    # 2. 领域知识（静态，可按 knowledge_filter 过滤用于消融实验）
    sections.append(format_domain_knowledge(
        domain_knowledge, stage=stage, knowledge_filter=knowledge_filter,
    ))

    # 3. 数据 Schema + 分布差异（静态）
    if schema_text is not None:
        sections.append(schema_text)
    elif data is not None:
        if isinstance(data, Stage1Data):
            table_name = (
                cfg.public_dataset.dataset_name
                if stage == "generic" and cfg.public_dataset.dataset_name
                else "BABD-13"
            )
            sections.append(generate_all_schema_text_stage1(data, table_name=table_name))
        else:
            sections.append(generate_all_schema_text(data))

    # 4. 验证规则
    sections.append(_format_verification_rules(cfg))

    # 5. SHAP 反馈（动态）
    if shap_feedback is not None:
        sections.append(_format_shap_feedback(shap_feedback))

    # 6. 失败记忆（动态）
    if failure_records:
        sections.append(_format_failure_memory(failure_records))

    # 7. 跨维度高分示例（动态）
    if cross_dim_examples:
        sections.append(_format_cross_dim_examples(cross_dim_examples))

    prompt = "\n\n---\n\n".join(s for s in sections if s.strip())
    logger.info(
        "System prompt assembled: %d chars, %d sections (stage=%s)",
        len(prompt), len(sections), stage,
    )
    return prompt


def build_full_prompt(
    cfg: KnowFeatConfig,
    domain_knowledge: DomainKnowledge,
    data: Optional[Stage2Data] = None,
    schema_text: Optional[str] = None,
    shap_feedback: Optional[ShapFeedback] = None,
    failure_records: Optional[list[FailureRecord]] = None,
    cross_dim_examples: Optional[list[CrossDimExample]] = None,
    round_num: int = 1,
    dimension: str = "",
    budget: int = 5,
    n_accepted: int = 0,
    n_candidates: int = 0,
    knowledge_filter: Optional[str] = None,
) -> tuple[str, str]:
    """组装 system prompt + 当前轮次的 user message。

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_message)
    """
    system = build_system_prompt(
        cfg=cfg,
        domain_knowledge=domain_knowledge,
        data=data,
        schema_text=schema_text,
        shap_feedback=shap_feedback,
        failure_records=failure_records,
        cross_dim_examples=cross_dim_examples,
        knowledge_filter=knowledge_filter,
    )

    user = build_round_instruction(
        round_num=round_num,
        dimension=dimension,
        budget=budget,
        n_accepted=n_accepted,
        n_candidates=n_candidates,
        max_rounds=cfg.search.max_rounds,
        max_features=cfg.search.max_features,
    )

    return system, user
