"""领域知识加载：风险指标 / 检测规则 / 交易模式 / 专家意见。

供 prompt/builder.py 组装 System Prompt 时注入。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from knowfeat.config import KnowFeatConfig, PathConfig

logger = logging.getLogger(__name__)


# ── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class RiskIndicator:
    """单条风险指标。"""
    index: int
    dimension: str          # 维度（如 "交易行为异常类风险"）
    risk_class: str         # 风险类（如 "金额与频次异常风险"）
    data_field: str         # 数据类
    direction: str          # 方向（风险描述）


@dataclass
class DetectionRule:
    """单条检测规则。"""
    rule_id: str            # RULE_001 etc.
    description: str        # 自然语言描述
    logic_expr: str         # 一阶逻辑表达式
    confidence: str         # 置信度
    rule_type: str          # behavior / composite / attribute
    pattern_category: str   # 异常模式类别
    python_hint: str = ""   # 自动生成的 Python 伪代码提示


@dataclass
class TransactionPattern:
    """交易模式。"""
    name: str
    category: str           # "normal" 或 "abnormal"
    description: str
    signals: list[str] = field(default_factory=list)


@dataclass
class DomainKnowledge:
    """Stage 2 全部领域知识容器。"""
    risk_indicators: list[RiskIndicator] = field(default_factory=list)
    detection_rules: list[DetectionRule] = field(default_factory=list)
    transaction_patterns: list[TransactionPattern] = field(default_factory=list)
    expert_opinion: str = ""

    # Stage 1 风险因子（JSON 文件列表）
    stage1_risk_factors: list[dict] = field(default_factory=list)

    # Stage 2 裁判文书提取的风险因子
    court_doc_risk_factors: list[dict] = field(default_factory=list)


# ── 1. 风险分类表 38 指标 ──────────────────────────────────────────────────

_DIMENSION_MAP = {
    "交易行为异常类风险": "交易行为异常类风险",
    "主体与关联类风险": "主体与关联类风险",
    "合规与场景类风险\n（深入分析）": "合规与场景类风险",
}


def load_risk_indicators(xlsx_path: Path) -> list[RiskIndicator]:
    """解析 风险分类表_38指标.xlsx → RiskIndicator 列表。

    Excel 结构：合并单元格，第 0 行是表头，1-38 行是数据，39 行是备注。
    列：序号 | 维度 | 风险类 | 数据类 | 方向
    """
    if not xlsx_path.exists():
        logger.warning("风险分类表不存在: %s", xlsx_path)
        return []

    df = pd.read_excel(xlsx_path, header=None, skiprows=1)
    df.columns = ["序号", "维度", "风险类", "数据类", "方向"]

    indicators: list[RiskIndicator] = []
    current_dim = ""
    current_class = ""

    for _, row in df.iterrows():
        idx = row["序号"]
        if pd.isna(idx) or not str(idx).strip().isdigit():
            continue
        idx = int(idx)

        if pd.notna(row["维度"]):
            raw_dim = str(row["维度"]).strip()
            current_dim = _DIMENSION_MAP.get(raw_dim, raw_dim)
        if pd.notna(row["风险类"]):
            current_class = str(row["风险类"]).strip()

        data_field = str(row["数据类"]).strip() if pd.notna(row["数据类"]) else ""
        direction = str(row["方向"]).strip() if pd.notna(row["方向"]) else ""

        indicators.append(RiskIndicator(
            index=idx,
            dimension=current_dim,
            risk_class=current_class,
            data_field=data_field,
            direction=direction,
        ))

    logger.info("Loaded %d risk indicators from %s", len(indicators), xlsx_path.name)
    return indicators


def format_risk_indicators(indicators: list[RiskIndicator]) -> str:
    """将风险指标格式化为 Markdown 文本，按维度 > 风险类分组。"""
    if not indicators:
        return "## 风险指标: 未加载\n"

    groups: dict[str, dict[str, list[RiskIndicator]]] = {}
    for ind in indicators:
        groups.setdefault(ind.dimension, {}).setdefault(ind.risk_class, []).append(ind)

    lines = [f"## 数字人民币交易风险指标（{len(indicators)} 项）\n"]
    for dim, classes in groups.items():
        lines.append(f"### {dim}")
        for cls, items in classes.items():
            lines.append(f"\n**{cls}**")
            for it in items:
                parts = [f"- [{it.index}] {it.data_field}"]
                if it.direction:
                    parts.append(f"  → {it.direction}")
                lines.append("".join(parts))
        lines.append("")

    return "\n".join(lines)


# ── 2. 检测规则 15 条 ──────────────────────────────────────────────────────

_RULE_PYTHON_HINTS: dict[str, str] = {
    "RULE_001": (
        "# 四类匿名钱包短时间大额取现\n"
        "wallets_4 = open_df[open_df['wallet_level'] == 4]\n"
        "atm_tx = tx_df[tx_df['transaction_mode'].isin(['ATM取现'])]\n"
        "# 按 src 聚合: 2h 窗口内 wallet 数 >= 10, 总金额 >= 100000"
    ),
    "RULE_002": (
        "# 休眠账户激活后大额交易\n"
        "first_tx = tx_df.groupby('src')['timestamp'].min()\n"
        "open_time = open_df.set_index('wallet_id')['wallet_open_timestamp']\n"
        "sleep_days = (first_tx - open_time).dt.days\n"
        "# sleep_days >= 90 且后续 tx_frequency >= 20/day"
    ),
    "RULE_003": (
        "# 资金快进快出（余额留存率低）\n"
        "in_amt = tx_df.groupby('dst')['amount'].sum()\n"
        "out_amt = tx_df.groupby('src')['amount'].sum()\n"
        "retention = 1 - out_amt / in_amt\n"
        "# retention <= 0.1 表示资金快速转出"
    ),
    "RULE_004": (
        "# 商户账户异常大额交易\n"
        "merchant = open_df[open_df['wallet_type'] == 3]['wallet_id']\n"
        "m_tx = tx_df[tx_df['dst'].isin(merchant)]\n"
        "# 1h 窗口内单商户收款 >= 100000"
    ),
    "RULE_005": (
        "# 跨地区流动交易\n"
        "# 按 src 统计交易涉及的不同 region_code 数量\n"
        "region_count = tx_df.merge(open_df[['wallet_id','region_code']], "
        "left_on='src', right_on='wallet_id').\\\n"
        "    groupby('src')['region_code'].nunique()\n"
        "# region_count >= 3"
    ),
    "RULE_006": (
        "# 分散转入集中转出\n"
        "daily = tx_df.groupby(['src', tx_df['timestamp'].dt.date]).agg(\n"
        "    count=('amount','count'), total=('amount','sum'))\n"
        "# count >= 20 and total >= 1000000\n"
        "# 入金笔数多、出金笔数少"
    ),
    "RULE_007": (
        "# 混合多种业务规避监管\n"
        "mode_count = tx_df.groupby('src')['transaction_mode'].nunique()\n"
        "# mode_count >= 3 且单笔金额刻意低于报告标准"
    ),
    "RULE_008": (
        "# 诈骗资金快速转移\n"
        "# 收到资金后 1h 内转出，余额不留\n"
        "# 按 wallet 排序交易时间，计算收入-支出间隔"
    ),
    "RULE_009": (
        "# 批量开户\n"
        "daily_open = open_df.groupby(\n"
        "    open_df['wallet_open_timestamp'].dt.date).size()\n"
        "# 同一天同地点 >= 5 个新开户"
    ),
    "RULE_010": (
        "# 与收入不符的大额交易\n"
        "# 交易金额 >= 收入 * 10（需要外部收入数据）"
    ),
    "RULE_011": (
        "# ATM 为主的交易，金额为特定倍数\n"
        "atm_ratio = tx_df.groupby('src').apply(\n"
        "    lambda g: (g['transaction_mode']=='ATM').mean())\n"
        "# atm_ratio 高且夜间交易占比高"
    ),
    "RULE_012": (
        "# 商户 POS 套现：整数金额占比高\n"
        "round_ratio = tx_df.groupby('dst').apply(\n"
        "    lambda g: (g['amount'] % 1000 == 0).mean())\n"
        "# round_ratio >= 0.8"
    ),
    "RULE_013": (
        "# 金字塔型资金网络\n"
        "# 周期性返利交易，时间间隔固定\n"
        "# 分析交易时间序列的周期性"
    ),
    "RULE_014": (
        "# 高风险地区小额频繁交易\n"
        "freq = tx_df.groupby('src')['amount'].agg(['count','mean'])\n"
        "# count 高 and mean 低 and 现金交易占比 >= 0.8"
    ),
    "RULE_015": (
        "# 网络赌博：分散转入分散转出，100 元整数倍\n"
        "multiple_100 = tx_df.groupby('src').apply(\n"
        "    lambda g: (g['amount'] % 100 == 0).mean())\n"
        "# multiple_100 >= 0.7 且深夜交易占比 >= 0.5"
    ),
}


def load_detection_rules(xlsx_path: Path) -> list[DetectionRule]:
    """解析 检测规则_15条.xlsx → DetectionRule 列表。"""
    if not xlsx_path.exists():
        logger.warning("检测规则表不存在: %s", xlsx_path)
        return []

    df = pd.read_excel(xlsx_path)
    rules: list[DetectionRule] = []

    for _, row in df.iterrows():
        rule_id = row.get("规则编号")
        if pd.isna(rule_id) or not str(rule_id).startswith("RULE_"):
            continue

        rule_id = str(rule_id).strip()
        rules.append(DetectionRule(
            rule_id=rule_id,
            description=str(row.get("自然语言描述", "")).strip(),
            logic_expr=str(row.get("一阶逻辑表达式", "")).strip(),
            confidence=str(row.get("置信度", "")).strip(),
            rule_type=str(row.get("规则类型", "")).strip(),
            pattern_category=str(row.get("异常模式类别", "")).strip(),
            python_hint=_RULE_PYTHON_HINTS.get(rule_id, ""),
        ))

    logger.info("Loaded %d detection rules from %s", len(rules), xlsx_path.name)
    return rules


def format_detection_rules(rules: list[DetectionRule]) -> str:
    """将检测规则格式化为 Markdown 文本。"""
    if not rules:
        return "## 检测规则: 未加载\n"

    lines = [f"## 检测规则（{len(rules)} 条）\n"]
    for r in rules:
        lines.append(f"### {r.rule_id} [{r.pattern_category}] ({r.rule_type})")
        lines.append(f"**描述**: {r.description}\n")
        lines.append(f"**逻辑**: `{r.logic_expr}`\n")
        if r.python_hint:
            lines.append("**Python 示例**:")
            lines.append(f"```python\n{r.python_hint}\n```\n")

    return "\n".join(lines)


# ── 3. 交易模式（8 种硬编码）────────────────────────────────────────────────

_TRANSACTION_PATTERNS: list[TransactionPattern] = [
    # ── 正常模式 (3) ──
    TransactionPattern(
        name="日常消费",
        category="normal",
        description="个人钱包在商户进行日常消费（餐饮、购物、交通），金额小、频率适中，主要在白天营业时段。",
        signals=["单笔 < 1000 元", "白天时段 (8-22 时)", "收款方为商户钱包"],
    ),
    TransactionPattern(
        name="工资发放与定期转账",
        category="normal",
        description="企业钱包定期向个人钱包发放工资或补贴，金额固定、时间规律（月初/月中），一对多转账。",
        signals=["企业→个人", "月周期", "金额相对固定", "一对多"],
    ),
    TransactionPattern(
        name="商户正常收款",
        category="normal",
        description="商户钱包正常营业收款，交易量与经营规模匹配，客户来源分散，营业时段内完成。",
        signals=["多对一", "营业时段", "交易量与历史均值匹配"],
    ),

    # ── 异常模式 (5) ──
    TransactionPattern(
        name="薅羊毛（批量开户套利）",
        category="abnormal",
        description=(
            "犯罪分子批量开立四类匿名钱包，套取运营机构的营销红包和消费券。"
            "特征：同一设备/IP/手机号段开立大量钱包，短时间内集中消费。"
        ),
        signals=[
            "同一设备开立 ≥ 3 个钱包",
            "同一手机号段短时间大量消费",
            "同一 IP 大量钱包短时间消费",
            "短期内大量钱包在同一商户消费",
        ],
    ),
    TransactionPattern(
        name="线下跑分（ATM 取现）",
        category="abnormal",
        description=(
            "不法分子开立批量四类钱包，通过 ATM 取现转移赃款。"
            "上游收到虚拟货币后将钱转入数币钱包，下游用账号密码在 ATM 取现。"
            "单次取现约 20 万元，钱包存活时间短，开立后长期休眠再突然激活。"
        ),
        signals=[
            "四类钱包 ATM 取现",
            "休眠→突然激活",
            "取现金额整数",
            "单日多次取现",
            "凌晨时段操作",
        ],
    ),
    TransactionPattern(
        name="线上跑分（钱包互转）",
        category="abnormal",
        description=(
            "匿名钱包→实名钱包→银行账户的链式转账，常见凌晨 12 点后操作。"
            "跑分金额 300-5000 元，持续 10-30 天，每日交易量达钱包限额上限。"
        ),
        signals=[
            "凌晨密集转账",
            "匿名→实名→银行账户链",
            "单笔 300-5000 元",
            "日交易量达限额",
            "持续 10-30 天",
        ],
    ),
    TransactionPattern(
        name="商户收款码跑分",
        category="abnormal",
        description=(
            "与商户勾结，利用收款码进行虚假消费洗钱，给商户约 8% 佣金。"
            "24 小时不间断交易，多为整数金额（5000、10000 元），"
            "单商户交易金额突增至日百万元。"
        ),
        signals=[
            "24 小时不间断",
            "整数金额（5k/1w）",
            "商户交易金额突增",
            "收款达商户上限",
            "交易金额与经营规模不符",
        ],
    ),
    TransactionPattern(
        name="贵金属购买洗钱",
        category="abnormal",
        description=(
            "通过数字人民币购买贵金属（如黄金），将赃款转化为高价值易变现实物，"
            "再邮寄给洗钱下游。形成'数币-贵金属-现金'洗钱闭环。"
        ),
        signals=[
            "大额购买贵金属",
            "交易对手为贵金属商户",
            "购买后无后续正常消费",
        ],
    ),
]


def get_transaction_patterns() -> list[TransactionPattern]:
    """返回 8 种交易模式（3 正常 + 5 异常）。"""
    return list(_TRANSACTION_PATTERNS)


def format_transaction_patterns(patterns: list[TransactionPattern]) -> str:
    """将交易模式格式化为 Markdown 文本。"""
    normal = [p for p in patterns if p.category == "normal"]
    abnormal = [p for p in patterns if p.category == "abnormal"]

    lines = [f"## 交易模式知识（{len(patterns)} 种）\n"]

    lines.append(f"### 正常交易模式（{len(normal)} 种）\n")
    for p in normal:
        lines.append(f"**{p.name}**: {p.description}")
        if p.signals:
            lines.append("- 信号: " + " | ".join(p.signals))
        lines.append("")

    lines.append(f"### 异常交易模式（{len(abnormal)} 种）\n")
    for p in abnormal:
        lines.append(f"**{p.name}**: {p.description}")
        if p.signals:
            lines.append("- 信号: " + " | ".join(p.signals))
        lines.append("")

    return "\n".join(lines)


# ── 4. 专家意见 ────────────────────────────────────────────────────────────

def load_expert_opinion(docx_path: Path) -> str:
    """加载邮储专家访谈 docx 文本。"""
    if not docx_path.exists():
        logger.warning("专家访谈文件不存在: %s", docx_path)
        return ""

    try:
        import docx
    except ImportError:
        logger.warning("python-docx 未安装，跳过专家意见加载")
        return ""

    doc = docx.Document(str(docx_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    logger.info("Loaded expert opinion: %d chars from %s", len(text), docx_path.name)
    return text


def format_expert_opinion(text: str) -> str:
    """将专家意见格式化为 Markdown 段落。"""
    if not text:
        return "## 专家意见: 未加载\n"
    return f"## 银行风控专家意见（邮储银行数币洗钱模式）\n\n{text}\n"


# ── 5. Stage 1 风险因子 ───────────────────────────────────────────────────

def load_stage1_risk_factors(risk_factors_dir: Path) -> list[dict]:
    """加载 Stage 1 Key-clue Extractor 输出的 JSON 风险因子文件。

    每个 JSON 文件格式:
    {
      "source": "...",
      "risk_factors": [
        {"pattern": "...", "dimension": [...], "observable_signals": [...], "data_hint": "..."}
      ]
    }
    """
    if not risk_factors_dir.exists():
        logger.warning("Stage 1 risk factors 目录不存在: %s", risk_factors_dir)
        return []

    factors: list[dict] = []
    for fp in sorted(risk_factors_dir.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            factors.append(data)
            logger.info("Loaded stage1 risk factor: %s (%d factors)",
                        fp.name, len(data.get("risk_factors", [])))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", fp.name, e)

    logger.info("Total stage1 risk factor files: %d", len(factors))
    return factors


def format_stage1_risk_factors(factors: list[dict]) -> str:
    """将 Stage 1 风险因子格式化为 Markdown 文本。"""
    if not factors:
        return "## Stage 1 风险因子: 未加载\n"

    lines = ["## 加密货币洗钱风险因子（Key-clue Extractor 输出）\n"]
    for doc in factors:
        source = doc.get("source", "未知来源")
        lines.append(f"### 来源: {source}\n")
        for rf in doc.get("risk_factors", []):
            pattern = rf.get("pattern", "")
            dims = rf.get("dimension", [])
            signals = rf.get("observable_signals", [])
            hint = rf.get("data_hint", "")

            lines.append(f"- **模式**: {pattern}")
            if dims:
                lines.append(f"  - 维度: {', '.join(dims)}")
            if signals:
                lines.append(f"  - 信号: {' | '.join(signals)}")
            if hint:
                lines.append(f"  - 数据提示: {hint}")
        lines.append("")

    return "\n".join(lines)


def format_court_doc_risk_factors(factors: list[dict]) -> str:
    """将裁判文书风险因子按维度聚合去重，生成压缩摘要。

    309 份文书 × ~6 个因子 = ~1920 条，全部展开约 40 万字符。
    按维度分组 → 信号去重 → 模式聚类，压缩至几千字。
    """
    if not factors:
        return "## 裁判文书风险因子: 未加载\n"

    total_rf = sum(len(doc.get("risk_factors", [])) for doc in factors)

    dim_patterns: dict[str, list[str]] = {}
    dim_signals: dict[str, set[str]] = {}
    dim_hints: dict[str, set[str]] = {}

    for doc in factors:
        for rf in doc.get("risk_factors", []):
            pattern = rf.get("pattern", "").strip()
            dims = rf.get("dimension", [])
            signals = rf.get("observable_signals", [])
            hint = rf.get("data_hint", "").strip()

            if not dims:
                dims = ["其他"]
            for dim in dims:
                dim_patterns.setdefault(dim, []).append(pattern)
                dim_signals.setdefault(dim, set()).update(
                    s.strip() for s in signals if s.strip()
                )
                if hint:
                    for h in hint.split(","):
                        h = h.strip()
                        if h:
                            dim_hints.setdefault(dim, set()).add(h)

    def _cluster_patterns(patterns: list[str], max_clusters: int = 15) -> list[tuple[str, int]]:
        """按前 20 字符粗聚类，每簇取最短的作为代表。"""
        from collections import defaultdict
        buckets: dict[str, list[str]] = defaultdict(list)
        for p in patterns:
            key = p[:20]
            buckets[key].append(p)
        clusters = []
        for _key, members in buckets.items():
            rep = min(members, key=len)
            clusters.append((rep, len(members)))
        clusters.sort(key=lambda x: -x[1])
        return clusters[:max_clusters]

    dim_order = ["金额频次", "时间地域", "行为序列", "主体关联"]
    lines = [
        f"## 裁判文书洗钱风险因子摘要"
        f"（{len(factors)} 份文书, {total_rf} 个因子, 按维度聚合去重）\n"
    ]

    for dim in dim_order:
        if dim not in dim_patterns:
            continue
        pats = dim_patterns[dim]
        sigs = dim_signals.get(dim, set())
        hints = dim_hints.get(dim, set())

        lines.append(f"### {dim}（{len(pats)} 个因子）\n")

        clusters = _cluster_patterns(pats)
        lines.append("**典型模式**:")
        for rep, cnt in clusters:
            lines.append(f"- {rep}（{cnt} 案）")

        if sigs:
            sig_list = sorted(sigs)
            lines.append(f"\n**可观测信号**（{len(sig_list)} 个去重）:")
            for s in sig_list[:30]:
                lines.append(f"- {s}")
            if len(sig_list) > 30:
                lines.append(f"- ...（另有 {len(sig_list) - 30} 个信号）")

        if hints:
            lines.append(f"\n**相关数据字段**: {', '.join(sorted(hints)[:20])}")

        lines.append("")

    remaining = set(dim_patterns.keys()) - set(dim_order)
    for dim in sorted(remaining):
        pats = dim_patterns[dim]
        lines.append(f"### {dim}（{len(pats)} 个因子）\n")
        clusters = _cluster_patterns(pats, max_clusters=5)
        for rep, cnt in clusters:
            lines.append(f"- {rep}（{cnt} 案）")
        lines.append("")

    return "\n".join(lines)


# ── 6. 顶层加载 ──────────────────────────────────────────────────────────

def load_domain_knowledge(cfg: KnowFeatConfig) -> DomainKnowledge:
    """加载全部领域知识。

    Parameters
    ----------
    cfg : KnowFeatConfig
        全局配置（包含文件路径）。

    Returns
    -------
    DomainKnowledge
        领域知识容器。
    """
    paths = cfg.paths

    dk = DomainKnowledge(
        risk_indicators=load_risk_indicators(paths.risk_indicators_xlsx),
        detection_rules=load_detection_rules(paths.detection_rules_xlsx),
        transaction_patterns=get_transaction_patterns(),
        expert_opinion=load_expert_opinion(paths.expert_interview_docx),
    )

    if cfg.stage == "stage1":
        dk.stage1_risk_factors = load_stage1_risk_factors(paths.stage1_risk_factors_dir)

    if cfg.stage == "stage2":
        dk.court_doc_risk_factors = load_stage1_risk_factors(paths.stage2_risk_factors_dir)

    logger.info(
        "Domain knowledge loaded: %d indicators, %d rules, %d patterns, "
        "expert=%d chars, stage1_factors=%d files, court_doc_factors=%d files",
        len(dk.risk_indicators), len(dk.detection_rules),
        len(dk.transaction_patterns), len(dk.expert_opinion),
        len(dk.stage1_risk_factors), len(dk.court_doc_risk_factors),
    )
    return dk


def format_domain_knowledge(dk: DomainKnowledge, stage: str = "stage2") -> str:
    """将 DomainKnowledge 格式化为完整 Markdown 文本，供 prompt 注入。

    Parameters
    ----------
    dk : DomainKnowledge
        领域知识容器。
    stage : str
        "stage1" 或 "stage2"，控制输出内容。

    Returns
    -------
    str
        格式化的领域知识 Markdown 文本。
    """
    sections: list[str] = ["# 领域知识\n"]

    if stage == "stage2":
        sections.append(format_risk_indicators(dk.risk_indicators))
        sections.append(format_detection_rules(dk.detection_rules))
        sections.append(format_transaction_patterns(dk.transaction_patterns))
        sections.append(format_expert_opinion(dk.expert_opinion))
        if dk.court_doc_risk_factors:
            sections.append(format_court_doc_risk_factors(dk.court_doc_risk_factors))
    elif stage == "stage1":
        sections.append(format_stage1_risk_factors(dk.stage1_risk_factors))

    text = "\n\n".join(sections)
    logger.info("Formatted domain knowledge: %d chars (stage=%s)", len(text), stage)
    return text
