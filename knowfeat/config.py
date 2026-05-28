"""KnowFeat 全局配置"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    api_key: str = os.environ.get("KNOWFEAT_API_KEY", "")
    base_url: str = os.environ.get("KNOWFEAT_BASE_URL", "https://api.deepseek.com/v1")
    model: str = os.environ.get("KNOWFEAT_MODEL", "deepseek-chat")
    temperature: float = 0.4
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3


# ---------------------------------------------------------------------------
# 文件路径
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    # Stage 2 数据
    accounts_csv: Path = PROJECT_ROOT / "data" / "accounts.csv"
    transactions_csv: Path = PROJECT_ROOT / "data" / "transactions.csv"
    close_csv: Path = PROJECT_ROOT / "data" / "close.csv"

    # 基线特征
    baseline_cache: Path = PROJECT_ROOT / "outputs" / "baseline_cache.parquet"
    baseline_feature_engineering: Path = PROJECT_ROOT / "inputs" / "baseline" / "feature_engineering.py"

    # 领域知识
    domain_knowledge_dir: Path = PROJECT_ROOT / "inputs" / "domain_knowledge"
    risk_indicators_xlsx: Path = PROJECT_ROOT / "inputs" / "domain_knowledge" / "风险分类表_38指标.xlsx"
    detection_rules_xlsx: Path = PROJECT_ROOT / "inputs" / "domain_knowledge" / "检测规则_15条.xlsx"
    expert_interview_docx: Path = PROJECT_ROOT / "inputs" / "domain_knowledge" / "邮储专家访谈_数币洗钱模式.docx"

    # Stage 1 知识
    stage1_knowledge_dir: Path = PROJECT_ROOT / "inputs" / "stage1_knowledge"
    stage1_risk_factors_dir: Path = PROJECT_ROOT / "inputs" / "stage1_knowledge" / "risk_factors"

    # Stage 2 裁判文书提取结果
    stage2_risk_factors_dir: Path = PROJECT_ROOT / "inputs" / "stage2_knowledge" / "risk_factors"

    # 裁判文书
    court_docs_dir: Path = PROJECT_ROOT / "裁判文书docx"

    # SimECNY 原始数据
    simecny_dir: Path = PROJECT_ROOT / "inputs" / "SimECNY"

    # 输出
    output_dir: Path = PROJECT_ROOT / "outputs"

    # 日志
    log_dir: Path = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# 验证阈值
# ---------------------------------------------------------------------------
@dataclass
class VerificationConfig:
    # L1 沙盒
    sandbox_timeout_sec: int = 30
    sandbox_max_retries: int = 2

    # L2 统计验证 — 噪声校准
    noise_n_random_features: int = 100
    noise_percentile: float = 95.0
    iv_min_fallback: float = 0.02
    ks_min_fallback: float = 0.02
    redundancy_max: float = 0.7
    missing_rate_max: float = 0.3

    # L3 模型效果
    cv_folds: int = 5
    precision_min: float = 0.90
    delta_recall_min: float = 0.0


# ---------------------------------------------------------------------------
# 搜索策略参数
# ---------------------------------------------------------------------------
@dataclass
class SearchConfig:
    max_rounds: int = 10
    max_features: int = 30
    convergence_rounds: int = 2

    # 四维度
    dimensions: list[str] = field(default_factory=lambda: [
        "金额频次", "时间地域", "行为序列", "主体关联",
    ])
    initial_budget_per_dim: int = 3

    # 失败记忆
    failure_memory_inject_n: int = 10

    # 跨维度传播
    cross_dim_top_k: int = 5

    # 特征复合打分权重
    score_weight_shap: float = 0.30
    score_weight_iv: float = 0.20
    score_weight_novelty: float = 0.20
    score_weight_delta_recall: float = 0.30


# ---------------------------------------------------------------------------
# 顶层配置
# ---------------------------------------------------------------------------
@dataclass
class KnowFeatConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    stage: str = "stage2"
    seed: int = 42
    verbose: bool = True


def load_config(stage: str = "stage2", **overrides) -> KnowFeatConfig:
    """创建配置实例，支持关键字覆盖。"""
    cfg = KnowFeatConfig(stage=stage)

    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        elif hasattr(cfg.llm, key):
            setattr(cfg.llm, key, value)
        elif hasattr(cfg.verification, key):
            setattr(cfg.verification, key, value)
        elif hasattr(cfg.search, key):
            setattr(cfg.search, key, value)

    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.log_dir.mkdir(parents=True, exist_ok=True)

    return cfg
