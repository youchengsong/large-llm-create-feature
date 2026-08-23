"""KnowFeat global configuration."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# LLM configuration
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
# File paths
# ---------------------------------------------------------------------------
@dataclass
class PathConfig:
    # Stage 2 data (SimECNY)
    accounts_csv: Path = PROJECT_ROOT / "data" / "accounts.csv"
    transactions_csv: Path = PROJECT_ROOT / "data" / "transactions.csv"
    close_csv: Path = PROJECT_ROOT / "data" / "close.csv"

    # Stage 1 data (BABD-13)
    babd13_csv: Path = Path("/root/autodl-tmp/archive/BABD-13.csv")

    # Baseline features
    baseline_cache: Path = PROJECT_ROOT / "outputs" / "baseline_cache.parquet"
    baseline_feature_engineering: Path = PROJECT_ROOT / "inputs" / "baseline" / "feature_engineering.py"

    # Domain knowledge
    domain_knowledge_dir: Path = PROJECT_ROOT / "inputs" / "domain_knowledge"
    risk_indicators_xlsx: Path = PROJECT_ROOT / "inputs" / "domain_knowledge" / "风险分类表_38指标.xlsx"
    detection_rules_xlsx: Path = PROJECT_ROOT / "inputs" / "domain_knowledge" / "检测规则_15条.xlsx"
    expert_interview_docx: Path = PROJECT_ROOT / "inputs" / "domain_knowledge" / "邮储专家访谈_数币洗钱模式.docx"

    # Stage 1 knowledge
    stage1_knowledge_dir: Path = PROJECT_ROOT / "inputs" / "stage1_knowledge"
    stage1_risk_factors_dir: Path = PROJECT_ROOT / "inputs" / "stage1_knowledge" / "risk_factors"

    # Stage 2 court document extraction results
    stage2_risk_factors_dir: Path = PROJECT_ROOT / "inputs" / "stage2_knowledge" / "risk_factors"

    # Court documents
    court_docs_dir: Path = PROJECT_ROOT / "裁判文书docx"

    # SimECNY raw data
    simecny_dir: Path = PROJECT_ROOT / "inputs" / "SimECNY"

    # Output
    output_dir: Path = PROJECT_ROOT / "outputs"

    # Logs
    log_dir: Path = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Verification thresholds
# ---------------------------------------------------------------------------
@dataclass
class VerificationConfig:
    # L1 sandbox
    sandbox_timeout_sec: int = 30
    sandbox_max_retries: int = 2

    # L2 statistical verification — noise calibration
    noise_n_random_features: int = 100
    noise_percentile: float = 95.0
    noise_multiplier: float = 3.0
    iv_min_floor: float = 0.005
    ks_min_floor: float = 0.005
    # Backward-compatible names used by older configs and integrations.  When
    # supplied, these aliases are normalized to the corresponding ``*_floor``.
    iv_min_fallback: Optional[float] = None
    ks_min_fallback: Optional[float] = None
    redundancy_max: float = 0.7
    missing_rate_max: float = 0.3

    # L3 model performance
    cv_folds: int = 5
    precision_min: float = 0.90
    delta_recall_min: float = 0.0
    accept_metric: str = "recall"  # "recall" (AML) or "auc" (public datasets)
    delta_auc_min: float = 0.0
    spw_override: Optional[float] = None  # None=auto, 1.0=no balancing

    def __post_init__(self) -> None:
        if self.iv_min_fallback is not None:
            self.iv_min_floor = self.iv_min_fallback
        if self.ks_min_fallback is not None:
            self.ks_min_floor = self.ks_min_fallback


# ---------------------------------------------------------------------------
# Search strategy parameters
# ---------------------------------------------------------------------------
@dataclass
class SearchConfig:
    max_rounds: int = 10
    max_features: int = 9999
    convergence_rounds: int = 2
    # Post-hoc pruning is expensive and changes the selected feature set, so
    # it must be explicitly enabled by experiment configs.
    enable_posthoc_selection: bool = False

    # Four dimensions (functional — used in prompts, do not translate)
    dimensions: list[str] = field(default_factory=lambda: [
        "金额频次", "时间地域", "行为序列", "主体关联",
    ])
    initial_budget_per_dim: int = 3

    # Failure memory
    failure_memory_inject_n: int = 10

    # Cross-dimension propagation
    cross_dim_top_k: int = 5

    # Feature composite scoring weights
    score_weight_shap: float = 0.30
    score_weight_iv: float = 0.20
    score_weight_novelty: float = 0.20
    score_weight_delta_recall: float = 0.30


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------
@dataclass
class PublicDatasetConfig:
    """Public dataset configuration (used when stage='generic')."""
    dataset_name: str = ""
    csv_path: Path = Path("")
    target_col: str = ""
    id_col: str = "row_id"
    domain: str = ""
    task: str = ""
    domain_knowledge_path: Path = Path("")


@dataclass
class KnowFeatConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    public_dataset: PublicDatasetConfig = field(default_factory=PublicDatasetConfig)
    stage: str = "stage2"  # "stage1", "stage2", or "generic"
    seed: int = 42
    verbose: bool = True


def load_config(stage: str = "stage2", **overrides) -> KnowFeatConfig:
    """Create a configuration instance with keyword overrides."""
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

    # Apply aliases supplied through ``load_config`` after field overrides.
    cfg.verification.__post_init__()

    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.log_dir.mkdir(parents=True, exist_ok=True)

    return cfg
