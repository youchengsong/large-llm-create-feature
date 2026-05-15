import os
from dataclasses import dataclass, field


@dataclass
class IslandConfig:
    name: str
    dimension: str
    description: str
    fields: list[str]


ISLANDS = [
    IslandConfig(
        name="金额频次",
        dimension="amount_frequency",
        description="金额分布、频率、拆分交易",
        fields=["amount", "interval", "hour", "transaction_motif"],
    ),
    IslandConfig(
        name="时间地域",
        dimension="time_region",
        description="夜间交易、IP跳变、跨地域",
        fields=["timestamp", "src_ip", "dst_ip", "region_code", "hour"],
    ),
    IslandConfig(
        name="行为序列",
        dimension="behavior_sequence",
        description="休眠激活、交易对手更换、交易用途偏离",
        fields=["timestamp", "transaction_mode", "zs_timestamp", "zs_channel"],
    ),
    IslandConfig(
        name="主体关联",
        dimension="entity_association",
        description="交叉转账网络、身份复用、设备共享",
        fields=[
            "wallet_open_cert",
            "wallet_open_tel",
            "src_device",
            "dst_device",
            "src",
            "dst",
        ],
    ),
]


@dataclass
class SearchParams:
    m: int = 4           # number of islands
    b: int = 3           # default candidates per island per round (used in round 1)
    B_total: int = 12    # total candidate budget per round (m * b)
    b_min: int = 1       # min candidates per island (dynamic allocation)
    b_max: int = 5       # max candidates per island (dynamic allocation)
    k: int = 5           # in-context examples via Boltzmann sampling
    T: int = 10          # max rounds
    tau: float = 1.0     # Boltzmann temperature
    intra_island_turns: int = 3  # multi-turn dialogue rounds per island per search round


@dataclass
class PendingPoolConfig:
    max_size: int = 20       # max features in pool
    max_age: int = 3         # purge after this many rounds
    max_rejects: int = 2     # purge after consecutive rejections


@dataclass
class ValidationThresholds:
    mi_min: float = 0.001
    corr_max: float = 0.9
    cramers_v_min: float = 0.05
    delta_prauc_min: float = 0.005
    bootstrap_p_max: float = 0.05


@dataclass
class ScoringWeights:
    shap: float = 0.35
    mi: float = 0.25
    redundancy: float = 0.25
    delta_prauc: float = 0.15


@dataclass
class XGBoostParams:
    scale_pos_weight: float = 19.0
    eval_metric: str = "aucpr"
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    device: str = os.environ.get("XGB_DEVICE", "cpu")


@dataclass
class LLMConfig:
    provider: str = "mock"  # openai / gemini / deepseek / local / mock
    model_name: str = ""
    api_key: str = ""
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120

    def __post_init__(self):
        if not self.api_key:
            env_map = {
                "openai": "OPENAI_API_KEY",
                "gemini": "GOOGLE_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
            }
            env_key = env_map.get(self.provider, "")
            if env_key:
                self.api_key = os.environ.get(env_key, "")


@dataclass
class PathConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    domain_knowledge_dir: str = "inputs/domain_knowledge"
    baseline_dir: str = "inputs/baseline"


@dataclass
class Config:
    islands: list[IslandConfig] = field(default_factory=lambda: list(ISLANDS))
    search: SearchParams = field(default_factory=SearchParams)
    pending_pool: PendingPoolConfig = field(default_factory=PendingPoolConfig)
    validation: ValidationThresholds = field(default_factory=ValidationThresholds)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    xgboost: XGBoostParams = field(default_factory=XGBoostParams)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)


DEFAULT_CONFIG = Config()
