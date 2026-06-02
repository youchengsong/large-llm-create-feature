"""数据加载：Stage 2 三表 / Stage 1 单表 / Generic 公开数据集 + 基线特征。"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from knowfeat.config import KnowFeatConfig, PathConfig, PublicDatasetConfig

logger = logging.getLogger(__name__)

DATE_COLS_ACCOUNTS = ["wallet_open_cert_expire", "wallet_open_date",
                      "wallet_open_timestamp", "realname_auth_time"]
DATE_COLS_TX = ["timestamp"]
DATE_COLS_CLOSE = ["zs_timestamp"]


@dataclass
class Stage2Data:
    """Stage 2 三表 + 基线特征的容器。"""
    open_df: pd.DataFrame
    tx_df: pd.DataFrame
    close_df: pd.DataFrame
    baseline_features: Optional[pd.DataFrame] = None
    label_col: str = "is_risk"


# ---------------------------------------------------------------------------
# Stage 1 (BABD-13): 单表数据
# ---------------------------------------------------------------------------

BABD13_ILLICIT_LABELS = {0, 2, 6, 7, 8, 9, 11}

BABD13_LABEL_MAP = {
    0: "Blackmail", 1: "Cyber-Security Service", 2: "Darknet Market",
    3: "Centralized Exchange", 4: "Infrastructure/Service", 5: "P2P Financial",
    6: "Gambling", 7: "Gov Criminal Blacklist", 8: "Money Laundering",
    9: "Ponzi Scheme", 10: "Mining Pool", 11: "Tumbler", 12: "Individual Wallet",
}


@dataclass
class Stage1Data:
    """Stage 1 单表 + 基线特征的容器 (BABD-13)。"""
    df: pd.DataFrame
    baseline_features: Optional[pd.DataFrame] = None
    label_col: str = "is_illicit"
    id_col: str = "account"
    feature_cols: list[str] = None

    def __post_init__(self):
        if self.feature_cols is None:
            self.feature_cols = [
                c for c in self.df.columns
                if c not in (self.id_col, "label", self.label_col)
            ]


def load_accounts(path: Path) -> pd.DataFrame:
    """加载 accounts.csv → open_df。"""
    logger.info("Loading accounts from %s", path)
    df = pd.read_csv(path)
    for col in DATE_COLS_ACCOUNTS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["wallet_id"] = df["wallet_id"].astype(int)
    logger.info("Accounts loaded: %d rows, %d cols", *df.shape)
    return df


def load_transactions(path: Path) -> pd.DataFrame:
    """加载 transactions.csv → tx_df，按 timestamp 排序。"""
    logger.info("Loading transactions from %s", path)
    df = pd.read_csv(path)
    for col in DATE_COLS_TX:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["is_risk"] = df["is_risk"].astype(int)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Transactions loaded: %d rows, %d cols", *df.shape)
    return df


def load_close(path: Path) -> pd.DataFrame:
    """加载 close.csv → close_df。"""
    logger.info("Loading close records from %s", path)
    df = pd.read_csv(path)
    for col in DATE_COLS_CLOSE:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["wallet_id"] = df["wallet_id"].astype(int)
    logger.info("Close records loaded: %d rows, %d cols", *df.shape)
    return df


def load_baseline_features(path: Path) -> pd.DataFrame:
    """加载 baseline_cache.parquet → baseline_features。

    该 parquet 是 feature_engineering.py 产出的已归一化特征矩阵，
    每行对应一笔交易，含 106 列（含 is_laundering 标签列），实际使用 96 个特征列。
    """
    logger.info("Loading baseline features from %s", path)
    df = pd.read_parquet(path)
    logger.info("Baseline features loaded: %d rows, %d cols", *df.shape)
    return df


def load_stage2(cfg: Optional[KnowFeatConfig] = None,
                paths: Optional[PathConfig] = None) -> Stage2Data:
    """一次性加载 Stage 2 全部数据。

    Parameters
    ----------
    cfg : KnowFeatConfig, optional
        顶层配置。如果提供，从中取 paths。
    paths : PathConfig, optional
        路径配置。如果 cfg 和 paths 都未提供，使用默认 PathConfig。

    Returns
    -------
    Stage2Data
        包含 open_df, tx_df, close_df, baseline_features 的容器。
    """
    if paths is None:
        paths = cfg.paths if cfg is not None else PathConfig()

    open_df = load_accounts(paths.accounts_csv)
    tx_df = load_transactions(paths.transactions_csv)
    close_df = load_close(paths.close_csv)

    baseline_features = None
    if paths.baseline_cache.exists():
        baseline_features = load_baseline_features(paths.baseline_cache)
    else:
        logger.warning("Baseline cache not found at %s, skipping.",
                       paths.baseline_cache)

    return Stage2Data(
        open_df=open_df,
        tx_df=tx_df,
        close_df=close_df,
        baseline_features=baseline_features,
    )


def get_label_series(data: Stage2Data) -> pd.Series:
    """从 baseline_features 中提取标签列（确保与特征行对齐）。"""
    if data.baseline_features is not None and data.label_col in data.baseline_features.columns:
        return data.baseline_features[data.label_col].astype(int).reset_index(drop=True)
    return data.tx_df[data.label_col].astype(int)


BASELINE_EXCLUDE_FEATURES = {
    # 标签 / ID 列
    "is_laundering", "is_risk", "tx_id", "tx_id_fe", "src", "dst",
    # 以下 10 个特征因数据泄漏或过拟合风险从基线中排除（不删除 parquet 原数据）
    "Min_Received",
    "Min_Sent",
    "S_src_counterparty_index",
    "Timestamp_Hour",
    "S_dst_counterparty_index",
    "1H_frequency_From",
    "1H_frequency_To",
    "wallets_per_cert_From",
    "min_inactive_period_From",
    "Max_Sent",
}


def get_baseline_feature_names(data: Stage2Data) -> list[str]:
    """返回基线特征列名（排除标签列和已排除特征）。"""
    if data.baseline_features is None:
        return []
    return [c for c in data.baseline_features.columns if c not in BASELINE_EXCLUDE_FEATURES]


# ---------------------------------------------------------------------------
# Stage 1 加载
# ---------------------------------------------------------------------------

def load_stage1(cfg: Optional[KnowFeatConfig] = None,
                paths: Optional[PathConfig] = None) -> Stage1Data:
    """加载 BABD-13 单表数据，执行 13-class → binary 标签映射。

    Returns
    -------
    Stage1Data
        包含 df (含 is_illicit 二值标签) 和 baseline_features 的容器。
    """
    if paths is None:
        paths = cfg.paths if cfg is not None else PathConfig()

    csv_path = paths.babd13_csv
    logger.info("Loading BABD-13 from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("BABD-13 loaded: %d rows, %d cols", *df.shape)

    # SW 列是类别型 (SA/WA)，编码为 0/1
    if "SW" in df.columns and not pd.api.types.is_numeric_dtype(df["SW"]):
        df["SW"] = (df["SW"] == "WA").astype(int)

    # 13-class → binary (use loc to avoid fragmentation warning)
    df = df.copy()
    df["is_illicit"] = df["label"].isin(BABD13_ILLICIT_LABELS).astype(int)
    n_pos = df["is_illicit"].sum()
    n_neg = len(df) - n_pos
    logger.info("Binary label: illicit=%d (%.2f%%), licit=%d",
                n_pos, 100 * n_pos / len(df), n_neg)

    # 基线特征 = 除 account 和 label 列外的所有列
    feature_cols = [c for c in df.columns if c not in ("account", "label", "is_illicit")]
    baseline_features = df[feature_cols].copy()

    return Stage1Data(
        df=df,
        baseline_features=baseline_features,
        label_col="is_illicit",
        id_col="account",
        feature_cols=feature_cols,
    )


def get_label_series_stage1(data: Stage1Data) -> pd.Series:
    """从 Stage 1 df 中提取标签列。"""
    return data.df[data.label_col].astype(int)


# ---------------------------------------------------------------------------
# Generic 公开数据集加载
# ---------------------------------------------------------------------------

def load_public_dataset(pub_cfg: PublicDatasetConfig) -> Stage1Data:
    """加载公开数据集为 Stage1Data 格式。

    Parameters
    ----------
    pub_cfg : PublicDatasetConfig
        公开数据集配置（csv_path, target_col, id_col 等）。

    Returns
    -------
    Stage1Data
        复用 Stage1Data 容器，id_col 默认为 "row_id"。
    """
    csv_path = pub_cfg.csv_path
    target_col = pub_cfg.target_col
    id_col = pub_cfg.id_col

    logger.info("Loading public dataset from %s (target=%s)", csv_path, target_col)
    df = pd.read_csv(csv_path)
    logger.info("Public dataset loaded: %d rows, %d cols", *df.shape)

    if id_col == "row_id" and "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        logger.info("Label-encoded '%s': %s", target_col, dict(zip(le.classes_, le.transform(le.classes_))))

    feature_cols = [
        c for c in df.columns
        if c not in (id_col, target_col)
    ]

    baseline_features = df[feature_cols].copy()

    return Stage1Data(
        df=df,
        baseline_features=baseline_features,
        label_col=target_col,
        id_col=id_col,
        feature_cols=feature_cols,
    )
