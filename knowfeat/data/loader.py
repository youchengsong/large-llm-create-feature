"""Stage 2 数据加载：三表 + 基线特征。"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from knowfeat.config import KnowFeatConfig, PathConfig

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
    """从 tx_df 中提取标签列。"""
    return data.tx_df[data.label_col].astype(int)


BASELINE_EXCLUDE_FEATURES = {
    # 标签 / ID 列
    "is_laundering", "is_risk", "tx_id", "src", "dst",
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
