from pathlib import Path

import pandas as pd


_FALLBACK_PATHS = {
    "open": [
        "accounts.csv",
        "../inputs/SimECNY/open/accounts.csv",
    ],
    "tx": [
        "transactions.csv",
        "../inputs/SimECNY/transaction/wallet_temporal_transactions_1105_1450_temp.csv",
    ],
    "close": [
        "close.csv",
        "../inputs/SimECNY/close/new_close_3.csv",
    ],
}


def _find_file(data_dir: str, key: str) -> Path:
    base = Path(data_dir)
    for rel in _FALLBACK_PATHS[key]:
        p = base / rel
        if p.exists():
            return p
    tried = [str(base / r) for r in _FALLBACK_PATHS[key]]
    raise FileNotFoundError(
        f"Cannot find {key} data. Tried: {tried}"
    )


def load_all_tables(
    data_dir: str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    open_df = pd.read_csv(_find_file(data_dir, "open"))
    tx_df = pd.read_csv(_find_file(data_dir, "tx"), on_bad_lines="skip")
    close_df = pd.read_csv(_find_file(data_dir, "close"))
    return open_df, tx_df, close_df


def extract_schema(df: pd.DataFrame, table_name: str) -> str:
    lines = [f"=== {table_name} ({len(df)} rows, {len(df.columns)} columns) ==="]
    for col in df.columns:
        dtype = df[col].dtype
        null_pct = df[col].isna().mean() * 100
        n_unique = df[col].nunique()

        if pd.api.types.is_bool_dtype(df[col]):
            true_pct = df[col].mean() * 100
            stats = f"True={true_pct:.1f}%"
        elif pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            stats = (
                f"min={desc['min']:.4g}, median={desc['50%']:.4g}, "
                f"max={desc['max']:.4g}, mean={desc['mean']:.4g}"
            )
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            stats = f"range=[{df[col].min()}, {df[col].max()}]"
        else:
            top_vals = df[col].dropna().value_counts().head(5)
            examples = ", ".join(f"{v}({c})" for v, c in top_vals.items())
            stats = f"top: {examples}"

        lines.append(
            f"  {col}: {dtype} | null={null_pct:.1f}% | unique={n_unique} | {stats}"
        )
    return "\n".join(lines)


def extract_label(tx_df: pd.DataFrame) -> pd.Series:
    if "is_risk" not in tx_df.columns:
        raise KeyError("Column 'is_risk' not found in transaction table")
    return tx_df["is_risk"].astype(int)


def extract_distribution_diff(
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
    label: pd.Series,
) -> str:
    """提取正常 vs 异常的类别条件分布差异（聚合统计量，不泄露原始数据）。"""
    import numpy as np

    lines = ["## 正常 vs 异常交易的分布差异（聚合统计量）\n"]
    risk = label.astype(bool)

    numeric_cols = ["amount", "interval", "hour"]
    for col in numeric_cols:
        if col not in tx_df.columns:
            continue
        normal_vals = tx_df.loc[~risk, col].dropna()
        abnormal_vals = tx_df.loc[risk, col].dropna()
        if len(normal_vals) == 0 or len(abnormal_vals) == 0:
            continue
        lines.append(
            f"- {col}: 正常中位数={normal_vals.median():.2f}, "
            f"异常中位数={abnormal_vals.median():.2f}, "
            f"正常均值={normal_vals.mean():.2f}, "
            f"异常均值={abnormal_vals.mean():.2f}"
        )

    if "hour" in tx_df.columns:
        night_hours = {22, 23, 0, 1, 2, 3, 4, 5}
        normal_night = tx_df.loc[~risk, "hour"].isin(night_hours).mean() * 100
        abnormal_night = tx_df.loc[risk, "hour"].isin(night_hours).mean() * 100
        lines.append(
            f"- 凌晨交易(22-5时)占比: 正常={normal_night:.1f}%, 异常={abnormal_night:.1f}%"
        )

    cat_cols = ["transaction_mode", "transaction_motif"]
    for col in cat_cols:
        if col not in tx_df.columns:
            continue
        for val in tx_df[col].dropna().unique()[:7]:
            normal_pct = (tx_df.loc[~risk, col] == val).mean() * 100
            abnormal_pct = (tx_df.loc[risk, col] == val).mean() * 100
            if abs(abnormal_pct - normal_pct) > 2:
                lines.append(f"- {col}={val}: 正常={normal_pct:.1f}%, 异常={abnormal_pct:.1f}%")

    if "wallet_level" in tx_df.columns:
        level_col = "src_wallet_level" if "src_wallet_level" in tx_df.columns else "wallet_level"
        if level_col in tx_df.columns:
            for lv in sorted(tx_df[level_col].dropna().unique()):
                normal_pct = (tx_df.loc[~risk, level_col] == lv).mean() * 100
                abnormal_pct = (tx_df.loc[risk, level_col] == lv).mean() * 100
                if abs(abnormal_pct - normal_pct) > 2:
                    lines.append(f"- {level_col}={int(lv)}: 正常={normal_pct:.1f}%, 异常={abnormal_pct:.1f}%")

    lines.append("\n### 开户表维度差异")
    abnormal_wallets = set(tx_df.loc[risk, "src"].dropna().unique())
    normal_wallets = set(tx_df.loc[~risk, "src"].dropna().unique()) - abnormal_wallets

    if "wallet_open_cert" in open_df.columns:
        cert_counts = open_df.groupby("wallet_open_cert")["wallet_id"].nunique()
        abn_certs = open_df[open_df["wallet_id"].isin(abnormal_wallets)]["wallet_open_cert"].dropna()
        norm_certs = open_df[open_df["wallet_id"].isin(normal_wallets)]["wallet_open_cert"].dropna()
        if len(abn_certs) > 0 and len(norm_certs) > 0:
            abn_multi = cert_counts.reindex(abn_certs.unique()).fillna(1).mean()
            norm_multi = cert_counts.reindex(norm_certs.unique()).fillna(1).mean()
            lines.append(f"- 同一身份证关联钱包数: 正常平均={norm_multi:.1f}, 异常平均={abn_multi:.1f}")

    if "wallet_open_tel" in open_df.columns:
        tel_counts = open_df.groupby("wallet_open_tel")["wallet_id"].nunique()
        abn_tels = open_df[open_df["wallet_id"].isin(abnormal_wallets)]["wallet_open_tel"].dropna()
        norm_tels = open_df[open_df["wallet_id"].isin(normal_wallets)]["wallet_open_tel"].dropna()
        if len(abn_tels) > 0 and len(norm_tels) > 0:
            abn_multi_t = tel_counts.reindex(abn_tels.unique()).fillna(1).mean()
            norm_multi_t = tel_counts.reindex(norm_tels.unique()).fillna(1).mean()
            lines.append(f"- 同一手机号关联钱包数: 正常平均={norm_multi_t:.1f}, 异常平均={abn_multi_t:.1f}")

    tx_per_wallet = tx_df.groupby("src")["tx_id"].count()
    abn_tx = tx_per_wallet.reindex(abnormal_wallets).dropna()
    norm_tx = tx_per_wallet.reindex(normal_wallets).dropna()
    if len(abn_tx) > 0 and len(norm_tx) > 0:
        lines.append(f"- 每钱包交易笔数: 正常平均={norm_tx.mean():.1f}, 异常平均={abn_tx.mean():.1f}")

    return "\n".join(lines)


def extract_join_statistics(
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
) -> str:
    """提取三表关联的覆盖率和拓扑统计。"""
    lines = ["## 三表关联统计\n"]

    open_ids = set(open_df["wallet_id"].dropna().unique())
    src_ids = set(tx_df["src"].dropna().unique())
    dst_ids = set(tx_df["dst"].dropna().unique())

    src_match = len(src_ids & open_ids) / max(len(src_ids), 1) * 100
    dst_match = len(dst_ids & open_ids) / max(len(dst_ids), 1) * 100
    lines.append(f"- 交易表 src → 开户表匹配率: {src_match:.1f}%")
    lines.append(f"- 交易表 dst → 开户表匹配率: {dst_match:.1f}%")

    close_ids = set(close_df["wallet_id"].dropna().unique())
    close_rate = len(close_ids & open_ids) / max(len(open_ids), 1) * 100
    lines.append(f"- 有销户记录的钱包占比: {close_rate:.1f}%")

    overlap = len(src_ids & dst_ids)
    total_wallets = len(src_ids | dst_ids)
    if total_wallets > 0:
        lines.append(f"- src/dst 重叠率: {overlap / total_wallets * 100:.1f}% 的钱包既发过也收过")

    lines.append(f"- 钱包总数(开户表): {len(open_ids)}")
    lines.append(f"- 交易涉及钱包数: {total_wallets}")
    lines.append(f"- 交易总笔数: {len(tx_df)}")
    lines.append(f"- 平均每钱包交易数: {len(tx_df) / max(total_wallets, 1):.1f}")

    return "\n".join(lines)
