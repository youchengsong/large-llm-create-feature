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
