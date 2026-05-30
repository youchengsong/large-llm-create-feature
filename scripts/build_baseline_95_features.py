"""
构建 95 个基线特征矩阵。

输入:
  - data/accounts.csv     (账户开立表)
  - data/transactions.csv (交易表)
  - data/close.csv        (注销表)

输出:
  - outputs/baseline_95_features.parquet  (1,384,512 行 × 95 列 + label)

流程:
  1. 调用原始 feature_engineering.py 生成 baseline_cache.parquet (106 列)
  2. 排除 10 个低质量特征 + 5 个 label/ID 列 → 保留 95 个特征
  3. 如果 baseline_cache.parquet 已存在，直接从中提取

依赖: pandas, numpy
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BASELINE_CACHE = OUTPUTS_DIR / "baseline_cache.parquet"

BASELINE_EXCLUDE_FEATURES = {
    "is_laundering", "is_risk", "tx_id", "src", "dst",
    "Min_Received", "Min_Sent", "S_src_counterparty_index",
    "Timestamp_Hour", "S_dst_counterparty_index",
    "1H_frequency_From", "1H_frequency_To",
    "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent",
}


def load_baseline_95():
    """加载 baseline_cache.parquet 并返回 (X_95, y, feature_names)。"""
    if not BASELINE_CACHE.exists():
        print("baseline_cache.parquet 不存在，请先运行 feature_engineering.py 生成。")
        print(f"  python {PROJECT_ROOT / 'inputs/baseline/feature_engineering.py'} \\")
        print(f"    {DATA_DIR / 'transactions.csv'} \\")
        print(f"    {DATA_DIR / 'accounts.csv'} \\")
        print(f"    {OUTPUTS_DIR / 'baseline_cache.parquet'} \\")
        print(f"    --close_path {DATA_DIR / 'close.csv'}")
        sys.exit(1)

    print(f"加载 {BASELINE_CACHE} ...")
    df = pd.read_parquet(BASELINE_CACHE)
    print(f"  原始: {df.shape[0]} 行, {df.shape[1]} 列")

    label_col = "is_risk" if "is_risk" in df.columns else "is_laundering"
    y = df[label_col].astype(int).values
    print(f"  标签列: {label_col}, 正样本: {y.sum()}, 负样本: {len(y) - y.sum()}")

    all_cols = set(df.columns)
    feature_cols = sorted([c for c in all_cols if c not in BASELINE_EXCLUDE_FEATURES])
    print(f"  排除 {len(all_cols) - len(feature_cols)} 列 → 保留 {len(feature_cols)} 个基线特征")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y, feature_cols


def main():
    X, y, feature_names = load_baseline_95()

    print(f"\n=== 95 个基线特征列表 ===")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:3d}. {name}")

    out_df = X.copy()
    out_df["is_laundering"] = y

    out_path = OUTPUTS_DIR / "baseline_95_features.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"\n已保存到 {out_path}")
    print(f"  形状: {out_df.shape[0]} 行, {out_df.shape[1]} 列 ({len(feature_names)} 特征 + 1 标签)")


if __name__ == "__main__":
    main()
