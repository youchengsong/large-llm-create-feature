"""从 138 万交易中采样 20 万条子集（保留全部异常），并生成对应的开户/销户子集。"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TARGET_TOTAL = 200_000
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_TOTAL,
                        help="目标交易总数 (default: 200000)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载交易表 ──────────────────────────────────────────────
    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    risk = tx[tx["is_risk"] == 1]
    normal = tx[tx["is_risk"] == 0]
    print(f"原始交易: {len(tx):,}  (异常={len(risk):,}, 正常={len(normal):,})")

    n_risk = len(risk)
    if n_risk >= args.target:
        raise ValueError(f"异常数 {n_risk} >= 目标 {args.target}，无需采样")

    n_normal_sample = args.target - n_risk
    normal_sampled = normal.sample(n=n_normal_sample, random_state=args.seed)
    tx_sub = pd.concat([risk, normal_sampled]).sort_index()
    print(f"采样后交易: {len(tx_sub):,}  (异常={n_risk:,}, 正常={n_normal_sample:,})")

    # ── 2. 提取涉及的钱包 ID ──────────────────────────────────────
    wallet_ids = set(tx_sub["src"].unique()) | set(tx_sub["dst"].unique())
    print(f"涉及钱包数: {len(wallet_ids):,}")

    # ── 3. 过滤开户表 ─────────────────────────────────────────────
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    accounts_sub = accounts[accounts["wallet_id"].isin(wallet_ids)]
    print(f"开户表: {len(accounts):,} → {len(accounts_sub):,}")

    # ── 4. 过滤销户表 ─────────────────────────────────────────────
    close = pd.read_csv(DATA_DIR / "close.csv")
    close_sub = close[close["wallet_id"].isin(wallet_ids)]
    print(f"销户表: {len(close):,} → {len(close_sub):,}")

    # ── 5. 备份原文件并保存 ─────────────────────────────────────
    backup_dir = DATA_DIR / "full_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ["transactions.csv", "accounts.csv", "close.csv"]:
        src_path = DATA_DIR / name
        dst_path = backup_dir / name
        if src_path.exists() and not dst_path.exists():
            src_path.rename(dst_path)
            print(f"  备份: {name} → full_backup/{name}")

    tx_sub.to_csv(out_dir / "transactions.csv", index=False)
    accounts_sub.to_csv(out_dir / "accounts.csv", index=False)
    close_sub.to_csv(out_dir / "close.csv", index=False)
    print(f"\n已保存到 {out_dir}/")
    print(f"  transactions.csv  {len(tx_sub):,} 行")
    print(f"  accounts.csv      {len(accounts_sub):,} 行")
    print(f"  close.csv         {len(close_sub):,} 行")

    # ── 6. 同步 baseline_cache ───────────────────────────────────
    baseline_path = PROJECT_ROOT / "outputs" / "baseline_cache.parquet"
    if baseline_path.exists():
        baseline_backup = backup_dir / "baseline_cache.parquet"
        if not baseline_backup.exists():
            import shutil
            shutil.copy2(baseline_path, baseline_backup)
            print(f"  备份: baseline_cache.parquet → full_backup/")
        bl = pd.read_parquet(baseline_path)
        bl_sub = bl.loc[bl.index.isin(tx_sub.index)]
        bl_sub.to_parquet(baseline_path)
        print(f"  baseline_cache.parquet  {len(bl_sub):,} 行 (已覆盖)")
    else:
        print("  baseline_cache.parquet 不存在，跳过")


if __name__ == "__main__":
    main()
