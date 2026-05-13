"""Validate the three generated tables (accounts, transactions, close)."""

import pandas as pd
import sys
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def validate():
    open_df = pd.read_csv(os.path.join(DATA_DIR, "accounts.csv"))
    tx_df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    close_df = pd.read_csv(os.path.join(DATA_DIR, "close.csv"))

    print("=== Table shapes ===")
    for name, df in [("accounts", open_df), ("transactions", tx_df), ("close", close_df)]:
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols")
        missing = df.isnull().sum()
        if missing.any():
            print(f"    missing: {missing[missing > 0].to_dict()}")

    print("\n=== Column check ===")
    expected_open = [
        "wallet_id", "wallet_type", "wallet_level", "wallet_open_cert",
        "wallet_open_cert_expire", "wallet_open_date", "wallet_open_timestamp",
        "wallet_open_tel", "realname_auth_time", "accociated_bank_account",
        "bank_account_number", "init_balance", "region_code", "is_abnormal",
        "open_device", "open_ip",
    ]
    expected_tx = [
        "tx_id", "timestamp", "src", "dst", "amount", "transaction_motif",
        "motif_id", "transaction_mode", "is_risk",
    ]
    expected_close = ["wallet_id", "zs_id", "zs_channel", "zs_timestamp",
                      "wallet_open_tel", "is_zs_laundering"]

    for name, df, expected in [
        ("accounts", open_df, expected_open),
        ("transactions", tx_df, expected_tx),
        ("close", close_df, expected_close),
    ]:
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            print(f"  {name} MISSING columns: {missing_cols}")
        else:
            print(f"  {name}: all expected columns present")

    print("\n=== is_risk distribution (transactions) ===")
    if "is_risk" in tx_df.columns:
        counts = tx_df["is_risk"].value_counts()
        total = len(tx_df)
        pos = counts.get(1, 0)
        print(f"  total={total}, positive={pos}, ratio={pos/total:.4%}")

    print("\n=== Referential integrity ===")
    wallet_ids = set(open_df["wallet_id"].astype(str))
    tx_src = set(tx_df["src"].astype(str))
    tx_dst = set(tx_df["dst"].astype(str))
    tx_wallets = tx_src | tx_dst
    unmatched = tx_wallets - wallet_ids
    print(f"  accounts wallet_ids: {len(wallet_ids)}")
    print(f"  tx unique wallets: {len(tx_wallets)}")
    print(f"  unmatched (tx wallets not in accounts): {len(unmatched)}")

    close_wallets = set(close_df["wallet_id"].astype(str))
    close_unmatched = close_wallets - wallet_ids
    print(f"  close wallets not in accounts: {len(close_unmatched)}")

    ok = len(unmatched) == 0 and len(close_unmatched) == 0
    print(f"\n{'PASS' if ok else 'WARN'}: referential integrity {'OK' if ok else 'has gaps'}")
    return ok


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
