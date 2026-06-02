"""Convert KnowFeat stage2_features.json to CSV for simecny evaluate.py."""
import argparse
import json
import sys
import logging
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowfeat.data.loader import load_accounts, load_transactions, load_close

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default=None,
                        help="Directory containing stage2_features.json (default: outputs/)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <features-dir>/simecny_knowfeat_features.csv)")
    args = parser.parse_args()

    feat_dir = Path(args.features_dir) if args.features_dir else PROJECT_ROOT / "outputs"
    if not feat_dir.is_absolute():
        feat_dir = PROJECT_ROOT / feat_dir

    features_json = feat_dir / "stage2_features.json"
    if not features_json.exists():
        logger.error("stage2_features.json not found at %s", features_json)
        sys.exit(1)

    with open(features_json) as f:
        features = json.load(f)

    logger.info("Loading simecny data...")
    data_dir = PROJECT_ROOT / "data"
    open_df = load_accounts(data_dir / "accounts.csv")
    tx_df = load_transactions(data_dir / "transactions.csv")
    close_df = load_close(data_dir / "close.csv")

    src_ids = tx_df["src"].values

    result_df = pd.DataFrame({"src": src_ids})
    ok = 0
    fail = 0

    for feat in features:
        name = feat["name"]
        code = feat["code"]
        logger.info("Executing feature: %s", name)
        try:
            ns = {"pd": pd, "np": np}
            exec(code, ns)
            func_candidates = [k for k, v in ns.items()
                                if callable(v) and k not in ("pd", "np") and not k.startswith("__")]
            func_name = func_candidates[0]
            feat_df = ns[func_name](open_df.copy(), tx_df.copy(), close_df.copy())
            if "wallet_id" in feat_df.columns:
                feat_df = feat_df.rename(columns={"wallet_id": "src"})
            feat_df["src"] = feat_df["src"].astype(int)
            result_df = result_df.merge(feat_df, on="src", how="left")
            ok += 1
            logger.info("  OK: %s, cols=%s", name, [c for c in feat_df.columns if c != "src"])
        except Exception as e:
            fail += 1
            logger.warning("  FAIL: %s -> %s", name, e)
            traceback.print_exc()

    feat_cols = [c for c in result_df.columns if c != "src"]
    out_df = result_df[feat_cols].fillna(0)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
    else:
        out_path = feat_dir / "simecny_knowfeat_features.csv"

    out_df.to_csv(out_path, index=False)
    logger.info("Done: %d ok, %d fail, saved to %s (%d features, %d rows)", ok, fail, out_path, len(feat_cols), len(out_df))

if __name__ == "__main__":
    main()
