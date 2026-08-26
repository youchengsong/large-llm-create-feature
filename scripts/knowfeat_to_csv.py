"""Convert KnowFeat generic_features.json to a features CSV for evaluate.py.

Usage:
    python scripts/knowfeat_to_csv.py --dataset credit_g
    python scripts/knowfeat_to_csv.py --dataset credit_g --features-dir outputs/credit_g_schema_only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_CONFIG = {
    "credit_g": {"csv": "data/public/credit_g.csv", "target": "class"},
    "diabetes": {"csv": "data/public/diabetes.csv", "target": "class"},
    "blood_transfusion": {"csv": "data/public/blood_transfusion.csv", "target": "Class"},
    "heart": {"csv": "data/public/heart.csv", "target": "HeartDisease"},
    "bank_marketing": {"csv": "data/public/bank_marketing.csv", "target": "y"},
    "breast_w": {"csv": "data/public/breast_w.csv", "target": "Class"},
    "adult": {"csv": "data/public/adult.csv", "target": "class"},
    "default_credit": {"csv": "data/public/default_credit.csv", "target": "target", "metrics": ["auc", "accuracy"]},
    "churn": {"csv": "data/public/churn.csv", "target": "target", "metrics": ["auc", "accuracy"]},
    "wine_quality": {"csv": "data/public/wine_quality.csv", "target": "class"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--features-dir", default=None,
                        help="Directory containing generic_features.json (default: outputs/{dataset})")
    args = parser.parse_args()

    ds = args.dataset
    cfg = DATASET_CONFIG[ds]

    feat_dir = Path(args.features_dir) if args.features_dir else PROJECT_ROOT / "outputs" / ds
    if not feat_dir.is_absolute():
        feat_dir = PROJECT_ROOT / feat_dir

    features_json = feat_dir / "generic_features.json"
    if not features_json.exists():
        logger.error("Features JSON not found: %s", features_json)
        sys.exit(1)

    df = pd.read_csv(PROJECT_ROOT / cfg["csv"])
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    with open(features_json) as f:
        features = json.load(f)

    logger.info("Loaded %d feature definitions from %s", len(features), features_json)

    result_df = pd.DataFrame({"row_id": df["row_id"]})
    success_count = 0

    for feat in features:
        name = feat["name"]
        code = feat["code"]
        try:
            exec_globals = {"pd": pd, "np": np}
            exec(code, exec_globals)
            func_candidates = [k for k, v in exec_globals.items()
                                if callable(v) and k not in ("pd", "np") and not k.startswith("__")]
            func_name = func_candidates[0]
            feat_result = exec_globals[func_name](df)
            if isinstance(feat_result, pd.DataFrame):
                new_cols = [c for c in feat_result.columns if c != "row_id"]
                if new_cols:
                    for c in new_cols:
                        result_df[c] = feat_result[c].values
                    success_count += 1
                    logger.info("  OK: %s -> %s", name, new_cols)
                else:
                    logger.warning("  SKIP: %s (no feature columns)", name)
            elif isinstance(feat_result, pd.Series):
                result_df[name] = feat_result.values
                success_count += 1
                logger.info("  OK: %s (Series)", name)
            else:
                logger.warning("  SKIP: %s (unexpected type: %s)", name, type(feat_result))
        except Exception as e:
            logger.warning("  FAIL: %s (%s)", name, e)

    out_csv = feat_dir / "knowfeat_features.csv"
    result_df.to_csv(out_csv, index=False)
    logger.info("Saved %d features to %s", success_count, out_csv)


if __name__ == "__main__":
    main()
