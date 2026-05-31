"""统一评估框架：80/20 train/test split × 5 random seeds × XGBoost。

用法:
    python scripts/evaluate.py --dataset credit_g --features-csv outputs/credit_g_features.csv
    python scripts/evaluate.py --dataset simecny --baseline  # 仅评估基线特征
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 2024]

DATASET_CONFIG = {
    "credit_g": {
        "csv": "data/public/credit_g.csv",
        "target": "class",
        "metrics": ["auc", "accuracy"],
        "domain_knowledge": "inputs/public_datasets/credit_g/domain_knowledge.json",
    },
    "diabetes": {
        "csv": "data/public/diabetes.csv",
        "target": "class",
        "metrics": ["auc", "accuracy"],
        "domain_knowledge": "inputs/public_datasets/diabetes/domain_knowledge.json",
    },
    "blood_transfusion": {
        "csv": "data/public/blood_transfusion.csv",
        "target": "Class",
        "metrics": ["auc", "accuracy"],
        "domain_knowledge": "inputs/public_datasets/blood_transfusion/domain_knowledge.json",
    },
    "heart": {
        "csv": "data/public/heart.csv",
        "target": "HeartDisease",
        "metrics": ["auc", "accuracy"],
        "domain_knowledge": "inputs/public_datasets/heart/domain_knowledge.json",
    },
    "bank_marketing": {
        "csv": "data/public/bank_marketing.csv",
        "target": "y",
        "metrics": ["auc", "accuracy"],
        "domain_knowledge": "inputs/public_datasets/bank_marketing/domain_knowledge.json",
    },
    "breast_w": {
        "csv": "data/public/breast_w.csv",
        "target": "Class",
        "metrics": ["auc", "accuracy"],
        "domain_knowledge": "inputs/public_datasets/breast_w/domain_knowledge.json",
    },
    "simecny": {
        "csv": None,  # uses baseline_cache.parquet
        "target": "is_risk",
        "metrics": ["recall", "precision", "f1", "auc"],
    },
}


def load_dataset(dataset_name: str, features_csv: str = None) -> tuple[pd.DataFrame, str]:
    """加载数据集，返回 (DataFrame, target_col)。"""
    cfg = DATASET_CONFIG[dataset_name]
    target = cfg["target"]

    if dataset_name == "simecny":
        baseline_path = PROJECT_ROOT / "outputs" / "baseline_cache.parquet"
        if not baseline_path.exists():
            raise FileNotFoundError(f"基线缓存不存在: {baseline_path}")
        df = pd.read_parquet(baseline_path)
    else:
        csv_path = PROJECT_ROOT / cfg["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"数据集不存在: {csv_path}\n请先运行: python scripts/download_public_datasets.py"
            )
        df = pd.read_csv(csv_path)

    if features_csv:
        feat_df = pd.read_csv(features_csv)
        if len(feat_df) == len(df):
            for col in feat_df.columns:
                if col not in df.columns:
                    df[col] = feat_df[col].values

    return df, target


def prepare_features(
    df: pd.DataFrame, target: str, feature_cols: list[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """准备特征矩阵和标签。"""
    if feature_cols is None:
        exclude = {target, "row_id", "wallet_id", "tx_id", "account",
                   "src", "dst", "is_laundering", "is_risk", "label", "is_illicit"}
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    y = df[target]
    if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
        from sklearn.preprocessing import LabelEncoder
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
    y = y.astype(int)

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes

    X = X.fillna(-999)
    import re
    X.columns = [re.sub(r'[\[\]<>]', '_', c) for c in X.columns]

    return X, y


def evaluate_single_seed(
    X: pd.DataFrame, y: pd.Series, seed: int, metrics_list: list[str],
) -> dict[str, float]:
    """单次 80/20 split 评估。"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, recall_score, precision_score, f1_score,
    )
    import xgboost as xgb

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        enable_categorical=True,
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {}
    for metric in metrics_list:
        if metric == "auc":
            results["auc"] = roc_auc_score(y_test, y_prob)
        elif metric == "accuracy":
            results["accuracy"] = accuracy_score(y_test, y_pred)
        elif metric == "recall":
            results["recall"] = recall_score(y_test, y_pred)
        elif metric == "precision":
            results["precision"] = precision_score(y_test, y_pred)
        elif metric == "f1":
            results["f1"] = f1_score(y_test, y_pred)

    return results


def evaluate_dataset(
    dataset_name: str,
    method_name: str = "baseline",
    features_csv: str = None,
    feature_cols: list[str] = None,
    seeds: list[int] = None,
) -> dict:
    """完整评估：5 seeds × metrics，返回标准化结果。"""
    if seeds is None:
        seeds = SEEDS

    cfg = DATASET_CONFIG[dataset_name]
    metrics_list = cfg["metrics"]

    df, target = load_dataset(dataset_name, features_csv)
    X, y = prepare_features(df, target, feature_cols)

    logger.info(
        "评估 %s / %s: X=(%d, %d), y 正例率=%.2f%%",
        dataset_name, method_name, *X.shape, y.mean() * 100,
    )

    all_results = []
    for seed in seeds:
        result = evaluate_single_seed(X, y, seed, metrics_list)
        all_results.append(result)
        logger.info("  seed=%d: %s", seed, {k: f"{v:.4f}" for k, v in result.items()})

    summary = {}
    for metric in metrics_list:
        values = [r[metric] for r in all_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    output = {
        "dataset": dataset_name,
        "method": method_name,
        "n_features": X.shape[1],
        "n_samples": len(y),
        "positive_rate": float(y.mean()),
        "seeds": seeds,
        "metrics": summary,
    }

    for metric in metrics_list:
        m = summary[metric]
        logger.info(
            "  %s: %.4f ± %.4f", metric, m["mean"], m["std"],
        )

    return output


def significance_test(
    results_a: dict, results_b: dict, metric: str, alpha: float = 0.05,
) -> dict:
    """Welch's t-test 对比两种方法。"""
    vals_a = results_a["metrics"][metric]["values"]
    vals_b = results_b["metrics"][metric]["values"]

    t_stat, p_value = stats.ttest_ind(vals_a, vals_b, equal_var=False)

    return {
        "metric": metric,
        "method_a": results_a["method"],
        "method_b": results_b["method"],
        "mean_a": np.mean(vals_a),
        "mean_b": np.mean(vals_b),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha,
    }


def save_results(results: dict, output_dir: Path = None):
    """保存评估结果到 JSON。"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{results['dataset']}_{results['method']}.json"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("结果已保存: %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="统一评估框架")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--method", default="baseline", help="方法名称")
    parser.add_argument("--features-csv", default=None, help="额外特征 CSV 文件")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline", action="store_true", help="仅评估基线特征")
    args = parser.parse_args()

    if args.baseline:
        args.method = "No-FE"

    results = evaluate_dataset(
        dataset_name=args.dataset,
        method_name=args.method,
        features_csv=args.features_csv,
        seeds=args.seeds,
    )

    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
