"""无泄漏最终评估（回应 ICDM feature-selection leakage 质疑）。

与旧 ``scripts/evaluate.py`` 的关键区别：
  旧协议：对全量数据做 5-seed 的 80/20 split，特征选择(L3)与最终指标共用同一批
          split → selection-on-test，指标乐观偏高；且“统一协议”把所有 baseline 也
          一并带进这个偏差。
  新协议：一个**冻结**的 dev/test 划分（knowfeat/data/split.py，全局 seed=42）。
          · 特征由 KnowFeat/baseline 在 D_dev 上生成与选择（本脚本不参与生成）。
          · 这里：在 D_dev 上训练 XGBoost，仅在从未参与任何决策的 D_test 上评分。
          · 用 bootstrap 重采样报告 test 指标的均值±方差（reviewer 要 uncertainty）。

特征以“可复现的确定性变换代码/列”提供：本脚本接收已算好的特征 CSV（整表 n 行，
与原始数据行对齐），按冻结划分切成 dev/test，绝不使用 test 的标签做任何选择。

用法::

    python scripts/evaluate_holdout.py --dataset simecny \
        --features-csv outputs/simecny_indicators/simecny_knowfeat_features.csv \
        --method KnowFeat

    python scripts/evaluate_holdout.py --dataset credit_g --baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowfeat.data.split import get_or_make_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 训练用的模型 seeds：仅影响“在固定 dev/test 上重复训练取平均”，
# 不再用于划分测试集，因此与旧协议的同名 seeds 语义完全不同。
MODEL_SEEDS = [42, 123, 456, 789, 2024]
N_BOOTSTRAP = 1000  # test 集 bootstrap 重采样次数

# 与旧 evaluate.py 保持一致的列排除表（标签/ID/已知泄漏特征）。
EXCLUDE_COLS = {
    "row_id", "wallet_id", "tx_id", "tx_id_fe", "account", "src", "dst",
    "is_laundering", "is_risk", "label", "is_illicit",
    "Min_Received", "Min_Sent", "S_src_counterparty_index", "Timestamp_Hour",
    "S_dst_counterparty_index", "1H_frequency_From", "1H_frequency_To",
    "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent",
}

DATASET_CONFIG = {
    "credit_g": {"csv": "data/public/credit_g.csv", "target": "class",
                 "metrics": ["auc", "accuracy"]},
    "diabetes": {"csv": "data/public/diabetes.csv", "target": "class",
                 "metrics": ["auc", "accuracy"]},
    "blood_transfusion": {"csv": "data/public/blood_transfusion.csv", "target": "Class",
                          "metrics": ["auc", "accuracy"]},
    "heart": {"csv": "data/public/heart.csv", "target": "HeartDisease",
              "metrics": ["auc", "accuracy"]},
    "bank_marketing": {"csv": "data/public/bank_marketing.csv", "target": "y",
                       "metrics": ["auc", "accuracy"]},
    "breast_w": {"csv": "data/public/breast_w.csv", "target": "Class",
                 "metrics": ["auc", "accuracy"]},
    "adult": {"csv": "data/public/adult.csv", "target": "class",
              "metrics": ["auc", "accuracy"]},
    "wine_quality": {"csv": "data/public/wine_quality.csv", "target": "class",
                     "metrics": ["auc", "accuracy"]},
    "default_credit": {"csv": "data/public/default_credit.csv", "target": "target",
                       "metrics": ["auc", "accuracy"]},
    "churn": {"csv": "data/public/churn.csv", "target": "target",
              "metrics": ["auc", "accuracy"]},
    "kc2": {"csv": "data/public/kc2.csv", "target": "target",
            "metrics": ["auc", "accuracy"]},
    "jm1": {"csv": "data/public/jm1.csv", "target": "target",
            "metrics": ["auc", "accuracy"]},
    "simecny": {"csv": None, "target": "is_risk",
                "metrics": ["recall", "precision", "f1", "auc"]},
    "elliptic": {"csv": "data/elliptic/elliptic_labeled.csv", "target": "label",
                 "metrics": ["recall", "precision", "f1", "auc"]},
}


def load_dataset(dataset_name: str, features_csv: str | None = None) -> tuple[pd.DataFrame, str]:
    """加载原始表并按行对齐拼接外部特征（不做任何 split）。"""
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
            raise FileNotFoundError(f"数据集不存在: {csv_path}")
        df = pd.read_csv(csv_path)

    if features_csv:
        feat_df = pd.read_csv(features_csv)
        if len(feat_df) != len(df):
            raise ValueError(
                f"特征 CSV 行数 {len(feat_df)} != 原始数据 {len(df)}，无法按行对齐。"
            )
        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

    return df, target


def prepare_xy(
    df: pd.DataFrame, target: str, feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """构造特征矩阵与标签（数值化 + 缺失填充），不做 split。"""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != target]

    X = df[feature_cols].copy()
    y = df[target]
    if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
        from sklearn.preprocessing import LabelEncoder
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
    y = y.astype(int).reset_index(drop=True)

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes
    X = X.fillna(-999).reset_index(drop=True)
    X.columns = [re.sub(r"[\[\]<>]", "_", str(c)) for c in X.columns]
    return X, y


def _fit_predict(X_dev, y_dev, X_test, seed, spw_override):
    """在 dev 上训练，返回 test 上的预测概率与硬预测。"""
    import xgboost as xgb

    if spw_override is not None:
        spw = spw_override
    else:
        spw = float((y_dev == 0).sum() / max((y_dev == 1).sum(), 1))
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        random_state=seed, eval_metric="logloss",
        enable_categorical=True, verbosity=0,
    )
    model.fit(X_dev, y_dev, verbose=False)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_prob, y_pred


def _metric_values(y_true, y_prob, y_pred, metrics_list):
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, recall_score, precision_score, f1_score,
    )
    out = {}
    for m in metrics_list:
        if m == "auc":
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        elif m == "accuracy":
            out["accuracy"] = float(accuracy_score(y_true, y_pred))
        elif m == "recall":
            out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        elif m == "precision":
            out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        elif m == "f1":
            out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out


def evaluate_holdout(
    dataset_name: str,
    method_name: str = "No-FE",
    features_csv: str | None = None,
    spw_override: float | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """在冻结 test 集上评估，bootstrap 报告不确定度。"""
    cfg = DATASET_CONFIG[dataset_name]
    metrics_list = cfg["metrics"]

    df, target = load_dataset(dataset_name, features_csv)
    X, y = prepare_xy(df, target)

    # 冻结划分：按数据集缓存一份，KnowFeat 生成侧与本评估侧共享同一 test 集。
    split_cache = PROJECT_ROOT / "outputs" / "splits" / f"{dataset_name}_split.json"
    split = get_or_make_split(len(y), labels=y, cache_path=split_cache)

    X_dev, y_dev = X.iloc[split.dev_idx], y.iloc[split.dev_idx]
    X_test, y_test = X.iloc[split.test_idx], y.iloc[split.test_idx]
    logger.info(
        "评估 %s / %s: dev=%d, test=%d, #feat=%d, test 正例率=%.3f%%, spw=%s",
        dataset_name, method_name, len(y_dev), len(y_test), X.shape[1],
        100 * y_test.mean(), spw_override if spw_override is not None else "auto",
    )

    # 多个 model seed 训练，test 上的预测概率取平均（降低单次训练随机性）。
    probs = []
    point_estimates = []
    for seed in MODEL_SEEDS:
        y_prob, y_pred = _fit_predict(X_dev, y_dev, X_test, seed, spw_override)
        probs.append(y_prob)
        point_estimates.append(_metric_values(y_test.values, y_prob, y_pred, metrics_list))
    mean_prob = np.mean(probs, axis=0)
    mean_pred = (mean_prob >= 0.5).astype(int)

    # bootstrap：对冻结 test 集有放回重采样，得到指标分布的均值±std。
    rng = np.random.RandomState(12345)
    yt = y_test.values
    n = len(yt)
    boot = {m: [] for m in metrics_list}
    for _ in range(n_bootstrap):
        bi = rng.randint(0, n, size=n)
        if yt[bi].sum() == 0 or yt[bi].sum() == len(bi):
            continue  # 跳过退化重采样（AUC 无定义）
        bm = _metric_values(yt[bi], mean_prob[bi], mean_pred[bi], metrics_list)
        for m in metrics_list:
            boot[m].append(bm[m])

    summary = {}
    for m in metrics_list:
        seed_vals = [pe[m] for pe in point_estimates]
        summary[m] = {
            "test_point": float(np.mean(seed_vals)),         # 冻结 test 上 seed 平均
            "test_seed_std": float(np.std(seed_vals)),       # 跨 model-seed 波动
            "bootstrap_mean": float(np.mean(boot[m])) if boot[m] else None,
            "bootstrap_std": float(np.std(boot[m])) if boot[m] else None,
            "bootstrap_ci95": (
                [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))]
                if boot[m] else None
            ),
        }
        s = summary[m]
        logger.info(
            "  %s: test=%.4f ±%.4f (seed) | bootstrap=%.4f ±%.4f, CI95=[%.4f, %.4f]",
            m, s["test_point"], s["test_seed_std"],
            s["bootstrap_mean"] or 0, s["bootstrap_std"] or 0,
            *(s["bootstrap_ci95"] or [0, 0]),
        )

    return {
        "dataset": dataset_name,
        "method": method_name,
        "protocol": "frozen_holdout_bootstrap",
        "split_seed": split.seed,
        "test_size": split.test_size,
        "n_dev": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "test_positive_rate": float(y_test.mean()),
        "model_seeds": MODEL_SEEDS,
        "n_bootstrap": n_bootstrap,
        "metrics": summary,
    }


def save_results(results: dict, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "eval_holdout"
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{results['dataset']}_{results['method']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存: %s", out)
    return out


def main():
    p = argparse.ArgumentParser(description="无泄漏冻结-holdout 评估")
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    p.add_argument("--method", default="No-FE")
    p.add_argument("--features-csv", default=None)
    p.add_argument("--baseline", action="store_true", help="仅评估基线特征")
    p.add_argument("--spw", type=float, default=None)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if args.baseline:
        args.method = "No-FE"

    results = evaluate_holdout(
        dataset_name=args.dataset,
        method_name=args.method,
        features_csv=args.features_csv,
        spw_override=args.spw,
        n_bootstrap=args.n_bootstrap,
    )
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
