import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

from src.config import DEFAULT_CONFIG


def run_baseline(
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
) -> pd.DataFrame:
    """调用 inputs/baseline/feature_engineering.py 的 feature_eng() 生成人工基线特征矩阵。

    将三个 DataFrame 写入临时 CSV，调用 feature_eng() 后读取结果返回。
    """
    import importlib.util

    baseline_path = Path(DEFAULT_CONFIG.paths.baseline_dir) / "feature_engineering.py"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline script not found: {baseline_path}")

    spec = importlib.util.spec_from_file_location("baseline_fe", str(baseline_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with tempfile.TemporaryDirectory() as tmpdir:
        tx_path = os.path.join(tmpdir, "transactions.csv")
        open_path = os.path.join(tmpdir, "accounts.csv")
        close_path_tmp = os.path.join(tmpdir, "close.csv")
        save_path = os.path.join(tmpdir, "features.csv")

        tx_df.to_csv(tx_path, index=False)
        open_df.to_csv(open_path, index=False)
        close_df.to_csv(close_path_tmp, index=False)

        mod.feature_eng(
            data_path=tx_path,
            account_path=open_path,
            save_path=save_path,
            close_path=close_path_tmp,
        )

        result = pd.read_csv(save_path)
    return result


_BASELINE_FEATURE_NAMES: list[str] = [
    "same_transactionaccount",
    "Timestamp_Hour",
    "transaction_count_directed",
    "transaction_count_Undirected",
    "accounttransaction_timeinterval",
    "transaction_density_Undirected",
    "send_concentration",
    "receive_concentration",
    "send_amount_concentration",
    "receive_amount_concentration",
    "counts_1H_From",
    "counts_1H_To",
    "1H_frequency_From",
    "1H_frequency_To",
    "Out_Degree",
    "In_Degree",
    "From_To_Ratio",
    "unique_outdegree",
    "unique_indegree",
    "unique_From_To_Ratio",
    "Total_Sent",
    "Max_Sent",
    "Min_Sent",
    "Mean_Sent",
    "Std_Sent",
    "volatility_Sent",
    "Total_Received",
    "Max_Received",
    "Min_Received",
    "Mean_Received",
    "Std_Received",
    "volatility_Received",
    "Receive_Send_Ratio",
    "Difference",
    "avg_max_sent_ratio",
    "avg_max_receive_ratio",
    "max_sent_receive_ratio",
    "volatility_ratio",
    "z_score_paid",
    "z_score_received",
    "large_paid",
    "large_received",
    "large_transaction_ratio",
    "activetime",
    "transaction_frequency",
    "last_direction",
    "last_amount",
    "avg_payment_time_diff",
    "avg_receipt_time_diff",
    "avg_inactive_period",
    "max_inactive_period",
    "min_inactive_period",
]


def get_baseline_feature_names() -> list[str]:
    """返回 40+ 人工特征名称列表。"""
    return list(_BASELINE_FEATURE_NAMES)


def compute_baseline_prauc(X_baseline: pd.DataFrame, y: pd.Series) -> float:
    """用 XGBoost 5 折分层 CV 计算基线 PR-AUC。"""
    cfg = DEFAULT_CONFIG.xgboost

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    scores: list[float] = []

    X = X_baseline.values if isinstance(X_baseline, pd.DataFrame) else np.asarray(X_baseline)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    for train_idx, val_idx in skf.split(X, y_arr):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        model = XGBClassifier(
            scale_pos_weight=cfg.scale_pos_weight,
            eval_metric=cfg.eval_metric,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        model.fit(X_train, y_train, verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, y_prob))

    return float(np.mean(scores))
