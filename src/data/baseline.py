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


_BASELINE_CODE_SAMPLES: dict[str, str] = {
    "Out_Degree": "tx_df.groupby('src')['tx_id'].count().reset_index().rename(columns={'tx_id':'Out_Degree','src':'wallet_id'})",
    "In_Degree": "tx_df.groupby('dst')['tx_id'].count().reset_index().rename(columns={'tx_id':'In_Degree','dst':'wallet_id'})",
    "From_To_Ratio": "out=tx_df.groupby('src')['tx_id'].count(); inp=tx_df.groupby('dst')['tx_id'].count(); ratio=(out/inp.reindex(out.index).fillna(1)).fillna(0)",
    "Total_Sent": "tx_df.groupby('src')['amount'].sum()",
    "Mean_Sent": "tx_df.groupby('src')['amount'].mean()",
    "Std_Sent": "tx_df.groupby('src')['amount'].std().fillna(0)",
    "Max_Sent": "tx_df.groupby('src')['amount'].max()",
    "Total_Received": "tx_df.groupby('dst')['amount'].sum()",
    "Receive_Send_Ratio": "received=tx_df.groupby('dst')['amount'].sum(); sent=tx_df.groupby('src')['amount'].sum(); ratio=received/sent.reindex(received.index).clip(lower=0.01)",
    "Timestamp_Hour": "tx_df['hour']  # 直接取交易小时",
    "large_transaction_ratio": "large=(tx_df['amount']>10000); tx_df.groupby('src')[large].mean()",
    "transaction_frequency": "tx_count=tx_df.groupby('src')['tx_id'].count(); active_days=tx_df.groupby('src')['timestamp'].nunique(); freq=tx_count/active_days.clip(lower=1)",
    "send_concentration": "top1=tx_df.groupby(['src','dst'])['amount'].sum().groupby('src').max(); total=tx_df.groupby('src')['amount'].sum(); conc=top1/total.clip(lower=0.01)",
    "avg_inactive_period": "tx_df.sort_values('timestamp').groupby('src')['interval'].mean()",
    "max_inactive_period": "tx_df.sort_values('timestamp').groupby('src')['interval'].max()",
}


def get_baseline_feature_code_samples() -> str:
    """返回基线特征代码摘要文本，供 System Prompt 注入。"""
    lines = ["以下是部分人工特征的实现逻辑（参考，不要重复，可以组合改进）：\n"]
    for name, code in _BASELINE_CODE_SAMPLES.items():
        lines.append(f"- **{name}**: `{code}`")
    return "\n".join(lines)


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
