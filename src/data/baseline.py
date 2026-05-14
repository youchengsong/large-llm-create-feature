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
    """轻量版基线特征提取，直接用 groupby 聚合代替 expanding window，避免 OOM。

    产出交易级别的特征矩阵（每行对应一笔交易），包含 ~15 个核心基线特征。
    """
    tx = tx_df.copy()
    tx["Timestamp"] = pd.to_datetime(tx["timestamp"])
    tx["is_laundering"] = tx["is_risk"].replace(
        {"False": 0, "True": 1, "0": 0, "1": 1, False: 0, True: 1}
    )

    tx["same_transactionaccount"] = (tx["src"] == tx["dst"]).astype(int)
    tx["Timestamp_Hour"] = tx["Timestamp"].dt.hour

    src_stats = tx.groupby("src")["amount"].agg(
        Out_Degree="count",
        Total_Sent="sum",
        Max_Sent="max",
        Min_Sent="min",
        Mean_Sent="mean",
        Std_Sent="std",
    ).fillna(0)
    src_stats["volatility_Sent"] = src_stats["Std_Sent"] / src_stats["Mean_Sent"].clip(lower=1e-9)
    src_stats["unique_outdegree"] = tx.groupby("src")["dst"].nunique()

    dst_stats = tx.groupby("dst")["amount"].agg(
        In_Degree="count",
        Total_Received="sum",
        Max_Received="max",
        Min_Received="min",
        Mean_Received="mean",
        Std_Received="std",
    ).fillna(0)
    dst_stats["volatility_Received"] = dst_stats["Std_Received"] / dst_stats["Mean_Received"].clip(lower=1e-9)
    dst_stats["unique_indegree"] = tx.groupby("dst")["src"].nunique()

    tx = tx.merge(src_stats, left_on="src", right_index=True, how="left")
    tx = tx.merge(dst_stats, left_on="dst", right_index=True, how="left")

    tx["From_To_Ratio"] = tx["Out_Degree"] / tx["In_Degree"].clip(lower=1)
    tx["unique_From_To_Ratio"] = tx["unique_outdegree"] / tx["unique_indegree"].clip(lower=1)
    tx["Receive_Send_Ratio"] = tx["Total_Received"] / tx["Total_Sent"].clip(lower=1)
    tx["Difference"] = tx["Total_Received"] - tx["Total_Sent"]
    tx["avg_max_sent_ratio"] = tx["Mean_Sent"] / (tx["Max_Sent"] + 1)
    tx["avg_max_receive_ratio"] = tx["Mean_Received"] / (tx["Max_Received"] + 1)
    tx["max_sent_receive_ratio"] = tx["Max_Sent"] / (tx["Max_Received"] + 1)
    tx["volatility_ratio"] = tx["volatility_Sent"] / (tx["volatility_Received"] + 1)

    tx["z_score_paid"] = (tx["amount"] - tx["Mean_Sent"]) / tx["Std_Sent"].clip(lower=1e-9)
    tx["z_score_received"] = (tx["amount"] - tx["Mean_Received"]) / tx["Std_Received"].clip(lower=1e-9)
    tx["large_paid"] = (tx["z_score_paid"].abs() > 3).astype(int)
    tx["large_received"] = (tx["z_score_received"].abs() > 3).astype(int)

    top1_sent = tx.groupby(["src", "dst"])["amount"].sum().groupby("src").max()
    total_sent = tx.groupby("src")["amount"].sum()
    send_conc = (top1_sent / total_sent.clip(lower=0.01)).rename("send_concentration")
    top1_recv = tx.groupby(["dst", "src"])["amount"].sum().groupby("dst").max()
    total_recv = tx.groupby("dst")["amount"].sum()
    recv_conc = (top1_recv / total_recv.clip(lower=0.01)).rename("receive_concentration")

    tx = tx.merge(send_conc, left_on="src", right_index=True, how="left")
    tx = tx.merge(recv_conc, left_on="dst", right_index=True, how="left")

    pair_count = tx.groupby(["src", "dst"])["tx_id"].transform("count")
    tx["transaction_count_directed"] = pair_count

    feature_cols = [
        "same_transactionaccount", "Timestamp_Hour",
        "transaction_count_directed",
        "send_concentration", "receive_concentration",
        "Out_Degree", "In_Degree", "From_To_Ratio",
        "unique_outdegree", "unique_indegree", "unique_From_To_Ratio",
        "Total_Sent", "Max_Sent", "Min_Sent", "Mean_Sent", "Std_Sent", "volatility_Sent",
        "Total_Received", "Max_Received", "Min_Received", "Mean_Received", "Std_Received", "volatility_Received",
        "Receive_Send_Ratio", "Difference",
        "avg_max_sent_ratio", "avg_max_receive_ratio", "max_sent_receive_ratio", "volatility_ratio",
        "z_score_paid", "z_score_received", "large_paid", "large_received",
        "is_laundering",
    ]
    available = [c for c in feature_cols if c in tx.columns]
    return tx[available].fillna(0).reset_index(drop=True)


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
