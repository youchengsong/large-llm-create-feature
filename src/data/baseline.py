import tempfile
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

from src.config import DEFAULT_CONFIG

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_baseline(
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
) -> pd.DataFrame:
    """调用完整特征工程流水线（Expanding Window），返回交易级特征矩阵。"""
    fe_path = _PROJECT_ROOT / "inputs" / "baseline" / "feature_engineering.py"
    if not fe_path.exists():
        raise FileNotFoundError(f"完整特征工程脚本不存在: {fe_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("feature_engineering", str(fe_path))
    fe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_module)

    with tempfile.TemporaryDirectory() as tmpdir:
        tx_path = os.path.join(tmpdir, "transactions.csv")
        open_path = os.path.join(tmpdir, "accounts.csv")
        close_path = os.path.join(tmpdir, "close.csv")
        out_path = os.path.join(tmpdir, "features.csv")

        tx_df.to_csv(tx_path, index=False)
        open_df.to_csv(open_path, index=False)
        close_df.to_csv(close_path, index=False)

        fe_module.feature_eng(
            data_path=tx_path,
            account_path=open_path,
            save_path=out_path,
            close_path=close_path,
        )

        original_path = os.path.join(
            tmpdir,
            os.path.splitext("features.csv")[0] + "_original.csv",
        )
        if os.path.exists(original_path):
            result_df = pd.read_csv(original_path)
        else:
            result_df = pd.read_csv(out_path)

    non_feature_cols = {
        "id", "amount", "is_laundering", "account_is_laundering_From",
        "account_is_laundering_To", "src", "dst", "Timestamp",
    }

    if "is_laundering" not in result_df.columns:
        if "is_risk" in tx_df.columns:
            label = tx_df["is_risk"].replace(
                {"False": 0, "True": 1, "0": 0, "1": 1, False: 0, True: 1}
            ).astype(int)
            if len(label) == len(result_df):
                result_df["is_laundering"] = label.values

    feature_cols = [c for c in result_df.columns if c not in non_feature_cols]
    out = result_df[feature_cols].fillna(0).reset_index(drop=True)

    if "is_laundering" in result_df.columns:
        out["is_laundering"] = result_df["is_laundering"].astype(int).values

    return out


def get_baseline_feature_names() -> list[str]:
    """返回基线特征名称列表（以实际 feature_engineering.py 输出为准）。"""
    return list(_BASELINE_FEATURE_NAMES)


_BASELINE_FEATURE_NAMES: list[str] = [
    "src_bank_account_number",
    "dst_bank_account_number",
    "src_wallet_level",
    "dst_wallet_level",
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
    "large_transaction_ratio_paid",
    "large_transaction_ratio_received",
    "large_transaction_ratio",
    "init_balance_From",
    "wallet_level_From",
    "wallet_type_From",
    "init_balance_To",
    "wallet_level_To",
    "wallet_type_To",
    "S_src_history_count",
    "S_src_geo_risk_202308",
    "S_src_time_risk_202308",
    "S_dst_history_count",
    "S_dst_geo_risk_202308",
    "S_dst_time_risk_202308",
]


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
            tree_method=cfg.tree_method,
            device=cfg.device,
        )
        model.fit(X_train, y_train, verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, y_prob))

    return float(np.mean(scores))
