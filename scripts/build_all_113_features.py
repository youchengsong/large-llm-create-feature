"""
构建 113 个特征矩阵的完全自包含脚本。

流程:
  1. 从原始 CSV 计算全部基线特征（来自 feature_engineering.py 的完整流水线）
  2. 删除 10 个低质量基线特征
  3. 从原始三表计算 18 个 KnowFeat 钱包级特征，通过 src 列 join 到交易级
  4. 输出最终 113 列特征矩阵

用法:
  python build_all_113_features.py <交易CSV> <账户CSV> <注销CSV> <输出CSV> [账户特征CSV]

依赖: pandas, numpy, sklearn (MinMaxScaler), 标准库 (sys, os, time, math, warnings, json)
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import math
import os
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

# ═══════════════════════════════════════════════════════════════════════════════
# 10 个需要从基线中删除的低质量特征
# ═══════════════════════════════════════════════════════════════════════════════
LOW_QUALITY_FEATURES = [
    "Min_Received", "Min_Sent", "S_src_counterparty_index",
    "Timestamp_Hour", "S_dst_counterparty_index",
    "1H_frequency_From", "1H_frequency_To",
    "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 基线特征工程 — 完整代码（来自 feature_engineering.py）
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_WALLET_LEVEL_LIMITS = {
    "1": {"single_transaction_limit": 50000.0, "daily_limit": 100000.0},
    "2": {"single_transaction_limit": 10000.0, "daily_limit": 50000.0},
    "3": {"single_transaction_limit": 2000.0, "daily_limit": 5000.0},
    "4": {"single_transaction_limit": 2000.0, "daily_limit": 5000.0},
}


def _load_wallet_level_limits(transaction_config_path):
    """从 transaction_config.json 读取 wallet_level_limits，缺省用内置表。"""
    import json

    limits = {k: dict(v) for k, v in DEFAULT_WALLET_LEVEL_LIMITS.items()}
    if not transaction_config_path or not os.path.exists(transaction_config_path):
        return limits
    try:
        with open(transaction_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        wl = cfg.get("wallet_level_limits")
        if not isinstance(wl, dict):
            return limits
        for k, v in wl.items():
            if not isinstance(v, dict):
                continue
            try:
                ks = str(int(float(k)))
            except (TypeError, ValueError):
                ks = str(k).strip()
            if "single_transaction_limit" not in v:
                continue
            limits[ks] = {
                "single_transaction_limit": float(v["single_transaction_limit"]),
                "daily_limit": float(v.get("daily_limit", 0.0)),
            }
    except Exception as e:
        print(f"  警告: 读取 wallet_level_limits 失败，使用默认值: {e}")
    return limits


def _norm_wallet_level(val):
    try:
        return str(int(float(val)))
    except (TypeError, ValueError):
        return str(val).strip() if val is not None and not (isinstance(val, float) and np.isnan(val)) else "1"


def _inst_code_from_bank(b):
    """机构标识：用 bank_account_number 字符串（如 6218、6221）。"""
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return ""
    s = str(b).strip()
    return s


def _compute_s_transaction_behavior(df_quantized, account_df, transaction_config_path=None, window_hours=24):
    """
    在最近 window_hours 小时交易子集上，为每个 wallet_id 计算：
    S_ibank_index, S_ibank_amount_ratio, S_counterparty_index,
    S_per_deal_match_times, S_daily_deal_match_times
    """
    limits = _load_wallet_level_limits(transaction_config_path)
    acc = account_df.copy()
    if "wallet_id" not in acc.columns:
        if "acct_id" in acc.columns:
            acc["wallet_id"] = acc["acct_id"].astype(str)
        else:
            return pd.DataFrame()
    acc["wallet_id"] = acc["wallet_id"].astype(str)

    if "bank_account_number" not in acc.columns:
        acc["bank_account_number"] = ""
    if "wallet_level" not in acc.columns:
        acc["wallet_level"] = 1

    bank_map = acc.set_index("wallet_id")["bank_account_number"].to_dict()
    level_map = acc.set_index("wallet_id")["wallet_level"].to_dict()

    df = df_quantized.copy()
    if "Timestamp" not in df.columns:
        return pd.DataFrame()
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    if ts.notna().sum() == 0:
        return pd.DataFrame()
    tmax = ts.max()
    t0 = tmax - pd.Timedelta(hours=window_hours)
    m = ts >= t0
    df = df.loc[m].copy()
    df["ts"] = ts.loc[m]
    if "From_Account" not in df.columns or "To_Account" not in df.columns:
        return pd.DataFrame()

    if "amount" in df.columns:
        amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    elif "Amount Paid" in df.columns:
        amt = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0.0)
    else:
        amt = pd.Series(0.0, index=df.index)

    F = df["From_Account"].astype(str)
    T = df["To_Account"].astype(str)
    inst_f = F.map(lambda w: _inst_code_from_bank(bank_map.get(w, "")))
    inst_t = T.map(lambda w: _inst_code_from_bank(bank_map.get(w, "")))
    cross = (inst_f != "") & (inst_t != "") & (inst_f != inst_t)

    out_ev = pd.DataFrame(
        {
            "wallet": F.values,
            "cp": T.values,
            "amt": amt.values,
            "cross": cross.values,
            "ts": df["ts"].values,
        }
    )
    in_ev = pd.DataFrame(
        {
            "wallet": T.values,
            "cp": F.values,
            "amt": amt.values,
            "cross": cross.values,
            "ts": df["ts"].values,
        }
    )
    ev = pd.concat([out_ev, in_ev], ignore_index=True)

    def single_half(w):
        wd = str(w)
        raw_lv = level_map.get(wd, 1)
        lv = _norm_wallet_level(raw_lv)
        row = limits.get(lv) or limits.get("1", DEFAULT_WALLET_LEVEL_LIMITS["1"])
        return 0.5 * float(row["single_transaction_limit"])

    def daily_half(w):
        wd = str(w)
        raw_lv = level_map.get(wd, 1)
        lv = _norm_wallet_level(raw_lv)
        row = limits.get(lv) or limits.get("1", DEFAULT_WALLET_LEVEL_LIMITS["1"])
        return 0.5 * float(row["daily_limit"])

    # --- 聚合 ---
    g = ev.groupby("wallet", sort=False)
    total_n = g.size()
    cross_n = ev.groupby("wallet", sort=False)["cross"].sum()
    total_amt = g["amt"].sum()
    cross_amt = ev.loc[ev["cross"]].groupby("wallet", sort=False)["amt"].sum()
    cross_amt = cross_amt.reindex(total_n.index).fillna(0.0)

    S_ibank_index = (cross_n / total_n.replace(0, np.nan)).fillna(0.0).astype(float)
    S_ibank_amount_ratio = (cross_amt / total_amt.replace(0, np.nan)).fillna(0.0).astype(float)

    n_cp = g["cp"].nunique()
    avg_cp = float(n_cp.mean()) if len(n_cp) > 0 else 0.0
    if avg_cp > 0:
        S_counterparty_index = (n_cp.astype(float) / avg_cp).fillna(0.0)
    else:
        S_counterparty_index = n_cp * 0.0

    ev["_thr_single"] = ev["wallet"].map(single_half)
    ev["_hit_single"] = (ev["amt"] >= ev["_thr_single"]).astype(int)
    S_per_deal_match_times = ev.groupby("wallet", sort=False)["_hit_single"].sum().astype(float)

    out_only = out_ev.copy()
    out_only["d"] = pd.to_datetime(out_only["ts"]).dt.normalize()
    daily_sum = out_only.groupby(["wallet", "d"], sort=False)["amt"].sum().reset_index()
    daily_sum["thr_d"] = daily_sum["wallet"].map(daily_half)
    daily_sum["hit_day"] = (daily_sum["amt"] >= daily_sum["thr_d"]).astype(int)
    S_daily_deal_match_times = daily_sum.groupby("wallet", sort=False)["hit_day"].sum().astype(float)

    out = pd.DataFrame(
        {
            "wallet_id": total_n.index.astype(str),
            "S_ibank_index": S_ibank_index.reindex(total_n.index).fillna(0.0).values,
            "S_ibank_amount_ratio": S_ibank_amount_ratio.reindex(total_n.index).fillna(0.0).values,
            "S_counterparty_index": S_counterparty_index.reindex(total_n.index).fillna(0.0).values,
            "S_per_deal_match_times": S_per_deal_match_times.reindex(total_n.index).fillna(0.0).values,
            "S_daily_deal_match_times": S_daily_deal_match_times.reindex(total_n.index).fillna(0.0).values,
        }
    )
    return out


def calculate_s_indicators(df_quantized, account_df, transaction_config_path=None):
    """
    计算账户 S 指标：
      - S_history_count / S_geo_risk / S_time_risk：来自账户开立表。
      - S_ibank_index / S_ibank_amount_ratio / S_counterparty_index / S_per_deal_match_times /
        S_daily_deal_match_times：来自全量交易表中「全局最近 24 小时」子窗口。
    """
    import json

    print(
        "    计算S指标（开户静态 + 最近24h交易行为: ibank / counterparty / 限额笔数等）..."
    )

    account_df = account_df.copy()

    if 'wallet_open_timestamp' not in account_df.columns:
        if 'wallet_open_date' in account_df.columns:
            account_df['wallet_open_timestamp'] = pd.to_datetime(
                account_df['wallet_open_date'], errors='coerce'
            )
        else:
            print("  警告: 账户开立数据中没有开立时间信息，无法计算 S_time_risk / S_history_count")
            return pd.DataFrame(columns=['wallet_id'])
    else:
        account_df['wallet_open_timestamp'] = pd.to_datetime(
            account_df['wallet_open_timestamp'], errors='coerce'
        )

    if 'wallet_open_tel' not in account_df.columns:
        print("  警告: 账户开立数据中没有手机号信息，S_history_count 将置 0")
        account_df['wallet_open_tel'] = ''

    if 'region_code' not in account_df.columns:
        print("  警告: 账户开立数据中没有地域代码信息，S_geo_risk 将置 0")
        account_df['region_code'] = 0

    if 'wallet_id' not in account_df.columns:
        if 'acct_id' in account_df.columns:
            account_df['wallet_id'] = account_df['acct_id'].astype(str)
        else:
            print("  警告: 账户开立数据中没有 wallet_id / acct_id，无法计算 S 指标")
            return pd.DataFrame(columns=['wallet_id'])
    else:
        account_df['wallet_id'] = account_df['wallet_id'].astype(str)

    high_risk_codes = ["540000", "650000"]
    if transaction_config_path and os.path.exists(transaction_config_path):
        try:
            with open(transaction_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'region_code' in config and 'high_risk_codes' in config['region_code']:
                    high_risk_codes = [str(c) for c in config['region_code']['high_risk_codes']]
                elif 'distribution_based_generation' in config:
                    nd = config['distribution_based_generation'].get('normal_wallet_distributions', {})
                    rc = nd.get('region_code', {})
                    if 'high_risk_codes' in rc:
                        high_risk_codes = [str(c) for c in rc['high_risk_codes']]
        except Exception as e:
            print(f"  警告: 读取高风险地区代码失败，使用默认值: {e}")
    high_risk_codes_set = set(str(c) for c in high_risk_codes)

    night_hours = {22, 23, 0, 1, 2, 3, 4, 5}

    account_df_sorted = account_df.sort_values('wallet_open_timestamp').copy()
    account_df_sorted['wallet_open_tel'] = account_df_sorted['wallet_open_tel'].astype(str)

    rows = []
    for _, row in account_df_sorted.iterrows():
        wid = str(row['wallet_id'])
        open_time = row['wallet_open_timestamp']
        phone = str(row.get('wallet_open_tel', ''))
        region_code = str(row.get('region_code', '0'))

        if pd.isna(open_time):
            S_history_count = 0
            S_geo = 0.0
            S_time = 0.0
        else:
            if phone and phone != 'nan':
                hist = account_df_sorted[
                    (account_df_sorted['wallet_open_tel'] == phone)
                    & (account_df_sorted['wallet_open_timestamp'] < open_time)
                ]
                S_history_count = len(hist)
            else:
                S_history_count = 0

            S_geo = 1.0 if region_code in high_risk_codes_set else 0.0

            hour = pd.to_datetime(open_time).hour
            S_time = 1.0 if hour in night_hours else 0.0

        rows.append(
            {
                "wallet_id": wid,
                "S_history_count": S_history_count,
                "S_geo_risk": S_geo,
                "S_time_risk": S_time,
            }
        )

    base = pd.DataFrame(rows)
    tx_metrics = _compute_s_transaction_behavior(
        df_quantized, account_df, transaction_config_path=transaction_config_path
    )
    extra_cols = [
        "S_ibank_index",
        "S_ibank_amount_ratio",
        "S_counterparty_index",
        "S_per_deal_match_times",
        "S_daily_deal_match_times",
    ]
    if tx_metrics is not None and len(tx_metrics) > 0:
        base = base.merge(tx_metrics, on="wallet_id", how="left")
    for c in extra_cols:
        if c not in base.columns:
            base[c] = 0.0
        else:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

    return base


def _resolve_transaction_config_path(data_path, account_path):
    """在交易或账户 CSV 同目录下查找 transaction_config.json。"""
    for d in (os.path.dirname(data_path or ""), os.path.dirname(account_path or "")):
        if not d:
            continue
        p = os.path.join(d, "transaction_config.json")
        if os.path.exists(p):
            return p
    return None


def compute_account_open_features_from_raw(account_df):
    """
    由 accounts 原始字段就地推导开立衍生特征：
    - open_hour / is_work_hour / is_night_hour
    - auth_delay_days
    - wallets_per_cert
    - days_to_expire_from_current
    - tel_reuse_count
    - log_balance
    """
    if "wallet_open_timestamp" in account_df.columns:
        open_ts = pd.to_datetime(account_df["wallet_open_timestamp"], errors="coerce")
        hour = open_ts.dt.hour
        account_df["open_hour"] = hour.fillna(0).astype(np.int64)
        m = hour.notna()
        account_df["is_work_hour"] = 0
        account_df["is_night_hour"] = 0
        work = (hour >= 9) & (hour < 18)
        account_df.loc[m, "is_work_hour"] = work[m].astype(np.int64)
        account_df.loc[m, "is_night_hour"] = (~work[m]).astype(np.int64)
    else:
        account_df["open_hour"] = 0
        account_df["is_work_hour"] = 0
        account_df["is_night_hour"] = 0

    if "realname_auth_time" in account_df.columns and "wallet_open_timestamp" in account_df.columns:
        auth = pd.to_datetime(account_df["realname_auth_time"], errors="coerce")
        open_ts = pd.to_datetime(account_df["wallet_open_timestamp"], errors="coerce")
        delay_days = (auth - open_ts).dt.total_seconds() / 86400.0
        account_df["auth_delay_days"] = pd.to_numeric(delay_days, errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        account_df["auth_delay_days"] = 0.0

    if "wallet_open_cert" in account_df.columns and "wallet_id" in account_df.columns:
        cert_key = account_df["wallet_open_cert"].fillna("").astype(str)
        account_df["wallets_per_cert"] = account_df.groupby(cert_key, sort=False)["wallet_id"].transform("count")
        account_df["wallets_per_cert"] = pd.to_numeric(account_df["wallets_per_cert"], errors="coerce").fillna(0).astype(np.int64)
    else:
        account_df["wallets_per_cert"] = 0

    if "wallet_open_cert_expire" in account_df.columns:
        exp = pd.to_datetime(account_df["wallet_open_cert_expire"], errors="coerce")
        today = pd.Timestamp.now(tz=None).normalize()
        account_df["days_to_expire_from_current"] = (exp - today).dt.days.fillna(0).astype(np.int64)
    else:
        account_df["days_to_expire_from_current"] = 0

    if "wallet_open_tel" in account_df.columns and "wallet_id" in account_df.columns:
        tel_key = account_df["wallet_open_tel"].fillna("").astype(str).str.strip()
        nonempty = tel_key.ne("")
        cnt = account_df.groupby(tel_key, sort=False)["wallet_id"].transform("count")
        account_df["tel_reuse_count"] = pd.to_numeric(cnt, errors="coerce").fillna(0).astype(np.int64)
        account_df.loc[~nonempty, "tel_reuse_count"] = 0
    else:
        account_df["tel_reuse_count"] = 0

    if "init_balance" in account_df.columns:
        bal = pd.to_numeric(account_df["init_balance"], errors="coerce").fillna(0.0)
        account_df["log_balance"] = np.where(bal > 0, np.log(bal), 0.0)
    else:
        account_df["log_balance"] = 0.0

    print(
        "  已从账户表原始字段计算: open_hour, is_work_hour, is_night_hour, auth_delay_days, "
        "wallets_per_cert, days_to_expire_from_current, tel_reuse_count, log_balance"
    )


def feature_eng(data_path, account_path, save_path, close_path=None, account_features_path=None):
    """
    特征工程主函数 -- 计算基线特征 + 删除低质量特征 + 追加 18 个 KnowFeat 特征。

    参数:
    - data_path: 交易数据路径
    - account_path: 账户开立数据路径
    - save_path: 归一化后的特征保存路径
    - close_path: 注销数据路径（可选）
    - account_features_path: 可选外部账户特征 CSV
    """
    print("正在进行特征工程（使用Rolling/Expanding Window，避免数据泄露）...")
    df = pd.read_csv(data_path)

    # 适配新的数据格式
    df['Timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_laundering'] = df['is_risk'].replace({'False': 0, 'True': 1, '0': 0, '1': 1, False: 0, True: 1})
    df['From_Account'] = df['src']
    df['To_Account'] = df['dst']

    # 创建id索引（使用tx_id作为id）
    id_index = df['tx_id'].copy()
    id_index.name = 'id'

    # 按时间排序（关键：确保时间顺序）
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # 保留原始列以便后续使用
    df_quantized = df[['Timestamp', 'From_Account', 'To_Account', 'amount']].copy()
    if 'src_bank_account_number' in df.columns:
        df_quantized['src_bank_account_number'] = df['src_bank_account_number']
    if 'dst_bank_account_number' in df.columns:
        df_quantized['dst_bank_account_number'] = df['dst_bank_account_number']
    if 'src_wallet_level' in df.columns:
        df_quantized['src_wallet_level'] = df['src_wallet_level']
    if 'dst_wallet_level' in df.columns:
        df_quantized['dst_wallet_level'] = df['dst_wallet_level']

    df_quantized['Amount Paid'] = df_quantized['amount']
    df_quantized['Amount Received'] = df_quantized['amount']
    df_quantized['is_laundering'] = df['is_laundering']

    # 交易双方是否为同一账户
    df_quantized['same_transactionaccount'] = (df_quantized['From_Account'] == df_quantized['To_Account']).astype(int)

    # 提取小时作为一个新特征
    df_quantized['Timestamp_Hour'] = df_quantized['Timestamp'].dt.hour

    print("计算账户对特征（使用Expanding Window）...")
    # 创建无向交易对
    df_quantized['account_pair'] = df_quantized.apply(lambda row: tuple(sorted([str(row['From_Account']), str(row['To_Account'])])), axis=1)

    # 使用Expanding Window计算账户对特征
    df_quantized['transaction_count_directed'] = df_quantized.groupby(['From_Account', 'To_Account']).cumcount()
    df_quantized['transaction_count_Undirected'] = df_quantized.groupby('account_pair').cumcount()

    # 特定两个账户间的交易时间间隔
    df_quantized['Timestamp_numeric'] = df_quantized['Timestamp'].astype('int64')
    df_quantized['pair_first_time_numeric'] = df_quantized.groupby('account_pair')['Timestamp_numeric'].transform('first')
    df_quantized['pair_last_time_numeric'] = df_quantized.groupby('account_pair')['Timestamp_numeric'].expanding().max().reset_index(level=0, drop=True)
    df_quantized['accounttransaction_timeinterval'] = (df_quantized['pair_last_time_numeric'] - df_quantized['pair_first_time_numeric']) / 1e9 / 60
    df_quantized['accounttransaction_timeinterval'] = df_quantized['accounttransaction_timeinterval'].fillna(0)
    df_quantized.drop(columns=['Timestamp_numeric', 'pair_first_time_numeric', 'pair_last_time_numeric'], inplace=True, errors='ignore')

    # 特定两个账户间的交易密度(无向)
    df_quantized['transaction_density_Undirected'] = df_quantized.apply(
        lambda row: row['transaction_count_Undirected'] / row['accounttransaction_timeinterval'] if row['accounttransaction_timeinterval'] > 0 else 0, axis=1)

    df_quantized.drop(columns=['account_pair', 'pair_first_time', 'pair_last_time'], inplace=True, errors='ignore')

    print("计算交易集中度特征（使用Expanding Window）...")
    # 用交易次数衡量交易集中度特征
    df_quantized['from_to_count'] = df_quantized.groupby(['From_Account', 'To_Account']).cumcount()
    df_quantized['from_total_count'] = df_quantized.groupby('From_Account').cumcount()
    df_quantized['send_concentration'] = df_quantized.apply(
        lambda row: row['from_to_count'] / row['from_total_count'] if row['from_total_count'] > 0 else 0,
        axis=1
    )

    df_quantized['to_from_count'] = df_quantized.groupby(['To_Account', 'From_Account']).cumcount()
    df_quantized['to_total_count'] = df_quantized.groupby('To_Account').cumcount()
    df_quantized['receive_concentration'] = df_quantized.apply(
        lambda row: row['to_from_count'] / row['to_total_count'] if row['to_total_count'] > 0 else 0,
        axis=1
    )

    # 用交易金额衡量交易集中度特征
    df_quantized['from_to_amount'] = df_quantized.groupby(['From_Account', 'To_Account'])['Amount Paid'].expanding().sum().reset_index(level=[0,1], drop=True).fillna(0)
    df_quantized['from_total_amount'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().sum().reset_index(level=0, drop=True).fillna(0)
    df_quantized['send_amount_concentration'] = df_quantized.apply(
        lambda row: row['from_to_amount'] / row['from_total_amount'] if row['from_total_amount'] > 0 else 0,
        axis=1
    )

    df_quantized['to_from_amount'] = df_quantized.groupby(['To_Account', 'From_Account'])['Amount Received'].expanding().sum().reset_index(level=[0,1], drop=True).fillna(0)
    df_quantized['to_total_amount'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().sum().reset_index(level=0, drop=True).fillna(0)
    df_quantized['receive_amount_concentration'] = df_quantized.apply(
        lambda row: row['to_from_amount'] / row['to_total_amount'] if row['to_total_amount'] > 0 else 0,
        axis=1
    )

    # 清理临时列
    df_quantized.drop(columns=['from_to_count', 'from_total_count', 'to_from_count', 'to_total_count',
                               'from_to_amount', 'from_total_amount', 'to_from_amount', 'to_total_amount'],
                     inplace=True, errors='ignore')

    # 添加原始顺序标记列
    df_quantized['original_order'] = np.arange(len(df_quantized))

    # 在排序后的副本上进行操作
    df_sorted = df_quantized.sort_values(by='Timestamp').copy()
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted['Index'] = df_sorted.index

    # 创建交易记录（使用排序后数据）
    from_accounts = df_sorted[['Index','Timestamp', 'From_Account']].rename(columns={'From_Account':'Account'})
    to_accounts = df_sorted[['Index','Timestamp','To_Account']].rename(columns={'To_Account':'Account'})
    from_accounts['Type'] = 'From'
    to_accounts['Type'] = 'To'
    all_accounts = pd.concat([from_accounts, to_accounts], ignore_index=True)

    # 按账户和时间排序
    all_accounts = all_accounts.sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    # 使用时间窗口计算
    all_accounts['counts_1H_From'] = all_accounts.groupby('Account', group_keys=False).apply(
        lambda g: g[g['Type']=='From'].rolling('1h', on='Timestamp', closed='left')['Type'].count().reindex(g.index, fill_value=0)
    ).reset_index(level=0, drop=True)
    all_accounts['counts_1H_From'] = all_accounts['counts_1H_From'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)

    all_accounts['counts_1H_To'] = all_accounts.groupby('Account', group_keys=False).apply(
        lambda g: g[g['Type']=='To'].rolling('1h', on='Timestamp', closed='left')['Type'].count().reindex(g.index, fill_value=0)
    ).reset_index(level=0, drop=True)
    all_accounts['counts_1H_To'] = all_accounts['counts_1H_To'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)

    # 合并结果回排序后的数据
    df_sorted = df_sorted.merge(
        all_accounts[all_accounts['Type']=='From'][['Index', 'counts_1H_From']],
        on='Index',
        how='left'
    )
    df_sorted = df_sorted.merge(
        all_accounts[all_accounts['Type']=='To'][['Index', 'counts_1H_To']],
        on='Index',
        how='left'
    )

    # 将结果合并回原始顺序
    df_quantized = df_quantized.merge(
        df_sorted[['original_order', 'counts_1H_From', 'counts_1H_To']],
        on='original_order',
        how='left'
    ).drop('original_order', axis=1)

    print("计算账户交易频率特征（使用Expanding Window）...")
    df_quantized['transaction_counts_From'] = df_quantized.groupby('From_Account').cumcount()
    df_quantized['transaction_counts_To'] = df_quantized.groupby('To_Account').cumcount()

    df_quantized['1H_frequency_From'] = df_quantized['counts_1H_From'] / df_quantized['transaction_counts_From'].replace(0, 1)
    df_quantized['1H_frequency_To'] = df_quantized['counts_1H_To'] / df_quantized['transaction_counts_To'].replace(0, 1)

    # 量化时间戳特征
    if pd.api.types.is_datetime64_any_dtype(df_quantized['Timestamp']):
        df_quantized['Timestamp'] = df_quantized['Timestamp'].astype('int64') / 10**9
    elif pd.api.types.is_numeric_dtype(df_quantized['Timestamp']):
        df_quantized['Timestamp'] = df_quantized['Timestamp'].astype('float64')
    else:
        df_quantized['Timestamp'] = pd.to_datetime(df_quantized['Timestamp'], errors='coerce').astype('int64') / 10**9

    print("计算账户度特征（使用Expanding Window）...")
    df_quantized['Out_Degree'] = df_quantized.groupby('From_Account').cumcount()
    df_quantized['In_Degree'] = df_quantized.groupby('To_Account').cumcount()
    df_quantized['From_To_Ratio'] = df_quantized.apply(
        lambda row: row['Out_Degree'] / row['In_Degree'] if row['In_Degree'] > 0 else 0,
        axis=1)

    df_quantized['unique_outdegree'] = df_quantized.groupby('From_Account')['To_Account'].expanding().apply(
        lambda x: len(x.unique()), raw=False).reset_index(level=0, drop=True)

    df_quantized['unique_indegree'] = df_quantized.groupby('To_Account')['From_Account'].expanding().apply(
        lambda x: len(x.unique()), raw=False).reset_index(level=0, drop=True)

    df_quantized['unique_From_To_Ratio'] = df_quantized.apply(
        lambda row: row['unique_outdegree'] / row['unique_indegree'] if row['unique_indegree'] > 0 else 0,
        axis=1)

    # 创建feature_df用于后续合并（从From_Account视角）
    feature_df_from = df_quantized[['From_Account', 'Out_Degree', 'In_Degree', 'From_To_Ratio',
                                    'unique_outdegree', 'unique_indegree', 'unique_From_To_Ratio']].copy()
    feature_df_from = feature_df_from.rename(columns={'From_Account': 'Account'})

    feature_df_to = df_quantized[['To_Account', 'Out_Degree', 'In_Degree', 'From_To_Ratio',
                                  'unique_outdegree', 'unique_indegree', 'unique_From_To_Ratio']].copy()
    feature_df_to = feature_df_to.rename(columns={'To_Account': 'Account'})

    feature_df = pd.concat([feature_df_from, feature_df_to]).drop_duplicates(subset=['Account'], keep='last')
    feature_df = feature_df.groupby('Account').last().reset_index()

    print("计算账户金额统计特征（使用Expanding Window）...")
    df_quantized['Total_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().sum().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Max_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().max().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Min_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().min().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Mean_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Std_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().std().reset_index(level=0, drop=True).fillna(0)
    df_quantized['volatility_Sent'] = df_quantized.apply(
        lambda row: row['Std_Sent'] / row['Mean_Sent'] if row['Mean_Sent'] > 0 else 0,
        axis=1
    )

    df_quantized['Total_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().sum().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Max_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().max().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Min_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().min().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Mean_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Std_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().std().reset_index(level=0, drop=True).fillna(0)
    df_quantized['volatility_Received'] = df_quantized.apply(
        lambda row: row['Std_Received'] / row['Mean_Received'] if row['Mean_Received'] > 0 else 0,
        axis=1
    )

    df_quantized['Receive_Send_Ratio'] = df_quantized['Total_Received'] / df_quantized['Total_Sent'].replace(0, 1)
    df_quantized['Receive_Send_Ratio'] = df_quantized['Receive_Send_Ratio'].replace([float('inf'), -float('inf')], 2)

    df_quantized['Difference'] = df_quantized['Total_Received'] - df_quantized['Total_Sent']

    df_quantized['avg_max_sent_ratio'] = df_quantized['Mean_Sent'] / (df_quantized['Max_Sent'] + 1)
    df_quantized['avg_max_receive_ratio'] = df_quantized['Mean_Received'] / (df_quantized['Max_Received'] + 1)

    df_quantized['max_sent_receive_ratio'] = df_quantized['Max_Sent'] / (df_quantized['Max_Received'] + 1)

    df_quantized['volatility_ratio'] = df_quantized['volatility_Sent'] / (df_quantized['volatility_Received'] + 1)

    # 计算异常大额交易占比
    df_quantized['amount_paid_std_expanding'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().std().reset_index(level=0, drop=True).fillna(0)
    df_quantized['amount_received_std_expanding'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().std().reset_index(level=0, drop=True).fillna(0)

    df_quantized['z_score_paid'] = df_quantized.apply(
        lambda row: (row['Amount Paid'] - row['Mean_Sent']) / row['amount_paid_std_expanding'] if row['amount_paid_std_expanding'] > 0 else 0,
        axis=1
    )
    df_quantized['z_score_received'] = df_quantized.apply(
        lambda row: (row['Amount Received'] - row['Mean_Received']) / row['amount_received_std_expanding'] if row['amount_received_std_expanding'] > 0 else 0,
        axis=1
    )

    df_quantized.drop(columns=['amount_paid_std_expanding', 'amount_received_std_expanding'], inplace=True, errors='ignore')

    df_quantized['large_paid'] = (df_quantized['z_score_paid'].abs() > 3).astype(int)
    df_quantized['large_received'] = (df_quantized['z_score_received'].abs() > 3).astype(int)

    df_quantized['large_transaction_ratio_paid'] = df_quantized.groupby('From_Account')['large_paid'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_quantized['large_transaction_ratio_received'] = df_quantized.groupby('To_Account')['large_received'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_quantized['large_transaction_ratio'] = (df_quantized['large_transaction_ratio_paid'] + df_quantized['large_transaction_ratio_received']) / 2

    # 创建金额特征DataFrame（从From_Account视角）
    amount_feature_from = df_quantized[['From_Account', 'Total_Sent', 'Max_Sent', 'Min_Sent', 'Mean_Sent',
                                       'Std_Sent', 'volatility_Sent', 'large_transaction_ratio_paid']].copy()
    amount_feature_from = amount_feature_from.rename(columns={'From_Account': 'Account', 'large_transaction_ratio_paid': 'large_transaction_ratio_paid_from'})

    amount_feature_to = df_quantized[['To_Account', 'Total_Received', 'Max_Received', 'Min_Received', 'Mean_Received',
                                     'Std_Received', 'volatility_Received', 'large_transaction_ratio_received']].copy()
    amount_feature_to = amount_feature_to.rename(columns={'To_Account': 'Account', 'large_transaction_ratio_received': 'large_transaction_ratio_received_to'})

    amount_feature = pd.merge(amount_feature_from.groupby('Account').last().reset_index(),
                              amount_feature_to.groupby('Account').last().reset_index(),
                              on='Account', how='outer').fillna(0)

    amount_feature['Receive_Send_Ratio'] = amount_feature['Total_Received'] / amount_feature['Total_Sent'].replace(0, 1)
    amount_feature['Receive_Send_Ratio'] = amount_feature['Receive_Send_Ratio'].replace([float('inf'), -float('inf')], 2)
    amount_feature['Difference'] = amount_feature['Total_Received'] - amount_feature['Total_Sent']
    amount_feature['avg_max_sent_ratio'] = amount_feature['Mean_Sent'] / (amount_feature['Max_Sent'] + 1)
    amount_feature['avg_max_receive_ratio'] = amount_feature['Mean_Received'] / (amount_feature['Max_Received'] + 1)
    amount_feature['max_sent_receive_ratio'] = amount_feature['Max_Sent'] / (amount_feature['Max_Received'] + 1)
    amount_feature['volatility_ratio'] = amount_feature['volatility_Sent'] / (amount_feature['volatility_Received'] + 1)
    amount_feature['large_transaction_ratio'] = (amount_feature['large_transaction_ratio_paid_from'] + amount_feature['large_transaction_ratio_received_to']) / 2
    amount_feature = amount_feature.drop(columns=['large_transaction_ratio_paid_from', 'large_transaction_ratio_received_to'], errors='ignore')

    feature_df = feature_df.merge(amount_feature, on='Account', how='outer').fillna(0)

    print("计算账户活跃时间特征（使用Expanding Window）...")
    from_accounts = df_quantized[['From_Account', 'Timestamp']].rename(columns={'From_Account': 'Account'})
    to_accounts = df_quantized[['To_Account', 'Timestamp']].rename(columns={'To_Account': 'Account'})
    all_accountstimestamp = pd.concat([from_accounts, to_accounts]).sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    if pd.api.types.is_datetime64_any_dtype(all_accountstimestamp['Timestamp']):
        all_accountstimestamp['Timestamp_numeric'] = all_accountstimestamp['Timestamp'].astype('int64') / 1e9
    else:
        all_accountstimestamp['Timestamp_numeric'] = all_accountstimestamp['Timestamp'].astype('float64')

    all_accountstimestamp['first_time_numeric'] = all_accountstimestamp.groupby('Account')['Timestamp_numeric'].transform('first')
    all_accountstimestamp['last_time_numeric'] = all_accountstimestamp.groupby('Account')['Timestamp_numeric'].expanding().max().reset_index(level=0, drop=True)
    all_accountstimestamp['activetime'] = all_accountstimestamp['last_time_numeric'] - all_accountstimestamp['first_time_numeric']
    all_accountstimestamp['activetime'] = all_accountstimestamp['activetime'].fillna(0)
    all_accountstimestamp.drop(columns=['Timestamp_numeric', 'first_time_numeric', 'last_time_numeric'], inplace=True, errors='ignore')

    all_accountstimestamp['transaction_count'] = all_accountstimestamp.groupby('Account').cumcount()

    all_accountstimestamp['transaction_frequency'] = all_accountstimestamp.apply(
        lambda row: row['transaction_count'] / row['activetime'] if row['activetime'] > 0 else 0,
        axis=1)

    account_activity = all_accountstimestamp.groupby('Account').last()[['activetime', 'transaction_frequency']].reset_index()
    account_activity = account_activity.drop_duplicates(subset=['Account'], keep='last')

    feature_df = feature_df.merge(account_activity, on='Account', how='outer').fillna(0)

    print("计算最后交易特征（使用Expanding Window）...")
    df_from = df_quantized[['From_Account', 'Timestamp','Amount Paid']].rename(columns={'From_Account': 'Account','Amount Paid': 'Amount'})
    df_from['direction'] = 1
    df_from = df_from.sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    df_to = df_quantized[['To_Account', 'Timestamp', 'Amount Received']].rename(columns={'To_Account': 'Account','Amount Received': 'Amount'})
    df_to['direction'] = 0
    df_to = df_to.sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    all_transactions = pd.concat([df_from, df_to]).sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    last_transaction = all_transactions.groupby('Account').last().reset_index()
    last_transaction = last_transaction[['Account', 'direction', 'Amount']].rename(columns={'direction': 'last_direction', 'Amount': 'last_amount'})

    feature_df = feature_df.merge(last_transaction, on='Account', how='outer').fillna(0)

    print("计算交易时间间隔特征（使用Expanding Window）...")
    payment_transactions = all_transactions[all_transactions['direction'] == 1].copy()
    receipt_transactions = all_transactions[all_transactions['direction'] == 0].copy()

    payment_transactions['time_diff'] = payment_transactions.groupby('Account')['Timestamp'].diff()
    receipt_transactions['time_diff'] = receipt_transactions.groupby('Account')['Timestamp'].diff()

    if pd.api.types.is_timedelta64_dtype(payment_transactions['time_diff']):
        payment_transactions['time_diff_numeric'] = payment_transactions['time_diff'].dt.total_seconds()
        receipt_transactions['time_diff_numeric'] = receipt_transactions['time_diff'].dt.total_seconds()
    else:
        payment_transactions['time_diff_numeric'] = payment_transactions['time_diff']
        receipt_transactions['time_diff_numeric'] = receipt_transactions['time_diff']

    payment_transactions['avg_payment_time_diff'] = payment_transactions.groupby('Account')['time_diff_numeric'].expanding().mean().reset_index(level=0, drop=True)
    receipt_transactions['avg_receipt_time_diff'] = receipt_transactions.groupby('Account')['time_diff_numeric'].expanding().mean().reset_index(level=0, drop=True)

    payment_transactions.drop(columns=['time_diff_numeric'], inplace=True, errors='ignore')
    receipt_transactions.drop(columns=['time_diff_numeric'], inplace=True, errors='ignore')

    average_payment_time_diff = payment_transactions.groupby('Account')['avg_payment_time_diff'].last().reset_index()
    average_receipt_time_diff = receipt_transactions.groupby('Account')['avg_receipt_time_diff'].last().reset_index()
    average_payment_time_diff = average_payment_time_diff.fillna(0)
    average_receipt_time_diff = average_receipt_time_diff.fillna(0)

    feature_df = feature_df.merge(average_payment_time_diff, on='Account', how='outer').merge(average_receipt_time_diff, on='Account', how='outer').fillna(0)

    # 计算账户的非活跃期长度
    all_transactions = all_transactions.sort_values(by=['Account', 'Timestamp'])
    all_transactions['time_diff'] = all_transactions.groupby('Account')['Timestamp'].diff()

    if pd.api.types.is_timedelta64_dtype(all_transactions['time_diff']):
        all_transactions['time_diff_numeric'] = all_transactions['time_diff'].dt.total_seconds()
    else:
        all_transactions['time_diff_numeric'] = all_transactions['time_diff']

    all_transactions['avg_inactive_period'] = all_transactions.groupby('Account')['time_diff_numeric'].expanding().mean().reset_index(level=0, drop=True)
    all_transactions['max_inactive_period'] = all_transactions.groupby('Account')['time_diff_numeric'].expanding().max().reset_index(level=0, drop=True)
    all_transactions['min_inactive_period'] = all_transactions.groupby('Account')['time_diff_numeric'].expanding().min().reset_index(level=0, drop=True)

    all_transactions.drop(columns=['time_diff_numeric'], inplace=True, errors='ignore')

    inactive_periods = all_transactions.groupby('Account')[['avg_inactive_period', 'max_inactive_period', 'min_inactive_period']].last().reset_index()
    inactive_periods = inactive_periods.fillna(0)

    feature_df = feature_df.merge(inactive_periods, on='Account', how='outer').fillna(0)

    # 读取账户开立数据
    print("读取账户开立数据...")
    account_df = pd.read_csv(account_path)
    account_df['is_laundering'] = account_df['is_abnormal'].replace({False: 0, True: 1, 'False': 0, 'True': 1, '0': 0, '1': 1})
    account_df = account_df.rename(columns={'is_laundering': 'account_is_laundering'})

    account_features_list = [
        'open_hour', 'is_work_hour', 'is_night_hour',
        'auth_delay_days', 'wallets_per_cert', 'days_to_expire_from_current',
        'tel_reuse_count', 'log_balance',
    ]

    print("由账户原始字段计算开立衍生特征...")
    compute_account_open_features_from_raw(account_df)

    # 可选：外部账户特征 CSV
    print("读取账户开立特征（可选外部文件）...")
    account_features_df = None
    if account_features_path and os.path.exists(account_features_path):
        print(f"  从文件读取: {account_features_path}")
        account_features_df = pd.read_csv(account_features_path)
        if 'wallet_id' not in account_features_df.columns:
            print("  警告: 外部特征文件无 wallet_id 列，忽略该文件")
            account_features_df = None

    base_account_cols = [
        'wallet_id', 'init_balance', 'wallet_level', 'bank_account_number',
        'account_is_laundering', 'wallet_type',
    ]
    extra_feat_cols = [c for c in account_features_list if c in account_df.columns]
    account_cols = base_account_cols + extra_feat_cols
    available_cols = [col for col in account_cols if col in account_df.columns]
    account_df = account_df[available_cols]

    if 'bank_account_number' in account_df.columns:
        account_df = account_df.rename(columns={'bank_account_number': 'bank_id'})

    if 'balance_limit' not in account_df.columns:
        account_df['balance_limit'] = 0

    feat_merge_cols = ['wallet_id'] + [c for c in account_features_list if c in account_df.columns]
    account_features_subset = (
        account_df[feat_merge_cols].copy() if len(feat_merge_cols) > 1 else account_df[['wallet_id']].copy()
    )
    if account_features_df is not None:
        account_features_df['wallet_id'] = account_features_df['wallet_id'].astype(str)
        for col in account_features_list:
            if col in account_features_df.columns and col not in account_features_subset.columns:
                account_features_subset = account_features_subset.merge(
                    account_features_df[['wallet_id', col]],
                    on='wallet_id',
                    how='left',
                )

    account_df = account_df.rename(columns={'wallet_id': 'acct_id'})

    print("合并账户开立信息和特征...")
    feature_df = pd.merge(account_df, feature_df, left_on='acct_id', right_on='Account', how='inner')
    feature_df.drop(columns=['acct_id'], inplace=True, errors='ignore')
    feature_df = feature_df.fillna(0)

    account_info_df = account_df.copy()
    account_info_df = account_info_df.rename(columns={'acct_id': 'Account'})
    _open_feat_cols_in_info = [c for c in account_features_list if c in account_info_df.columns]
    if _open_feat_cols_in_info:
        account_info_df = account_info_df.drop(columns=_open_feat_cols_in_info)

    df_quantized = df_quantized.merge(account_info_df, left_on='From_Account', right_on='Account', how='left', suffixes=('', '_From'))
    df_quantized = df_quantized.drop(columns=['Account'], errors='ignore')

    df_quantized = df_quantized.merge(account_info_df, left_on='To_Account', right_on='Account', how='left', suffixes=('_From', '_To'))
    df_quantized = df_quantized.drop(columns=['Account'], errors='ignore')

    # 合并账户开立特征
    if len(feat_merge_cols) > 1:
        print("合并账户开立特征到交易表（open_hour, is_work_hour 等）...")
        account_features_subset['wallet_id'] = account_features_subset['wallet_id'].astype(str)
        df_quantized['From_Account'] = df_quantized['From_Account'].astype(str)
        df_quantized['To_Account'] = df_quantized['To_Account'].astype(str)

        account_features_from = account_features_subset.copy()
        account_features_from = account_features_from.rename(columns={'wallet_id': 'From_Account'})
        for feat in account_features_list:
            if feat in account_features_from.columns:
                account_features_from = account_features_from.rename(columns={feat: f'{feat}_From'})

        df_quantized = df_quantized.merge(
            account_features_from,
            on='From_Account',
            how='left',
        )

        account_features_to = account_features_subset.copy()
        account_features_to = account_features_to.rename(columns={'wallet_id': 'To_Account'})
        for feat in account_features_list:
            if feat in account_features_to.columns:
                account_features_to = account_features_to.rename(columns={feat: f'{feat}_To'})

        df_quantized = df_quantized.merge(
            account_features_to,
            on='To_Account',
            how='left',
        )

        for feat in account_features_list:
            if f'{feat}_From' in df_quantized.columns:
                df_quantized[f'{feat}_From'] = df_quantized[f'{feat}_From'].fillna(0)
            if f'{feat}_To' in df_quantized.columns:
                df_quantized[f'{feat}_To'] = df_quantized[f'{feat}_To'].fillna(0)

        merged_features_count = len([f for f in account_features_list if f in account_features_subset.columns])
        print(f"  已合并 {merged_features_count} 个账户开立特征（From和To各{merged_features_count}个，共{merged_features_count*2}列）")
    else:
        print("  警告: 账户表中无 wallet_id 或未生成任何开立衍生列，跳过开立特征合并")

    # 计算S指标
    print("计算S指标（S_history_count, S_geo_risk等）...")
    timestamp_backup = df_quantized['Timestamp'].copy()

    if pd.api.types.is_numeric_dtype(df_quantized['Timestamp']):
        df_quantized['Timestamp'] = pd.to_datetime(df_quantized['Timestamp'], unit='s', errors='coerce')
    else:
        df_quantized['Timestamp'] = pd.to_datetime(df_quantized['Timestamp'], errors='coerce')

    account_df_for_s = pd.read_csv(account_path)
    if 'wallet_open_timestamp' not in account_df_for_s.columns:
        if 'wallet_open_date' in account_df_for_s.columns:
            account_df_for_s['wallet_open_timestamp'] = pd.to_datetime(account_df_for_s['wallet_open_date'], errors='coerce')

    if 'account_is_laundering' not in account_df_for_s.columns:
        if 'is_abnormal' in account_df_for_s.columns:
            account_df_for_s['account_is_laundering'] = account_df_for_s['is_abnormal'].replace({False: 0, True: 1, 'False': 0, 'True': 1, '0': 0, '1': 1})
        else:
            account_df_for_s['account_is_laundering'] = 0

    transaction_config_path = None
    data_dir = os.path.dirname(data_path)
    potential_config_path = os.path.join(data_dir, "transaction_config.json")
    if os.path.exists(potential_config_path):
        transaction_config_path = potential_config_path
    else:
        account_dir = os.path.dirname(account_path)
        potential_config_path = os.path.join(account_dir, "transaction_config.json")
        if os.path.exists(potential_config_path):
            transaction_config_path = potential_config_path

    s_indicators_df = calculate_s_indicators(df_quantized, account_df_for_s, transaction_config_path)

    df_quantized['Timestamp'] = timestamp_backup

    # 合并S指标到df_quantized
    if s_indicators_df is not None and len(s_indicators_df) > 0:
        print(f"  成功计算S指标，共 {len(s_indicators_df)} 个钱包的S指标")
        print(f"  S指标列: {list(s_indicators_df.columns)}")

        s_indicators_df['wallet_id'] = s_indicators_df['wallet_id'].astype(str)
        df_quantized['From_Account'] = df_quantized['From_Account'].astype(str)
        df_quantized['To_Account'] = df_quantized['To_Account'].astype(str)

        s_indicators_from = s_indicators_df.copy()
        s_indicators_from = s_indicators_from.rename(columns={
            'wallet_id': 'From_Account',
            **{col: f'S_src_{col.replace("S_", "")}' for col in s_indicators_from.columns if col != 'wallet_id'}
        })
        df_quantized = df_quantized.merge(
            s_indicators_from,
            on='From_Account',
            how='left'
        )

        s_indicators_to = s_indicators_df.copy()
        s_indicators_to = s_indicators_to.rename(columns={
            'wallet_id': 'To_Account',
            **{col: f'S_dst_{col.replace("S_", "")}' for col in s_indicators_to.columns if col != 'wallet_id'}
        })
        df_quantized = df_quantized.merge(
            s_indicators_to,
            on='To_Account',
            how='left'
        )

        s_indicator_cols = [col for col in s_indicators_df.columns if col != 'wallet_id']
        for col in s_indicator_cols:
            src_col = f'S_src_{col.replace("S_", "")}'
            dst_col = f'S_dst_{col.replace("S_", "")}'
            if src_col in df_quantized.columns:
                df_quantized[src_col] = df_quantized[src_col].fillna(0)
            if dst_col in df_quantized.columns:
                df_quantized[dst_col] = df_quantized[dst_col].fillna(0)

        print(f"  已合并 {len(s_indicator_cols)} 个S指标（From和To各{len(s_indicator_cols)}个，共{len(s_indicator_cols)*2}列）")
        src_cols = [f'S_src_{col.replace("S_", "")}' for col in s_indicator_cols]
        dst_cols = [f'S_dst_{col.replace("S_", "")}' for col in s_indicator_cols]
        print(f"  S指标列名: {src_cols + dst_cols}")
    else:
        print("  警告: calculate_s_indicators 返回了空DataFrame，S指标将不会被合并")
        s_indicator_cols = [
            'S_history_count',
            'S_geo_risk',
            'S_time_risk',
            'S_ibank_index',
            'S_ibank_amount_ratio',
            'S_counterparty_index',
            'S_per_deal_match_times',
            'S_daily_deal_match_times'
        ]
        for col in s_indicator_cols:
            df_quantized[f'S_src_{col.replace("S_", "")}'] = 0.0
            df_quantized[f'S_dst_{col.replace("S_", "")}'] = 0.0
        print(f"  已创建默认S指标列（全为0）: {len(s_indicator_cols)*2} 列")

    # 将 feature_df 中仅存在于账户级汇总表的列合并回 df_quantized
    print("合并 feature_df 中的账户级汇总特征到 df_quantized...")
    _feat_from_df_cols = [
        'activetime', 'transaction_frequency', 'last_direction', 'last_amount',
        'avg_payment_time_diff', 'avg_receipt_time_diff',
        'avg_inactive_period', 'max_inactive_period', 'min_inactive_period',
    ]
    _feat_from_df_cols = [c for c in _feat_from_df_cols if c in feature_df.columns]
    if _feat_from_df_cols:
        _fj = feature_df[['Account'] + _feat_from_df_cols].copy()
        _fj['Account'] = _fj['Account'].astype(str)
        df_quantized['From_Account'] = df_quantized['From_Account'].astype(str)
        df_quantized['To_Account'] = df_quantized['To_Account'].astype(str)
        _m_from = _fj.rename(
            columns={'Account': 'From_Account', **{c: f'{c}_From' for c in _feat_from_df_cols}}
        )
        df_quantized = df_quantized.merge(_m_from, on='From_Account', how='left')
        _m_to = _fj.rename(
            columns={'Account': 'To_Account', **{c: f'{c}_To' for c in _feat_from_df_cols}}
        )
        df_quantized = df_quantized.merge(_m_to, on='To_Account', how='left')
        for _c in _feat_from_df_cols:
            _a, _b = f'{_c}_From', f'{_c}_To'
            if _a in df_quantized.columns:
                df_quantized[_a] = df_quantized[_a].fillna(0)
            if _b in df_quantized.columns:
                df_quantized[_b] = df_quantized[_b].fillna(0)
        print(f"  已合并 {len(_feat_from_df_cols)} 个特征 x 2（From/To）: {', '.join(_feat_from_df_cols)}")
    else:
        print("  警告: feature_df 中未找到 activetime/last_* 等预期列，跳过 feature_df 合并")

    df_new = df_quantized.copy()

    # 删除账户ID列和重复特征
    print("删除账户ID列和重复特征...")

    columns_to_drop = ['From_Account', 'To_Account']
    for col in columns_to_drop:
        if col in df_new.columns:
            df_new = df_new.drop(columns=[col])

    duplicate_features_to_drop = [
        'Amount Paid', 'Amount Received',
        'bank_id_From', 'bank_id_To',
        'transaction_counts_From', 'transaction_counts_To',
        'balance_limit_From', 'balance_limit_To',
    ]

    features_dropped = []
    for feature in duplicate_features_to_drop:
        if feature in df_new.columns:
            df_new = df_new.drop(columns=[feature])
            features_dropped.append(feature)

    if len(features_dropped) > 0:
        print(f"  已删除 {len(features_dropped)} 个重复特征: {', '.join(features_dropped)}")
    else:
        print("  未发现需要删除的重复特征")

    # 将标签从数据集中分离出来
    labels = df_new['is_laundering']
    account_is_laundering_From = df_new.get('account_is_laundering_From', pd.Series([0] * len(df_new)))
    account_is_laundering_To = df_new.get('account_is_laundering_To', pd.Series([0] * len(df_new)))

    # 保存原始特征（归一化之前）
    print("保存原始特征（归一化之前）...")
    df_original = df_new.copy()
    if 'Timestamp' in df_original.columns:
        df_original = df_original.drop(columns=['Timestamp'])
    df_original['is_laundering'] = labels
    df_original['account_is_laundering_From'] = account_is_laundering_From
    df_original['account_is_laundering_To'] = account_is_laundering_To
    df_original = pd.concat([id_index.reset_index(drop=True), df_original], axis=1)

    save_dir = os.path.dirname(save_path)
    save_basename = os.path.basename(save_path)
    save_name_without_ext = os.path.splitext(save_basename)[0]
    original_features_path = os.path.join(save_dir, f"{save_name_without_ext}_original.csv")

    df_original.to_csv(original_features_path, index=False)
    print(f"  原始特征已保存到: {original_features_path}")

    # 准备归一化的数据
    if 'Timestamp' in df_new.columns:
        df_new = df_new.drop(columns=['Timestamp'])
    df_new = df_new.drop(['is_laundering'], axis=1, errors='ignore')
    if 'account_is_laundering_From' in df_new.columns:
        df_new = df_new.drop(['account_is_laundering_From'], axis=1)
    if 'account_is_laundering_To' in df_new.columns:
        df_new = df_new.drop(['account_is_laundering_To'], axis=1)

    # ═══════════════════════════════════════════════════════════════════════
    # 删除 10 个低质量基线特征
    # ═══════════════════════════════════════════════════════════════════════
    print("\n删除 10 个低质量基线特征...")
    lq_dropped = []
    for feat_name in LOW_QUALITY_FEATURES:
        if feat_name in df_new.columns:
            df_new = df_new.drop(columns=[feat_name])
            lq_dropped.append(feat_name)
    print(f"  已删除 {len(lq_dropped)} 个低质量特征: {', '.join(lq_dropped)}")

    # ═══════════════════════════════════════════════════════════════════════
    # 计算并追加 18 个 KnowFeat 特征
    # ═══════════════════════════════════════════════════════════════════════
    print("\n计算 18 个 KnowFeat 特征...")

    # 从原始 CSV 加载三表（用于 KnowFeat 计算，不使用处理后的 df_quantized）
    raw_open_df = pd.read_csv(account_path)
    raw_tx_df = pd.read_csv(data_path)
    raw_close_df = pd.read_csv(close_path) if close_path and os.path.exists(close_path) else pd.DataFrame()

    # 确保 raw_tx_df 有必要的列
    if 'timestamp' in raw_tx_df.columns:
        raw_tx_df['timestamp'] = pd.to_datetime(raw_tx_df['timestamp'], errors='coerce')
    if 'hour' not in raw_tx_df.columns and 'timestamp' in raw_tx_df.columns:
        raw_tx_df['hour'] = raw_tx_df['timestamp'].dt.hour
    if 'interval' not in raw_tx_df.columns and 'timestamp' in raw_tx_df.columns:
        raw_tx_df = raw_tx_df.sort_values(['src', 'timestamp'])
        raw_tx_df['interval'] = raw_tx_df.groupby('src')['timestamp'].diff().dt.total_seconds().fillna(0)
    if 'transaction_mode' not in raw_tx_df.columns:
        raw_tx_df['transaction_mode'] = 'single'

    # src_series 用于 join（与 df_new 行对齐）
    src_series = df['src'].values

    knowfeat_success = 0
    knowfeat_names = []

    for feat_name, feat_func in KNOWFEAT_FUNCTIONS:
        try:
            feat_df = feat_func(raw_open_df, raw_tx_df, raw_close_df)
            # 钱包级 -> 交易级：通过 src 列 join
            merge_df = pd.DataFrame({"wallet_id": src_series})
            merged = merge_df.merge(feat_df, on="wallet_id", how="left")
            feat_values = pd.to_numeric(merged[feat_name], errors="coerce").fillna(0).values
            df_new[feat_name] = feat_values
            knowfeat_names.append(feat_name)
            knowfeat_success += 1
            print(f"  [OK] {feat_name} ({feat_df.shape[0]} wallets)")
        except Exception as e:
            print(f"  [FAIL] {feat_name}: {e}")

    print(f"\n成功计算 {knowfeat_success}/{len(KNOWFEAT_FUNCTIONS)} 个 KnowFeat 特征")
    print(f"最终特征数: {df_new.shape[1]} 列")

    # 对数据集进行归一化
    print("对特征进行归一化...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df_new)
    df_normalized = scaler.transform(df_new)

    df_normalized = pd.DataFrame(df_normalized, columns=df_new.columns)
    df_normalized['is_laundering'] = labels.tolist()
    df_normalized['account_is_laundering_From'] = account_is_laundering_From.tolist()
    df_normalized['account_is_laundering_To'] = account_is_laundering_To.tolist()
    df_normalized = pd.concat([id_index.reset_index(drop=True), df_normalized], axis=1)

    # 在最终特征中保留原始的 src / dst 列
    if {'tx_id', 'src', 'dst'}.issubset(set(df.columns)):
        src_dst_map = df[['tx_id', 'src', 'dst']].copy()
        src_dst_map = src_dst_map.rename(columns={'tx_id': 'id'})
        dup_ids = src_dst_map['id'].duplicated(keep=False)
        if dup_ids.any():
            n_dup = int(dup_ids.sum())
            print(f"  警告: 交易数据中存在重复 tx_id（共 {n_dup} 行），merge 前已按 id 保留首行")
        src_dst_map = src_dst_map.drop_duplicates(subset=['id'], keep='first')
        df_normalized = df_normalized.merge(src_dst_map, on='id', how='left')
    else:
        print("  警告: 原始数据中未同时发现 'tx_id', 'src', 'dst' 三列，无法在输出中保留 src/dst")

    # 排序用时间戳
    _ts_row = df[['tx_id', 'timestamp']].drop_duplicates(subset=['tx_id'], keep='first').copy()
    _ts_map = pd.Series(
        pd.to_datetime(_ts_row['timestamp']).values,
        index=_ts_row['tx_id'].astype(str).values,
    )
    df_normalized['_ts_sort'] = df_normalized['id'].astype(str).map(_ts_map)
    if df_normalized['_ts_sort'].isna().any():
        n_bad = int(df_normalized['_ts_sort'].isna().sum())
        print(f"  警告: 有 {n_bad} 行 id 未映射到原始 timestamp")

    # 删除指定的列
    print("删除指定列...")
    columns_to_remove = [
        'id', 'init_balance_From', 'init_balance_To', 'amount',
        'account_is_laundering_From', 'account_is_laundering_To',
    ]
    columns_exist = [col for col in columns_to_remove if col in df_normalized.columns]
    if columns_exist:
        df_normalized = df_normalized.drop(columns=columns_exist)
        print(f"  已删除列: {', '.join(columns_exist)}")
    else:
        print("  未找到需要删除的列")

    # 按照时间顺序排序
    print("按时间顺序排序...")
    df_normalized = df_normalized.sort_values('_ts_sort').reset_index(drop=True)
    df_normalized = df_normalized.drop(columns=['_ts_sort'])

    # 保存
    df_normalized.to_csv(save_path, index=False)
    print(f"归一化后的完整特征已保存到: {save_path}")

    # 打印特征列表摘要
    all_feat_cols = [c for c in df_normalized.columns if c not in ('is_laundering', 'src', 'dst')]
    print(f"\n特征工程已完成。共 {len(all_feat_cols)} 个特征列。")
    if knowfeat_names:
        print(f"其中 KnowFeat 新特征 ({len(knowfeat_names)} 个): {', '.join(knowfeat_names)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 18 个 KnowFeat 特征函数（内联定义，完全自包含）
# ═══════════════════════════════════════════════════════════════════════════════

def knowfeat_small_test_round_amount_ratio(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    src_features = tx.groupby('src').agg(
        tx_count=('amount', 'count'),
        small_tx_count=('amount_abs', lambda x: (x <= 5).sum()),
        round_100_count=('amount_abs', lambda x: (x % 100 == 0).sum()),
    ).reset_index()
    src_features['small_tx_ratio'] = src_features['small_tx_count'] / (src_features['tx_count'] + 1e-10)
    src_features['round_100_ratio'] = src_features['round_100_count'] / (src_features['tx_count'] + 1e-10)
    src_features['small_test_round_amount_ratio'] = src_features['small_tx_ratio'] * (1 + src_features['round_100_ratio'])
    result = src_features[['src', 'small_test_round_amount_ratio']].copy()
    result.columns = ['wallet_id', 'small_test_round_amount_ratio']
    return result


def knowfeat_amount_concentration_entropy(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()

    def calc_top3_ratio(x):
        counts = x.value_counts()
        if len(counts) == 0:
            return 0.0
        return counts.head(3).sum() / len(x)

    def calc_entropy(x):
        counts = x.value_counts()
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs + 1e-10))

    src_features = tx.groupby('src')['amount_abs'].agg([
        ('amount_top3_ratio', calc_top3_ratio),
        ('amount_entropy', calc_entropy),
        ('tx_count', 'count')
    ]).reset_index()
    max_entropy = np.log2(src_features['tx_count'].clip(lower=2))
    src_features['norm_entropy'] = src_features['amount_entropy'] / (max_entropy + 1e-10)
    src_features['amount_concentration_entropy'] = src_features['amount_top3_ratio'] * (1 - src_features['norm_entropy'].clip(0, 1))
    result = src_features[['src', 'amount_concentration_entropy']].copy()
    result.columns = ['wallet_id', 'amount_concentration_entropy']
    return result


def knowfeat_top3_hour_concentration(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['hour'] = tx['hour'].astype(int)

    def calc_top3_hour_ratio(group):
        counts = group.value_counts()
        if len(counts) == 0:
            return 0.0
        return counts.head(3).sum() / len(group)

    result = tx.groupby('src')['hour'].apply(calc_top3_hour_ratio).reset_index()
    result.columns = ['wallet_id', 'top3_hour_concentration']
    return result


def knowfeat_fan_in_fan_out_ratio(open_df, tx_df, close_df):
    tx = tx_df.copy()

    def calc_mode_ratios(group):
        total = len(group)
        return pd.Series({
            'fan_in_ratio': (group == 'fan_in').sum() / total,
            'fan_out_ratio': (group == 'fan_out').sum() / total,
            'many_to_many_ratio': (group == 'many_to_many').sum() / total,
            'one_to_many_ratio': (group == 'one_to_many').sum() / total,
        })

    mode_stats = tx.groupby('src')['transaction_mode'].apply(calc_mode_ratios).unstack()
    mode_stats['fan_in_fan_out_ratio'] = mode_stats['fan_in_ratio'] / (mode_stats['fan_out_ratio'] + 1e-10)
    mode_stats['abnormal_mode_ratio'] = mode_stats['fan_in_ratio'] + mode_stats['many_to_many_ratio'] + mode_stats['one_to_many_ratio']
    result_val = mode_stats['fan_in_fan_out_ratio'] * (1 + mode_stats['abnormal_mode_ratio'])
    result = pd.DataFrame({'wallet_id': mode_stats.index, 'fan_in_fan_out_ratio': result_val.values})
    return result


def knowfeat_mode_transition_entropy(open_df, tx_df, close_df):
    tx = tx_df.copy()

    def calc_normalized_entropy(group):
        counts = group.value_counts()
        probs = counts / len(group)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy

    src_entropy = tx.groupby('src')['transaction_mode'].apply(calc_normalized_entropy).reset_index()
    src_entropy.columns = ['wallet_id', 'mode_transition_entropy']
    src_entropy['mode_transition_entropy'] = 1.0 - src_entropy['mode_transition_entropy']
    return src_entropy


def knowfeat_new_counterparty_concentration(open_df, tx_df, close_df):
    pair_counts = tx_df.groupby(['src', 'dst']).size().reset_index()
    pair_counts.columns = ['src', 'dst', 'pair_count']
    src_stats = pair_counts.groupby('src').agg(
        total_counterparties=('dst', 'count'),
        new_counterparties=('pair_count', lambda x: (x == 1).sum())
    ).reset_index()
    src_stats['new_counterparty_ratio'] = src_stats['new_counterparties'] / (src_stats['total_counterparties'] + 1e-10)
    src_stats['new_counterparty_concentration'] = src_stats['new_counterparty_ratio'] / (src_stats['total_counterparties'] + 1.0)
    result = src_stats[['src', 'new_counterparty_concentration']].copy()
    result.columns = ['wallet_id', 'new_counterparty_concentration']
    return result


def knowfeat_counterparty_level_diversity_risk(open_df, tx_df, close_df):
    level_diversity = tx_df.groupby('src')['dst_wallet_level'].nunique().reset_index()
    level_diversity.columns = ['wallet_id', 'dst_level_diversity']
    src_self_level = tx_df.groupby('src')['src_wallet_level'].first().reset_index()
    src_self_level.columns = ['wallet_id', 'self_wallet_level']
    result = level_diversity.merge(src_self_level, on='wallet_id', how='left')
    diversity_risk = 1.0 - result['dst_level_diversity'] / 4.0
    level_risk = 1.0 - result['self_wallet_level'] / 4.0
    result['counterparty_level_diversity_risk'] = diversity_risk * (1.0 + level_risk)
    return result[['wallet_id', 'counterparty_level_diversity_risk']]


def knowfeat_amount_range_ratio_risk(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx_pos = tx[tx['amount'] > 0].copy()
    if len(tx_pos) == 0:
        return pd.DataFrame({'wallet_id': tx_df['src'].unique(), 'amount_range_ratio_risk': 0.0})
    range_stats = tx_pos.groupby('src').agg(
        max_amount=('amount', 'max'), min_positive=('amount', 'min'), tx_count=('amount', 'count'),
    ).reset_index()
    range_stats['amount_range_ratio'] = range_stats['max_amount'] / (range_stats['min_positive'] + 1e-10)
    range_stats['amount_range_ratio_risk'] = np.log1p(range_stats['amount_range_ratio'])
    range_stats.loc[range_stats['tx_count'] < 2, 'amount_range_ratio_risk'] = 0.0
    result = range_stats[['src', 'amount_range_ratio_risk']].copy()
    result.columns = ['wallet_id', 'amount_range_ratio_risk']
    return result


def knowfeat_high_risk_region_off_hour_ratio(open_df, tx_df, close_df):
    high_risk_regions = [540000, 650000]
    open_df = open_df.copy()
    open_df['is_high_risk_region'] = open_df['region_code'].isin(high_risk_regions).astype(int)
    tx = tx_df.copy()
    tx['is_off_hour'] = ((tx['hour'] >= 18) | (tx['hour'] < 8)).astype(int)
    tx_with_region = tx.merge(open_df[['wallet_id', 'is_high_risk_region']],
                               left_on='src', right_on='wallet_id', how='left')
    src_stats = tx_with_region.groupby('src').agg(
        total_tx=('tx_id', 'count'), off_hour_tx=('is_off_hour', 'sum'), high_risk_flag=('is_high_risk_region', 'first')
    ).reset_index()
    src_stats['off_hour_ratio'] = src_stats['off_hour_tx'] / (src_stats['total_tx'] + 1e-10)
    src_stats['high_risk_region_off_hour_ratio'] = src_stats['high_risk_flag'] * src_stats['off_hour_ratio']
    result = src_stats[['src', 'high_risk_region_off_hour_ratio']].copy()
    result.columns = ['wallet_id', 'high_risk_region_off_hour_ratio']
    return result


def knowfeat_rhythm_stability_risk(open_df, tx_df, close_df):
    src_stats = tx_df.groupby('src').agg(
        interval_mean=('interval', 'mean'), interval_std=('interval', 'std'),
        long_ratio=('interval', lambda x: (x > 300).mean()),
    ).reset_index()
    src_stats['interval_cv'] = src_stats['interval_std'] / (src_stats['interval_mean'] + 1e-10)
    src_stats['rhythm_stability'] = 1.0 / (1.0 + src_stats['interval_cv'])
    src_stats['rhythm_stability_risk'] = src_stats['long_ratio'] * src_stats['rhythm_stability']
    result = src_stats[['src', 'rhythm_stability_risk']].copy()
    result.columns = ['wallet_id', 'rhythm_stability_risk']
    return result


def knowfeat_net_flow_direction_risk(open_df, tx_df, close_df):
    tx_src = tx_df[['src', 'timestamp', 'amount']].copy()
    tx_src.columns = ['wallet_id', 'timestamp', 'amount']
    tx_src['direction'] = -1
    tx_dst = tx_df[['dst', 'timestamp', 'amount']].copy()
    tx_dst.columns = ['wallet_id', 'timestamp', 'amount']
    tx_dst['direction'] = 1
    all_tx = pd.concat([tx_src, tx_dst]).sort_values(['wallet_id', 'timestamp'])

    def calc_flow_pattern(group):
        if len(group) < 2:
            return pd.Series({'net_flow_ratio': 0.0, 'switch_ratio': 0.0})
        directions = group['direction'].values
        amounts = group['amount'].abs().values * directions
        total_in = amounts[amounts > 0].sum()
        total_out = abs(amounts[amounts < 0].sum())
        total = total_in + total_out
        net_flow_ratio = (total_in - total_out) / total if total > 0 else 0.0
        switches = (directions[1:] != directions[:-1]).sum()
        switch_ratio = switches / (len(directions) - 1)
        return pd.Series({'net_flow_ratio': net_flow_ratio, 'switch_ratio': switch_ratio})

    wallet_counts = pd.concat([tx_df.groupby('src').size(), tx_df.groupby('dst').size()]).groupby(level=0).sum()
    active = wallet_counts[wallet_counts >= 2].index
    all_tx_active = all_tx[all_tx['wallet_id'].isin(active)]
    flow_pattern = all_tx_active.groupby('wallet_id').apply(calc_flow_pattern).reset_index()
    flow_pattern['net_flow_abs'] = flow_pattern['net_flow_ratio'].abs()
    flow_pattern['net_flow_direction_risk'] = flow_pattern['net_flow_abs'] * (1.0 - flow_pattern['switch_ratio'])
    return flow_pattern[['wallet_id', 'net_flow_direction_risk']].copy()


def knowfeat_enterprise_counterparty_concentration(open_df, tx_df, close_df):
    wallet_type_map = open_df[['wallet_id', 'wallet_type']].set_index('wallet_id')['wallet_type'].to_dict()
    tx = tx_df[['src', 'dst']].copy()
    tx['dst_wtype'] = tx['dst'].map(wallet_type_map)
    tx = tx.dropna(subset=['dst_wtype'])

    def calc_enterprise_concentration(group):
        counts = group.value_counts()
        total = len(group)
        enterprise_ratio = counts.get(2, 0) / total
        probs = counts / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        norm_entropy = entropy / max_entropy
        return enterprise_ratio * (1 - norm_entropy)

    result = tx.groupby('src')['dst_wtype'].apply(calc_enterprise_concentration).reset_index()
    result.columns = ['wallet_id', 'enterprise_counterparty_concentration']
    return result


def knowfeat_first_small_test_ratio(open_df, tx_df, close_df):
    tx_sorted = tx_df.sort_values(['src', 'timestamp'])
    first_tx = tx_sorted.groupby('src').first().reset_index()
    first_tx['amount_abs'] = first_tx['amount'].abs()
    first_tx['first_small_test'] = (first_tx['amount_abs'] <= 5).astype(int)
    result = first_tx[['src', 'first_small_test']].copy()
    result.columns = ['wallet_id', 'first_small_test_ratio']
    return result


def knowfeat_night_amount_short_term_risk(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['is_deep_night'] = (tx['hour'] >= 22) | (tx['hour'] < 6)
    tx['amount_abs'] = tx['amount'].abs()
    deep_night_amt = tx[tx['is_deep_night']].groupby('src')['amount_abs'].sum().reset_index()
    deep_night_amt.columns = ['src', 'deep_night_amount']
    total_amt = tx.groupby('src')['amount_abs'].sum().reset_index()
    total_amt.columns = ['src', 'total_amount']
    amt_ratio = total_amt.merge(deep_night_amt, on='src', how='left').fillna(0)
    amt_ratio['deep_night_amt_ratio'] = amt_ratio['deep_night_amount'] / (amt_ratio['total_amount'] + 1e-10)
    time_span = tx.groupby('src')['timestamp'].agg(['min', 'max']).reset_index()
    time_span['time_span_days'] = (time_span['max'] - time_span['min']).dt.days
    result = amt_ratio.merge(time_span[['src', 'time_span_days']], on='src')
    result['night_amount_short_term_risk'] = result['deep_night_amt_ratio'] * (1.0 / (result['time_span_days'] + 1.0))
    result = result[['src', 'night_amount_short_term_risk']].copy()
    result.columns = ['wallet_id', 'night_amount_short_term_risk']
    return result


def knowfeat_amount_bin_skew_index(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    src_tiny = tx.groupby('src')['amount_abs'].apply(lambda x: (x <= 1).mean()).reset_index()
    src_tiny.columns = ['src', 'ultra_tiny_ratio']
    src_mid = tx.groupby('src')['amount_abs'].apply(lambda x: ((x >= 1000) & (x <= 50000)).mean()).reset_index()
    src_mid.columns = ['src', 'mid_amount_ratio']
    src_large = tx.groupby('src')['amount_abs'].apply(lambda x: (x >= 50000).mean()).reset_index()
    src_large.columns = ['src', 'large_amount_ratio']
    result = src_tiny.merge(src_mid, on='src').merge(src_large, on='src')
    result['amount_bin_skew_index'] = result['ultra_tiny_ratio'] / (result['mid_amount_ratio'] + result['large_amount_ratio'] + 0.01)
    result = result[['src', 'amount_bin_skew_index']].copy()
    result.columns = ['wallet_id', 'amount_bin_skew_index']
    return result


def knowfeat_counterparty_type_amount_divergence(open_df, tx_df, close_df):
    wallet_type = open_df[['wallet_id', 'wallet_type']].set_index('wallet_id')
    tx = tx_df.copy()
    tx['dst_wallet_type'] = tx['dst'].map(wallet_type['wallet_type'])
    tx_pos = tx[tx['amount'] > 0].copy()
    if len(tx_pos) == 0:
        return pd.DataFrame({'wallet_id': tx_df['src'].unique(), 'counterparty_type_amount_divergence': 0.0})
    total_amt = tx_pos.groupby('src')['amount'].sum().reset_index()
    total_amt.columns = ['wallet_id', 'total_amount']
    type1_amt = tx_pos[tx_pos['dst_wallet_type'] == 1].groupby('src')['amount'].sum().reset_index()
    type1_amt.columns = ['wallet_id', 'amount_to_type1']
    type3_amt = tx_pos[tx_pos['dst_wallet_type'] == 3].groupby('src')['amount'].sum().reset_index()
    type3_amt.columns = ['wallet_id', 'amount_to_type3']
    merged = pd.merge(total_amt, type1_amt, on='wallet_id', how='left').fillna(0)
    merged = pd.merge(merged, type3_amt, on='wallet_id', how='left').fillna(0)
    merged['type1_amount_ratio'] = merged['amount_to_type1'] / (merged['total_amount'] + 1e-10)
    merged['type3_amount_ratio'] = merged['amount_to_type3'] / (merged['total_amount'] + 1e-10)
    merged['counterparty_type_amount_divergence'] = merged['type1_amount_ratio'] - merged['type3_amount_ratio']
    return merged[['wallet_id', 'counterparty_type_amount_divergence']].copy()


def knowfeat_tiny_amount_concentration(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()

    def calc_tiny_concentration(group):
        tiny = group[group <= 1]
        if len(tiny) == 0:
            return 0.0
        value_counts = tiny.value_counts()
        if len(value_counts) == 0:
            return 0.0
        return value_counts.iloc[0] / len(tiny)

    result = tx.groupby('src')['amount_abs'].apply(calc_tiny_concentration).reset_index()
    result.columns = ['wallet_id', 'tiny_amount_concentration']
    return result


def knowfeat_time_concentration_risk_index(open_df, tx_df, close_df):
    tx = tx_df.copy()
    hour_counts = tx.groupby(['src', 'hour']).size().reset_index(name='count')
    src_total = tx.groupby('src').size().reset_index(name='total')
    hour_pct = hour_counts.merge(src_total, on='src')
    hour_pct['pct'] = hour_pct['count'] / hour_pct['total']
    max_hour_pct = hour_pct.groupby('src')['pct'].max().reset_index()
    max_hour_pct.columns = ['src', 'max_hour_pct']
    active_hours = tx.groupby('src')['hour'].nunique().reset_index()
    active_hours.columns = ['src', 'active_hours']
    result = max_hour_pct.merge(active_hours, on='src')
    result['time_concentration_risk_index'] = result['max_hour_pct'] / (result['active_hours'] + 1e-10)
    result = result[['src', 'time_concentration_risk_index']].copy()
    result.columns = ['wallet_id', 'time_concentration_risk_index']
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# KnowFeat 特征注册表（名称 -> 函数）
# ═══════════════════════════════════════════════════════════════════════════════

KNOWFEAT_FUNCTIONS = [
    ("small_test_round_amount_ratio", knowfeat_small_test_round_amount_ratio),
    ("amount_concentration_entropy", knowfeat_amount_concentration_entropy),
    ("top3_hour_concentration", knowfeat_top3_hour_concentration),
    ("fan_in_fan_out_ratio", knowfeat_fan_in_fan_out_ratio),
    ("mode_transition_entropy", knowfeat_mode_transition_entropy),
    ("new_counterparty_concentration", knowfeat_new_counterparty_concentration),
    ("counterparty_level_diversity_risk", knowfeat_counterparty_level_diversity_risk),
    ("amount_range_ratio_risk", knowfeat_amount_range_ratio_risk),
    ("high_risk_region_off_hour_ratio", knowfeat_high_risk_region_off_hour_ratio),
    ("rhythm_stability_risk", knowfeat_rhythm_stability_risk),
    ("net_flow_direction_risk", knowfeat_net_flow_direction_risk),
    ("enterprise_counterparty_concentration", knowfeat_enterprise_counterparty_concentration),
    ("first_small_test_ratio", knowfeat_first_small_test_ratio),
    ("night_amount_short_term_risk", knowfeat_night_amount_short_term_risk),
    ("amount_bin_skew_index", knowfeat_amount_bin_skew_index),
    ("counterparty_type_amount_divergence", knowfeat_counterparty_type_amount_divergence),
    ("tiny_amount_concentration", knowfeat_tiny_amount_concentration),
    ("time_concentration_risk_index", knowfeat_time_concentration_risk_index),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("用法: python build_all_113_features.py <交易CSV> <账户CSV> <注销CSV> <输出CSV> [账户特征CSV]")
        sys.exit(1)

    data_path = sys.argv[1]
    account_path = sys.argv[2]
    close_path = sys.argv[3]
    save_path = sys.argv[4]
    account_features_path = sys.argv[5] if len(sys.argv) > 5 else None

    start_time = time.time()
    feature_eng(data_path, account_path, save_path, close_path=close_path, account_features_path=account_features_path)
    end_time = time.time()
    runtime = end_time - start_time
    length = pd.read_csv(data_path).shape[0]
    print(f"程序运行总时间: {runtime:.2f} 秒")
    print(f"每条数据特征工程时间: {runtime/length:.4f} 秒")
