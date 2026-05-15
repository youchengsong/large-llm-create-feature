import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import math
import psutil
import os
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

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
      - S_history_count / S_geo_risk_202308 / S_time_risk_202308：来自账户开立表（历史无交易泄露）。
      - S_ibank_index / S_ibank_amount_ratio / S_counterparty_index / S_per_deal_match_times /
        S_daily_deal_match_times：来自全量交易表中「全局最近 24 小时」子窗口（各钱包在该窗口内的
        转入+转出笔数/金额等）；跨机构以本方与对手方 bank_account_number 是否不同判定；
        限额来自 transaction_config.json 的 wallet_level_limits，缺省用内置 1–4 档表。

    返回:
    - DataFrame: wallet_id + 上述各 S_* 列
    """
    import json

    print(
        "    计算S指标（开户静态 + 最近24h交易行为: ibank / counterparty / 限额笔数等）..."
    )

    # 确保 account_df 有必要的列
    account_df = account_df.copy()

    # 开户时间
    if 'wallet_open_timestamp' not in account_df.columns:
        if 'wallet_open_date' in account_df.columns:
            account_df['wallet_open_timestamp'] = pd.to_datetime(
                account_df['wallet_open_date'], errors='coerce'
            )
        else:
            print("  警告: 账户开立数据中没有开立时间信息，无法计算 S_time_risk_202308 / S_history_count")
            return pd.DataFrame(columns=['wallet_id'])
    else:
        account_df['wallet_open_timestamp'] = pd.to_datetime(
            account_df['wallet_open_timestamp'], errors='coerce'
        )

    # 手机号
    if 'wallet_open_tel' not in account_df.columns:
        print("  警告: 账户开立数据中没有手机号信息，S_history_count 将置 0")
        account_df['wallet_open_tel'] = ''

    # 地域代码
    if 'region_code' not in account_df.columns:
        print("  警告: 账户开立数据中没有地域代码信息，S_geo_risk_202308 将置 0")
        account_df['region_code'] = 0

    # wallet_id
    if 'wallet_id' not in account_df.columns:
        if 'acct_id' in account_df.columns:
            account_df['wallet_id'] = account_df['acct_id'].astype(str)
        else:
            print("  警告: 账户开立数据中没有 wallet_id / acct_id，无法计算 S 指标")
            return pd.DataFrame(columns=['wallet_id'])
    else:
        account_df['wallet_id'] = account_df['wallet_id'].astype(str)

    # 高风险地区代码
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

    # 夜间时段
    night_hours = {22, 23, 0, 1, 2, 3, 4, 5}

    # 为计算 S_history_count，先按开户时间排序
    account_df_sorted = account_df.sort_values('wallet_open_timestamp').copy()
    account_df_sorted['wallet_open_tel'] = account_df_sorted['wallet_open_tel'].astype(str)

    rows = []
    for _, row in account_df_sorted.iterrows():
        wid = str(row['wallet_id'])
        open_time = row['wallet_open_timestamp']
        phone = str(row.get('wallet_open_tel', ''))
        region_code = str(row.get('region_code', '0'))

        if pd.isna(open_time):
            # 没有开立时间：全部置 0
            S_history_count = 0
            S_geo = 0.0
            S_time = 0.0
        else:
            # S_history_count：同一手机号、开立时间早于当前钱包的数量
            if phone and phone != 'nan':
                hist = account_df_sorted[
                    (account_df_sorted['wallet_open_tel'] == phone)
                    & (account_df_sorted['wallet_open_timestamp'] < open_time)
                ]
                S_history_count = len(hist)
            else:
                S_history_count = 0

            # S_geo_risk_202308：是否高风险地区
            S_geo = 1.0 if region_code in high_risk_codes_set else 0.0

            # S_time_risk_202308：开户时间是否在夜间
            hour = pd.to_datetime(open_time).hour
            S_time = 1.0 if hour in night_hours else 0.0

        rows.append(
            {
                "wallet_id": wid,
                "S_history_count": S_history_count,
                "S_geo_risk_202308": S_geo,
                "S_time_risk_202308": S_time,
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


def compute_account_open_features_from_raw(account_df, transaction_config_path=None):
    """
    由 accounts 原始字段就地推导开立衍生特征（与 account_features_list 对齐）：
    - open_hour / is_work_hour / is_night_hour：由 wallet_open_timestamp，工作时间 9<=hour<18
    - auth_delay_days：realname_auth_time - wallet_open_timestamp（天，负值截为 0）
    - wallets_per_cert：同一 wallet_open_cert 下 wallet 数量
    - days_to_expire_from_current：wallet_open_cert_expire - 当前日期（剩余天数，已过期为负）
    - tel_reuse_count：同一 wallet_open_tel（手机号）上注册了多少个钱包（计数，无手机号则为 0）
    - is_high_risk_region：region_code 是否属于高风险行政区代码；默认与配置均为 ["540000","650000"]
    - log_balance：对 init_balance 取自然对数 ln，余额 <=0 时为 0
    """
    import json

    high_risk_codes = ["540000", "650000"]
    if transaction_config_path and os.path.exists(transaction_config_path):
        try:
            with open(transaction_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                if "region_code" in config and "high_risk_codes" in config.get("region_code", {}):
                    high_risk_codes = [str(c) for c in config["region_code"]["high_risk_codes"]]
                elif "distribution_based_generation" in config:
                    nd = config["distribution_based_generation"].get("normal_wallet_distributions", {})
                    rc = nd.get("region_code", {})
                    if "high_risk_codes" in rc:
                        high_risk_codes = [str(c) for c in rc["high_risk_codes"]]
        except Exception as e:
            print(f"  警告: 读取 transaction_config 高风险地区失败，使用默认: {e}")
    risk_set = set(str(c) for c in high_risk_codes)

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

    if "region_code" in account_df.columns:
        # 与 high_risk_codes 同一套字符串编码对齐（如 540000、650000）
        rc_num = pd.to_numeric(account_df["region_code"], errors="coerce").fillna(0).astype(np.int64).astype(str)
        account_df["is_high_risk_region"] = rc_num.isin(risk_set).astype(np.int64)
    else:
        account_df["is_high_risk_region"] = 0

    if "init_balance" in account_df.columns:
        bal = pd.to_numeric(account_df["init_balance"], errors="coerce").fillna(0.0)
        account_df["log_balance"] = np.where(bal > 0, np.log(bal), 0.0)
    else:
        account_df["log_balance"] = 0.0

    print(
        "  已从账户表原始字段计算: open_hour, is_work_hour, is_night_hour, auth_delay_days, "
        "wallets_per_cert, days_to_expire_from_current, tel_reuse_count, is_high_risk_region, log_balance"
    )


def feature_eng(data_path, account_path, save_path, close_path=None, account_features_path=None):
    """
    特征工程主函数
    
    参数:
    - data_path: 交易数据路径
    - account_path: 账户开立数据路径
    - save_path: 归一化后的特征保存路径
    - close_path: 注销数据路径（可选，如果为None则自动发现）
    - account_features_path: 可选。外部账户特征 CSV；仅当某列在账户表中不存在时从该文件补列（账户表内已计算的列优先）。
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
    
    # 保留原始列以便后续使用（包括银行账户号码和钱包级别）
    df_quantized = df[['Timestamp', 'From_Account', 'To_Account', 'amount']].copy()
    # 添加银行账户号码和钱包级别信息（如果存在）
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

    #交易双方是否为同一账户
    df_quantized['same_transactionaccount'] = (df_quantized['From_Account'] == df_quantized['To_Account']).astype(int)

    #提取小时作为一个新特征
    df_quantized['Timestamp_Hour'] = df_quantized['Timestamp'].dt.hour

    print("计算账户对特征（使用Expanding Window）...")
    # 创建无向交易对
    df_quantized['account_pair'] = df_quantized.apply(lambda row: tuple(sorted([str(row['From_Account']), str(row['To_Account'])])), axis=1)
    
    # 使用Expanding Window计算账户对特征
    # 账户间的交易次数(有向) - 使用expanding count
    df_quantized['transaction_count_directed'] = df_quantized.groupby(['From_Account', 'To_Account']).cumcount()  # 从0开始
    
    # 每对账户的交易次数(无向) - 使用expanding count
    df_quantized['transaction_count_Undirected'] = df_quantized.groupby('account_pair').cumcount()  # 从0开始
    
    #特定两个账户间的交易时间间隔 - 使用expanding window计算min和max
    # 将Timestamp转换为数值（纳秒）进行计算
    df_quantized['Timestamp_numeric'] = df_quantized['Timestamp'].astype('int64')
    df_quantized['pair_first_time_numeric'] = df_quantized.groupby('account_pair')['Timestamp_numeric'].transform('first')
    df_quantized['pair_last_time_numeric'] = df_quantized.groupby('account_pair')['Timestamp_numeric'].expanding().max().reset_index(level=0, drop=True)
    # 直接计算时间差（纳秒转换为分钟）
    df_quantized['accounttransaction_timeinterval'] = (df_quantized['pair_last_time_numeric'] - df_quantized['pair_first_time_numeric']) / 1e9 / 60
    df_quantized['accounttransaction_timeinterval'] = df_quantized['accounttransaction_timeinterval'].fillna(0)
    df_quantized.drop(columns=['Timestamp_numeric', 'pair_first_time_numeric', 'pair_last_time_numeric'], inplace=True, errors='ignore')
    
    #特定两个账户间的交易密度(无向)
    df_quantized['transaction_density_Undirected'] = df_quantized.apply(
        lambda row: row['transaction_count_Undirected'] / row['accounttransaction_timeinterval'] if row['accounttransaction_timeinterval'] > 0 else 0, axis=1)

    df_quantized.drop(columns=['account_pair', 'pair_first_time', 'pair_last_time'], inplace=True, errors='ignore')

    print("计算交易集中度特征（使用Expanding Window）...")
    # 用交易次数衡量交易集中度特征 - 使用expanding window
    # 对于每个From_Account，计算到当前To_Account的交易次数占比
    df_quantized['from_to_count'] = df_quantized.groupby(['From_Account', 'To_Account']).cumcount()  # 从0开始
    df_quantized['from_total_count'] = df_quantized.groupby('From_Account').cumcount()  # 从0开始
    df_quantized['send_concentration'] = df_quantized.apply(
        lambda row: row['from_to_count'] / row['from_total_count'] if row['from_total_count'] > 0 else 0,
        axis=1
    )
    
    df_quantized['to_from_count'] = df_quantized.groupby(['To_Account', 'From_Account']).cumcount()  # 从0开始
    df_quantized['to_total_count'] = df_quantized.groupby('To_Account').cumcount()  # 从0开始
    df_quantized['receive_concentration'] = df_quantized.apply(
        lambda row: row['to_from_count'] / row['to_total_count'] if row['to_total_count'] > 0 else 0,
        axis=1
    )

    # 用交易金额衡量交易集中度特征 - 使用expanding window
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
    df_sorted['Index'] = df_sorted.index  # 创建基于排序后数据的索引列

    # 创建交易记录（使用排序后数据）
    from_accounts = df_sorted[['Index','Timestamp', 'From_Account']].rename(columns={'From_Account':'Account'})
    to_accounts = df_sorted[['Index','Timestamp','To_Account']].rename(columns={'To_Account':'Account'})
    from_accounts['Type'] = 'From'
    to_accounts['Type'] = 'To'
    all_accounts = pd.concat([from_accounts, to_accounts], ignore_index=True)

    # 按账户和时间排序（保证rolling计算正确性）
    all_accounts = all_accounts.sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    # 使用时间窗口计算（确保使用排序后的时间序列）
    # 使用closed='left'排除当前时间点，使用历史1小时内的交易统计
    # 修复NaN/inf错误：先填充NaN和inf，再转换为int
    all_accounts['counts_1H_From'] = all_accounts.groupby('Account', group_keys=False).apply(
        lambda g: g[g['Type']=='From'].rolling('1H', on='Timestamp', closed='left')['Type'].count().reindex(g.index, fill_value=0)
    ).reset_index(level=0, drop=True)
    # 处理NaN和inf值
    all_accounts['counts_1H_From'] = all_accounts['counts_1H_From'].fillna(0).replace([np.inf, -np.inf], 0).astype(int)

    all_accounts['counts_1H_To'] = all_accounts.groupby('Account', group_keys=False).apply(
        lambda g: g[g['Type']=='To'].rolling('1H', on='Timestamp', closed='left')['Type'].count().reindex(g.index, fill_value=0)
    ).reset_index(level=0, drop=True)
    # 处理NaN和inf值
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
    # 短时间内的高频交易行为 - 使用expanding window计算账户的累计交易次数
    # 计算账户的累计交易次数
    df_quantized['transaction_counts_From'] = df_quantized.groupby('From_Account').cumcount()  # 从0开始
    df_quantized['transaction_counts_To'] = df_quantized.groupby('To_Account').cumcount()  # 从0开始

    # 计算短时间内频繁交易特征
    df_quantized['1H_frequency_From'] = df_quantized['counts_1H_From'] / df_quantized['transaction_counts_From'].replace(0, 1)
    df_quantized['1H_frequency_To'] = df_quantized['counts_1H_To'] / df_quantized['transaction_counts_To'].replace(0, 1)

    # 量化时间戳特征
    # 将Timestamp转换为Unix时间戳（秒）
    # 如果Timestamp是datetime类型，转换为纳秒再除以10^9得到秒
    # 如果Timestamp已经是数值类型，假设已经是秒，直接使用
    if pd.api.types.is_datetime64_any_dtype(df_quantized['Timestamp']):
        df_quantized['Timestamp'] = df_quantized['Timestamp'].astype('int64') / 10**9
    elif pd.api.types.is_numeric_dtype(df_quantized['Timestamp']):
        # 如果已经是数值类型，假设已经是秒，直接使用
        df_quantized['Timestamp'] = df_quantized['Timestamp'].astype('float64')
    else:
        # 如果是字符串或其他类型，先转换为datetime再转换为秒
        df_quantized['Timestamp'] = pd.to_datetime(df_quantized['Timestamp'], errors='coerce').astype('int64') / 10**9

    print("计算账户度特征（使用Expanding Window）...")
    # 使用Expanding Window计算度特征
    # 计算出度（From_Account 的累计次数）
    df_quantized['Out_Degree'] = df_quantized.groupby('From_Account').cumcount()  # 从0开始
    # 计算入度（To_Account 的累计次数）
    df_quantized['In_Degree'] = df_quantized.groupby('To_Account').cumcount()  # 从0开始
    # 计算出入度之比
    df_quantized['From_To_Ratio'] = df_quantized.apply(
        lambda row: row['Out_Degree'] / row['In_Degree'] if row['In_Degree'] > 0 else 0,
        axis=1)
    
    # 计算unique出度（向多少账户转过帐）- 使用expanding nunique
    # 注意：expanding().nunique()在pandas中可能不支持，需要使用自定义方法
    df_quantized['unique_outdegree'] = df_quantized.groupby('From_Account')['To_Account'].expanding().apply(
        lambda x: len(x.unique()), raw=False).reset_index(level=0, drop=True)
    
    # 计算unique入度（从多少账户收过款）
    df_quantized['unique_indegree'] = df_quantized.groupby('To_Account')['From_Account'].expanding().apply(
        lambda x: len(x.unique()), raw=False).reset_index(level=0, drop=True)
    
    # 计算unique出入度之比
    df_quantized['unique_From_To_Ratio'] = df_quantized.apply(
        lambda row: row['unique_outdegree'] / row['unique_indegree'] if row['unique_indegree'] > 0 else 0,
        axis=1)
    
    # 创建feature_df用于后续合并（从From_Account视角）
    feature_df_from = df_quantized[['From_Account', 'Out_Degree', 'In_Degree', 'From_To_Ratio', 
                                    'unique_outdegree', 'unique_indegree', 'unique_From_To_Ratio']].copy()
    feature_df_from = feature_df_from.rename(columns={'From_Account': 'Account'})
    
    # 创建feature_df用于后续合并（从To_Account视角）
    feature_df_to = df_quantized[['To_Account', 'Out_Degree', 'In_Degree', 'From_To_Ratio',
                                  'unique_outdegree', 'unique_indegree', 'unique_From_To_Ratio']].copy()
    feature_df_to = feature_df_to.rename(columns={'To_Account': 'Account'})
    
    # 合并（取每笔交易时的值）
    feature_df = pd.concat([feature_df_from, feature_df_to]).drop_duplicates(subset=['Account'], keep='last')
    feature_df = feature_df.groupby('Account').last().reset_index()

    print("计算账户金额统计特征（使用Expanding Window）...")
    # 使用Expanding Window计算账户发送的金额统计
    df_quantized['Total_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().sum().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Max_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().max().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Min_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().min().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Mean_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Std_Sent'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().std().reset_index(level=0, drop=True).fillna(0)
    df_quantized['volatility_Sent'] = df_quantized.apply(
        lambda row: row['Std_Sent'] / row['Mean_Sent'] if row['Mean_Sent'] > 0 else 0,
        axis=1
    )
    
    # 使用Expanding Window计算账户收到的金额统计
    df_quantized['Total_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().sum().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Max_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().max().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Min_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().min().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Mean_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_quantized['Std_Received'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().std().reset_index(level=0, drop=True).fillna(0)
    df_quantized['volatility_Received'] = df_quantized.apply(
        lambda row: row['Std_Received'] / row['Mean_Received'] if row['Mean_Received'] > 0 else 0,
        axis=1
    )
    
    #计算账户收付款比例
    df_quantized['Receive_Send_Ratio'] = df_quantized['Total_Received'] / df_quantized['Total_Sent'].replace(0, 1)
    df_quantized['Receive_Send_Ratio'] = df_quantized['Receive_Send_Ratio'].replace([float('inf'), -float('inf')], 2)
    
    #计算账户收付款差额
    df_quantized['Difference'] = df_quantized['Total_Received'] - df_quantized['Total_Sent']
    
    #计算账户平均交易金额与最大交易金额的比率
    df_quantized['avg_max_sent_ratio'] = df_quantized['Mean_Sent'] / (df_quantized['Max_Sent'] + 1)
    df_quantized['avg_max_receive_ratio'] = df_quantized['Mean_Received'] / (df_quantized['Max_Received'] + 1)
    
    #计算账户最大收款金额和最大付款金额的比率
    df_quantized['max_sent_receive_ratio'] = df_quantized['Max_Sent'] / (df_quantized['Max_Received'] + 1)
    
    #计算账户收付款波动性比例
    df_quantized['volatility_ratio'] = df_quantized['volatility_Sent'] / (df_quantized['volatility_Received'] + 1)

    # 计算异常大额交易占比 - 使用expanding window计算z-score
    # 计算expanding标准差
    df_quantized['amount_paid_std_expanding'] = df_quantized.groupby('From_Account')['Amount Paid'].expanding().std().reset_index(level=0, drop=True).fillna(0)
    df_quantized['amount_received_std_expanding'] = df_quantized.groupby('To_Account')['Amount Received'].expanding().std().reset_index(level=0, drop=True).fillna(0)
    
    # 使用expanding标准差计算z-score
    df_quantized['z_score_paid'] = df_quantized.apply(
        lambda row: (row['Amount Paid'] - row['Mean_Sent']) / row['amount_paid_std_expanding'] if row['amount_paid_std_expanding'] > 0 else 0,
        axis=1
    )
    df_quantized['z_score_received'] = df_quantized.apply(
        lambda row: (row['Amount Received'] - row['Mean_Received']) / row['amount_received_std_expanding'] if row['amount_received_std_expanding'] > 0 else 0,
        axis=1
    )
    
    # 清理临时列
    df_quantized.drop(columns=['amount_paid_std_expanding', 'amount_received_std_expanding'], inplace=True, errors='ignore')
    
    # 标记Z-Score大于3或小于-3的值为异常值
    df_quantized['large_paid'] = (df_quantized['z_score_paid'].abs() > 3).astype(int)
    df_quantized['large_received'] = (df_quantized['z_score_received'].abs() > 3).astype(int)
    
    # 异常值比例 - 使用expanding mean
    df_quantized['large_transaction_ratio_paid'] = df_quantized.groupby('From_Account')['large_paid'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_quantized['large_transaction_ratio_received'] = df_quantized.groupby('To_Account')['large_received'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    df_quantized['large_transaction_ratio'] = (df_quantized['large_transaction_ratio_paid'] + df_quantized['large_transaction_ratio_received']) / 2
    
    # 创建金额特征DataFrame（从From_Account视角）
    amount_feature_from = df_quantized[['From_Account', 'Total_Sent', 'Max_Sent', 'Min_Sent', 'Mean_Sent', 
                                       'Std_Sent', 'volatility_Sent', 'large_transaction_ratio_paid']].copy()
    amount_feature_from = amount_feature_from.rename(columns={'From_Account': 'Account', 'large_transaction_ratio_paid': 'large_transaction_ratio_paid_from'})
    
    # 创建金额特征DataFrame（从To_Account视角）
    amount_feature_to = df_quantized[['To_Account', 'Total_Received', 'Max_Received', 'Min_Received', 'Mean_Received',
                                     'Std_Received', 'volatility_Received', 'large_transaction_ratio_received']].copy()
    amount_feature_to = amount_feature_to.rename(columns={'To_Account': 'Account', 'large_transaction_ratio_received': 'large_transaction_ratio_received_to'})
    
    # 合并金额特征
    amount_feature = pd.merge(amount_feature_from.groupby('Account').last().reset_index(),
                              amount_feature_to.groupby('Account').last().reset_index(),
                              on='Account', how='outer').fillna(0)
    
    # 计算合并后的比率特征
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
    # 使用Expanding Window计算账户活跃时间
    # 将from_account和to_account合并为一个列
    from_accounts = df_quantized[['From_Account', 'Timestamp']].rename(columns={'From_Account': 'Account'})
    to_accounts = df_quantized[['To_Account', 'Timestamp']].rename(columns={'To_Account': 'Account'})
    all_accountstimestamp = pd.concat([from_accounts, to_accounts]).sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    # 计算每个账户的第一个和最后一个交易之间的时间差（使用expanding window）
    # 注意：Timestamp在第459行已经被转换为Unix时间戳（秒），所以直接使用即可
    # 将Timestamp转换为数值进行计算
    if pd.api.types.is_datetime64_any_dtype(all_accountstimestamp['Timestamp']):
        # 如果是datetime类型，转换为纳秒再除以10^9得到秒
        all_accountstimestamp['Timestamp_numeric'] = all_accountstimestamp['Timestamp'].astype('int64') / 1e9
    else:
        # 如果已经是数值类型（秒），直接使用
        all_accountstimestamp['Timestamp_numeric'] = all_accountstimestamp['Timestamp'].astype('float64')
    
    all_accountstimestamp['first_time_numeric'] = all_accountstimestamp.groupby('Account')['Timestamp_numeric'].transform('first')
    all_accountstimestamp['last_time_numeric'] = all_accountstimestamp.groupby('Account')['Timestamp_numeric'].expanding().max().reset_index(level=0, drop=True)
    # 直接计算时间差（秒）
    all_accountstimestamp['activetime'] = all_accountstimestamp['last_time_numeric'] - all_accountstimestamp['first_time_numeric']
    all_accountstimestamp['activetime'] = all_accountstimestamp['activetime'].fillna(0)
    all_accountstimestamp.drop(columns=['Timestamp_numeric', 'first_time_numeric', 'last_time_numeric'], inplace=True, errors='ignore')
    
    # 计算每个账户的交易次数（使用cumcount）
    all_accountstimestamp['transaction_count'] = all_accountstimestamp.groupby('Account').cumcount()  # 从0开始

    # 计算交易频率（交易次数/活跃时间）
    all_accountstimestamp['transaction_frequency'] = all_accountstimestamp.apply(
        lambda row: row['transaction_count'] / row['activetime'] if row['activetime'] > 0 else 0,
        axis=1)
    
    # 获取每个账户的最新值
    account_activity = all_accountstimestamp.groupby('Account').last()[['activetime', 'transaction_frequency']].reset_index()
    account_activity = account_activity.drop_duplicates(subset=['Account'], keep='last')

    feature_df = feature_df.merge(account_activity, on='Account', how='outer').fillna(0)

    print("计算最后交易特征（使用Expanding Window）...")
    # 查看账户最后一笔交易是收款0 还是付款1, 以及其金额 - 使用expanding window
    df_from = df_quantized[['From_Account', 'Timestamp','Amount Paid']].rename(columns={'From_Account': 'Account','Amount Paid': 'Amount'})
    df_from['direction'] = 1
    df_from = df_from.sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    df_to = df_quantized[['To_Account', 'Timestamp', 'Amount Received']].rename(columns={'To_Account': 'Account','Amount Received': 'Amount'})
    df_to['direction'] = 0
    df_to = df_to.sort_values(['Account', 'Timestamp']).reset_index(drop=True)

    # 合并付款和收款记录
    all_transactions = pd.concat([df_from, df_to]).sort_values(['Account', 'Timestamp']).reset_index(drop=True)
    
    # 使用expanding window获取每个账户的最后交易信息
    last_transaction = all_transactions.groupby('Account').last().reset_index()
    last_transaction = last_transaction[['Account', 'direction', 'Amount']].rename(columns={'direction': 'last_direction', 'Amount': 'last_amount'})

    feature_df = feature_df.merge(last_transaction, on='Account', how='outer').fillna(0)

    print("计算交易时间间隔特征（使用Expanding Window）...")
    # 查看付款、收款交易之间的平均时间差 - 使用expanding window
    payment_transactions = all_transactions[all_transactions['direction'] == 1].copy()
    receipt_transactions = all_transactions[all_transactions['direction'] == 0].copy()
    
    # 计算每两个连续付款交易之间的时间差
    # 注意：Timestamp 可能已经是数值类型（秒），需要判断
    payment_transactions['time_diff'] = payment_transactions.groupby('Account')['Timestamp'].diff()
    receipt_transactions['time_diff'] = receipt_transactions.groupby('Account')['Timestamp'].diff()
    
    # 将timedelta转换为数值（秒）进行计算
    # 如果 time_diff 已经是数值类型（Timestamp 被转换为秒），直接使用；否则使用 .dt.total_seconds()
    if pd.api.types.is_timedelta64_dtype(payment_transactions['time_diff']):
        payment_transactions['time_diff_numeric'] = payment_transactions['time_diff'].dt.total_seconds()
        receipt_transactions['time_diff_numeric'] = receipt_transactions['time_diff'].dt.total_seconds()
    else:
        # Timestamp 已经是数值（秒），time_diff 也是秒
        payment_transactions['time_diff_numeric'] = payment_transactions['time_diff']
        receipt_transactions['time_diff_numeric'] = receipt_transactions['time_diff']
    
    # 计算每个账户的平均时间差（使用expanding mean）
    payment_transactions['avg_payment_time_diff'] = payment_transactions.groupby('Account')['time_diff_numeric'].expanding().mean().reset_index(level=0, drop=True)
    receipt_transactions['avg_receipt_time_diff'] = receipt_transactions.groupby('Account')['time_diff_numeric'].expanding().mean().reset_index(level=0, drop=True)
    
    # 清理临时列
    payment_transactions.drop(columns=['time_diff_numeric'], inplace=True, errors='ignore')
    receipt_transactions.drop(columns=['time_diff_numeric'], inplace=True, errors='ignore')
    
    # 获取每个账户的最新值
    average_payment_time_diff = payment_transactions.groupby('Account')['avg_payment_time_diff'].last().reset_index()
    average_receipt_time_diff = receipt_transactions.groupby('Account')['avg_receipt_time_diff'].last().reset_index()
    average_payment_time_diff = average_payment_time_diff.fillna(0)
    average_receipt_time_diff = average_receipt_time_diff.fillna(0)

    feature_df = feature_df.merge(average_payment_time_diff, on='Account', how='outer').merge(average_receipt_time_diff, on='Account', how='outer').fillna(0)

    # 计算账户的非活跃期长度 - 使用expanding window
    all_transactions = all_transactions.sort_values(by=['Account', 'Timestamp'])
    all_transactions['time_diff'] = all_transactions.groupby('Account')['Timestamp'].diff()
    
    # 将timedelta转换为数值（秒）进行计算
    # 如果 time_diff 已经是数值类型（Timestamp 被转换为秒），直接使用；否则使用 .dt.total_seconds()
    if pd.api.types.is_timedelta64_dtype(all_transactions['time_diff']):
        all_transactions['time_diff_numeric'] = all_transactions['time_diff'].dt.total_seconds()
    else:
        # Timestamp 已经是数值（秒），time_diff 也是秒
        all_transactions['time_diff_numeric'] = all_transactions['time_diff']
    
    # 计算非活跃期的统计数据（使用expanding统计）
    all_transactions['avg_inactive_period'] = all_transactions.groupby('Account')['time_diff_numeric'].expanding().mean().reset_index(level=0, drop=True)
    all_transactions['max_inactive_period'] = all_transactions.groupby('Account')['time_diff_numeric'].expanding().max().reset_index(level=0, drop=True)
    all_transactions['min_inactive_period'] = all_transactions.groupby('Account')['time_diff_numeric'].expanding().min().reset_index(level=0, drop=True)
    
    # 清理临时列
    all_transactions.drop(columns=['time_diff_numeric'], inplace=True, errors='ignore')
    
    # 获取每个账户的最新值
    inactive_periods = all_transactions.groupby('Account')[['avg_inactive_period', 'max_inactive_period', 'min_inactive_period']].last().reset_index()
    inactive_periods = inactive_periods.fillna(0)

    feature_df = feature_df.merge(inactive_periods, on='Account', how='outer').fillna(0)

    # 读取账户开立数据
    print("读取账户开立数据...")
    account_df = pd.read_csv(account_path)
    # 适配新的列名：wallet_id -> acct_id, is_abnormal -> is_laundering
    account_df['is_laundering'] = account_df['is_abnormal'].replace({False: 0, True: 1, 'False': 0, 'True': 1, '0': 0, '1': 1})
    account_df = account_df.rename(columns={'is_laundering': 'account_is_laundering'})

    account_features_list = [
        'open_hour', 'is_work_hour', 'is_night_hour',
        'auth_delay_days', 'wallets_per_cert', 'days_to_expire_from_current',
        'tel_reuse_count', 'is_high_risk_region', 'log_balance',
    ]

    cfg_path_early = _resolve_transaction_config_path(data_path, account_path)
    print("由账户原始字段计算开立衍生特征...")
    compute_account_open_features_from_raw(account_df, transaction_config_path=cfg_path_early)

    # 可选：外部账户特征 CSV（仅补充账户表中仍不存在的列；已计算列优先）
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

    # 重命名以匹配旧代码的期望
    if 'bank_account_number' in account_df.columns:
        account_df = account_df.rename(columns={'bank_account_number': 'bank_id'})
    # 保持 wallet_type 原名，不重命名为 account_type

    # 添加balance_limit列（如果不存在）
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
    # 合并账户开立信息到feature_df
    feature_df = pd.merge(account_df, feature_df, left_on='acct_id', right_on='Account', how='inner')
    feature_df.drop(columns=['acct_id'], inplace=True, errors='ignore')
    feature_df = feature_df.fillna(0)

    # 将账户开立信息合并到df_quantized（用于From_Account和To_Account）
    # 注意：account_df的wallet_id已经被rename为acct_id，需要重新创建account_info_df
    account_info_df = account_df.copy()
    account_info_df = account_info_df.rename(columns={'acct_id': 'Account'})
    
    # 合并From_Account的账户信息
    df_quantized = df_quantized.merge(account_info_df, left_on='From_Account', right_on='Account', how='left', suffixes=('', '_From'))
    df_quantized = df_quantized.drop(columns=['Account'], errors='ignore')
    
    # 合并To_Account的账户信息
    df_quantized = df_quantized.merge(account_info_df, left_on='To_Account', right_on='Account', how='left', suffixes=('_From', '_To'))
    df_quantized = df_quantized.drop(columns=['Account'], errors='ignore')
    
    # 合并账户开立特征（由账户表计算 + 可选外部 CSV 补缺列）
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
    print("计算S指标（S_history_count, S_geo_risk_202308等）...")
    # 注意：df_quantized的Timestamp在第459行已经被转换为Unix时间戳（秒）
    # 但calculate_s_indicators需要datetime类型，所以需要先恢复Timestamp为datetime
    # 保存当前Timestamp（可能是数值类型）
    timestamp_backup = df_quantized['Timestamp'].copy()
    
    # 恢复Timestamp为datetime类型（用于S指标计算）
    if pd.api.types.is_numeric_dtype(df_quantized['Timestamp']):
        # 如果是数值类型（Unix时间戳，秒），转换为datetime
        df_quantized['Timestamp'] = pd.to_datetime(df_quantized['Timestamp'], unit='s', errors='coerce')
    else:
        # 如果已经是datetime类型，直接使用
        df_quantized['Timestamp'] = pd.to_datetime(df_quantized['Timestamp'], errors='coerce')
    
    # 需要重新读取account_df以获取完整信息（因为之前可能被修改了）
    account_df_for_s = pd.read_csv(account_path)
    if 'wallet_open_timestamp' not in account_df_for_s.columns:
        if 'wallet_open_date' in account_df_for_s.columns:
            account_df_for_s['wallet_open_timestamp'] = pd.to_datetime(account_df_for_s['wallet_open_date'], errors='coerce')
    
    # 确保有必要的列
    if 'account_is_laundering' not in account_df_for_s.columns:
        if 'is_abnormal' in account_df_for_s.columns:
            account_df_for_s['account_is_laundering'] = account_df_for_s['is_abnormal'].replace({False: 0, True: 1, 'False': 0, 'True': 1, '0': 0, '1': 1})
        else:
            account_df_for_s['account_is_laundering'] = 0
    
    # 尝试从data_path目录查找transaction_config.json
    transaction_config_path = None
    data_dir = os.path.dirname(data_path)
    potential_config_path = os.path.join(data_dir, "transaction_config.json")
    if os.path.exists(potential_config_path):
        transaction_config_path = potential_config_path
    else:
        # 尝试从account_path目录查找
        account_dir = os.path.dirname(account_path)
        potential_config_path = os.path.join(account_dir, "transaction_config.json")
        if os.path.exists(potential_config_path):
            transaction_config_path = potential_config_path
    
    s_indicators_df = calculate_s_indicators(df_quantized, account_df_for_s, transaction_config_path)
    
    # S指标计算完成后，恢复Timestamp为数值类型（Unix时间戳，秒）
    # 这样后续的特征工程流程可以继续使用数值类型的Timestamp
    df_quantized['Timestamp'] = timestamp_backup
    
    # 合并S指标到df_quantized（为From_Account和To_Account分别合并）
    if s_indicators_df is not None and len(s_indicators_df) > 0:
        print(f"  成功计算S指标，共 {len(s_indicators_df)} 个钱包的S指标")
        print(f"  S指标列: {list(s_indicators_df.columns)}")
        
        # 确保wallet_id类型一致
        s_indicators_df['wallet_id'] = s_indicators_df['wallet_id'].astype(str)
        df_quantized['From_Account'] = df_quantized['From_Account'].astype(str)
        df_quantized['To_Account'] = df_quantized['To_Account'].astype(str)
        
        # 合并From_Account的S指标（添加S_src前缀）
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
        
        # 合并To_Account的S指标（添加S_dst前缀）
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
        
        # 填充缺失值
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
        # 即使没有S指标，也要创建默认的S指标列（全为0）
        s_indicator_cols = [
            'S_history_count',
            'S_geo_risk_202308',
            'S_time_risk_202308',
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
    
    # 将 feature_df 中仅存在于账户级汇总表的列合并回 df_quantized（按 From / To 对齐）
    # df_quantized 上已有逐笔 expanding 特征；feature_df 另有 activetime、last_*、inactive 等账户汇总列
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
        print(f"  已合并 {len(_feat_from_df_cols)} 个特征 × 2（From/To）: {', '.join(_feat_from_df_cols)}")
    else:
        print("  警告: feature_df 中未找到 activetime/last_* 等预期列，跳过 feature_df 合并")

    df_new = df_quantized.copy()

    # 删除账户ID列和重复特征（在特征工程阶段完成）
    print("删除账户ID列和重复特征...")
    
    # 删除账户ID列
    columns_to_drop = ['From_Account', 'To_Account']
    for col in columns_to_drop:
        if col in df_new.columns:
            df_new = df_new.drop(columns=[col])
    
    # 删除重复特征
    duplicate_features_to_drop = [
        # 金额重复：保留 amount，删除 Amount Paid 和 Amount Received（它们在单笔交易中相等）
        'Amount Paid', 'Amount Received',
        # 银行账户号码重复：保留原始列，删除合并后的列
        'bank_id_From', 'bank_id_To',
        # 钱包级别保留：wallet_level_From 和 wallet_level_To 保留（不删除）
        # 交易计数重复：保留 Out_Degree 和 In_Degree（更通用的名称），删除 transaction_counts_From/To
        'transaction_counts_From', 'transaction_counts_To',
        # balance_limit 列（如果全为0或与 same_transactionaccount 相同，删除）
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
    # 删除Timestamp列（如果存在），因为它已经被转换为Unix时间戳数值，不应该作为特征保存
    # 避免保存CSV时被错误解释为1970-01-01这样的日期
    if 'Timestamp' in df_original.columns:
        df_original = df_original.drop(columns=['Timestamp'])
    df_original['is_laundering'] = labels
    df_original['account_is_laundering_From'] = account_is_laundering_From
    df_original['account_is_laundering_To'] = account_is_laundering_To
    df_original = pd.concat([id_index.reset_index(drop=True), df_original], axis=1)
    
    # 生成原始特征文件路径
    save_dir = os.path.dirname(save_path)
    save_basename = os.path.basename(save_path)
    save_name_without_ext = os.path.splitext(save_basename)[0]
    original_features_path = os.path.join(save_dir, f"{save_name_without_ext}_original.csv")
    
    df_original.to_csv(original_features_path, index=False)
    print(f"  原始特征已保存到: {original_features_path}")
    
    # 准备归一化的数据（移除标签列和Timestamp列）
    # 删除Timestamp列（如果存在），因为它已经被转换为Unix时间戳数值，不应该作为特征进行归一化
    if 'Timestamp' in df_new.columns:
        df_new = df_new.drop(columns=['Timestamp'])
    df_new = df_new.drop(['is_laundering'], axis=1, errors='ignore')
    if 'account_is_laundering_From' in df_new.columns:
        df_new = df_new.drop(['account_is_laundering_From'], axis=1)
    if 'account_is_laundering_To' in df_new.columns:
        df_new = df_new.drop(['account_is_laundering_To'], axis=1)

    # 对数据集进行归一化
    print("对特征进行归一化...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df_new)
    df_normalized = scaler.transform(df_new)

    ## 将归一化的数据转化为DataFrame 对象，并且设置列名
    df_normalized = pd.DataFrame(df_normalized, columns=df_new.columns)
    df_normalized['is_laundering'] = labels.tolist()
    df_normalized['account_is_laundering_From'] = account_is_laundering_From.tolist()
    df_normalized['account_is_laundering_To'] = account_is_laundering_To.tolist()
    df_normalized = pd.concat([id_index.reset_index(drop=True), df_normalized], axis=1)

    # 在最终特征中保留原始的 src / dst 列（按 tx_id 对齐）
    # 注意：此处不对 src / dst 做任何归一化或变换，只作原样保留，便于后续按节点划分或回溯原交易
    if {'tx_id', 'src', 'dst'}.issubset(set(df.columns)):
        src_dst_map = df[['tx_id', 'src', 'dst']].copy()
        src_dst_map = src_dst_map.rename(columns={'tx_id': 'id'})
        dup_ids = src_dst_map['id'].duplicated(keep=False)
        if dup_ids.any():
            n_dup = int(dup_ids.sum())
            print(f"  警告: 交易数据中存在重复 tx_id（共 {n_dup} 行），merge 前已按 id 保留首行，避免行数膨胀与时间戳错位")
        src_dst_map = src_dst_map.drop_duplicates(subset=['id'], keep='first')
        df_normalized = df_normalized.merge(src_dst_map, on='id', how='left')
    else:
        print("  警告: 原始数据中未同时发现 'tx_id', 'src', 'dst' 三列，无法在输出中保留 src/dst")

    # 排序用时间戳：在删除 id 之前按 tx_id 映射，避免 merge 后行数与 df 不一致导致无法对齐
    _ts_row = df[['tx_id', 'timestamp']].drop_duplicates(subset=['tx_id'], keep='first').copy()
    _ts_map = pd.Series(
        pd.to_datetime(_ts_row['timestamp']).values,
        index=_ts_row['tx_id'].astype(str).values,
    )
    df_normalized['_ts_sort'] = df_normalized['id'].astype(str).map(_ts_map)
    if df_normalized['_ts_sort'].isna().any():
        n_bad = int(df_normalized['_ts_sort'].isna().sum())
        print(f"  警告: 有 {n_bad} 行 id 未映射到原始 timestamp（排序时 NaT 会靠前）")

    # 删除指定的列
    print("删除指定列...")
    # 注意：
    # - wallet_type_From 和 wallet_type_To 保留，不删除
    # - 此处不删除 src / dst，按照需求在最终输出中保留原始节点列
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
    
    # 仅保存完整的归一化特征文件（不在此处划分训练/测试集）
    df_normalized.to_csv(save_path, index=False)
    print(f"归一化后的完整特征已保存到: {save_path}")
    
    print("\n特征工程已完成。")

if __name__ == "__main__":
    # 交易特征工程模式
    if len(sys.argv) < 5:
        print("交易特征工程模式需要至少4个参数:")
        print("  python feature_engineering_1230.py <交易CSV文件> <账户CSV文件> <注销CSV文件> <输出CSV文件> [账户特征CSV文件]")
        sys.exit(1)
    #  python feature_engineering_0116.py tr.csv  open.csv  close.csv  feature_0122.csv
    data_path = sys.argv[1]
    account_path = sys.argv[2]
    close_path = sys.argv[3]
    save_path = sys.argv[4]
    # 可选的账户特征文件路径（第5个参数）
    account_features_path = sys.argv[5] if len(sys.argv) > 5 else None
    
    start_time = time.time()
    feature_eng(data_path, account_path, save_path, close_path=close_path, account_features_path=account_features_path)
    end_time = time.time()
    runtime = end_time - start_time
    length = pd.read_csv(data_path).shape[0]
    print(f"程序运行总时间: {runtime:.2f} 秒")
    print(f"每条数据特征工程时间: {runtime/length:.4f} 秒")

