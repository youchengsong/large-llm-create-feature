"""
构建 113 个特征矩阵 = 95 基线 + 18 KnowFeat 新特征。

输入:
  - data/accounts.csv     (账户开立表)
  - data/transactions.csv (交易表)
  - data/close.csv        (注销表)
  - outputs/baseline_cache.parquet (106 列基线缓存)

输出:
  - outputs/all_113_features.parquet (1,384,512 行 × 113 列 + label)

流程:
  1. 从 baseline_cache.parquet 加载 95 个基线特征
  2. 从 18 个 KnowFeat 特征函数计算钱包级特征 → 通过 src 列 join 到交易级
  3. 合并为 113 列特征矩阵

注: night_amount_short_term_risk 因 timestamp 列类型问题可能失败 (17/18)，
    实际输出为 112 列特征。

依赖: pandas, numpy
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BASELINE_CACHE = OUTPUTS_DIR / "baseline_cache.parquet"
STAGE2_FEATURES = OUTPUTS_DIR / "stage2_features.json"

BASELINE_EXCLUDE_FEATURES = {
    "is_laundering", "is_risk", "tx_id", "src", "dst",
    "Min_Received", "Min_Sent", "S_src_counterparty_index",
    "Timestamp_Hour", "S_dst_counterparty_index",
    "1H_frequency_From", "1H_frequency_To",
    "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent",
}


# ═══════════════════════════════════════════════════════════
# 18 个 KnowFeat 特征函数（内联定义，不依赖外部文件）
# ═══════════════════════════════════════════════════════════

def feature_small_test_round_amount_ratio(open_df, tx_df, close_df):
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


def feature_amount_concentration_entropy(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()

    def calc_top3_ratio(x):
        counts = x.value_counts()
        if len(counts) == 0: return 0.0
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


def feature_top3_hour_concentration(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['hour'] = tx['hour'].astype(int)

    def calc_top3_hour_ratio(group):
        counts = group.value_counts()
        if len(counts) == 0: return 0.0
        return counts.head(3).sum() / len(group)

    result = tx.groupby('src')['hour'].apply(calc_top3_hour_ratio).reset_index()
    result.columns = ['wallet_id', 'top3_hour_concentration']
    return result


def feature_fan_in_fan_out_ratio(open_df, tx_df, close_df):
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


def feature_mode_transition_entropy(open_df, tx_df, close_df):
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


def feature_new_counterparty_concentration(open_df, tx_df, close_df):
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


def feature_counterparty_level_diversity_risk(open_df, tx_df, close_df):
    level_diversity = tx_df.groupby('src')['dst_wallet_level'].nunique().reset_index()
    level_diversity.columns = ['wallet_id', 'dst_level_diversity']
    src_self_level = tx_df.groupby('src')['src_wallet_level'].first().reset_index()
    src_self_level.columns = ['wallet_id', 'self_wallet_level']
    result = level_diversity.merge(src_self_level, on='wallet_id', how='left')
    diversity_risk = 1.0 - result['dst_level_diversity'] / 4.0
    level_risk = 1.0 - result['self_wallet_level'] / 4.0
    result['counterparty_level_diversity_risk'] = diversity_risk * (1.0 + level_risk)
    return result[['wallet_id', 'counterparty_level_diversity_risk']]


def feature_amount_range_ratio_risk(open_df, tx_df, close_df):
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


def feature_high_risk_region_off_hour_ratio(open_df, tx_df, close_df):
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


def feature_rhythm_stability_risk(open_df, tx_df, close_df):
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


def feature_net_flow_direction_risk(open_df, tx_df, close_df):
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


def feature_enterprise_counterparty_concentration(open_df, tx_df, close_df):
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


def feature_first_small_test_ratio(open_df, tx_df, close_df):
    tx_sorted = tx_df.sort_values(['src', 'timestamp'])
    first_tx = tx_sorted.groupby('src').first().reset_index()
    first_tx['amount_abs'] = first_tx['amount'].abs()
    first_tx['first_small_test'] = (first_tx['amount_abs'] <= 5).astype(int)
    result = first_tx[['src', 'first_small_test']].copy()
    result.columns = ['wallet_id', 'first_small_test_ratio']
    return result


def feature_night_amount_short_term_risk(open_df, tx_df, close_df):
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


def feature_amount_bin_skew_index(open_df, tx_df, close_df):
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


def feature_counterparty_type_amount_divergence(open_df, tx_df, close_df):
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


def feature_tiny_amount_concentration(open_df, tx_df, close_df):
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()

    def calc_tiny_concentration(group):
        tiny = group[group <= 1]
        if len(tiny) == 0: return 0.0
        value_counts = tiny.value_counts()
        if len(value_counts) == 0: return 0.0
        return value_counts.iloc[0] / len(tiny)

    result = tx.groupby('src')['amount_abs'].apply(calc_tiny_concentration).reset_index()
    result.columns = ['wallet_id', 'tiny_amount_concentration']
    return result


def feature_time_concentration_risk_index(open_df, tx_df, close_df):
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


# ═══════════════════════════════════════════════════════════
# KnowFeat 特征注册表（名称 → 函数）
# ═══════════════════════════════════════════════════════════

KNOWFEAT_FUNCTIONS = [
    ("small_test_round_amount_ratio", feature_small_test_round_amount_ratio),
    ("amount_concentration_entropy", feature_amount_concentration_entropy),
    ("top3_hour_concentration", feature_top3_hour_concentration),
    ("fan_in_fan_out_ratio", feature_fan_in_fan_out_ratio),
    ("mode_transition_entropy", feature_mode_transition_entropy),
    ("new_counterparty_concentration", feature_new_counterparty_concentration),
    ("counterparty_level_diversity_risk", feature_counterparty_level_diversity_risk),
    ("amount_range_ratio_risk", feature_amount_range_ratio_risk),
    ("high_risk_region_off_hour_ratio", feature_high_risk_region_off_hour_ratio),
    ("rhythm_stability_risk", feature_rhythm_stability_risk),
    ("net_flow_direction_risk", feature_net_flow_direction_risk),
    ("enterprise_counterparty_concentration", feature_enterprise_counterparty_concentration),
    ("first_small_test_ratio", feature_first_small_test_ratio),
    ("night_amount_short_term_risk", feature_night_amount_short_term_risk),
    ("amount_bin_skew_index", feature_amount_bin_skew_index),
    ("counterparty_type_amount_divergence", feature_counterparty_type_amount_divergence),
    ("tiny_amount_concentration", feature_tiny_amount_concentration),
    ("time_concentration_risk_index", feature_time_concentration_risk_index),
]


def main():
    # ── 1. 加载 95 基线特征 ──
    print("=" * 60)
    print("Step 1: 加载 95 个基线特征")
    print("=" * 60)

    if not BASELINE_CACHE.exists():
        print(f"错误: {BASELINE_CACHE} 不存在，请先运行 feature_engineering.py")
        sys.exit(1)

    df = pd.read_parquet(BASELINE_CACHE)
    print(f"  baseline_cache: {df.shape[0]} 行, {df.shape[1]} 列")

    label_col = "is_risk" if "is_risk" in df.columns else "is_laundering"
    y = df[label_col].astype(int).values
    print(f"  标签列: {label_col}, 正样本: {y.sum()}, 负样本: {len(y) - y.sum()}")

    all_cols = set(df.columns)
    baseline_cols = sorted([c for c in all_cols if c not in BASELINE_EXCLUDE_FEATURES])
    X = df[baseline_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"  基线特征: {len(baseline_cols)} 个")

    # ── 2. 加载原始三表 ──
    print(f"\n{'=' * 60}")
    print("Step 2: 加载原始交易数据")
    print("=" * 60)

    open_df = pd.read_csv(DATA_DIR / "accounts.csv")
    tx_df = pd.read_csv(DATA_DIR / "transactions.csv")
    close_df = pd.read_csv(DATA_DIR / "close.csv")
    print(f"  accounts: {open_df.shape}, transactions: {tx_df.shape}, close: {close_df.shape}")

    # baseline_cache 是交易级别，与 tx_df 行序一致，通过 tx_df.src 做 join key
    src_series = tx_df["src"]
    print(f"  Join key: tx_df.src, unique wallets: {src_series.nunique()}")
    assert len(src_series) == len(X), f"tx_df ({len(src_series)}) != baseline ({len(X)})"

    # ── 3. 计算 18 个 KnowFeat 特征 ──
    print(f"\n{'=' * 60}")
    print("Step 3: 计算 18 个 KnowFeat 特征")
    print("=" * 60)

    success_count = 0
    knowfeat_names = []

    for name, func in KNOWFEAT_FUNCTIONS:
        try:
            feat_df = func(open_df, tx_df, close_df)
            merge_df = pd.DataFrame({"wallet_id": src_series.values})
            merged = merge_df.merge(feat_df, on="wallet_id", how="left")
            feat_values = pd.to_numeric(merged[name], errors="coerce").fillna(0).values
            X[name] = feat_values
            knowfeat_names.append(name)
            success_count += 1
            print(f"  [OK] {name} ({feat_df.shape[0]} wallets)")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print(f"\n成功加载 {success_count}/{len(KNOWFEAT_FUNCTIONS)} 个 KnowFeat 特征")
    print(f"最终特征矩阵: {X.shape} ({len(baseline_cols)} 基线 + {len(knowfeat_names)} KnowFeat)")

    # ── 4. 保存 ──
    print(f"\n{'=' * 60}")
    print("Step 4: 保存结果")
    print("=" * 60)

    out_df = X.copy()
    out_df["is_laundering"] = y

    out_path = OUTPUTS_DIR / "all_113_features.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"已保存到 {out_path}")
    print(f"  形状: {out_df.shape[0]} 行, {out_df.shape[1]} 列")

    # ── 5. 打印特征列表 ──
    print(f"\n=== 全部 {len(baseline_cols) + len(knowfeat_names)} 个特征 ===")
    print(f"\n--- 基线特征 ({len(baseline_cols)} 个) ---")
    for i, name in enumerate(baseline_cols, 1):
        print(f"  {i:3d}. {name}")
    print(f"\n--- KnowFeat 新特征 ({len(knowfeat_names)} 个) ---")
    for i, name in enumerate(knowfeat_names, 1):
        print(f"  {len(baseline_cols) + i:3d}. {name}")


if __name__ == "__main__":
    main()
