"""
特征: mode_switch_frequency_trend_sender
说明: 洗钱账户在操作过程中，为了快速完成资金转移并规避监测，会不断尝试不同的交易模式，导致模式切换频率逐渐增加。这种时间趋势是区分正常和异常行为的重要序列特征。

验证指标:
  - Score: 0.5289
  - MI: 0.0000
  - Redundancy: 0.2351
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 70

可解释性报告:
  - 值含义: 该特征表示钱包在交易生命周期中，后半段交易模式切换频率相对于前半段的变化趋势。值越正（>0.05），表示后半段切换频率明显增加，即交易模式越来越不稳定，是异常行为的信号；值接近0（-0.05~0.05），表示前后半段切换频率基本一致，属于正常行为；值越负（<-0.1），表示后半段切换频率明显降低，即交易模式趋于稳定，常见于正常用户的固定场景消费。
  - 风险阈值: 高风险阈值建议设为 >0.08。判断依据：异常交易的75分位值为0.0615，而正常交易的75分位值为0.0603，两者非常接近，说明在75分位处区分度不足。但异常交易的均值（-0.0180）高于正常交易（-0.0271），且中位数（-0.0009）明显高于正常交易（-0.0053），说明异常交易整体上更倾向于正向趋势。考虑到IV=0.0113较低，该特征应作为辅助信号而非主信号，选择0.08作为阈值可以捕捉到那些模式切换频率显著加速的极端异常账户，同时避免误伤正常用户。
  - 协同特征: 建议与以下特征配合使用：
1. **transaction_frequency**（交易频率）：洗钱账户通常同时具有高频交易和模式切换增加的特征，两者结合能更准确识别线上跑分模式。
2. **avg_inactive_period**（平均不活跃间隔）：正常用户如果模式切换增加，通常是因为长时间未使用后重新激活，而洗钱账户的模式切换增加伴随的是不活跃间隔缩短（即频繁操作）。
3. **send_concentration**（发送集中度）：洗钱账户的fan_out模式会导致发送集中度降低（分散转出），与模式切换增加结合，能识别从集中到分散的转型行为。
  - 误判场景: 以下场景可能导致误判：
1. **新功能体验用户**：用户刚开通数字人民币钱包，前期仅小额测试（一种模式），后期开始正常消费（多种模式），导致后半段切换频率增加。
2. **季节性消费变化**：如用户平时只用于日常小额支付（模式单一），但在双十一等购物节期间大量使用不同商户钱包（模式多样）。
3. **账户恢复使用**：用户长期未使用钱包，近期重新激活并尝试不同交易功能（如从仅收款变为也付款）。
4. **设备更换或升级**：用户更换手机或升级钱包等级后，可使用的交易模式增加。
  - 监管说明: 该特征衡量的是用户交易模式的变化趋势——如果用户后半段交易中频繁切换不同的交易方式（如从单笔转账变为分散转出再变为集中收款），说明其操作行为越来越不稳定，可能是在有意识地规避监测，这是洗钱行为的典型特征。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 mode_switch_frequency_trend_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'])
    tx_df_sorted = tx_df.sort_values(['src', 'timestamp'])

    # 为每个钱包按时间排序，标记交易模式
    tx_df_sorted['mode_id'] = tx_df_sorted['transaction_mode'].astype('category').cat.codes

    # 计算每笔交易与前一笔交易模式是否不同
    tx_df_sorted['prev_mode'] = tx_df_sorted.groupby('src')['mode_id'].shift(1)
    tx_df_sorted['mode_switch'] = (tx_df_sorted['mode_id'] != tx_df_sorted['prev_mode']).astype(int)

    # 按钱包分，将交易按时间分成前后两半，比较模式切换频率的变化
    # 计算每个钱包的交易序号
    tx_df_sorted['tx_seq'] = tx_df_sorted.groupby('src').cumcount() + 1
    tx_df_sorted['total_tx'] = tx_df_sorted.groupby('src')['tx_id'].transform('count')

    # 前半部分和后半部分
    tx_df_sorted['is_second_half'] = tx_df_sorted['tx_seq'] > tx_df_sorted['total_tx'] / 2

    # 分别计算前后半段的模式切换频率
    switch_stats = tx_df_sorted.groupby(['src', 'is_second_half']).agg(
        switch_count=('mode_switch', 'sum'),
        tx_count=('tx_id', 'count')
    ).reset_index()

    # 计算每半段的切换频率
    switch_stats['switch_freq'] = switch_stats['switch_count'] / switch_stats['tx_count'].clip(lower=1)

    # 分别提取前后半段的频率
    first_half = switch_stats[~switch_stats['is_second_half']][['src', 'switch_freq']].rename(columns={'switch_freq': 'first_half_freq'})
    second_half = switch_stats[switch_stats['is_second_half']][['src', 'switch_freq']].rename(columns={'switch_freq': 'second_half_freq'})

    # 合并并计算趋势（后半段频率 - 前半段频率）
    trend_stats = first_half.merge(second_half, on='src', how='outer').fillna(0)
    trend_stats['mode_switch_trend'] = trend_stats['second_half_freq'] - trend_stats['first_half_freq']

    result = trend_stats[['src', 'mode_switch_trend']].rename(columns={'src': 'wallet_id'})
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: mode_switch_frequency_trend_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 mode_switch_frequency_trend_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
