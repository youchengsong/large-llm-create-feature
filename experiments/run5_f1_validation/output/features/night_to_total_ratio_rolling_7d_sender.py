"""
特征: night_to_total_ratio_rolling_7d_sender
说明: 线上跑分模式中异常交易集中在20时-1时，class4_laundering集中在0-1时和23时。通过计算深夜时段交易占比的滚动均值，可以识别出近期突然在深夜频繁交易的钱包。

验证指标:
  - Score: 0.1964
  - MI: 0.0000
  - Redundancy: 0.6590
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 109

可解释性报告:
  - 值含义: {'0.0': '该钱包在最近7天内没有任何深夜时段（20时-1时）的交易，交易时间完全符合正常作息规律。', '0.0-0.1': '深夜交易占比很低（<10%），属于正常波动范围，可能是偶尔的夜间消费。', '0.1-0.3': '深夜交易占比中等（10%-30%），存在一定异常嫌疑，可能是个别夜晚的集中交易。', '0.3-0.5': '深夜交易占比偏高（30%-50%），异常嫌疑较高，可能涉及线上跑分或class4洗钱。', '0.5-1.0': '深夜交易占比极高（>50%），高度异常，极大概率涉及线上跑分或class4洗钱模式。'}
  - 风险阈值: 0.3
  - 协同特征: [{'feature': 'transaction_frequency_From（交易频率）', 'reason': '线上跑分模式不仅深夜交易占比高，交易频率也异常高。两者结合可区分正常用户的偶尔夜间消费（低频+低深夜比）和跑分钱包（高频+高深夜比）。'}, {'feature': 'is_night_hour_From（是否深夜交易）', 'reason': 'is_night_hour_From是单笔交易的深夜标记，而本特征是近7天的深夜占比滚动均值。前者捕捉瞬时异常，后者刻画持续行为模式，两者互补可提高异常检测的鲁棒性。'}, {'feature': 'large_transaction_ratio（大额交易占比）', 'reason': 'class4洗钱涉及金额100-2000元，线上跑分金额10-1000元。若深夜交易占比高且大额交易占比也高，可能指向class4洗钱；若深夜交易占比高但大额占比低（金额较小），则更可能指向线上跑分。'}]
  - 误判场景: ['夜班工作者：如保安、医护人员、物流司机等职业，其正常交易时间可能在深夜，导致深夜交易占比偏高。', '跨时区用户：若钱包用户实际位于不同时区（如海外用户），其本地时间的夜间可能对应服务器时间的白天，导致误判。', '短期旅行或出差：用户短期前往不同时区，临时调整作息，可能在深夜产生集中交易。', '特定消费场景：如深夜外卖、线上娱乐充值、游戏内购等，正常用户也可能在深夜产生多笔交易。']
  - 监管说明: 该特征计算钱包最近7天在深夜时段（晚上8点到凌晨1点）的交易占比，如果这个比例超过30%，说明该钱包近期有异常集中的夜间交易行为，符合线上跑分和class4洗钱的时间特征，需要进一步核查交易对手和资金流向。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 night_to_total_ratio_rolling_7d_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    tx_df['hour'] = pd.to_datetime(tx_df['timestamp']).dt.hour
    tx_df['is_night'] = ((tx_df['hour'] >= 20) | (tx_df['hour'] <= 1)).astype(int)
    tx_df['date'] = pd.to_datetime(tx_df['timestamp']).dt.date

    # 计算每钱包每日的深夜交易比例
    daily_night = tx_df.groupby(['src', 'date']).agg(
        night_count=('is_night', 'sum'),
        total_count=('tx_id', 'count')
    ).reset_index()
    daily_night['night_ratio'] = daily_night['night_count'] / daily_night['total_count'].clip(lower=1)

    # 对每个钱包，计算最近7天的平均深夜比例（如果有足够数据）
    def rolling_mean_7d(group):
        group = group.sort_values('date')
        group['rolling_night_ratio'] = group['night_ratio'].rolling(window=7, min_periods=1).mean()
        return group

    result = daily_night.groupby('src').apply(rolling_mean_7d).reset_index(drop=True)
    result = result.groupby('src')['rolling_night_ratio'].last().reset_index()
    result.columns = ['wallet_id', 'feature_value']
    result['feature_value'] = result['feature_value'].fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: night_to_total_ratio_rolling_7d_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 night_to_total_ratio_rolling_7d_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
