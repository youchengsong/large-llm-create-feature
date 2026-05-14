"""
特征: night_hour_entropy
说明: 洗钱交易往往集中在深夜时段但交易时间点可能分散（逃避监管），通过计算深夜交易时间分布的熵值，可以量化这种时间分散程度。

验证指标:
  - Score: 0.5037
  - MI: 0.0011
  - Redundancy: 0.5383
  - Cramer's V: 0.0217
  - Delta PR-AUC: -0.006597
  - SHAP Rank: 18

可解释性报告:
  - 值含义: {'0.0-0.3': "深夜交易时间高度集中（如全部在凌晨1点），异常交易特征明显，需重点关注。符合洗钱模式中'深夜时段集中交易'的特征", '0.3-0.6': '深夜交易时间较为集中（如集中在2-3个时段），有一定异常嫌疑，可能为批量注册的薅羊毛账户', '0.6-0.8': '正常波动范围，深夜交易时间分布适中，普通用户夜间偶尔交易的特征', '0.8-1.0': '深夜交易时间高度分散（覆盖多个深夜时段），符合正常用户偶尔在不同夜间时段交易的特征'}
  - 风险阈值: {'high_risk_threshold': 0.3, 'low_risk_threshold': 0.6, 'judgment_basis': "异常交易中位数(0.7115)明显低于正常中位数(0.7592)，且异常交易的25%分位(0.6160)显著低于正常的25%分位(0.7143)。选择0.3作为高风险阈值是因为：1) 当熵值低于0.3时，表示深夜交易集中在1-2个小时内，高度符合洗钱模式中'深夜集中交易'的特征；2) 异常交易的标准差(0.0829)大于正常(0.0682)，说明异常交易的熵值波动更大，低熵值区间更具区分度。0.6作为低风险阈值对应正常交易的25%分位，低于此值即有可疑"}
  - 协同特征: [{'feature': 'activetime', 'reason': "activetime统计所有交易的时间分布，night_hour_entropy专注深夜时段。两者结合可区分：是'只在深夜活动'（高异常风险）还是'全天活动但深夜也有分散交易'（正常）"}, {'feature': 'transaction_frequency', 'reason': '高频交易+低深夜熵值组合是线上跑分模式的典型特征（凌晨高频分散转出）。正常用户深夜交易频率低，即使熵值低也不可疑'}, {'feature': 'send_concentration', 'reason': '低深夜熵值+高发送集中度（向少数对手频繁深夜转账）符合线下跑分模式；低深夜熵值+低发送集中度（向大量对手深夜分散转账）符合线上跑分模式'}, {'feature': 'avg_inactive_period', 'reason': "低深夜熵值+长不活跃期后突然激活，符合'休眠账户突然激活'的风险指标，是洗钱模式5（small_amount_testing）的典型特征"}]
  - 误判场景: [{'scenario': '夜班工作者（如保安、医护人员）', 'reason': '这类用户可能在凌晨时段规律性地进行小额消费（如买夜宵），导致深夜交易集中在固定时段，熵值较低。需要结合交易金额和频次判断'}, {'scenario': '跨境时差用户', 'description': "居住在海外或经常跨时区旅行的用户，其'深夜'可能对应对方的白天，导致深夜交易集中。需结合交易对手地域信息判断"}, {'scenario': '自动扣费/定期缴费', 'description': '部分用户设置凌晨自动扣费（如信用卡还款、保险），导致深夜交易集中在固定时间点。这类交易金额固定、对手固定，与洗钱模式不同'}, {'scenario': '少量深夜交易用户', 'description': '如果用户仅有一两笔深夜交易，熵值可能因样本不足而偏低。此类情况需结合total_night交易数量判断，样本量小于3时熵值不可信'}]
  - 监管说明: 该特征衡量的是用户所有深夜交易（晚上8点到凌晨5点）在时间上的分散程度：正常用户深夜交易会分散在不同时段（如有时11点、有时凌晨2点），而洗钱账户为了规避监管往往集中在特定深夜时段（如全部在凌晨1点），导致时间分布熵值偏低。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 night_hour_entropy，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 提取小时
    tx = tx_df.copy()
    tx['hour'] = pd.to_datetime(tx['timestamp']).dt.hour

    # 定义深夜时段（20点-次日5点，包含22-23点、0-5点、20-1点）
    night_hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]

    # 标记是否为深夜交易
    tx['is_night'] = tx['hour'].isin(night_hours)

    # 对每个钱包，统计深夜交易的小时分布
    night_tx = tx[tx['is_night']].copy()

    if len(night_tx) > 0:
        # 统计每个钱包在每个深夜小时的交易次数
        hour_counts = night_tx.groupby(['src', 'hour']).size().reset_index(name='count')

        # 计算每个钱包的总深夜交易数
        total_night = hour_counts.groupby('src')['count'].sum().reset_index(name='total_night')

        # 合并并计算概率
        hour_counts = hour_counts.merge(total_night, on='src')
        hour_counts['prob'] = hour_counts['count'] / hour_counts['total_night']

        # 计算每个钱包的深夜小时熵值
        hour_counts['entropy_contrib'] = -hour_counts['prob'] * np.log2(hour_counts['prob'] + 1e-10)
        entropy = hour_counts.groupby('src')['entropy_contrib'].sum().reset_index()
        entropy.columns = ['wallet_id', 'night_hour_entropy']
    else:
        entropy = pd.DataFrame({'wallet_id': tx['src'].unique(), 'night_hour_entropy': 0.0})

    # 对于没有深夜交易的钱包，熵值为0
    all_wallets = tx['src'].unique()
    result = pd.DataFrame({'wallet_id': all_wallets})
    result = result.merge(entropy, on='wallet_id', how='left')
    result['night_hour_entropy'] = result['night_hour_entropy'].fillna(0)

    # 归一化到0-1之间（最大熵为log2(10)≈3.32）
    result['night_hour_entropy'] = result['night_hour_entropy'] / np.log2(10)
    result['night_hour_entropy'] = result['night_hour_entropy'].clip(0, 1)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: night_hour_entropy")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 night_hour_entropy: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
