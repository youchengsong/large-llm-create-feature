"""
特征: last_24h_tx_ratio
说明: 异常交易模式中 small_amount_testing 涉及休眠账户（>=90天）突然激活，最后24小时交易占比会异常高。正常钱包的交易分布更均匀，最后24小时占比通常较低。这个特征可以直接反映近期突然活跃的异常行为。

验证指标:
  - Score: 0.2640
  - MI: 0.0009
  - Redundancy: 0.5263
  - Cramer's V: 0.0196
  - Delta PR-AUC: -0.016160
  - SHAP Rank: 30

可解释性报告:
  - 值含义: {'0~0.0074': '交易分布非常均匀，最后24小时几乎没有集中交易，属于典型的长期稳定使用模式', '0.0074~0.0199': '正常交易波动范围，最后24小时交易占比在合理区间内', '0.0199~0.0317': '交易开始向近期集中，可能为正常用户月末集中消费或异常激活的临界区', '0.0317以上': '交易高度集中在最后24小时，与休眠账户突然激活的small_amount_testing模式高度吻合'}
  - 风险阈值: 0.0317
  - 协同特征: [{'feature': 'activetime', 'reason': 'last_24h_tx_ratio高值表示交易集中在近期，结合activetime（活跃时段）可区分：若同时activetime在深夜时段（0-5点），则高度吻合small_amount_testing模式；若activetime在白天，则可能是正常用户集中消费。'}, {'feature': 'avg_inactive_period', 'reason': 'last_24h_tx_ratio高+avg_inactive_period长（>90天）的组合，精准捕获休眠账户突然激活的异常行为，两者互为佐证。'}, {'feature': 'transaction_count_directed', 'reason': 'last_24h_tx_ratio高但transaction_count_directed低（交易笔数少），说明是少量交易集中在最后24小时，更符合测试性小额交易特征；若交易笔数也多，则可能是正常用户集中消费。'}]
  - 误判场景: ['新注册用户：刚开通钱包的用户，其所有交易自然集中在近期，导致last_24h_tx_ratio偏高（可能接近1），但实际上属于正常使用。可通过结合wallet_open_date（开户日期）过滤：开户时间<7天的用户可降低此特征的权重。', '短期促销活动用户：参与限时优惠活动的用户可能在短时间内集中交易，导致该特征偏高。可通过结合transaction_motif中的normal_small_high_freq模式判断。', '偶尔使用用户：长期不使用钱包，偶尔登录使用一次的用户，其交易也集中在最后24小时，但交易笔数极少（1-2笔），可通过结合transaction_count_directed区分。']
  - 监管说明: 该特征衡量的是钱包最近一天内的交易占其全部交易的比例，比例异常高意味着该钱包突然变得活跃，可能与休眠账户突然启动进行小额测试交易的可疑行为有关。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 last_24h_tx_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 找到每个钱包的最后交易时间
    last_tx_time = tx_df.groupby('src')['timestamp'].max().reset_index()
    last_tx_time.columns = ['wallet_id', 'last_tx_time']
    last_tx_time['last_tx_time'] = pd.to_datetime(last_tx_time['last_tx_time'])

    # 计算每个钱包最后24小时内的交易笔数
    last_24h = pd.to_datetime(last_tx_time['last_tx_time']) - pd.Timedelta(hours=24)
    tx_df['timestamp_dt'] = pd.to_datetime(tx_df['timestamp'])

    tx_with_last = tx_df.merge(last_tx_time, left_on='src', right_on='wallet_id', suffixes=('', '_last'))
    tx_with_last['is_last_24h'] = (tx_with_last['timestamp_dt'] >= tx_with_last['last_tx_time'] - pd.Timedelta(hours=24))

    # 计算每个钱包总交易笔数和最后24小时笔数
    total_tx = tx_df.groupby('src')['tx_id'].count().reset_index()
    total_tx.columns = ['wallet_id', 'total_tx']
    last_24h_tx = tx_with_last[tx_with_last['is_last_24h']].groupby('src')['tx_id'].count().reset_index()
    last_24h_tx.columns = ['wallet_id', 'last_24h_tx']

    result = total_tx.merge(last_24h_tx, on='wallet_id', how='left')
    result['last_24h_tx'] = result['last_24h_tx'].fillna(0)
    result['last_24h_tx_ratio'] = result['last_24h_tx'] / result['total_tx'].clip(lower=1)
    result = result[['wallet_id', 'last_24h_tx_ratio']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: last_24h_tx_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 last_24h_tx_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
