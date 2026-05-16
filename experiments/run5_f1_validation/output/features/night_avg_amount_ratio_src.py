"""
特征: night_avg_amount_ratio_src
说明: 夜间交易金额如果显著偏离白天交易金额，可能表明夜间交易有特殊目的（如测试金额或跑分标准金额）。正常用户在夜间的交易金额通常与白天相似或更小，而洗钱交易可能有特定的金额范围。

验证指标:
  - Score: 0.3741
  - MI: 0.0000
  - Redundancy: 0.5288
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 87

可解释性报告:
  - 值含义: {'0-0.1': '夜间交易金额远低于白天平均水平，典型的正常用户行为模式（夜间小额消费或偶尔转账）', '0.1-0.5': '夜间交易金额与白天接近，可能是夜间加班、跨境时差或夜班工作者', '0.5-1.0': '夜间交易金额明显高于白天，可能涉及特殊目的交易（如洗钱跑分、赌博充值）', '>1.0': '夜间交易金额超过白天平均金额，高度异常，可能为批量自动化交易或逃避监管的集中交易'}
  - 风险阈值: 高风险阈值：>0.5。选择依据：异常交易75%分位数为0.0661，而正常交易75%分位数为0.1208，表明异常交易夜间金额普遍低于正常。但异常交易均值0.1896 vs 正常0.2482，说明部分异常交易夜间金额显著偏高。结合领域知识（线上跑分模式凌晨20时-1时高频交易，金额10-1000元），当夜间平均金额超过白天均值50%以上时，洗钱嫌疑显著增加。
  - 协同特征: [{'feature': 'night_transaction_count_ratio_src', 'reason': "夜间交易占比与夜间金额偏离度结合，可区分'少量大额夜间交易'（高金额比+低数量比）和'大量小额夜间跑分'（低金额比+高数量比）两种不同模式"}, {'feature': 'amount_std_ratio', 'reason': "夜间金额偏离度+金额波动性，可识别'固定金额跑分'（低波动+高偏离）和'随机金额洗钱'（高波动+高偏离）"}, {'feature': 'fan_out_count', 'reason': '结合分散转出模式，夜间高金额比+fan_out模式=典型的线上跑分洗钱特征（分散转入后集中转出）'}]
  - 误判场景: ['夜班工作者（如保安、医生、外卖骑手）在夜间有规律性消费，金额可能高于白天均值', '跨境电商卖家，因时差原因夜间处理国际收款，金额通常较大', '学生群体，白天上课晚上集中购物或缴费，单笔金额可能超过白天零散消费的平均值', '夜间打赏主播的用户，金额可能远高于日常消费水平']
  - 监管说明: 该特征通过对比用户夜间交易金额与其全天平均金额的偏离程度，来识别那些在深夜时段进行异常大额交易的洗钱行为——因为正常人在深夜通常只进行小额消费，而洗钱分子却可能在深夜进行大额资金转移以规避监控。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 night_avg_amount_ratio_src，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 标记夜间交易(22-5时)
    tx_df['is_night'] = ((tx_df['hour'] >= 22) | (tx_df['hour'] <= 5)).astype(int)

    # 计算每个src的夜间交易平均金额
    night_avg = tx_df[tx_df['is_night']==1].groupby('src')['amount'].mean().reset_index()
    night_avg.columns = ['wallet_id', 'night_avg_amount']

    # 计算每个src的整体平均金额
    total_avg = tx_df.groupby('src')['amount'].mean().reset_index()
    total_avg.columns = ['wallet_id', 'total_avg_amount']

    # 合并
    result = night_avg.merge(total_avg, on='wallet_id', how='outer')
    # 对于没有夜间交易的，night_avg_amount设为NaN，填充为0
    result['night_avg_amount'] = result['night_avg_amount'].fillna(0)
    result['total_avg_amount'] = result['total_avg_amount'].fillna(0)

    # 计算比值，避免除零
    result['night_avg_amount_ratio_src'] = result['night_avg_amount'] / (result['total_avg_amount'] + 0.01)
    result = result[['wallet_id', 'night_avg_amount_ratio_src']]

    # 清理
    del tx_df['is_night']

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: night_avg_amount_ratio_src")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 night_avg_amount_ratio_src: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
