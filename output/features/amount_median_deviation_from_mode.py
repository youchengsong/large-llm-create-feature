"""
特征: amount_median_deviation_from_mode
说明: 根据领域知识，small_amount_testing模式的金额为0.01-0.1元，而normal_small_high_freq模式的金额为1-100元，regular_large_low_freq模式的金额为100-1000元。如果钱包的交易中位数金额偏离其交易模式的典型金额范围，可能表明异常行为。该特征能捕捉到异常交易中中位数与模式典型值的偏离。

验证指标:
  - Score: 0.0878
  - MI: 0.0010
  - Redundancy: 0.8789
  - Cramer's V: 0.0132
  - Delta PR-AUC: 0.008155
  - SHAP Rank: 39

可解释性报告:
  - 值含义: {'0-1': '钱包交易金额中位数与所属交易模式的典型金额范围高度一致，表明该钱包的交易行为完全符合其交易模式的预期特征，通常为正常用户。', '1-5': '钱包交易金额中位数略有偏离，但仍处于合理波动范围内，可能由于正常用户的交易习惯变化或小额偶然性交易导致，需结合其他特征综合判断。', '5-20': '钱包交易金额中位数明显偏离其交易模式的典型金额范围，存在一定风险，可能涉及异常交易模式切换（如从正常消费转向测试性交易），建议进一步监控。', '20-50': '钱包交易金额中位数严重偏离典型范围，高风险信号，常见于洗钱模式，如利用small_amount_testing模式进行小额测试后转移大额资金，或merchant_laundering模式中金额异常波动。', '>50': '极端偏离，几乎可以确定钱包的交易行为与注册模式完全不符，极大概率是洗钱账户，尤其是批量注册的4类钱包或休眠账户突然激活后进行异常交易。'}
  - 风险阈值: 20
  - 协同特征: [{'feature': 'activetime', 'reason': 'amount_median_deviation_from_mode高值结合activetime异常（如凌晨活跃），可有效识别small_amount_testing和online_laundering等深夜交易模式，避免仅凭金额偏离产生的误判。'}, {'feature': 'From_To_Ratio', 'reason': '高偏离值配合高From_To_Ratio（发送远多于接收），可锁定fan_out模式的洗钱账户，如线上跑分场景中大量分散转出。'}, {'feature': 'transaction_frequency', 'reason': '高偏离值配合高交易频率，可区分merchant_laundering（高频小额）与regular_large_low_freq（低频大额），提高对商户收款码跑分的检测精度。'}, {'feature': 'In_Degree', 'reason': '高偏离值配合低In_Degree（接收方数量少），可识别fan_in模式的洗钱账户，如多账户集中转入后一次性转出。'}]
  - 误判场景: ['正常商户钱包在促销活动期间，可能临时改变交易金额范围（如从regular_large_low_freq转向normal_small_high_freq），导致中位数偏离典型模式，但实际为正常经营行为。', '新注册用户由于交易历史短，中位数可能受少数几笔交易影响而偏离，尤其是首次大额充值后的小额消费，容易触发高偏离值。', '跨场景使用的钱包（如个人钱包同时用于日常消费和偶尔大额转账），其整体中位数可能偏离单一模式的典型范围，但并非洗钱行为。', '数据统计误差：当钱包交易笔数极少（如少于5笔）时，中位数易受极端值影响，导致偏离值虚高，需结合交易笔数特征进行过滤。']
  - 监管说明: 该特征衡量钱包的交易金额是否与其声称的用途一致，例如一个声称用于小额消费的钱包突然出现大额转账，就会被标记为可疑，这类似于银行监控中'交易行为与客户画像不符'的预警逻辑。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_median_deviation_from_mode，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 定义各交易模式的典型中位数金额范围
    mode_median_ranges = {
        'single_transaction': (1, 5000),
        'normal_small_high_freq': (1, 100),
        'regular_large_low_freq': (100, 1000),
        'small_amount_testing': (0.01, 0.1),
        'merchant_laundering': (10, 1000),
        'class4_laundering': (100, 2000)
    }

    # 计算每个钱包在各交易模式下的中位数金额
    wallet_mode_medians = tx_df.groupby(['src', 'transaction_motif'])['amount'].median().reset_index()
    wallet_mode_medians.columns = ['wallet_id', 'transaction_motif', 'mode_median']

    # 计算钱包整体中位数金额
    wallet_overall_median = tx_df.groupby('src')['amount'].median().reset_index()
    wallet_overall_median.columns = ['wallet_id', 'overall_median']

    # 合并
    df = wallet_overall_median.merge(wallet_mode_medians, on='wallet_id', how='left')

    # 计算每个钱包与其各交易模式典型金额范围的偏离程度
    # 对于每个模式，计算中位数偏离该模式典型范围中心值的程度
    mode_centers = {k: (v[0] + v[1]) / 2 for k, v in mode_median_ranges.items()}

    df['center'] = df['transaction_motif'].map(mode_centers)
    df['deviation'] = np.abs(df['mode_median'] - df['center']) / (df['center'] + 1)  # 相对偏离

    # 取最大偏离作为特征值
    result = df.groupby('wallet_id')['deviation'].max().reset_index()
    result.columns = ['wallet_id', 'amount_median_deviation_from_mode']
    result['amount_median_deviation_from_mode'] = result['amount_median_deviation_from_mode'].fillna(0)

    # 对没有交易模式映射的钱包，使用与整体中位数的偏离
    # 计算整体中位数的典型值（所有钱包中位数的中位数）
    typical_overall_median = wallet_overall_median['overall_median'].median()
    result['overall_median'] = result['wallet_id'].map(wallet_overall_median.set_index('wallet_id')['overall_median'])
    result['amount_median_deviation_from_mode'] = result.apply(
        lambda row: np.abs(row['overall_median'] - typical_overall_median) / (typical_overall_median + 1) 
        if row['amount_median_deviation_from_mode'] == 0 else row['amount_median_deviation_from_mode'],
        axis=1
    )

    result = result[['wallet_id', 'amount_median_deviation_from_mode']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_median_deviation_from_mode")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_median_deviation_from_mode: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
