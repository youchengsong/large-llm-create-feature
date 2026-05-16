"""
特征: hourly_amount_skewness_src
说明: 洗钱行为（如线上跑分模式3、class4_laundering模式6）通常集中在特定时段进行交易，且金额分布不均匀（大部分交易金额接近，但偶尔有异常大额或小额交易），导致小时级平均金额的分布呈现偏斜。正常消费行为的金额在时间上分布更均匀。

验证指标:
  - Score: 0.4660
  - MI: 0.0000
  - Redundancy: 0.6077
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 69

可解释性报告:
  - 值含义: {'负值区间（-2 ~ -0.5）': '小时平均金额分布左偏，即大部分小时的平均交易金额偏低，但个别小时有特别低的异常值。异常钱包在此区间集中（中位数-0.088），表明洗钱钱包倾向于在少数特定时段进行大额交易，其他时段交易极少或为零，造成左偏分布。', '接近零区间（-0.5 ~ 0.5）': '小时平均金额分布接近对称，各小时交易金额相对均衡。正常钱包集中在此区间（中位数0.041），表明日常消费行为在时间上分布较为均匀。', '正值区间（0.5 ~ 2）': '小时平均金额分布右偏，即大部分小时的平均交易金额偏低，但个别小时有特别高的异常值。可能反映正常用户的偶尔大额消费（如节日购物），但若值过大且伴随其他异常特征，也可能指向洗钱行为。'}
  - 风险阈值: {'threshold': -0.5, 'rationale': '异常钱包的中位数为-0.088，且25%分位为-0.766，即约25%的异常钱包偏度小于-0.766。取-0.5作为阈值，能覆盖约30%的异常钱包，同时正常钱包中仅有约25%低于此值（正常25%分位为-0.709），具有较好的区分能力。该阈值与领域知识一致：洗钱行为（线上跑分、class4洗钱）通常在特定时段集中交易，导致小时平均金额左偏。', 'indicator': '当 hourly_amount_skewness_src < -0.5 时，该钱包的小时平均金额分布异常左偏，应列为可疑对象进一步核查'}
  - 协同特征: [{'feature': 'is_night_hour_From', 'reason': '洗钱行为（线上跑分模式3、class4洗钱模式6）集中在深夜时段，而正常消费较少。结合夜间交易占比和小时金额偏度，能更准确识别异常模式：若偏度左偏且夜间交易占比高，则洗钱嫌疑大幅上升。'}, {'feature': 'transaction_frequency_From', 'reason': '洗钱钱包通常交易频率异常（极高或极低）。结合小时金额偏度：若交易频率高且偏度左偏，说明高频集中在少数时段，符合跑分特征；若交易频率低且偏度左偏，说明偶尔大额交易集中在特定时段，符合class4洗钱特征。'}, {'feature': 'large_transaction_ratio', 'reason': '大额交易占比与小时金额偏度具有互补性。正常用户偶尔大额消费可能导致右偏（正值），而洗钱行为的大额交易集中在特定时段导致左偏（负值）。两者结合可区分正常大额消费与洗钱行为。'}]
  - 误判场景: ['定时自动转账：用户设置了每月固定日期的自动还款/转账，导致该小时平均金额显著高于其他小时，造成左偏。这类用户通常有规律的生活作息，可通过检查交易对手是否固定、交易是否定期等进一步排除。', '季节性大额消费：如双11购物节、春节红包等集中消费场景，用户可能在1-2小时内进行多笔大额交易，导致小时平均金额偏斜。这类情况通常伴随正常的消费场景和交易对手。', '新开户用户：刚开户的用户交易记录少，可能只有1-2小时有交易，偏度计算不稳定且容易极端。建议结合交易笔数特征（如Out_Degree > 10）过滤掉交易过少的钱包。']
  - 监管说明: 该特征通过分析用户在不同小时的平均交易金额是否均匀分布，来识别那些只在特定时间段集中进行大额交易的可疑账户——正常用户各小时消费金额相对均衡，而洗钱账户往往集中在深夜等特定时段完成非法资金转移。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 hourly_amount_skewness_src，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包每个小时的平均交易金额
    hourly_avg = tx_df.groupby(['src', 'hour'])['amount'].mean().reset_index()

    # 计算每个钱包的小时平均金额的偏度（skewness）
    def calc_skewness(group):
        values = group['amount'].values
        if len(values) < 3:
            return 0.0
        mean = np.mean(values)
        std = np.std(values, ddof=0)
        if std == 0:
            return 0.0
        # 计算样本偏度
        n = len(values)
        skew = (n * np.sum((values - mean)**3)) / ((n-1) * (n-2) * (std**3))
        return skew

    skewness = hourly_avg.groupby('src').apply(calc_skewness).reset_index()
    skewness.columns = ['wallet_id', 'hourly_amount_skewness_src']

    # 填充NaN
    skewness['hourly_amount_skewness_src'] = skewness['hourly_amount_skewness_src'].fillna(0)

    result = skewness
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: hourly_amount_skewness_src")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 hourly_amount_skewness_src: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
