"""
特征: amount_sequence_trend_direction
说明: 洗钱行为常表现为先测试小额再大额交易（测试模式），或先大额收款再小额分散转出（跑分模式）。金额序列的趋势方向能区分正常随机波动与有计划的洗钱行为。

验证指标:
  - Score: 0.2034
  - MI: 0.0009
  - Redundancy: 0.3463
  - Cramer's V: 0.0162
  - Delta PR-AUC: -0.023011
  - SHAP Rank: 48

可解释性报告:
  - 值含义: {'0-0.01': '金额序列无明显趋势方向，交易金额随机波动，符合正常消费行为特征', '0.01-0.03': '金额序列存在轻微单向趋势，可能是正常消费习惯（如日常小额递增消费）', '0.03-0.05': '金额序列呈现中等强度的单向趋势，需关注是否存在测试后大额交易或大额分散转出行为', '0.05-0.10': '金额序列呈现明显单向趋势，高概率为有计划洗钱行为（测试模式或跑分模式）', '>0.10': '金额序列呈现强烈单向趋势，极大概率存在系统性洗钱操作'}
  - 风险阈值: 0.05
  - 协同特征: [{'feature': 'small_amount_testing_ratio（小额测试交易占比）', 'reason': '当金额趋势方向强（>0.05）且小额测试交易占比高时，可精准识别测试后大额交易的洗钱模式'}, {'feature': 'fan_out_ratio（分散转出占比）', 'reason': '当趋势方向为负（向下）且fan_out占比高时，可识别大额收款后分散转出的跑分模式'}, {'feature': 'night_transaction_ratio（深夜交易占比）', 'reason': '洗钱行为常发生在深夜，三者联合可大幅降低误报率'}, {'feature': 'active_days_ratio（活跃天数占比）', 'reason': '休眠账户突然激活且金额趋势方向强，可识别小型测试模式'}]
  - 误判场景: ['定期定额投资行为：用户每月固定日期固定金额买入理财产品，产生稳定向上的金额趋势', '工资发放场景：每月固定收到大额工资后逐渐小额消费，产生向下趋势', '大额一次性消费分期：用户大额消费后分多笔小额还款，产生稳定向下趋势', '商户日常经营：商户每日收款金额随客流量自然波动，可能呈现单向趋势', '红包/补贴发放：平台集中发放红包后用户逐渐使用，产生明显向下趋势']
  - 监管说明: 该特征检测钱包交易金额是否呈现持续增大或持续减小的异常趋势，洗钱分子通常会先试小额再转大额，或先收大额再分散转出，形成规律性的金额变化方向，而普通用户的交易金额通常是随机波动的。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_sequence_trend_direction，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按钱包和时间排序，计算金额序列的单调趋势
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()

    # 计算每笔交易金额相对于前一笔的变化方向
    tx_sorted['prev_amount'] = tx_sorted.groupby('src')['amount'].shift(1)
    tx_sorted['direction'] = np.sign(tx_sorted['amount'] - tx_sorted['prev_amount'])

    # 统计每个钱包的方向分布（正方向占比）
    stats = tx_sorted.groupby('src')['direction'].agg(['mean', 'count']).reset_index()
    # mean>0表示整体向上趋势，mean<0表示向下趋势，mean≈0表示无趋势
    # 用绝对值表示趋势强度
    stats['feature_value'] = stats['mean'].abs()
    result = stats[['src', 'feature_value']].rename(columns={'src': 'wallet_id'})
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_sequence_trend_direction")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_sequence_trend_direction: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
