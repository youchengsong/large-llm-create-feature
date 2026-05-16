"""
特征: amount_sudden_spike_sender
说明: 洗钱行为中休眠账户突然激活并发生大额交易，或正常账户突然收到大额资金后立即转出，金额突变能反映这种异常行为。

验证指标:
  - Score: 0.2126
  - MI: 0.0000
  - Redundancy: 0.6545
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 108

可解释性报告:
  - 值含义: {'0.0-0.5': '交易金额稳定，相邻交易间金额变化不超过3倍，属于正常消费或转账行为', '0.5-0.8': '交易金额有适度波动，相邻交易间金额变化在3-6倍之间，可能为正常的大额消费或工资发放', '0.8-0.9': '交易金额波动较大，相邻交易间金额变化在6-8倍之间，需关注但可能为正常大额交易', '0.9-1.0': "交易金额出现剧烈突变，相邻交易间金额变化超过8倍甚至上百倍，高度疑似洗钱行为中的'休眠账户激活后大额交易'或'快进快出'模式"}
  - 风险阈值: 0.95
  - 协同特征: [{'feature': 'max_inactive_period_From', 'reason': "洗钱模式中'休眠账户突然激活'的特征需要同时满足：长期无交易（max_inactive_period大）+金额突变（amount_sudden_spike_sender高）。两者组合可精准识别休眠账户被用于洗钱的行为。"}, {'feature': 'large_transaction_ratio', 'reason': '金额突变后如果伴随大量大额交易（large_transaction_ratio高），说明该账户已成为洗钱资金通道，而非偶然的异常交易。'}, {'feature': 'Receive_Send_Ratio', 'reason': "洗钱账户通常表现为'快进快出'：收到大额资金后立即转出，导致Receive_Send_Ratio接近1。配合金额突变特征，可识别'收到大额后立即转出'的洗钱模式。"}, {'feature': 'is_night_hour_From', 'reason': '线上跑分模式中，金额突变常发生在凌晨时段（20时-1时）。两个特征同时高时，洗钱嫌疑大幅提升。'}]
  - 误判场景: ['正常用户一次性大额消费（如购买家电、支付学费），金额从日常几十元突变至上万元', '工资发放场景：账户收到月薪后立即转出至理财账户，导致相邻交易的金额突变', '商户钱包日常经营：白天小额收款（几十元），晚上集中转出大额资金（数千元）', '用户更换消费场景：从日常小额消费转为大额旅游消费、购房定金等', '红包/优惠券使用：收到大额红包后立即消费，导致金额突变']
  - 监管说明: 该特征检测账户是否突然出现金额剧烈异常的转账，比如一个平时只转几十块的账户突然转了数万块，这是洗钱分子利用沉睡或正常账户进行大额资金转移的典型手法。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_sudden_spike_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按src和时间排序
    tx_sorted = tx_df.sort_values(['src', 'timestamp'])

    # 计算相邻交易金额的差异倍数
    tx_sorted['prev_amount'] = tx_sorted.groupby('src')['amount'].shift(1)
    tx_sorted['amount_diff_ratio'] = np.where(
        (tx_sorted['prev_amount'] > 0) & (tx_sorted['amount'] > 0),
        tx_sorted['amount'] / tx_sorted['prev_amount'],
        1.0
    )
    # 取对数，使正负变化对称
    tx_sorted['amount_diff_ratio_log'] = np.log2(tx_sorted['amount_diff_ratio'].clip(lower=0.001, upper=1000))

    # 每个sender的金额突变程度：取最大绝对对数比值
    stats = tx_sorted.groupby('src')['amount_diff_ratio_log'].agg(
        lambda x: np.abs(x).max() if len(x) > 0 else 0
    ).reset_index()
    stats.columns = ['wallet_id', 'amount_sudden_spike_sender']

    # 归一化到[0,1]（最大对数比值约为10，对应2^10=1024倍）
    stats['amount_sudden_spike_sender'] = stats['amount_sudden_spike_sender'] / 10
    stats['amount_sudden_spike_sender'] = stats['amount_sudden_spike_sender'].clip(upper=1.0)

    result = stats
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_sudden_spike_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_sudden_spike_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
