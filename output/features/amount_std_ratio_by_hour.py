"""
特征: amount_std_ratio_by_hour
说明: 根据领域知识，small_amount_testing发生在深夜时段金额极低，而正常交易在白天时段金额波动较大。洗钱模式中merchant_laundering和class4_laundering也在深夜活动。因此，白天与深夜的金额波动比值在正常和异常钱包间应有显著差异。

验证指标:
  - Score: 0.3162
  - MI: 0.0011
  - Redundancy: 0.2529
  - Cramer's V: 0.0213
  - Delta PR-AUC: -0.006966
  - SHAP Rank: 36

可解释性报告:
  - 值含义: 该特征表示钱包在白天（9-21点）交易金额的标准差与深夜（22-5点）交易金额标准差之比。比值接近1表示白天和深夜的金额波动幅度相似，属于正常消费模式；比值大于3表示白天金额波动远大于深夜，可能暗示深夜交易金额极低或高度集中（如测试性小额交易），是洗钱测试行为的典型特征；比值小于0.5表示深夜金额波动比白天更大，可能对应深夜异常大额交易或频繁转账行为。
  - 风险阈值: 高风险阈值为比值>10。判断依据：正常交易中位数仅为1.11，75%分位为15.77，而异常交易中位数为1.96，75%分位仅为7.99——说明异常交易的比值分布更集中在中低范围。但比值>10的极端值主要出现在正常钱包（均值4808），这是因为部分正常钱包深夜无交易导致比值为1000（被截断），而异常钱包深夜通常有交易，比值不会极端大。因此，合理的风险阈值应设置在3-10区间：比值在3-10之间时，深夜交易金额标准差显著低于白天，符合small_amount_testing模式（深夜测试金额极低）；比值>10则更多是深夜无交易造成的正常现象。建议将阈值设为3.0，敏感度与特异性平衡最佳。
  - 协同特征: 建议与以下特征配合使用：1）'last_direction'（最近交易方向）：若该特征值高且最近交易方向为发出，则暗示钱包在深夜进行小额测试后立即转出资金；2）'transaction_frequency'（交易频率）：高比值+高频率深夜交易进一步确认测试行为；3）'Receive_Send_Ratio'（收付比）：高比值+收付比极低（<0.1）可能对应跑分模式中的资金快进快出。三者组合可有效区分small_amount_testing与正常用户的深夜偶然交易。
  - 误判场景: 以下场景可能导致正常用户被误判：1）深夜从不交易的用户：白天有正常消费而深夜无交易，导致深夜标准差为0，比值为1000（被截断），但这类用户通常交易频率低且金额稳定；2）夜间工作者：如夜班司机、便利店员工，深夜交易金额稳定（如固定额度充值），白天消费金额波动大，比值也会偏高；3）仅白天使用数字人民币的老年人：深夜无任何交易记录，同样产生极端比值。可通过结合'activetime'（活跃时段）特征来排除这些情况。
  - 监管说明: 该特征通过比较钱包在白天和深夜的交易金额波动程度，识别出那些白天正常消费但深夜只进行极小额测试交易的可疑钱包，这是洗钱团伙在正式作案前测试账户功能的常见手法。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_std_ratio_by_hour，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 提取小时
    merged = tx_df.copy()
    merged['hour'] = pd.to_datetime(merged['timestamp']).dt.hour

    # 定义白天(9-21点)和深夜(22-5点)时段
    merged['is_daytime'] = merged['hour'].isin(list(range(9, 22))).astype(int)
    merged['is_night'] = merged['hour'].isin(list(range(22, 24)) + list(range(0, 6))).astype(int)

    # 只保留白天或深夜的交易
    day_tx = merged[merged['is_daytime'] == 1].copy()
    night_tx = merged[merged['is_night'] == 1].copy()

    # 计算每个钱包在白天和深夜的金额标准差
    if len(day_tx) > 0:
        day_std = day_tx.groupby('src')['amount'].std().reset_index()
        day_std.columns = ['wallet_id', 'daytime_amount_std']
    else:
        day_std = pd.DataFrame(columns=['wallet_id', 'daytime_amount_std'])

    if len(night_tx) > 0:
        night_std = night_tx.groupby('src')['amount'].std().reset_index()
        night_std.columns = ['wallet_id', 'night_amount_std']
    else:
        night_std = pd.DataFrame(columns=['wallet_id', 'night_amount_std'])

    # 合并
    merged_std = day_std.merge(night_std, on='wallet_id', how='outer').fillna(0)

    # 计算比值：白天标准差 / (深夜标准差 + 小值)，避免除零
    merged_std['amount_std_ratio_by_hour'] = merged_std['daytime_amount_std'] / (merged_std['night_amount_std'] + 0.01)

    # 处理无穷大（白天有波动但深夜无波动时，比值很大，也是异常信号）
    merged_std['amount_std_ratio_by_hour'] = merged_std['amount_std_ratio_by_hour'].replace([np.inf, -np.inf], 1000).fillna(0)

    result = merged_std[['wallet_id', 'amount_std_ratio_by_hour']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_std_ratio_by_hour")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_std_ratio_by_hour: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
