"""
特征: late_night_active_day_ratio
说明: 洗钱模式中，small_amount_testing（测试交易）和 class4_laundering 都明确在深夜时段进行。如果钱包频繁在深夜交易，尤其是每天深夜都有交易，则异常概率高。深夜活跃天数占比比单纯的平均小时更能捕捉持续性异常行为。

验证指标:
  - Score: 0.6000
  - MI: 0.0014
  - Redundancy: 0.8282
  - Cramer's V: 0.0239
  - Delta PR-AUC: 0.007109
  - SHAP Rank: 15

可解释性报告:
  - 值含义: 该特征表示钱包在深夜时段（22:00-05:59）活跃的天数占其总活跃天数的比例。正常交易均值约0.34，异常交易均值约0.20，说明正常用户也有一定比例的深夜交易，但异常用户深夜交易占比反而更低——这与预期相反，需要谨慎解读。
  - 风险阈值: 高风险阈值为 < 0.1 或 > 0.6。依据：异常交易中位数0.177，25%分位0.1；正常交易25%分位0.183。值<0.1可能代表钱包在深夜几乎不交易（极端正常），但值>0.6则代表钱包几乎只在深夜交易，符合small_amount_testing和class4_laundering的深夜特征。建议以0.6为高风险阈值。
  - 协同特征: 建议与以下特征配合使用：
1. Timestamp_Hour + transaction_frequency：深夜高频交易组合可识别线上跑分模式
2. small_amount_testing_flag（若有）：低金额+深夜高活跃度=测试交易
3. Out_Degree + In_Degree：深夜fan_out模式=典型洗钱
4. avg_inactive_period：休眠账户突然激活+深夜交易=高风险组合
  - 误判场景: 可能的误判场景：
1. 夜班工作者（如保安、护士、物流人员）正常在深夜消费
2. 跨境时差用户：海外用户在当地白天交易，对应北京时间深夜
3. 数字人民币红包/优惠活动在深夜发放时产生的正常交易
4. 自动还款/定时转账任务在凌晨执行
  - 监管说明: 该特征衡量一个钱包在深夜（晚上10点到早上6点）交易的频繁程度——如果某个钱包几乎只在深夜进行交易，可能是在利用夜间监管薄弱时段进行测试交易或洗钱，但需要注意夜班工作者等正常场景也可能产生类似行为。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 late_night_active_day_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    # 提取交易小时
    tx_df_copy = tx_df.copy()
    tx_df_copy['hour'] = pd.to_datetime(tx_df_copy['timestamp']).dt.hour
    tx_df_copy['date'] = pd.to_datetime(tx_df_copy['timestamp']).dt.date
    # 深夜时段：22-23点 和 0-5点
    tx_df_copy['is_late_night'] = ((tx_df_copy['hour'] >= 22) | (tx_df_copy['hour'] <= 5)).astype(int)
    # 计算每个钱包的总活跃天数和深夜活跃天数
    total_days = tx_df_copy.groupby('src')['date'].nunique().reset_index()
    total_days.columns = ['wallet_id', 'total_active_days']
    late_night_days = tx_df_copy[tx_df_copy['is_late_night']==1].groupby('src')['date'].nunique().reset_index()
    late_night_days.columns = ['wallet_id', 'late_night_active_days']
    # 合并计算比例
    result = total_days.merge(late_night_days, on='wallet_id', how='left').fillna(0)
    result['late_night_active_day_ratio'] = result['late_night_active_days'] / result['total_active_days'].clip(lower=1)
    result = result[['wallet_id', 'late_night_active_day_ratio']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: late_night_active_day_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 late_night_active_day_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
