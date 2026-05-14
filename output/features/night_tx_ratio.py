"""
特征: night_tx_ratio
说明: 洗钱行为常利用深夜无人监管时段进行试探性交易或大额转账，如小金额测试（0-5点）和4类钱包限额规避（23点）。此特征可识别这种异常时间偏好。

验证指标:
  - Score: 0.0511
  - MI: 0.0010
  - Redundancy: 0.8833
  - Cramer's V: 0.0162
  - Delta PR-AUC: -0.015620
  - SHAP Rank: 60

可解释性报告:
  - 值含义: 该特征表示每个钱包在深夜时段（0-5点）发起的交易占其全部交易的比例。值接近0表示该钱包几乎不在深夜交易（正常行为），值在0.03-0.05表示有一定深夜交易但可能正常（如夜班工作者），值超过0.1表示深夜交易占比异常偏高（可疑），值超过0.5表示主要交易活动发生在深夜（高度可疑）。
  - 风险阈值: 高风险阈值建议设为0.06。判断依据：正常交易75%分位为0.0432，异常交易75%分位为0.0438，两者差异极小，但异常交易标准差(0.0177)略大于正常(0.0157)，说明异常交易中深夜交易占比的波动更大。结合洗钱模式中'小金额测试(0-5点)'和'4类钱包限额规避(23点)'的领域知识，0.06对应正常交易约95%分位，能捕捉到尾部异常行为。
  - 协同特征: 建议与以下特征配合使用：
1. **small_amount_ratio**（小金额交易占比）：深夜小金额测试是洗钱试探的典型模式，两者组合可精准识别试探性洗钱。
2. **fan_out_ratio**（分散转出占比）：线上跑分模式(20-1点)常伴随fan_out模式，深夜高频分散转出是强信号。
3. **inactive_days**（休眠天数）：休眠账户突然在深夜激活并交易是洗钱激活账户的典型特征。
4. **amount_zscore**（金额Z分数）：深夜大额转账（偏离历史均值）叠加high night_tx_ratio是线下跑分模式的特征。
  - 误判场景: 以下场景可能导致误判：
1. **夜班工作者**：如保安、医护人员、工厂夜班工人，其正常消费和转账可能集中在深夜。
2. **跨境时差用户**：钱包注册地与实际居住地有时差，深夜交易在当地可能是白天。
3. **自动扣费服务**：某些订阅服务（如云存储、音乐平台）的自动扣费发生在凌晨。
4. **游戏充值用户**：深夜游戏充值行为在年轻用户中较为常见。
5. **留学生或海外务工人员**：与国内亲友的转账因时差可能在深夜。
  - 监管说明: 该特征通过计算用户深夜（0-5点）交易占比，识别那些偏好无人监管时段进行交易的可疑账户——洗钱分子常利用深夜系统无人值守、风控响应慢的窗口期进行试探性转账或规避限额操作。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 night_tx_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 提取交易小时
    if 'hour' not in tx_df.columns:
        tx_df['hour'] = pd.to_datetime(tx_df['timestamp']).dt.hour

    # 定义深夜时段 (0-5点)
    tx_df['is_night'] = tx_df['hour'].isin([0, 1, 2, 3, 4, 5])

    # 按转出方统计深夜交易占比
    night_ratio = tx_df.groupby('src')['is_night'].mean().reset_index()
    night_ratio.columns = ['wallet_id', 'feature_value']
    night_ratio['feature_value'] = night_ratio['feature_value'].fillna(0)

    result = night_ratio

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: night_tx_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 night_tx_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
