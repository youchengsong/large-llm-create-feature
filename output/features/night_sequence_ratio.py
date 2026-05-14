"""
特征: night_sequence_ratio
说明: 异常交易行为序列中深夜时段交易占比高（small_amount_testing在0-5点、22-23点；class4_laundering在23-1点；online_laundering在20-1点），正常交易深夜占比极低。该特征直接量化行为序列的时间异常程度。

验证指标:
  - Score: 0.0847
  - MI: 0.0011
  - Redundancy: 0.8683
  - Cramer's V: 0.0138
  - Delta PR-AUC: -0.004736
  - SHAP Rank: 47

可解释性报告:
  - 值含义: {'0-0.05': '几乎无深夜交易，符合正常用户作息规律，风险极低。', '0.05-0.12': '少量深夜交易，可能是偶尔的夜间消费或转账，属于正常波动范围。', '0.12-0.2': '中等偏高的深夜交易比例，可能存在夜间活跃特征，需结合其他特征判断是否异常。', '0.2-0.5': '显著偏高的深夜交易比例，高度可疑——用户近半数交易发生在深夜，与正常生活规律严重不符。', '0.5-1.0': '极高深夜交易比例，强烈暗示洗钱行为（如线上跑分、小金额测试、4类钱包夜间洗钱等模式）。'}
  - 风险阈值: 0.15
  - 协同特征: [{'feature': 'transaction_frequency', 'reason': '深夜高频交易是线上跑分的典型特征，两者结合可精准识别凌晨时段的高频异常交易模式。'}, {'feature': 'activetime', 'reason': 'activetime衡量用户活跃时段分布，与night_sequence_ratio互补——前者反映整体活跃时段，后者聚焦深夜异常，联合使用可区分正常夜猫子用户与洗钱行为。'}, {'feature': 'send_concentration', 'reason': '洗钱行为常伴随集中转出（如分散转入后集中转出），深夜交易+高发送集中度是商户收款码跑分的典型特征。'}, {'feature': 'large_transaction_ratio', 'reason': '深夜大额交易（如class4_laundering模式中日限额5000元）与深夜小金额测试（small_amount_testing）的模式差异可通过此组合区分。'}]
  - 误判场景: ['夜班工作者（如医护人员、安保人员、24小时便利店员工）在深夜正常消费或转账。', '跨境用户因时差导致当地时间深夜但北京时间白天（如海外留学生深夜与国内家人转账）。', '习惯性夜间购物的用户（如深夜刷电商平台、购买数字产品）。', '偶尔参加夜间活动（如聚会、演唱会）后产生的深夜交易。']
  - 监管说明: 该特征衡量用户在深夜时段（0-5点、22-23点）的交易占比——正常人的交易大多在白天，如果某账户近一半交易发生在深夜，说明其行为模式与正常生活规律严重偏离，可能是利用夜间监管薄弱时段进行洗钱操作，如小金额测试、线上跑分或4类钱包限额洗钱。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 night_sequence_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 提取小时
    tx_df_copy = tx_df.copy()
    tx_df_copy['hour'] = pd.to_datetime(tx_df_copy['timestamp']).dt.hour

    # 定义深夜时段：0-5点 和 22-23点
    night_mask = ((tx_df_copy['hour'] >= 0) & (tx_df_copy['hour'] <= 5)) | ((tx_df_copy['hour'] >= 22) & (tx_df_copy['hour'] <= 23))

    # 计算每个钱包深夜交易占比
    tx_df_copy['is_night'] = night_mask.astype(int)

    night_ratio = tx_df_copy.groupby('src')['is_night'].mean().reset_index()
    night_ratio.columns = ['wallet_id', 'feature_value']

    result = night_ratio
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: night_sequence_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 night_sequence_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
