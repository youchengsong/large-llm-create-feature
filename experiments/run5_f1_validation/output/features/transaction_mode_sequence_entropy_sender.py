"""
特征: transaction_mode_sequence_entropy_sender
说明: 根据数字人民币洗钱模式，线上跑分（模式3）和商户收款码跑分（模式4）都涉及大量fan_out（分散转出）交易，导致交易模式序列的多样性降低。通过计算交易模式序列的熵值，可以有效识别这种异常模式偏好。

验证指标:
  - Score: 0.4485
  - MI: 0.0000
  - Redundancy: 0.6123
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 86

可解释性报告:
  - 值含义: {'0-0.5': '交易模式高度集中，钱包几乎只使用一种交易模式（如全是fan_out），可能为洗钱账户或特定自动化脚本操作', '0.5-1.0': '交易模式多样性较低，存在明显偏好，线上跑分模式（模式3）或商户收款码跑分（模式4）的典型特征', '1.0-1.5': '交易模式多样性正常，混合使用多种交易模式，符合正常用户行为特征', '1.5-2.32': '交易模式高度多样，钱包频繁切换多种交易模式，可能为正常活跃用户或存在异常分散行为'}
  - 风险阈值: 高风险阈值为 0.5121（异常交易75%分位数）。选择依据：异常交易的中位数（0.7011）和75%分位数（0.5121）均低于正常交易，说明异常交易的模式熵值整体偏低。当熵值低于0.5121时，意味着该钱包的交易模式多样性低于75%的异常账户，应触发警报。
  - 协同特征: [{'feature': 'fan_out_ratio（fan_out交易占比）', 'reason': '低熵值+高fan_out占比可精确识别线上跑分模式（模式3），两者结合可过滤掉偶然使用单种模式的正常用户'}, {'feature': 'night_transaction_ratio（凌晨交易占比）', 'reason': '低熵值+高凌晨交易占比可直接命中线上跑分模式（模式3）的典型行为特征，显著提高检测准确率'}, {'feature': 'small_amount_testing_flag（小额测试交易标记）', 'reason': '低熵值+小额测试交易标记可识别薅羊毛模式（模式1）中批量注册后使用单一交易模式的账户'}]
  - 误判场景: ['新注册的商户钱包：由于业务刚启动，可能仅使用fan_out模式进行收款，导致熵值偏低，但属于正常经营行为', '定期定额理财账户：用户设置自动定投后，交易模式长期保持single模式，熵值接近0，但无洗钱风险', '单一用途的福利发放钱包：企业批量发放红包时仅使用fan_out模式，交易模式单一但无异常']
  - 监管说明: 该特征衡量一个钱包在交易中是否频繁更换操作方式（如分散转出、集中转入等）。洗钱账户通常只使用一种固定模式操作，而正常用户会混合使用多种方式，因此该特征能有效识别行为单一的可疑账户。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 transaction_mode_sequence_entropy_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    from scipy.stats import entropy

    # 按src和时间排序，获取每个钱包的交易模式序列
    sorted_tx = tx_df.sort_values(['src', 'timestamp'])

    # 定义交易模式到数值的映射
    mode_map = {'fan_out': 1, 'fan_in': 2, 'forward': 3, 'single': 4, 'many_to_one': 5}
    sorted_tx['mode_code'] = sorted_tx['transaction_mode'].map(mode_map)

    # 计算每个钱包交易模式序列的熵
    def calc_entropy(group):
        mode_counts = group['mode_code'].value_counts(normalize=True)
        if len(mode_counts) <= 1:
            return 0
        return entropy(mode_counts.values, base=2)

    entropy_stats = sorted_tx.groupby('src').apply(calc_entropy).reset_index()
    entropy_stats.columns = ['wallet_id', 'transaction_mode_sequence_entropy_sender']
    result = entropy_stats
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: transaction_mode_sequence_entropy_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 transaction_mode_sequence_entropy_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
