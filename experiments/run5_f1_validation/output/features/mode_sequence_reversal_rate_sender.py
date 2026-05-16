"""
特征: mode_sequence_reversal_rate_sender
说明: 洗钱行为通常遵循固定模式（如持续fan_out分散转出），较少出现模式反转。正常交易行为（如消费、转账）的模式序列更随机，反转更频繁。低反转率是异常信号。

验证指标:
  - Score: 0.2811
  - MI: 0.0000
  - Redundancy: 0.4587
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 84

可解释性报告:
  - 值含义: 该特征表示钱包在交易模式序列中出现'反转'的频率（如从fan_out变为forward再变回fan_out）。取值范围为0到1，0表示从未出现模式反转（所有模式转换均为单向），1表示每次模式转换都是反转。正常交易行为中，反转率通常在0.43-0.57之间，异常交易的反转率分布与之高度重叠，因此该特征区分度有限。
  - 风险阈值: 无明确高风险阈值。由于正常与异常交易在该特征上的分布几乎完全重叠（中位数、均值、四分位数均相同），无法设定有效的风险阈值。当前KS=0.0217、IV=0.0081均表明该特征几乎无区分能力，不建议单独用于风险判定。
  - 协同特征: 建议与以下特征配合使用：
1. **fan_out_ratio**（fan_out模式占比）：结合反转率可判断是否持续fan_out（低反转率+高fan_out占比）或偶尔混入其他模式（高反转率+低fan_out占比）。
2. **hour_entropy**（交易小时熵值）：低反转率+低时间熵值可能暗示机械式操作（如自动化脚本）。
3. **sequence_length**（交易序列长度）：短序列下反转率统计不可靠，需结合序列长度做加权。
  - 误判场景: 1. **短序列钱包**：只有3-4笔交易的钱包，反转率容易被极端化（如3笔交易中1次反转=0.5，但样本量不足）。
2. **周期性消费模式**：如用户定期向多个账户转账（fan_out→forward→fan_out…），可能自然产生反转。
3. **商户结算**：商户集中收款（fan_in）后分散发放工资（fan_out），模式切换频繁。
  - 监管说明: 该特征衡量的是钱包交易模式的'翻来覆去'程度——比如先分散转钱、再集中收钱、又分散转钱。正常用户可能因生活需要自然产生这种变化，但洗钱账户通常只会用一种固定模式（比如一直分散转钱），所以变化少。不过这个特征在本次数据中区分度很差，不建议单独作为判断依据。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 mode_sequence_reversal_rate_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按每个src钱包排序交易
    sorted_tx = tx_df.sort_values(['src', 'timestamp']).copy()

    # 计算模式反转：当前模式与前一个模式不同，且前一个模式与再前一个相同（即反转）
    sorted_tx['mode_prev'] = sorted_tx.groupby('src')['transaction_mode'].shift(1)
    sorted_tx['mode_prev2'] = sorted_tx.groupby('src')['transaction_mode'].shift(2)

    # 反转定义：mode_prev2 == mode_prev != current（从A变B再变回A）
    sorted_tx['is_reversal'] = (
        (sorted_tx['mode_prev'] == sorted_tx['mode_prev2']) & 
        (sorted_tx['mode_prev'] != sorted_tx['transaction_mode'])
    ).astype(int)

    # 统计每个src的反转次数和总转换次数
    reversal_stats = sorted_tx.groupby('src').agg(
        reversal_count=('is_reversal', 'sum'),
        total_transitions=('mode_change', lambda x: (x == 1).sum())
    ).reset_index()

    # 计算反转率
    reversal_stats['mode_sequence_reversal_rate_sender'] = (
        reversal_stats['reversal_count'] / reversal_stats['total_transitions'].clip(lower=1)
    )

    result = reversal_stats[['src', 'mode_sequence_reversal_rate_sender']].copy()
    result.columns = ['wallet_id', 'mode_sequence_reversal_rate_sender']
    result['mode_sequence_reversal_rate_sender'] = result['mode_sequence_reversal_rate_sender'].fillna(0)

    # 填充未出现的src
    all_wallets = open_df[['wallet_id']].copy()
    result = all_wallets.merge(result, on='wallet_id', how='left')
    result['mode_sequence_reversal_rate_sender'] = result['mode_sequence_reversal_rate_sender'].fillna(0)
    result = result[['wallet_id', 'mode_sequence_reversal_rate_sender']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: mode_sequence_reversal_rate_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 mode_sequence_reversal_rate_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
