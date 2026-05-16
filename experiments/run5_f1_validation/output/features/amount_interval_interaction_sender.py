"""
特征: amount_interval_interaction_sender
说明: 根据领域知识，'休眠账户突然激活并发生大额交易'是洗钱的重要特征。通过计算每笔交易金额与上一笔交易间隔的乘积或比值，可以量化这种'长时间等待后的大额交易'模式。具体来说，对于一个钱包，计算其所有交易中'金额*间隔'的变异系数(CV)，高CV值意味着有些交易同时具有大金额和长时间隔，而其他交易则正常，这正是洗钱行为的典型特征。这与已有特征avg_inactive_period(单纯的平均间隔)和Std_Sent(单纯的金额波动)有本质区别。

验证指标:
  - Score: 0.2866
  - MI: 0.0000
  - Redundancy: 0.3136
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 92

可解释性报告:
  - 值含义: 该特征衡量钱包交易的'金额-时间间隔'乘积变异程度。取值在[0, +∞)，值越小表示钱包所有交易的金额与间隔关系越稳定一致，值越大表示存在部分交易同时具有'大金额+长时间隔'或'小金额+短间隔'的极端组合。正常钱包的中位数约3.05，异常钱包中位数约3.04，差异极小，说明该特征在区分正常与异常交易上几乎无效。
  - 风险阈值: 无有效风险阈值。KS=0.0208远低于0.2的有效区分标准，IV=0.0058远低于0.02的弱预测力阈值。正常与异常交易的分布几乎完全重叠（均值差异仅0.009，标准差差异0.14），无法找到任何有意义的切割点能将两类样本分开。Delta F1仅为0.000016，表明该特征加入模型后对性能无实际提升。
  - 协同特征: 不建议与其他特征配合使用。该特征在所有验证指标上均表现极差，属于无效特征。若强行使用，可与以下特征组合尝试：
- avg_inactive_period（平均交易间隔）：两者都涉及时间维度，但本特征已被证明无区分度
- Std_Sent（发送金额标准差）：两者都涉及金额波动，但本特征对异常交易无捕捉能力
- 凌晨交易占比：结合时间异常特征，但本特征无法增强任何识别效果
  - 误判场景: 以下场景可能导致正常钱包产生该特征的高值，但实际并非洗钱：
1. 正常用户规律性消费：用户每月固定大额支付房租（金额大、间隔30天），同时日常小额消费（金额小、间隔短），导致金额-间隔乘积的变异系数升高
2. 季节性消费：用户平时消费少，但在双十一、春节等节日集中大额消费，产生类似'长时间隔后大额交易'的模式
3. 工资入账场景：用户每月收到固定工资（大额、长间隔），同时日常频繁小额消费，同样会产生高CV值
4. 新激活钱包：刚注册钱包的用户，第一笔交易与后续交易间隔差异大（注册后很久才首次交易），但这是正常行为
  - 监管说明: 该特征试图通过分析每笔交易的金额与距离上一笔交易的时间间隔之间的波动关系来识别洗钱行为，但实际验证表明正常交易和异常交易在该指标上几乎完全一样，无法有效区分，因此不建议在反洗钱监测中使用此特征。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_interval_interaction_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按src和时间排序，计算每笔交易与其上一笔交易的间隔(已存在interval字段)
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()

    # 计算每笔交易的金额*间隔(以天为单位)，间隔为0时设为1避免无穷大
    tx_sorted['interval_days'] = tx_sorted['interval'].clip(lower=1) / (24 * 3600)  # 转换为天
    tx_sorted['amount_interval_product'] = tx_sorted['amount'] * tx_sorted['interval_days']

    # 对每个钱包，计算金额*间隔的变异系数(CV)
    # 先按钱包聚合
    wallet_stats = tx_sorted.groupby('src')['amount_interval_product'].agg(['mean', 'std', 'count']).reset_index()
    wallet_stats.columns = ['wallet_id', 'mean_product', 'std_product', 'count_product']

    # 计算CV，只考虑至少有2笔交易的钱包
    wallet_stats['amount_interval_interaction_sender'] = np.where(
        (wallet_stats['count_product'] >= 2) & (wallet_stats['mean_product'] > 0),
        wallet_stats['std_product'] / wallet_stats['mean_product'],
        0
    )
    wallet_stats['amount_interval_interaction_sender'] = wallet_stats['amount_interval_interaction_sender'].fillna(0)

    result = wallet_stats[['wallet_id', 'amount_interval_interaction_sender']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_interval_interaction_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_interval_interaction_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
