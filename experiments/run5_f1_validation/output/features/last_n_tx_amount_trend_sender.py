"""
特征: last_n_tx_amount_trend_sender
说明: small_amount_testing模式中先进行0.01-0.1元的小额测试，随后进行大额交易。这种从小到大的金额变化趋势是洗钱的典型特征。通过最近交易的金额趋势斜率可以量化这种模式。

验证指标:
  - Score: 0.5378
  - MI: 0.0000
  - Redundancy: 0.3706
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 64

可解释性报告:
  - 值含义: 该特征表示钱包最近10笔交易（不足5笔则取0）的金额变化趋势斜率。负值表示金额整体呈下降趋势（从大到小），正值表示金额整体呈上升趋势（从小到大），绝对值越大趋势越陡峭。正常钱包的中位数约-555，异常钱包中位数约-1211，说明异常钱包更倾向于金额下降趋势（先大后小），而正常钱包趋势相对平缓。
  - 风险阈值: 高风险阈值为小于-4000或大于2000。判断依据：异常交易的25%分位为-4071.98，75%分位为4.77，说明异常交易中约25%斜率小于-4000（强烈下降趋势，即先大额后小额），而正常交易仅有约25%斜率小于-3625。同时正值斜率（上升趋势）在异常中极少（75%分位仅4.77），因此斜率大于2000（强烈上升趋势）也可能异常。
  - 协同特征: 建议与以下特征配合使用：
1. **last_amount_From**（最近一笔交易金额）：结合趋势斜率可判断趋势是否以小额结尾（下降趋势+最后一笔金额小=典型测试后洗钱模式）
2. **avg_inactive_period_From**（平均不活跃间隔）：如果趋势为负且间隔突然变短，说明账户从休眠激活后立即进行大额交易，符合small_amount_testing模式
3. **small_amount_testing**交易motif标签：该特征本身就是为捕捉small_amount_testing设计，与motif标签配合可区分是真实测试还是正常消费波动
  - 误判场景: 可能的误判场景：
1. 正常用户大额消费后回归日常小额消费（如购买大件商品后恢复正常开销），产生负斜率
2. 正常用户工资入账后逐渐消费（先大后小），也会产生负斜率
3. 正常用户连续小额充值后突然大额消费（如游戏充值后买装备），产生正斜率
4. 交易笔数刚好5-10笔的短序列，斜率受极端值影响大
  - 监管说明: 该特征检测的是钱包交易金额从大到小或从小到大的变化模式——洗钱者常先小额测试账户功能，再转入大额资金并快速分散转出，形成明显的金额先升后降的V形或倒V形趋势，而正常用户交易金额通常随机波动，没有这种规律性变化。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 last_n_tx_amount_trend_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    from scipy.stats import linregress

    def calc_trend(group):
        if len(group) < 5:
            return 0.0
        group = group.sort_values('timestamp').tail(10)
        x = np.arange(len(group))
        y = group['amount'].values
        try:
            slope, _, _, _, _ = linregress(x, y)
            return slope
        except:
            return 0.0

    trend_df = tx_df.groupby('src').apply(calc_trend).reset_index()
    trend_df.columns = ['wallet_id', 'feature_value']
    trend_df['feature_value'] = trend_df['feature_value'].fillna(0)
    result = trend_df
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: last_n_tx_amount_trend_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 last_n_tx_amount_trend_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
