"""
特征: fan_out_forward_ratio_change_sender
说明: 根据洗钱模式中的'线下跑分'和'线上跑分'描述，异常交易倾向于持续使用fan_out模式进行分散转出，很少切换回forward模式。通过计算fan_out->forward与forward->fan_out的转移比例，可以量化这种单向性。

验证指标:
  - Score: 0.3792
  - MI: 0.0000
  - Redundancy: 0.2586
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 66

可解释性报告:
  - 值含义: 该特征衡量钱包从'分散转出(fan_out)'模式切换到'顺序转发(forward)'模式的倾向性，与从'顺序转发'切换到'分散转出'的倾向性的比值（取对数平滑）。值越大，表示钱包更倾向于从fan_out切换到forward（行为模式多样化，更接近正常用户）；值越小（接近0或负值），表示钱包更倾向于从forward切换到fan_out，即持续使用fan_out模式进行分散转出，符合'线上跑分'和'线下跑分'中'分散转出后很少切换回正常转发模式'的异常行为特征。
  - 风险阈值: 高风险阈值为0.1074（异常交易中位数）以下。判断依据：异常交易中位数为0.1074，显著低于正常交易中位数0.6931，说明异常钱包的fan_out->forward转移概率远低于forward->fan_out转移概率，即它们几乎只从forward切换到fan_out，而很少从fan_out切回forward，这种单向性正是洗钱交易中'持续分散转出'的典型特征。
  - 协同特征: [{'feature': 'total_fan_out_ratio', 'reason': "互补性：total_fan_out_ratio衡量fan_out模式占总交易的比例，而本特征衡量fan_out与forward之间的切换方向性。两者结合可以区分'偶然使用fan_out但很快切换回forward的正常用户'和'持续使用fan_out的异常用户'。"}, {'feature': 'night_hour_transaction_ratio', 'reason': "互补性：线上跑分常在凌晨（20时-1时）进行，结合凌晨交易比例可以更精准地识别'凌晨持续fan_out分散转出'的异常模式。"}, {'feature': 'avg_inactive_period', 'reason': "互补性：线下跑分模式中，交易间隔较短（收到大额后立即转出），结合平均不活跃周期可以区分'快速周转的异常交易'和'正常分散转账'。"}]
  - 误判场景: [{'scenario': '正常商户的批量结算', 'explanation': '正常商户在营业结束后，可能将多笔收款（fan_out模式）集中转给财务账户，然后财务再按比例转发（forward模式）给多个供应商。这种业务场景也会产生较高的fan_out->forward切换，导致特征值偏高，被误判为高风险。'}, {'scenario': '个人用户的资金归集', 'explanation': '个人用户可能先分散收款（fan_in）后，再集中转出（fan_out），然后下一次又收到多笔款项后顺序转发（forward）。这种正常的资金管理行为也可能导致较高的forward->fan_out切换，特征值偏低，被误判为异常。'}]
  - 监管说明: 该指标衡量钱包在'分散转出'和'顺序转发'两种交易模式之间的切换倾向，异常钱包通常只会从'顺序转发'切换到'分散转出'，很少反向切换，表现出'只出不转'的单向性特征，这与洗钱活动中持续将资金分散转出的行为模式高度吻合。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 fan_out_forward_ratio_change_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按src和时间排序
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()

    # 获取下一模式
    tx_sorted['next_mode'] = tx_sorted.groupby('src')['transaction_mode'].shift(-1)

    # 过滤最后一条
    transitions = tx_sorted.dropna(subset=['next_mode']).copy()

    # 标记特定转移类型
    transitions['fan_out_to_forward'] = ((transitions['transaction_mode'] == 'fan_out') & 
                                          (transitions['next_mode'] == 'forward')).astype(int)
    transitions['forward_to_fan_out'] = ((transitions['transaction_mode'] == 'forward') & 
                                          (transitions['next_mode'] == 'fan_out')).astype(int)

    # 统计每个钱包的转移次数
    transition_stats = transitions.groupby('src').agg(
        fan_out_to_forward_count=('fan_out_to_forward', 'sum'),
        forward_to_fan_out_count=('forward_to_fan_out', 'sum'),
        total_fan_out=('transaction_mode', lambda x: (x == 'fan_out').sum()),
        total_forward=('transaction_mode', lambda x: (x == 'forward').sum())
    ).reset_index()
    transition_stats.columns = ['wallet_id', 'fan_out_to_forward_count', 'forward_to_fan_out_count', 
                                'total_fan_out', 'total_forward']

    # 计算比率：fan_out后转forward的概率 / forward后转fan_out的概率
    # 使用平滑避免除零
    transition_stats['fan_out_to_forward_rate'] = transition_stats['fan_out_to_forward_count'] / transition_stats['total_fan_out'].clip(lower=1)
    transition_stats['forward_to_fan_out_rate'] = transition_stats['forward_to_fan_out_count'] / transition_stats['total_forward'].clip(lower=1)

    # 计算比值，加平滑
    transition_stats['fan_out_forward_ratio_change'] = (
        (transition_stats['fan_out_to_forward_rate'] + 0.001) / 
        (transition_stats['forward_to_fan_out_rate'] + 0.001)
    )

    # 取对数使分布更对称
    transition_stats['fan_out_forward_ratio_change'] = np.log1p(transition_stats['fan_out_forward_ratio_change'])

    result = transition_stats[['wallet_id', 'fan_out_forward_ratio_change']]

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: fan_out_forward_ratio_change_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 fan_out_forward_ratio_change_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
