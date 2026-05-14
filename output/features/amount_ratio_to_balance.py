"""
特征: amount_ratio_to_balance
说明: 根据领域知识，洗钱模式中small_amount_testing金额仅0.01-0.1元，而正常交易金额1-5000元。对比钱包初始余额，测试交易的金额占比极低，而正常交易占比适中。同时，洗钱模式中也有大额交易接近限额，金额占比高。这个比值能同时捕捉两种异常。

验证指标:
  - Score: 0.2869
  - MI: 0.0013
  - Redundancy: 0.3649
  - Cramer's V: 0.0165
  - Delta PR-AUC: -0.023529
  - SHAP Rank: 20

可解释性报告:
  - 值含义: {'0.0-0.2': '极低异常度：钱包交易金额与余额的比值处于正常范围，既没有异常小的测试交易，也没有异常大的大额交易，属于正常用户的典型行为模式', '0.2-0.5': '低异常度：存在少量极端比值交易，可能是正常用户偶尔的小额测试或大额转账，洗钱嫌疑较低', '0.5-0.7': '中等异常度：约一半的交易是极端比值（极小或极大），需关注是否存在small_amount_testing或大额洗钱行为', '0.7-0.9': '高异常度：大部分交易呈现极端比值特征，强烈暗示存在系统性洗钱行为，需结合其他特征进一步排查', '0.9-1.0': '极高异常度：几乎全部交易均为极端比值，极大概率是专门的洗钱钱包，应立即触发预警'}
  - 风险阈值: {'primary_threshold': 0.7, 'secondary_threshold': 0.9, 'justification': '从分布差异看，异常交易的中位数(0.7125)明显高于正常交易的中位数(0.5094)，且异常交易的25%分位(0.3710)已接近正常交易的75%分位(0.3090)。取0.7作为主阈值，可覆盖约50%的异常交易（中位数以上），同时仅误报约25%的正常交易（正常75%分位为0.9034，说明0.7以下仍有大量正常交易）。0.9作为次级阈值，对应正常交易的75%分位，超过此值的正常交易仅25%，但异常交易中仍有相当比例，适合作为强制预警线。', 'business_logic': 'small_amount_testing的金额为0.01-0.1元，若初始余额100元，比值仅0.0001-0.001；merchant_laundering的金额10-1000元，若余额仅20元，比值可达0.5以上。两种极端情况分别对应极小和极大比值，0.7的阈值意味着70%的交易呈现极端比值特征'}
  - 协同特征: [{'feature': 'Timestamp_Hour', 'reason': 'small_amount_testing集中在深夜(0-5点,22-23点)，merchant_laundering也在夜间(20-1点)。当amount_ratio_to_balance高且交易时间集中在夜间时，洗钱嫌疑大幅提升'}, {'feature': 'transaction_mode', 'reason': '异常交易中fan_out(54.3%)和fan_in(22.2%)占比远高于正常交易。高amount_ratio_to_balance结合fan_out/fan_in模式，可精准识别资金归集或分散转出的洗钱路径'}, {'feature': 'avg_inactive_period', 'reason': 'small_amount_testing会激活休眠账户(>=90天无交易)。若高amount_ratio_to_balance且avg_inactive_period很大（账户长期休眠后突然活跃），则测试交易嫌疑极高'}, {'feature': 'Out_Degree / In_Degree', 'reason': 'merchant_laundering表现为商户大量收款后转出，fan_out模式导致Out_Degree高；class4_laundering涉及多账户交叉转账。结合度数特征可区分不同类型的洗钱'}]
  - 误判场景: [{'scenario': '新开户用户首次大额充值', 'explanation': '新钱包初始余额低（如0.01元），首次充值10000元会导致比值高达1000000，远超0.5的large_ratio阈值。但这是正常行为，可通过账户注册时长特征排除'}, {'scenario': '余额极低的小额消费钱包', 'explanation': '钱包初始余额仅1元，日常消费10元就会产生10的比值，被标记为large_ratio。这类用户虽然比值高，但交易频率和模式正常，不会同时满足夜间交易等条件'}, {'scenario': '批量发放红包或补贴', 'explanation': '企业向员工批量发放小额红包（如0.1元），若员工钱包余额也低（如0.5元），比值0.2可能落在中等区间。但这类交易有明确发放方和用途，交易对手固定'}, {'scenario': '余额极低的测试环境钱包', 'explanation': '测试钱包余额0.01元，任何0.01元以上的交易都会产生极大比值。这类钱包通常交易笔数少、无规律，可通过交易总数和活跃天数辅助判断'}]
  - 监管说明: 该特征衡量每笔交易金额占钱包初始余额的比例是否异常极端——洗钱测试交易金额极小（0.01元）而余额相对较大，或洗钱大额交易接近钱包限额（比例超过50%），正常用户的交易比例通常处于中间范围，不会频繁出现这种极端情况。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_ratio_to_balance，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每笔交易的金额与发送方钱包初始余额的比值
    # 先获取每个钱包的初始余额
    init_balance = open_df[['wallet_id', 'init_balance']].copy()
    init_balance.columns = ['wallet_id', 'init_balance']

    # 合并到交易表
    merged = tx_df.merge(init_balance, left_on='src', right_on='wallet_id', how='left')

    # 计算比值，避免除零
    merged['amount_ratio'] = merged['amount'] / merged['init_balance'].clip(lower=0.01)

    # 对每个钱包计算：极端小比值(测试交易)和极端大比值(大额洗钱)的占比
    # 定义阈值：比值<0.001为极小(测试交易)，比值>0.5为极大(接近余额的大额交易)
    merged['is_tiny_ratio'] = (merged['amount_ratio'] < 0.001).astype(int)
    merged['is_large_ratio'] = (merged['amount_ratio'] > 0.5).astype(int)

    # 聚合到钱包级别
    wallet_stats = merged.groupby('src').agg(
        tiny_ratio_count=('is_tiny_ratio', 'sum'),
        large_ratio_count=('is_large_ratio', 'sum'),
        total_tx=('tx_id', 'count')
    ).reset_index()

    wallet_stats.columns = ['wallet_id', 'tiny_ratio_count', 'large_ratio_count', 'total_tx']

    # 计算极端比值占比之和（极小额+极大额），作为异常程度的度量
    wallet_stats['amount_ratio_to_balance'] = (wallet_stats['tiny_ratio_count'] + wallet_stats['large_ratio_count']) / wallet_stats['total_tx'].clip(lower=1)

    result = wallet_stats[['wallet_id', 'amount_ratio_to_balance']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_ratio_to_balance")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_ratio_to_balance: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
