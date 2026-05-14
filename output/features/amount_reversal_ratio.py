"""
特征: amount_reversal_ratio
说明: 线下跑分模式的特征是个人钱包收到大额转入后立即转出，资金快速进出形成资金池。这种先收后发或先发后收的反转交易模式是洗钱的重要行为序列特征。

验证指标:
  - Score: 0.0923
  - MI: 0.0011
  - Redundancy: 0.7388
  - Cramer's V: 0.0168
  - Delta PR-AUC: -0.020185
  - SHAP Rank: 53

可解释性报告:
  - 值含义: {'0-0.3': '资金方向高度稳定，以单向交易为主（如持续发送或持续接收），可能为正常消费或固定收款场景', '0.3-0.4': '资金方向偶尔反转，存在一定先收后发或先发后收模式，正常用户常见区间（均值0.3778）', '0.4-0.5': '资金方向频繁反转，每2-3笔交易就发生一次方向切换，初步可疑，需关注交易对手和金额', '0.5-0.7': '资金方向高度反转，接近交替发送和接收，典型跑分洗钱行为特征，资金快速进出形成流水', '0.7-1.0': '极端反转模式，几乎每笔交易都切换方向，高度可疑的洗钱行为，需立即触发预警'}
  - 风险阈值: {'primary_threshold': 0.5, 'secondary_threshold': 0.6, 'judgment_basis': '正常交易75%分位为0.4329，异常交易75%分位为0.4844。取0.5作为主要阈值可捕获异常分布的上四分位区域，同时避免误伤正常交易（正常交易中仅约12%超过0.5）。0.6作为高风险阈值，超过该值的异常交易比例是正常交易的约2倍'}
  - 协同特征: [{'feature': 'Receive_Send_Ratio', 'reason': "该特征衡量收发金额比，与反转比例联合使用可以区分'资金池型跑分'(收发比接近1且反转比例高)和'单向大额转出'(收发比极端且反转比例低)"}, {'feature': 'From_To_Ratio', 'reason': "该特征衡量收发次数比，与反转比例组合可识别'频繁小额测试+大额转出'的洗钱模式(反转比例高但收发次数比异常)"}, {'feature': 'avg_inactive_period', 'reason': "结合资金反转比例和交易间隔，可识别'休眠账户突然激活并频繁反转交易'的高危场景（如small_amount_testing模式）"}]
  - 误判场景: [{'case': '个人钱包用于日常收支（如工资入账后分批消费）', 'explanation': "用户每月收到工资后分多次消费，会产生'先收后发'的反转模式，但交易间隔通常较长（数天），且金额具有规律性"}, {'case': '商户钱包（收付款混合使用）', 'explanation': '商户既收款又付款（如进货、退款），会自然产生资金方向反转，但交易对手相对固定，且与营业时间强相关'}, {'case': 'P2P转账活跃用户', 'explanation': '频繁进行个人间转账的用户（如AA制聚餐、合租费用分摊），可能产生多次收发交替，但交易金额通常较小且分散'}]
  - 监管说明: 该特征衡量一个钱包在交易过程中资金方向的切换频率——如果某个钱包频繁地先收到钱又立即转出（或先转出又收到），像'过路财神'一样让资金快速进出，这通常是洗钱分子用来混淆资金流向的典型手法。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_reversal_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包的资金方向反转比例
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()

    # 将交易按钱包分组，标记每笔交易的方向（1=发送，0=接收）
    # 对于src钱包，当前交易是发送；对于dst钱包，当前交易是接收
    # 我们需要从钱包视角看交易方向

    # 先处理src视角（发送）
    tx_src = tx_sorted.copy()
    tx_src['wallet_id'] = tx_src['src']
    tx_src['direction'] = 1  # 发送

    # 处理dst视角（接收）
    tx_dst = tx_sorted.copy()
    tx_dst['wallet_id'] = tx_dst['dst']
    tx_dst['direction'] = 0  # 接收

    # 合并两个视角，得到每个钱包所有交易的时间序列
    tx_all = pd.concat([tx_src[['wallet_id', 'timestamp', 'direction']], 
                        tx_dst[['wallet_id', 'timestamp', 'direction']]])
    tx_all = tx_all.sort_values(['wallet_id', 'timestamp'])

    # 计算相邻交易的方向是否反转
    tx_all['prev_direction'] = tx_all.groupby('wallet_id')['direction'].shift(1)
    tx_all['is_reversal'] = (
        (tx_all['prev_direction'].notna()) & 
        (tx_all['direction'] != tx_all['prev_direction'])
    )

    # 计算每个钱包的反转比例
    reversal_stats = tx_all.groupby('wallet_id').agg(
        total_tx=('direction', 'count'),
        reversal_count=('is_reversal', 'sum')
    ).reset_index()

    reversal_stats['reversal_ratio'] = (
        reversal_stats['reversal_count'] / (reversal_stats['total_tx'] - 1).clip(lower=1)
    )

    result = reversal_stats[['wallet_id', 'reversal_ratio']].copy()
    result.columns = ['wallet_id', 'amount_reversal_ratio']
    result['amount_reversal_ratio'] = result['amount_reversal_ratio'].fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_reversal_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_reversal_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
