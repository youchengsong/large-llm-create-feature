"""
特征: dst_concentration_change_over_time
说明: 异常交易中fan_out模式（54.3%）和fan_in模式（22.2%）占比远高于正常，表明异常账户在不同时期可能切换交易模式（如先分散测试再集中洗钱），导致交易对手集中度在时序上发生剧烈变化。

验证指标:
  - Score: 0.4252
  - MI: 0.0011
  - Redundancy: 0.5247
  - Cramer's V: 0.0273
  - Delta PR-AUC: -0.011662
  - SHAP Rank: 14

可解释性报告:
  - 值含义: {'0-0.02': '交易对手集中度在前后半段基本稳定，属于正常用户行为模式，通常对应日常消费或固定收款场景。', '0.02-0.08': '存在轻微变化，可能因季节性消费或偶尔更换交易对手导致，风险较低。', '0.08-0.15': '交易对手集中度发生中等幅度变化，需关注账户是否在切换交易策略（如从分散测试转向集中交易）。', '0.15-0.25': '高变化区间，强烈暗示账户行为模式发生根本性转变，常见于先进行小额分散测试，再集中向少数目标账户转移资金的洗钱行为。', '>0.25': '极端变化，几乎可以确定为异常行为，通常对应从完全随机交易到高度集中交易的剧烈切换。'}
  - 风险阈值: 0.12
  - 协同特征: [{'feature': 'send_concentration', 'reason': '该特征衡量整体交易对手集中度，与dst_concentration_change_over_time配合可区分是长期高集中度（正常商户）还是突然变得高集中（可疑）。两者同时高值时，基本可判定为洗钱账户。'}, {'feature': 'transaction_frequency', 'reason': '高频交易且集中度剧烈变化的账户，大概率是跑分/洗钱行为。低频但集中度变化的账户可能是正常用户更换常用收款方。'}, {'feature': 'Timestamp_Hour', 'reason': '若集中度变化发生在深夜时段（0-5点），且伴随高变化值，几乎可确认为线上跑分模式。'}, {'feature': 'large_transaction_ratio', 'reason': '大额交易占比高且集中度剧烈变化的账户，符合线下跑分特征：先接收大额资金，再快速分散转出。'}]
  - 误判场景: ['个体工商户更换主要收款账户：例如商户从使用个人钱包收款转为使用对公账户，导致集中度在前后半段发生较大变化。', '季节性业务调整：如电商卖家在促销季（双11）前后，交易对手从少数批发商变为大量个人消费者，集中度会先高后低。', '账户功能升级：用户从1类钱包升级为2类钱包后，交易限额提升，可能从日常小额消费转向大额转账，导致集中度变化。', '家庭账户共用：家庭共同使用的钱包，不同家庭成员在不同时间段使用，导致交易对手集中度自然变化。']
  - 监管说明: 该特征通过比较账户前半段和后半段交易对手的集中程度变化，识别出那些先进行小额分散测试、再集中向少数账户转移资金的洗钱行为模式。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 dst_concentration_change_over_time，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按src和时间排序交易
    sorted_tx = tx_df.sort_values(['src', 'timestamp']).copy()

    # 为每个src的交易编号（按时间顺序）
    sorted_tx['tx_seq'] = sorted_tx.groupby('src').cumcount()
    sorted_tx['total_tx'] = sorted_tx.groupby('src')['tx_id'].transform('count')
    sorted_tx['half_point'] = sorted_tx['total_tx'] // 2

    # 分为前半段和后半段
    first_half = sorted_tx[sorted_tx['tx_seq'] < sorted_tx['half_point']].copy()
    second_half = sorted_tx[sorted_tx['tx_seq'] >= sorted_tx['half_point']].copy()

    # 计算前半段交易对手集中度（前3个对手的交易金额占比）
    def calc_concentration(group):
        if len(group) == 0:
            return np.nan
        # 按dst汇总金额
        dst_amounts = group.groupby('dst')['amount'].sum().sort_values(ascending=False)
        total = dst_amounts.sum()
        if total == 0:
            return np.nan
        # 前3个对手的金额占比
        top3_sum = dst_amounts.head(3).sum()
        return top3_sum / total

    first_conc = first_half.groupby('src').apply(calc_concentration).reset_index()
    first_conc.columns = ['wallet_id', 'first_half_concentration']

    second_conc = second_half.groupby('src').apply(calc_concentration).reset_index()
    second_conc.columns = ['wallet_id', 'second_half_concentration']

    # 合并并计算变化幅度
    conc_change = first_conc.merge(second_conc, on='wallet_id', how='outer')
    conc_change['dst_concentration_change_over_time'] = (conc_change['second_half_concentration'] - 
                                                           conc_change['first_half_concentration']).abs()
    conc_change['dst_concentration_change_over_time'] = conc_change['dst_concentration_change_over_time'].fillna(0)

    result = conc_change[['wallet_id', 'dst_concentration_change_over_time']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: dst_concentration_change_over_time")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 dst_concentration_change_over_time: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
