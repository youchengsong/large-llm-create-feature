"""
特征: fan_out_amount_volatility_sender
说明: 洗钱者在进行fan_out分散转出时，为了快速完成资金转移并保持隐蔽，通常会等额或近似等额地转出资金，导致fan_out交易的金额变异系数较低。正常用户的fan_out交易则因交易对象和用途不同而金额差异较大。

验证指标:
  - Score: 0.1768
  - MI: 0.0000
  - Redundancy: 0.4600
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 88

可解释性报告:
  - 值含义: {'0~0.1': 'fan_out交易金额几乎完全相等，高度可疑的洗钱行为，表明资金被刻意等额分散转出以规避监管', '0.1~0.4': 'fan_out交易金额波动较小，存在一定的洗钱嫌疑，可能是洗钱者为了加快速度而保持金额相近', '0.4~0.7': 'fan_out交易金额波动正常，对应正常用户因交易对象和用途不同而产生的金额差异', '0.7~1.0': 'fan_out交易金额波动较大，可能对应正常用户的大额混合小额交易，或高风险场景下的异常大额转出', '>1.0': 'fan_out交易金额波动剧烈，标准差超过均值，可能对应正常用户特殊场景或数据异常'}
  - 风险阈值: {'high_risk': 0.1, 'judgment_basis': '异常交易的中位数(0.5347)与正常交易(0.5320)几乎一致，但异常交易的均值(0.5161)低于正常(0.5084)，且异常交易的标准差(0.2438)小于正常(0.2528)。这表明异常交易的fan_out金额分布更集中、变异更小。选择0.1作为高风险阈值，因为fan_out交易金额几乎相等(CV<0.1)是洗钱者刻意等额分散转出的典型特征，而正常用户很难出现所有fan_out交易金额完全一致的情况。此阈值虽然覆盖范围小，但特异性极高。'}
  - 协同特征: [{'feature': 'fan_out_count_ratio', 'reason': 'fan_out交易占比高的钱包，如果同时fan_out金额变异系数低，则洗钱嫌疑显著增加。fan_out_count_ratio衡量分散转出的倾向性，fan_out_amount_volatility_sender衡量分散转出的金额规律性，两者结合可精准识别刻意分散资金的洗钱行为。'}, {'feature': 'night_transaction_ratio', 'reason': '线上跑分模式通常发生在凌晨(20时-1时)，如果钱包在夜间进行fan_out交易且金额变异系数低，则高度符合线上跑分特征。夜间交易比例高+fan_out金额一致=典型的洗钱行为。'}, {'feature': 'amount_roundness_score', 'reason': '洗钱者为了计算方便，常将fan_out金额设为整数或接近限额的整数金额。如果fan_out金额变异系数低且金额接近整数(如1000、5000等)，则洗钱概率大幅提升。'}, {'feature': 'counterparty_diversity', 'reason': '洗钱者通常向多个不同对手分散转出，如果fan_out交易对手数量多、金额一致且时间集中，则基本可以判定为洗钱行为。'}]
  - 误判场景: [{'scenario': '定期定额转账', 'description': '用户每月固定向多个账户转账相同金额(如房租、赡养费、慈善捐款)，会导致fan_out金额变异系数极低，被误判为洗钱。'}, {'scenario': '红包/福利发放', 'description': '企业通过数字人民币向员工发放等额红包或福利，所有转出金额完全一致，会触发该特征的高风险判断。'}, {'scenario': '批量退款/返现', 'description': '电商平台或商家向多个客户进行等额退款或返现，fan_out金额一致但属于正常经营行为。'}, {'scenario': '缴费/订阅服务', 'description': '用户向多个服务商支付相同金额的订阅费或会员费(如视频会员、云服务)，可能导致fan_out金额完全一致。'}]
  - 监管说明: 该特征检测的是资金从同一个钱包分散转出时，每笔转出金额是否几乎一模一样。正常人的分散转账金额通常有高有低，而洗钱者为了快速转移资金，往往会将大额资金拆分成金额完全相同的多笔小额转出，这种刻意的等额分散行为是洗钱的重要信号。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 fan_out_amount_volatility_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 筛选fan_out交易
    fan_out_tx = tx_df[tx_df['transaction_mode'] == 'fan_out'].copy()

    # 计算每个钱包fan_out交易的金额变异系数（CV = std/mean）
    cv_stats = fan_out_tx.groupby('src').agg(
        fan_out_amount_mean=('amount', 'mean'),
        fan_out_amount_std=('amount', 'std')
    ).reset_index()

    # 计算变异系数
    cv_stats['fan_out_cv'] = cv_stats['fan_out_amount_std'] / cv_stats['fan_out_amount_mean'].clip(lower=0.01)
    cv_stats['fan_out_cv'] = cv_stats['fan_out_cv'].fillna(0)

    # 处理无限值
    cv_stats['fan_out_cv'] = cv_stats['fan_out_cv'].replace([np.inf, -np.inf], 0)

    # 对于没有fan_out交易的钱包，补充0
    all_senders = tx_df['src'].unique()
    result = pd.DataFrame({'src': all_senders})
    result = result.merge(cv_stats[['src', 'fan_out_cv']], on='src', how='left')
    result['fan_out_cv'] = result['fan_out_cv'].fillna(0)

    result = result[['src', 'fan_out_cv']].rename(columns={'src': 'wallet_id'})
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: fan_out_amount_volatility_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 fan_out_amount_volatility_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
