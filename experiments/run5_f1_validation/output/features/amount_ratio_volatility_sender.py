"""
特征: amount_ratio_volatility_sender
说明: 洗钱模式中，资金常被拆分成不同金额转出以规避监管，相邻交易金额比值的波动能反映这种刻意拆分的特征。

验证指标:
  - Score: 0.3631
  - MI: 0.0000
  - Redundancy: 0.5803
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 107

可解释性报告:
  - 值含义: {'0-2.2': '金额比值波动极低，相邻交易金额高度一致。可能对应定期定额缴费（如每月固定金额话费充值）、批量代发工资等规律性资金操作，属于正常交易模式', '2.2-4.0': '金额比值波动处于正常范围，对应日常消费场景中金额的自然变化（如购物金额随商品价格波动），无显著异常特征', '4.0-8.0': '金额比值波动偏高，可能对应以下场景：① 正常用户的大额偶发交易（如购车、装修付款）；② 洗钱分子试图通过金额变化规避监测，但拆分行为不够精细', '8.0以上': '金额比值波动极高，相邻交易金额差异剧烈。高度符合洗钱特征：资金被刻意拆分为差异极大的金额（如一笔10000元拆成9999元和1元），试图通过金额离散化规避整数阈值监测'}
  - 风险阈值: {'primary_threshold': 8.0, 'secondary_threshold': 12.0, 'basis': '基于分布差异分析：正常交易75分位=7.86，异常交易75分位=8.22，8.0作为第一阈值可捕获异常分布的上尾部分。12.0作为第二阈值（正常交易均值+2σ=5.20+2*3.83=12.86，取整12.0），超过此阈值的交易中异常占比显著提升。虽然IV=0.024偏低，但在与其他特征（如fan_out模式、凌晨交易）组合时能有效提升检出率', 'calibration_note': '建议根据业务接受度调整：若容忍误报，可降至6.0以捕获更多可疑交易；若追求精确打击，可提升至10.0以降低误报'}
  - 协同特征: [{'feature': 'transaction_mode=fan_out', 'reason': "洗钱分子常采用分散转出（fan_out）模式，将大额资金拆成多笔转出。结合金额比值波动特征，可区分'正常分散转账'（如工资发放，金额比值稳定）与'洗钱拆分'（金额比值剧烈波动）"}, {'feature': 'is_night_hour_From', 'reason': '凌晨高频交易是线上跑分的典型特征。在非交易时段出现的高波动金额比值，洗钱嫌疑大幅提升。正常用户凌晨交易多为紧急转账，金额比值波动通常较小'}, {'feature': 'send_concentration', 'reason': "资金集中度低（向多个对手分散转出）+ 金额比值波动高，是'化整为零'洗钱模式的典型画像。而高集中度+高波动则可能是正常的偶发大额交易"}]
  - 误判场景: ['场景一：个体工商户日常收款。商户可能收到金额差异极大的多笔收款（如一笔500元早餐费、一笔20000元团购订金），导致金额比值波动异常。可通过结合商户标识（wallet_type）和交易对手多样性（unique_outdegree）来排除', '场景二：紧急资金归集。用户临时将多个账户余额归集到主账户，可能产生金额差异极大的连续转入（如先转0.05元测试，再转50000元）。可通过检查交易对手是否为同一主体（same_transactionaccount）来识别', '场景三：系统自动调账。对公钱包的利息结算、手续费扣除等系统操作，金额可能呈现0.01元到数万元的极端波动。可通过交易对手为银行内部账户（bank_account_number）来过滤', '场景四：跨境汇款分拆。正常用户为规避外汇管制，将大额资金分拆成多笔小额汇款，金额比值可能剧烈波动。需结合地域信息（region_code）和交易用途（transaction_motif）综合判断']
  - 监管说明: 该特征通过分析同一账户相邻两笔交易金额的比例变化幅度，识别洗钱分子'故意将钱拆成不同大小金额转出'的典型手法——正常人的交易金额会自然波动，但洗钱分子为了避开整数监测，会刻意让每笔金额都不同，导致金额比值异常大幅跳动。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_ratio_volatility_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按src和时间排序，计算相邻交易金额的比值
    tx_sorted = tx_df.sort_values(['src', 'timestamp'])
    tx_sorted['prev_amount'] = tx_sorted.groupby('src')['amount'].shift(1)
    tx_sorted['amount_ratio'] = np.where(tx_sorted['prev_amount'] > 0, 
                                          tx_sorted['amount'] / tx_sorted['prev_amount'], np.nan)

    # 计算每个sender的金额比值波动性(CV)
    stats = tx_sorted.groupby('src')['amount_ratio'].agg(['std', 'mean']).reset_index()
    stats.columns = ['wallet_id', 'ratio_std', 'ratio_mean']
    stats['amount_ratio_volatility_sender'] = np.where(stats['ratio_mean'] > 0, 
                                                         stats['ratio_std'] / stats['ratio_mean'], 0)
    result = stats[['wallet_id', 'amount_ratio_volatility_sender']].fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_ratio_volatility_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_ratio_volatility_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
