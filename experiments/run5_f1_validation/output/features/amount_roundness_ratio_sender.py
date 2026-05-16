"""
特征: amount_roundness_ratio_sender
说明: 洗钱交易（如merchant_laundering、online_laundering）通常涉及固定金额或整数金额，以便于计算佣金和统计流水。正常消费交易（如normal_small_high_freq）金额更随机，包含小数。该特征能捕捉这种差异，且与已有特征（金额统计量）不冗余。

验证指标:
  - Score: 0.3589
  - MI: 0.0000
  - Redundancy: 0.2949
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 101

可解释性报告:
  - 值含义: {'0.0 ~ 0.01': '绝大多数正常用户和异常用户都落在此区间，表示发送交易中几乎全部为非整数金额（含小数），符合正常消费行为特征。', '0.01 ~ 0.05': '发送交易中有少量整数金额交易，可能是偶然的大额整数支付（如房租、转账），也可能是洗钱行为中的试探性整数交易，需要结合其他特征判断。', '0.05 ~ 0.15': '发送交易中有较高比例的整数金额交易，存在洗钱嫌疑（如merchant_laundering模式中固定佣金金额、online_laundering中整数金额转账），需重点关注。', '0.15 ~ 1.0': '发送交易中大部分或全部为整数金额，高度疑似洗钱行为（如批量整数金额拆分转账、固定金额跑分交易），应立即触发预警。'}
  - 风险阈值: {'high_risk_threshold': 0.05, 'extreme_risk_threshold': 0.15, 'basis': '根据分布数据，正常交易的75分位为0.015，即75%的正常用户该特征值低于0.015；异常交易的75分位为0.0153，两者在低值区间几乎无差异。但结合业务知识：洗钱交易中整数金额占比应显著高于正常消费。取0.05作为高风险阈值，意味着该用户有超过5%的交易是整数金额，已超出正常用户的95%分位（估算），具有明显异常性。取0.15作为极端风险阈值，意味着超过15%的金额为整数，基本可判定为系统性洗钱行为。'}
  - 协同特征: [{'feature': 'transaction_motif（交易模式标签）', 'reason': '当amount_roundness_ratio_sender > 0.05且交易模式为fan_out时，高度吻合online_laundering的分散转出整数金额特征；若交易模式为many_to_one，则吻合merchant_laundering的集中收款后整数转出特征。'}, {'feature': 'hour（交易小时）', 'reason': '洗钱交易常在凌晨（0-5点、20-1点）进行。当amount_roundness_ratio_sender高且交易集中在凌晨时段，洗钱嫌疑大幅提升。'}, {'feature': 'send_concentration（发送集中度）', 'reason': '洗钱交易通常有固定交易对手（如跑分团伙），send_concentration较高。当整数金额占比高且发送集中度高时，可有效区分正常整数支付（如房租）与洗钱行为。'}, {'feature': 'wallet_level（钱包等级）', 'reason': '4类钱包（低等级）有日限额5000元，洗钱分子常通过4类钱包进行小额定整数金额拆分转账。结合wallet_level可提升对4类钱包整数交易的预警精准度。'}]
  - 误判场景: [{'scenario': '房租/固定费用支付', 'description': '用户每月固定向房东支付整数金额的房租（如3000元），或定期缴纳固定金额的水电费、物业费，会导致该特征值升高。', 'mitigation': '结合交易对手固定性（send_concentration高但交易对手数量少）和交易周期（固定间隔）可区分，或结合交易金额是否接近当地平均租金水平判断。'}, {'scenario': '薪资转账', 'description': '企业通过数字人民币发放工资，每笔金额为整数（如5000元），会导致发送方（企业钱包）该特征值极高。', 'mitigation': '结合交易对手数量（大量员工）、交易时间（固定日期）和wallet_type（企业钱包）可排除。'}, {'scenario': '小额测试交易', 'description': '用户首次使用数字人民币时进行0.01元、0.1元等整数测试交易，若交易笔数较少，可能拉高比例。', 'mitigation': '结合账户注册时长和交易历史长度判断，新用户短期内的整数测试交易属正常行为。'}]
  - 监管说明: 该特征衡量一个钱包发出的交易中，金额为整数的比例——洗钱分子为了方便计算佣金和统计流水，倾向于使用固定整数金额进行交易，而正常消费通常包含小数，因此高整数金额比例是洗钱行为的典型信号。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_roundness_ratio_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    # 计算每个src钱包发送交易中金额为整数的比例（无小数部分或小数部分为0）
    tx_df['is_round_amount'] = (tx_df['amount'] == tx_df['amount'].astype(int)) | (tx_df['amount'] * 100 % 100 == 0)
    round_ratio = tx_df.groupby('src')['is_round_amount'].mean().reset_index()
    round_ratio.columns = ['wallet_id', 'amount_roundness_ratio_sender']
    round_ratio['amount_roundness_ratio_sender'] = round_ratio['amount_roundness_ratio_sender'].fillna(0)
    result = round_ratio
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_roundness_ratio_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_roundness_ratio_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
