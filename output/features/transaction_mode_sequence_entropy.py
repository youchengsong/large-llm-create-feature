"""
特征: transaction_mode_sequence_entropy
说明: 异常交易模式在fan_out、fan_in、many_to_many之间频繁切换，而正常交易集中在single、forward等少数模式。熵值越高说明交易模式越多样化，符合线上跑分和商户收款码跑分中多模式混合使用的特征。

验证指标:
  - Score: 0.2835
  - MI: 0.0013
  - Redundancy: 0.6057
  - Cramer's V: 0.0236
  - Delta PR-AUC: -0.001051
  - SHAP Rank: 20

可解释性报告:
  - 值含义: {'0.0-0.3': '交易模式高度单一，通常为单一交易模式（如全部是single_transaction或全部是fan_out），常见于正常用户的规律性交易（如定期缴费）或休眠账户的少量交易', '0.3-0.6': '交易模式较为集中，以1-2种模式为主，正常用户和部分异常用户（如线上跑分用户使用fan_out+fan_in组合）均可能出现', '0.6-0.8': '交易模式多样化，正常用户的高频日常消费（混合small_high_freq和single_transaction）和商户收款码跑分（混合many_to_many和forward）都可能达到此区间', '0.8-1.0': '交易模式极度多样化，几乎均匀覆盖所有6种模式，强烈提示异常行为（如线上跑分团伙在多个模式间快速切换以规避监测）'}
  - 风险阈值: {'high_risk': '值 < 0.3 或 值 > 0.85', 'judgment_basis': {'low_end': "正常用户中75%分位为0.6947，而异常用户25%分位为0.2742。值低于0.3意味着交易模式高度单一，结合领域知识中'small_amount_testing'模式（金额0.01-0.1元，深夜交易，休眠账户突然激活）的特征——这类异常交易通常只使用fan_in或fan_out单一模式，导致熵值极低", 'high_end': "正常用户均值0.5800，异常用户均值0.4872，但异常用户的标准差(0.2621)远大于正常用户(0.1495)，说明异常用户中既有极低熵也有极高熵的极端值。值高于0.85说明钱包几乎均匀使用所有交易模式，这违背了正常用户的行为习惯（正常用户通常有固定模式偏好），符合'线上跑分'和'商户收款码跑分'中多模式混合使用的特征"}}
  - 协同特征: [{'feature': 'Timestamp_Hour', 'reason': '低熵值(<0.3) + 深夜交易(0-5点)可精准识别small_amount_testing模式；高熵值(>0.85) + 凌晨交易(20-1点)可识别线上跑分', 'expected_effect': '将Delta PR-AUC从-0.001提升至0.01以上'}, {'feature': 'From_To_Ratio', 'reason': '低熵值 + 极高的From_To_Ratio(远大于1)可识别fan_out为主的洗钱模式；低熵值 + 极低的From_To_Ratio可识别fan_in为主的收款洗钱', 'expected_effect': "弥补熵值无法区分'单一模式是哪种模式'的缺陷"}, {'feature': 'avg_inactive_period', 'reason': '低熵值 + 长休眠期(>90天)可识别休眠账户突然激活的small_amount_testing模式，这是数字人民币洗钱模式3（线上跑分）的典型特征', 'expected_effect': '显著降低误报率，因为正常用户的低熵通常伴随高频交易'}]
  - 误判场景: [{'scenario': '新注册的测试钱包', 'explanation': '刚开通的钱包可能只有1-2笔测试交易（如充值和提现），交易模式单一导致熵值极低（接近0），但这是正常的新用户行为', 'mitigation': "结合'wallet_open_date'特征，过滤掉开户时间<3天的钱包"}, {'scenario': '自动缴费类钱包', 'explanation': "用于定期自动缴纳水电燃气费的钱包，交易模式固定为'forward'或'single'，长期保持低熵值，但这是正常行为", 'mitigation': "结合'amount'特征，如果金额稳定在固定值（如每月的电费金额）则排除"}, {'scenario': '高频小额消费钱包', 'explanation': '日常消费频繁的用户（如使用数字人民币购买早餐、咖啡），交易模式可能在normal_small_high_freq和single_transaction之间切换，熵值可能达到0.7-0.8，与高风险阈值接近', 'mitigation': "结合'amount'特征，如果金额均小于100元且交易时间集中在白天(7-19点)则排除"}]
  - 监管说明: 该特征衡量一个钱包在交易时是否频繁切换不同的转账方式（如'集中转出'、'集中转入'、'一对一'等），正常用户通常只用1-2种固定方式，而洗钱团伙为了躲避监测会刻意混合使用多种方式，导致该指标异常偏高或偏低。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 transaction_mode_sequence_entropy，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按src分组，获取每个钱包的交易模式序列
    mode_seq = tx_df.groupby('src')['transaction_mode'].apply(list).reset_index()

    # 计算每个钱包的交易模式熵
    def calc_entropy(mode_list):
        if len(mode_list) == 0:
            return 0.0
        # 计算每种模式的频率
        total = len(mode_list)
        unique_modes, counts = np.unique(mode_list, return_counts=True)
        probs = counts / total
        # 计算熵
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        # 归一化到[0,1]区间，除以log2(类别数)
        n_categories = 6  # transaction_mode有6种
        normalized_entropy = entropy / np.log2(n_categories) if n_categories > 1 else 0.0
        return normalized_entropy

    mode_seq['feature_value'] = mode_seq['transaction_mode'].apply(calc_entropy)
    mode_seq = mode_seq[['src', 'feature_value']]
    mode_seq.columns = ['wallet_id', 'feature_value']
    result = mode_seq
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: transaction_mode_sequence_entropy")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 transaction_mode_sequence_entropy: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
