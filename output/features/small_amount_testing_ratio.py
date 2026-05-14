"""
特征: small_amount_testing_ratio
说明: 洗钱分子在批量注册或使用新账户时，通常会先进行极小额测试交易（0.01-0.1元）来验证账户功能是否正常，这种模式在正常用户中几乎不存在。

验证指标:
  - Score: 0.2826
  - MI: 0.0004
  - Redundancy: 0.1982
  - Cramer's V: 0.0209
  - Delta PR-AUC: -0.007117
  - SHAP Rank: 29

可解释性报告:
  - 值含义: {'0.0': '钱包从未发起过极小金额（≤0.1元）测试交易，行为正常', '0.0-0.01': '极少量的极小金额交易，可能是正常用户的偶然小额测试或转账失误', '0.01-0.05': '有一定比例的极小金额交易，需警惕测试行为，建议结合其他特征判断', '0.05-0.10': '较高比例的极小金额交易，强洗钱测试嫌疑，特别是配合深夜交易时', '>0.10': '极高比例的极小金额交易，几乎可以确定为批量测试账户，强烈风险信号'}
  - 风险阈值: 0.01（即1%的测试交易占比）
  - 协同特征: [{'feature': 'activetime（活跃时间段）', 'reason': '洗钱测试交易多在深夜0-5点发生，结合activetime可区分是正常测试还是深夜可疑测试'}, {'feature': 'transaction_frequency（交易频率）', 'reason': '测试账户通常短时间内发起多笔极小金额交易，高频+小额测试是强信号'}, {'feature': 'avg_inactive_period（平均休眠期）', 'reason': '休眠账户突然激活并做测试交易是典型洗钱模式，结合可大幅提升检测精度'}, {'feature': 'unique_outdegree（唯一对手方数）', 'reason': '测试交易通常面向多个不同账户，高唯一对手方数+高测试占比表明是批量测试行为'}]
  - 误判场景: ['新注册用户首次使用钱包时，可能进行0.01元的小额转账验证账户功能，属于正常行为', '商户钱包在进行收款码功能测试时，可能产生多笔0.01-0.1元的测试交易', '系统自动对账或红包发放时，可能产生极小金额的测试记录', '用户误操作（如输入错误金额）后取消但已产生交易记录']
  - 监管说明: 该特征衡量钱包是否频繁进行极小金额（≤0.1元）的测试交易——洗钱分子在批量注册新账户后，常通过发送几分钱来测试账户是否可用，而正常用户几乎不会这样做。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 small_amount_testing_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包发起的交易中，极小金额（<=0.1元）的交易占比
    wallet_tx = tx_df[['src', 'amount']].copy()
    wallet_tx['is_small_test'] = (wallet_tx['amount'] <= 0.1).astype(int)

    # 统计每个钱包的交易总数和测试交易数
    test_stats = wallet_tx.groupby('src').agg(
        total_tx=('amount', 'count'),
        test_tx=('is_small_test', 'sum')
    ).reset_index()

    test_stats['small_amount_testing_ratio'] = test_stats['test_tx'] / test_stats['total_tx'].clip(lower=1)
    test_stats.columns = ['wallet_id', 'total_tx', 'test_tx', 'small_amount_testing_ratio']
    result = test_stats[['wallet_id', 'small_amount_testing_ratio']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: small_amount_testing_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 small_amount_testing_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
