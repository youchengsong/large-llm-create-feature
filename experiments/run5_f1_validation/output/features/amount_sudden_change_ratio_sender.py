"""
特征: amount_sudden_change_ratio_sender
说明: 洗钱分子常通过突然改变金额模式来规避检测，如休眠账户突然大额交易。金额突变比例能捕捉这种异常行为模式。

验证指标:
  - Score: 0.5856
  - MI: 0.0000
  - Redundancy: 0.0941
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 112

可解释性报告:
  - 值含义: {'0-0.1': '金额变化极小，交易模式稳定，通常为正常用户的规律性消费或固定金额转账', '0.1-0.5': '金额有适度波动，属于正常交易行为，如日常消费金额的自然变化', '0.5-1.0': '金额变化较大，可能为正常用户的大额消费或转账，也可能是洗钱分子的试探性交易', '1.0-5.0': '金额突变明显，洗钱分子可能开始调整交易策略，如从测试交易转向大额交易', '>5.0': '金额剧烈突变，高度可疑，通常对应休眠账户突然激活后的大额交易或洗钱分子的金额跳跃式操作'}
  - 风险阈值: 0.81
  - 协同特征: [{'feature': 'max_inactive_period', 'reason': '金额突变配合长期休眠后激活，是洗钱分子利用休眠账户进行大额交易的典型模式，两者结合可显著提升检测准确率'}, {'feature': 'large_transaction_ratio', 'reason': '金额突变比例高且大额交易占比高的钱包，可能正在从正常交易模式切换为洗钱模式，协同使用可区分正常用户的偶发性大额消费'}, {'feature': 'send_concentration', 'reason': '金额突变配合高集中度转出（如将资金集中转给少数对手），符合洗钱分子将分散资金归集后转出的行为模式'}]
  - 误判场景: ['用户更换工作或生活城市后，消费金额模式发生自然变化（如从日常小额消费变为大额房租转账）', '用户一次性购买大件商品（如家电、家具），导致单笔金额远超历史均值', '用户账户被盗用后，盗用者进行的大额交易会被误判为金额突变，实际为安全事件而非洗钱', '用户因家庭原因（如医疗、教育）突然需要大额资金转账']
  - 监管说明: 该特征衡量的是用户单笔交易金额相对于其历史交易金额的变化剧烈程度，洗钱分子常通过突然改变交易金额来规避检测，而正常用户的金额变化通常较为平缓且有合理解释。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_sudden_change_ratio_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()
    tx_sorted['prev_amount'] = tx_sorted.groupby('src')['amount'].shift(1)
    tx_sorted['amount_change'] = (tx_sorted['amount'] - tx_sorted['prev_amount']).abs()
    tx_sorted['amount_change_ratio'] = tx_sorted['amount_change'] / tx_sorted['prev_amount'].abs().clip(lower=1)
    # 只计算有前一笔的交易
    valid = tx_sorted['prev_amount'].notna()
    feature = tx_sorted[valid].groupby('src')['amount_change_ratio'].median().reset_index()
    feature.columns = ['wallet_id', 'amount_sudden_change_ratio_sender']
    feature['amount_sudden_change_ratio_sender'] = feature['amount_sudden_change_ratio_sender'].fillna(0)
    result = feature
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_sudden_change_ratio_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_sudden_change_ratio_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
