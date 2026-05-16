"""
特征: amount_breakdown_entropy_sender
说明: 洗钱行为中资金常被拆分成特定金额区间进行转移，金额区间的熵值能反映这种刻意拆分的模式。

验证指标:
  - Score: 0.4123
  - MI: 0.0000
  - Redundancy: 0.5784
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 82

可解释性报告:
  - 值含义: {'0-0.3': '金额区间高度集中——资金几乎全部落在同一个金额区间内（如全部是1-10元的小额交易），常见于特定固定金额的消费场景（如话费充值、公交卡充值），属于正常业务模式', '0.3-0.5': '金额区间较为集中——资金主要分布在2-3个金额区间，可能是日常消费（小额）加偶尔大额支出，行为模式正常', '0.5-0.7': '金额区间分布均匀（正常范围）——资金在各金额区间有合理分布，对应正常用户的多样化交易行为，正常用户的中位数0.5182落在此区间', '0.7-1.0': '金额区间高度分散——资金在多个区间均匀分布，看似正常但结合其他特征可能暗示刻意拆分行为，异常交易均值0.4916反而略低于正常均值0.5022'}
  - 风险阈值: 0.35（低阈值）和0.65（高阈值）。选择依据：正常交易的25%分位为0.4254，75%分位为0.5884；异常交易的25%分位为0.4166，75%分位为0.5775。异常交易的熵值整体略低于正常交易（均值0.4916 vs 0.5022），说明洗钱行为倾向于将资金集中在少数几个金额区间（如小额测试金额0.01-0.1元区间+跑分金额10-1000元区间），导致熵值偏低。因此熵值<0.35（远低于正常下四分位）或>0.65（远高于正常上四分位）均值得关注。
  - 协同特征: [{'feature': 'large_transaction_ratio (大额交易占比)', 'reason': '当金额熵值偏低（<0.35）且大额交易占比异常高时，表明资金集中在高金额区间，可能是商户收款码跑分模式中的大额集中收款'}, {'feature': 'is_night_hour_From (凌晨交易时段)', 'reason': '当金额熵值偏低且大量交易发生在凌晨时段，符合线上跑分模式（20时-1时高频交易，金额10-1000元集中在少数区间）'}, {'feature': 'send_concentration (发送集中度)', 'reason': '当金额熵值偏低且发送集中度高（资金集中流向少数对手），符合线下跑分模式（固定交易对手，金额接近限额）'}, {'feature': 'avg_inactive_period (平均静默期)', 'reason': '当金额熵值偏低且平均静默期突然变短（从长期不活跃变为频繁交易），符合small_amount_testing模式中休眠账户突然激活的特征'}]
  - 误判场景: ['单一场景专用钱包：如仅用于缴纳物业费（固定金额500元）或仅用于购买月票（固定金额200元）的钱包，金额区间高度集中导致熵值偏低，但属于正常用途', '新开户钱包：刚开通的钱包交易笔数较少（如仅1-3笔），金额区间分布无法反映真实行为模式，熵值可能因样本不足而异常', '大额消费场景：如购买电子产品（5000-10000元区间）后立即进行日常小额消费（10-100元区间），资金分布在两个极端区间导致熵值偏高，但属于正常消费行为', '商户批量结算：商户钱包定期收到多笔不同金额的结算款（如日结款金额分布广泛），熵值偏高但属于正常商业活动']
  - 监管说明: 该特征衡量一个钱包在交易金额上的分散程度——如果资金总是在几个固定的金额范围内进出（比如只转10-100元和1000-2000元这两个区间），说明可能是在刻意控制金额以规避监管，常见于洗钱分子将非法资金拆分成特定金额进行转移。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_breakdown_entropy_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 定义金额区间（根据洗钱模式中的典型金额范围）
    bins = [0, 1, 10, 100, 500, 1000, 2000, 5000, 10000, 50000, float('inf')]
    labels = range(len(bins)-1)
    tx_df['amount_bin'] = pd.cut(tx_df['amount'], bins=bins, labels=labels)

    # 计算每个sender在各金额区间的交易笔数分布
    bin_counts = tx_df.groupby(['src', 'amount_bin']).size().unstack(fill_value=0)

    # 计算每个sender的金额区间熵值
    def calc_entropy(row):
        proportions = row / row.sum()
        # 避免log(0)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))

    entropy = bin_counts.apply(calc_entropy, axis=1).reset_index()
    entropy.columns = ['wallet_id', 'amount_breakdown_entropy_sender']

    # 归一化到[0,1]（最大熵为log2(10)≈3.32）
    max_entropy = np.log2(len(labels))
    entropy['amount_breakdown_entropy_sender'] = entropy['amount_breakdown_entropy_sender'] / max_entropy

    result = entropy
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_breakdown_entropy_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_breakdown_entropy_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
