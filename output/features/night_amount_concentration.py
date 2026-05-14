"""
特征: night_amount_concentration
说明: 异常交易（如小额测试、线上跑分）常发生在深夜，且金额模式与正常交易不同（测试金额极小，跑分金额相对较大）。深夜交易金额的集中度能有效区分正常用户（深夜少量消费）和异常用户（深夜集中进行特定金额交易）。

验证指标:
  - Score: 0.0223
  - MI: 0.0011
  - Redundancy: 0.8960
  - Cramer's V: 0.0176
  - Delta PR-AUC: -0.020475
  - SHAP Rank: 50

可解释性报告:
  - 值含义: {'0': '该钱包没有深夜交易记录，深夜金额集中度为0，属于正常行为模式（非深夜活跃用户）', '0.0015-0.1016': '正常交易用户的深夜金额集中度区间（25%-75%分位），表示深夜交易金额占全天总金额的比例适中，符合正常消费习惯（如深夜小额消费）', '0.0139-0.0895': '异常交易用户的深夜金额集中度区间（25%-75%分位），与正常用户区间高度重叠，说明该特征在区分能力上有限', '>0.1016': '深夜交易金额占全天比例过高（超过正常用户75%分位），可能表示该用户有异常的高比例深夜交易行为，如线上跑分、小额测试等', '0.0498': '异常交易中位数，低于正常用户中位数（0.0714），说明异常用户的深夜交易金额集中度反而略低，可能因为异常用户全天都有交易，稀释了深夜占比'}
  - 风险阈值: {'primary_threshold': 0.15, 'judgment_basis': '深夜无交易（值为0）也可能是异常信号——部分异常模式（如小额测试）发生在凌晨且全天交易量极小，导致深夜占比被稀释。建议结合交易次数特征综合判断。', 'secondary_threshold': 0.0}
  - 协同特征: [{'feature': 'activetime（活跃时段）', 'reason': 'activetime直接统计交易活跃时段，与night_amount_concentration互补。若activetime显示深夜活跃且night_amount_concentration高，则深夜洗钱嫌疑更大；若activetime显示全天活跃但night_amount_concentration高，则可能是正常用户夜间加班消费。'}, {'feature': 'transaction_count_Undirected（总交易次数）', 'reason': '深夜金额集中度高但交易次数极少的用户（如只有1-2笔深夜交易），可能是正常夜间消费；而深夜金额集中度高且交易次数多的用户，更可能是线上跑分等批量操作。'}, {'feature': 'z_score_paid/z_score_received（金额Z分数）', 'reason': '深夜金额集中度高且Z分数异常的账户，说明深夜交易金额与用户历史行为不符，更可能是异常行为。例如正常用户深夜消费金额通常较小，若突然有大额深夜交易则值得关注。'}, {'feature': 'large_transaction_ratio（大额交易占比）', 'reason': '深夜金额集中度高且大额交易占比也高的账户，可能涉嫌深夜大额洗钱（如商户跑分）；而深夜金额集中度高但大额交易占比低的账户，更可能是小额测试行为。'}]
  - 误判场景: ['夜班工作者：如医护人员、工厂夜班、夜间配送等职业，其深夜消费（外卖、打车等）是正常行为，可能导致深夜金额集中度高，但无洗钱意图。', '跨境电商或外汇交易者：由于时差原因，部分用户可能在深夜进行跨境交易，导致深夜金额占比高，但属于正常业务需求。', '夜间娱乐场所从业者：如酒吧、KTV、夜店等场所的消费和转账集中在深夜，属于行业特性。', '零花钱发放场景：家长在深夜给孩子发红包或零花钱，可能造成深夜金额集中度高，但属于正常家庭行为。', '深夜自动扣款：如订阅服务、保险扣费等在凌晨执行，可能导致深夜金额占比异常，但属于正常交易。']
  - 监管说明: 该特征衡量的是用户深夜（22点至次日5点）交易金额占全天总金额的比例。虽然异常交易（如线上跑分、小额测试）确实常在深夜发生，但该特征区分度有限，因为许多正常用户（如夜班工作者、跨时区交易者）也会有较高的深夜交易比例。建议将该特征与其他特征（如交易次数、金额异常程度）配合使用，以降低误判率。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 night_amount_concentration，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 1. 提取小时信息
    tx_df = tx_df.copy()
    tx_df['hour'] = pd.to_datetime(tx_df['timestamp']).dt.hour

    # 2. 定义深夜时段（22-23点 和 0-5点）
    night_mask = (tx_df['hour'] >= 22) | (tx_df['hour'] <= 5)

    # 3. 计算每个src的深夜总金额和全天总金额
    night_amount = tx_df[night_mask].groupby('src')['amount'].sum()
    total_amount = tx_df.groupby('src')['amount'].sum()

    # 4. 计算深夜金额集中度
    concentration = night_amount / total_amount.clip(lower=0.01)
    concentration = concentration.fillna(0)

    # 5. 确保所有src都有值（没有深夜交易的设为0）
    stats = concentration.reset_index()
    stats.columns = ['wallet_id', 'feature_value']
    stats = stats.fillna(0)
    result = stats
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: night_amount_concentration")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 night_amount_concentration: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
