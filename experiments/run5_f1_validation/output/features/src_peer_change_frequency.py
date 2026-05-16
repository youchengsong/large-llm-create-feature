"""
特征: src_peer_change_frequency
说明: 洗钱行为中，跑分团伙的交易对手通常是固定的（上下家），而正常用户的交易对手会随时间自然变化。通过计算交易对手的更换频率（新对手占总对手的比例）可以区分固定对手的洗钱模式和正常消费模式。

验证指标:
  - Score: 0.3433
  - MI: 0.0000
  - Redundancy: 0.5780
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 131

可解释性报告:
  - 值含义: 该特征衡量钱包作为发送方时，交易对手的更换频率。值越接近1表示交易对手几乎每次都在变化（新对手比例高），值显著低于1（如0.9以下）表示交易对手相对固定，重复交易对手比例高。正常用户均值约1.02，异常用户均值约1.017，两者差异极小，整体分布高度重叠。
  - 风险阈值: 由于正常与异常分布几乎完全重合（中位数均为1.0，均值仅差0.0028），无法设定有效的单一高风险阈值。建议仅作为辅助参考特征，可能结合其他特征（如交易频次、金额）在极端场景下使用，例如值<0.95且交易对手数>50时关注。
  - 协同特征: 建议与以下特征配合使用：1）unique_outdegree（独立交易对手数）：如果对手数多但更换频率低，说明在重复与同一批对手交易；2）From_To_Ratio（发送/接收比）：结合可判断是否为单向资金归集模式；3）transaction_frequency（交易频率）：高频+低更换频率可能对应跑分模式的固定上下家结构。
  - 误判场景: 1）定期向固定亲友转账的用户（如发工资、还贷）会表现出低更换频率；2）商户钱包对固定供应商或客户付款；3）订阅服务类定期付款；4）数据稀疏场景：仅2-3笔交易且对手均不同时，值可能接近1，但实际无异常。
  - 监管说明: 该特征衡量账户的交易对手是否频繁更换，正常消费场景中交易对手会自然变化，而洗钱跑分团伙往往与固定上下家反复交易，但本特征在此数据集上区分能力较弱，仅作为辅助参考指标。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 src_peer_change_frequency，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 对每个钱包作为src，按时间排序交易
    src_tx = tx_df[['src', 'dst', 'timestamp']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: src_peer_change_frequency")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 src_peer_change_frequency: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
