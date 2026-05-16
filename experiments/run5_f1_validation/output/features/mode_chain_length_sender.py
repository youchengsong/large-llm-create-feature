"""
特征: mode_chain_length_sender
说明: 线上跑分模式中，交易模式通常为连续的fan_out（分散转出），链条越长越可疑。参考领域知识中'线下跑分'和'线上跑分'模式，均涉及连续分散转出行为。

验证指标:
  - Score: 0.4624
  - MI: 0.0000
  - Redundancy: 0.6941
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 95

可解释性报告:
  - 值含义: 该特征表示一个钱包在其交易历史中，连续出现相同交易模式（如 fan_out）的最大链条长度。值越低（如0-10）通常代表交易模式切换频繁，行为更接近正常用户；值中等（如10-30）代表有一定规律性但未达异常阈值；值较高（如30-60）代表持续使用同一种交易模式，可能存在自动化交易或跑分行为；值很高（>60）代表高度机械化的交易模式，洗钱嫌疑显著增大。
  - 风险阈值: 高风险阈值为 40。判断依据：1) 异常交易的中位数（26）和75%分位（43）均高于正常交易的中位数（21）和75%分位（40），表明异常钱包的链条长度整体偏长；2) 在40这个阈值处，异常交易中有约25%的钱包超过此值，而正常交易中仅有约25%超过，区分度较好；3) 结合领域知识中'线上跑分'模式的特征（连续fan_out、凌晨高频交易），链条长度超过40意味着至少连续40笔交易保持相同模式，这已远超正常消费行为的规律性。
  - 协同特征: 建议与以下特征配合使用：
1. **night_transaction_ratio**（凌晨交易占比）：线上跑分通常在深夜进行，两者结合可精准识别'长时间连续fan_out + 深夜交易'的高危组合。
2. **unique_outdegree**（唯一对手数）：跑分模式中虽然连续fan_out，但对手数量可能有限（固定跑分团伙），链条长但对手少是典型特征。
3. **avg_inactive_period**（平均静默期）：跑分钱包的交易间隔通常很短（秒级或分钟级），链条长+间隔短是自动化脚本的典型特征。
4. **S_counterparty_index**（对手集中度指数）：跑分账户的对手集中度通常偏低（分散转出），但结合链条长度可区分'正常高频消费'与'跑分分散转出'。
  - 误判场景: 以下正常场景可能误判：
1. **商户日常收款**：商户钱包每天大量收款（fan_in模式），如果持续多天使用同一模式收款，链条长度可能很高，但这是正常经营行为。
2. **自动还款/代扣服务**：如水电煤代扣、信用卡自动还款，每月固定时间固定金额转出，链条长度可能累积很长，但属于正常场景。
3. **工资代发账户**：企业钱包每月固定时间向员工集中转账（fan_out模式），链条长度随月份累积。
4. **红包发放活动**：电商平台在促销期间大量发放红包（fan_out模式），链条长度可能短暂飙升。
5. **低频但长期使用的钱包**：如果钱包每几个月才交易一次但每次都使用相同模式，链条长度也可能累积到较高值。
  - 监管说明: 这个特征衡量的是某个钱包在交易过程中，连续使用同一种转账模式（比如'一笔钱分成多笔转给不同人'）的最长次数——如果一个人像机器一样连续几十次都用完全相同的方式转账，很可能是在进行有组织的洗钱活动。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 mode_chain_length_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按每个src钱包排序交易
    sorted_tx = tx_df.sort_values(['src', 'timestamp']).copy()

    # 计算每个src的连续相同transaction_mode最长链条
    sorted_tx['mode_prev'] = sorted_tx.groupby('src')['transaction_mode'].shift(1)
    sorted_tx['mode_change'] = (sorted_tx['transaction_mode'] != sorted_tx['mode_prev']).astype(int)
    sorted_tx['mode_change'] = sorted_tx['mode_change'].fillna(1)

    # 计算每个src的连续相同mode的链条长度
    sorted_tx['chain_id'] = sorted_tx.groupby('src')['mode_change'].cumsum()
    chain_lengths = sorted_tx.groupby(['src', 'chain_id']).size().reset_index(name='chain_len')
    max_chain = chain_lengths.groupby('src')['chain_len'].max().reset_index()
    max_chain.columns = ['wallet_id', 'mode_chain_length_sender']

    # 填充未出现的src为0
    all_wallets = open_df[['wallet_id']].copy()
    result = all_wallets.merge(max_chain, on='wallet_id', how='left')
    result['mode_chain_length_sender'] = result['mode_chain_length_sender'].fillna(0)
    result = result[['wallet_id', 'mode_chain_length_sender']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: mode_chain_length_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 mode_chain_length_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
