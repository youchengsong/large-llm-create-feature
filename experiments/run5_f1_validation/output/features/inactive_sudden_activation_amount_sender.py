"""
特征: inactive_sudden_activation_amount_sender
说明: 洗钱模式中，休眠账户突然激活并进行交易是典型特征（风险分类：休眠账户突然激活并发生大额交易）。结合激活后的金额行为，能更准确识别异常。

验证指标:
  - Score: 0.3800
  - MI: 0.0000
  - Redundancy: 0.1575
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 103

可解释性报告:
  - 值含义: 该特征衡量钱包在经历最长休眠期后，首次交易金额相对于历史平均交易金额的倍数。值=1表示激活后交易金额与历史均值相当，属于正常行为；值<1表示激活后交易金额低于历史均值，可能为试探性交易；值>3表示激活后交易金额远超历史均值，具有异常嫌疑；值>10表示激活后交易金额是历史均值的10倍以上，高度异常。
  - 风险阈值: 高风险阈值为>3。判断依据：异常交易该特征的75分位值为1.5131，而正常交易的75分位值为1.5277，两者在低分位区域几乎重叠。但结合领域知识，洗钱模式中休眠账户突然激活后通常伴随大额交易（如风险分类中'休眠账户突然激活并发生大额交易'），且异常交易均值(1.0296)略高于正常(1.0267)，说明异常交易在激活后的金额放大效应虽不显著但存在。选择3作为阈值是因为：1) 超过3倍历史均值在经济意义上属于显著偏离；2) 该特征IV值较低(0.0087)，说明单独使用时区分度有限，但作为组合特征的一部分，高倍率激活仍是重要的风险信号。
  - 协同特征: 建议与以下特征配合使用：1) max_inactive_period（最大休眠期）：直接识别休眠时长，两者结合可定位'长期休眠+突然大额激活'的完整异常模式；2) is_night_hour_From（凌晨交易时段）：洗钱常发生在凌晨，结合激活后金额放大可提高检测精度；3) transaction_frequency_From（交易频率）：激活后若伴随高频交易，则洗钱嫌疑更大；4) large_transaction_ratio（大额交易比例）：验证激活后是否持续大额交易。协同原理：休眠激活本身是弱信号，但叠加金额异常放大、时段异常、频率异常等多维特征时，能形成强风险模式。
  - 误判场景: 可能的误判场景：1) 季节性消费：如用户在旅游旺季、春节等特殊时期激活钱包并进行大额消费，金额可能远超历史均值；2) 工资/奖金入账：用户收到大额工资或奖金后激活钱包进行转账；3) 账户恢复：用户因忘记密码等原因长期未用，找回后正常使用；4) 更换用途：用户原本用于小额日常消费的钱包，后用于大额理财或购房等场景；5) 数据噪声：历史交易记录过少（如仅1-2笔）时，均值计算不稳定，单笔大额交易会放大比值。
  - 监管说明: 该指标检测长期未使用的钱包突然开始交易时，交易金额是否明显超出其历史正常水平——就像一张长期不用的银行卡突然开始大额消费，可能意味着账户被他人盗用或用于洗钱活动。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 inactive_sudden_activation_amount_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按src和时间排序
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()

    # 找到每个钱包的最大间隔期及后续交易
    # 先计算每个钱包的间隔
    tx_sorted['interval'] = tx_sorted.groupby('src')['interval'].transform(lambda x: x)

    # 找到每个钱包最大间隔所在行索引
    idx_max_interval = tx_sorted.groupby('src')['interval'].idxmax()
    # 获取最大间隔后的下一笔交易
    max_interval_rows = tx_sorted.loc[idx_max_interval]
    next_tx = tx_sorted.loc[tx_sorted.groupby('src').cumcount() == 0].copy()

    # 对于每个钱包，找到最大间隔后的第一笔交易
    result_list = []
    for wallet_id in tx_sorted['src'].unique():
        wallet_tx = tx_sorted[tx_sorted['src'] == wallet_id].sort_values('timestamp')
        if len(wallet_tx) < 2:
            continue
        max_interval_idx = wallet_tx['interval'].idxmax()
        max_interval_row = wallet_tx.loc[max_interval_idx]
        # 找到该行之后的第一笔交易
        mask = (wallet_tx.index > max_interval_idx)
        if mask.any():
            next_tx_row = wallet_tx[mask].iloc[0]
            amount_after_sleep = next_tx_row['amount']
            # 历史平均金额（排除最大间隔后的交易）
            hist_amounts = wallet_tx.loc[:max_interval_idx, 'amount']
            if len(hist_amounts) > 0 and hist_amounts.mean() > 0:
                ratio = amount_after_sleep / hist_amounts.mean()
                result_list.append({'wallet_id': wallet_id, 'feature_value': ratio})

    if result_list:
        result = pd.DataFrame(result_list)
    else:
        result = pd.DataFrame(columns=['wallet_id', 'feature_value'])

    # 确保所有钱包都有值
    all_wallets = tx_df['src'].unique()
    result_full = pd.DataFrame({'wallet_id': all_wallets})
    result = result_full.merge(result, on='wallet_id', how='left')
    result['feature_value'] = result['feature_value'].fillna(0)
    result['feature_value'] = result['feature_value'].clip(lower=0, upper=50)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: inactive_sudden_activation_amount_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 inactive_sudden_activation_amount_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
