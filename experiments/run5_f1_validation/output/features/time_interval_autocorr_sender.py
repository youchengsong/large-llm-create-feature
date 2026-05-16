"""
特征: time_interval_autocorr_sender
说明: 线上跑分和线下跑分模式中，交易通常有规律的时间间隔（如固定时间批量转出），导致时间间隔序列自相关高。正常消费行为的时间间隔更随机，自相关低。高自相关是异常信号。

验证指标:
  - Score: 0.2907
  - MI: 0.0000
  - Redundancy: 0.3063
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 90

可解释性报告:
  - 值含义: 该特征衡量钱包交易时间间隔的规律性，取值范围[-1,1]。值接近1表示时间间隔高度规律（如每5分钟转出一笔），值接近-1表示间隔交替变化（如长-短-长-短），值接近0表示间隔完全随机。正常消费行为通常呈现随机间隔（值接近0），异常洗钱行为（特别是线上跑分、线下跑分）往往有固定操作节奏（值显著偏离0）
  - 风险阈值: 高风险阈值建议设为|值|>0.3。依据：正常交易的标准差为0.1666，异常交易标准差为0.1545，两者差异极小（0.0121），说明该特征区分度有限。但结合领域知识，线上跑分模式要求固定时间间隔转出（如每10分钟转一笔），这类行为的自相关值应>0.5。考虑到KS=0.0326且IV=0.0096（弱预测力），建议将该特征作为辅助信号而非独立判断依据
  - 协同特征: 建议与以下特征配合使用：1）`transaction_frequency`（交易频率）：高频率+高自相关=批量自动化操作嫌疑，低频率+高自相关=定时定额转账（如工资代发，属正常）2）`is_night_hour`（是否凌晨交易）：凌晨高自相关=线上跑分典型模式，白天高自相关=可能为定期缴费或理财定投 3）`S_src_counterparty_index`（交易对手分散度）：高自相关+低对手分散度=固定对手循环转账（跑分特征），高自相关+高对手分散度=商户批量收款（可能正常）
  - 误判场景: 以下正常场景可能误判为高风险：1）定期定额理财定投（每月固定日期固定金额转入理财账户，自相关接近1）2）工资代发（企业每月固定日期批量转出，时间间隔高度规律）3）水电煤自动缴费（每月固定日期扣款）4）P2P网贷还款（借款人每月固定日期还款）5）商户定期结算（如外卖平台每2小时结算一次，时间间隔固定）
  - 监管说明: 该指标衡量账户交易是否像机器一样有规律：正常人的花钱节奏是随机的，但洗钱分子为了操作效率，往往固定时间批量转账，导致交易间隔高度规律

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 time_interval_autocorr_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按每个src钱包排序交易，取间隔
    sorted_tx = tx_df.sort_values(['src', 'timestamp']).copy()

    # 只保留有至少3笔交易的钱包（才能计算自相关）
    wallet_tx_counts = sorted_tx.groupby('src').size()
    valid_wallets = wallet_tx_counts[wallet_tx_counts >= 3].index

    # 计算每个src的时间间隔序列的自相关（lag=1）
    def calc_autocorr(intervals):
        if len(intervals) < 3:
            return 0
        # 计算lag-1自相关
        arr = intervals.values
        mean = np.mean(arr)
        if np.std(arr) == 0:
            return 0
        # 使用numpy的corrcoef
        corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        return corr if not np.isnan(corr) else 0

    # 对每个src计算间隔自相关
    filtered_tx = sorted_tx[sorted_tx['src'].isin(valid_wallets)].copy()
    autocorr_vals = filtered_tx.groupby('src')['interval'].apply(calc_autocorr).reset_index()
    autocorr_vals.columns = ['wallet_id', 'time_interval_autocorr_sender']
    autocorr_vals['time_interval_autocorr_sender'] = autocorr_vals['time_interval_autocorr_sender'].fillna(0)

    # 填充所有钱包
    all_wallets = open_df[['wallet_id']].copy()
    result = all_wallets.merge(autocorr_vals, on='wallet_id', how='left')
    result['time_interval_autocorr_sender'] = result['time_interval_autocorr_sender'].fillna(0)
    result = result[['wallet_id', 'time_interval_autocorr_sender']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: time_interval_autocorr_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 time_interval_autocorr_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
