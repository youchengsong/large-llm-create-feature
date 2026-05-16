"""
特征: src_sleep_activation_flag
说明: 领域知识明确提到'休眠账户(>=90天)突然激活'是风险模式small_amount_testing的特征，且'账户状态'风险指标也强调'休眠账户突然激活并发生大额交易'。通过判断钱包是否存在间隔超过90天的交易记录，可以捕捉这种激活行为。

验证指标:
  - Score: 0.1063
  - MI: 0.0000
  - Redundancy: 0.4133
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 149

可解释性报告:
  - 值含义: 该特征为二值特征，0表示钱包从未出现休眠后激活行为，1表示钱包至少存在一次间隔超过90天的交易记录（即休眠账户重新活跃）。均值在异常交易中略高（0.1541 vs 0.1282），但整体区分度有限，说明绝大多数交易无论正常还是异常均未触发该标志。
  - 风险阈值: 高风险阈值为1（即特征值为1表示触发休眠激活）。判断依据：领域知识明确指出'休眠账户(>=90天)突然激活'是small_amount_testing等风险模式的核心特征，且IV=0.0056、KS=0.0259表明虽然整体区分度弱，但该特征在异常交易中占比更高，适合作为辅助信号而非独立判断。
  - 协同特征: 建议与以下特征配合使用：
1. **small_amount_testing相关特征**（如交易金额极小、凌晨时段占比）：休眠激活后若伴随小额试探性交易，可显著提高洗钱嫌疑。
2. **交易频次特征**（如transaction_frequency、1H_frequency_From）：休眠激活后突然出现高频交易，与正常用户行为模式差异更大。
3. **交易对手集中度**（如send_concentration）：休眠后若资金快速集中转出至固定对手，符合'线下跑分'模式。
协同原理：单一休眠激活信号可能被正常用户换设备/换地点的偶然行为触发，但结合金额、时间、对手等异常特征可大幅降低误报。
  - 误判场景: 可能的误判场景包括：
1. **正常用户更换手机或重新安装App**：部分用户可能因设备丢失、更换手机号等原因超过90天未使用，重新登录后首次交易即触发。
2. **季节性/低频使用者**：如仅节假日使用数字人民币的用户，可能自然间隔超过90天。
3. **钱包被遗忘后找回**：用户忘记曾注册过钱包，时隔数月重新发现并使用。
4. **跨年度结算或退款**：某些商业场景下，退款或结算周期可能跨越90天。
  - 监管说明: 该特征标记了那些超过90天未使用、突然又发起交易的钱包，这类账户突然活跃可能是洗钱团伙利用闲置账户进行非法资金转移的信号，需结合交易金额、时间、对手等信息综合判断。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 src_sleep_activation_flag，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按src和时间排序，计算交易间隔
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()

    # 计算每笔交易与其前一笔交易的时间间隔（秒）
    tx_sorted['prev_timestamp'] = tx_sorted.groupby('src')['timestamp'].shift(1)
    tx_sorted['interval_seconds'] = (pd.to_datetime(tx_sorted['timestamp']) - pd.to_datetime(tx_sorted['prev_timestamp'])).dt.total_seconds()

    # 定义休眠阈值：90天 = 90*24*3600 = 7776000秒
    sleep_threshold = 90 * 24 * 3600

    # 标记是否出现休眠后激活（间隔超过90天）
    tx_sorted['is_sleep_activation'] = (tx_sorted['interval_seconds'] > sleep_threshold).astype(int)

    # 统计每个钱包是否有休眠激活记录
    sleep_activation = tx_sorted.groupby('src')['is_sleep_activation'].max().reset_index()
    sleep_activation.columns = ['wallet_id', 'src_sleep_activation_flag']

    # 同时计算休眠激活次数（可能多次激活）
    sleep_count = tx_sorted.groupby('src')['is_sleep_activation'].sum().reset_index()
    sleep_count.columns = ['wallet_id', 'sleep_activation_count']

    # 合并，优先使用是否激活的标志
    result = sleep_activation[['wallet_id', 'src_sleep_activation_flag']].copy()
    result.columns = ['wallet_id', 'feature_value']
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: src_sleep_activation_flag")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 src_sleep_activation_flag: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
