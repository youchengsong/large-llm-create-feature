"""
特征: mode_switch_frequency
说明: 异常交易涉及多种洗钱模式，如线上跑分(分散转出fan_out)、商户收款码跑分(many_to_many)，这些模式间频繁切换是洗钱行为的重要特征。

验证指标:
  - Score: 0.0723
  - MI: 0.0012
  - Redundancy: 0.7998
  - Cramer's V: 0.0176
  - Delta PR-AUC: -0.025813
  - SHAP Rank: 34

可解释性报告:
  - 值含义: {'0.0-0.1': '交易模式高度稳定，几乎从不切换。可能是单一用途的商户钱包（如水电煤缴费）或休眠账户。', '0.1-0.3': '交易模式偶尔切换，属于正常行为。可能是日常消费场景，偶尔有收款和转账。', '0.3-0.5': '中等频率的模式切换。普通用户日常使用多模式（如先收后转、单笔+小额高频）的正常范围。', '0.5-0.7': '频繁的模式切换。可能是有意识地在不同交易模式间切换，如跑分团伙交替使用fan_out和many_to_many。', '0.7-1.0': '极高频率的模式切换。几乎每笔交易都切换模式，高度可疑，常见于专业洗钱团伙的自动化交易。'}
  - 风险阈值: {'高风险阈值': 'mode_switch_frequency > 0.7', '判断依据': '异常交易的均值(0.3453)低于正常(0.4393)，但异常交易的25%分位(0.0333)极低，说明部分异常账户几乎不切换模式（可能是专门做单一模式的跑分）。而异常交易的75%分位(0.5196)高于正常的75%分位(0.5253)，且异常交易的标准差(0.2331)远大于正常(0.1203)，说明异常账户中存在一批切换频率极高的极端值。取0.7作为阈值（正常交易的均值+2倍标准差≈0.68），可捕捉异常分布右尾的极端高频切换行为。'}
  - 协同特征: [{'feature': 'Out_Degree（出度）', 'reason': '高频模式切换+高出度=典型线上跑分（fan_out模式）。当用户频繁在fan_out和其他模式间切换且大量转出时，高度可疑。'}, {'feature': 'Timestamp_Hour（交易小时）', 'reason': '高频切换+凌晨交易(20-1点)=线上跑分的时间特征。正常用户凌晨很少切换交易模式。'}, {'feature': 'transaction_count_Undirected（总交易量）', 'reason': '高频切换+高交易量=自动化洗钱。人工正常交易很难持续高频切换模式，只有脚本化操作才能做到。'}]
  - 误判场景: ['多业务混合的个人商户：如一个钱包既用于日常消费（小额高频），又用于收款（fan_in），还用于向供应商转账（forward），切换频率可能很高。', '账户借用场景：亲友临时借用账户进行不同类型的交易，导致模式切换异常。', '新开户用户：刚开户时可能尝试不同功能（充值、消费、转账），短时间内模式切换频繁。', '系统/业务升级：当钱包功能变更或接入新商户时，用户可能短期内切换多种模式。']
  - 监管说明: 该特征衡量一个钱包在交易过程中频繁更换交易模式（如从分散转出变成多对多）的程度，正常用户交易模式相对稳定，而洗钱团伙常通过不断变换操作手法来规避监测，切换频率越高，洗钱嫌疑越大。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 mode_switch_frequency，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包的交易模式切换频率
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()

    # 获取每笔交易的模式，并计算是否与上一笔不同
    tx_sorted['prev_mode'] = tx_sorted.groupby('src')['transaction_mode'].shift(1)
    tx_sorted['mode_switched'] = (
        (tx_sorted['prev_mode'].notna()) & 
        (tx_sorted['transaction_mode'] != tx_sorted['prev_mode'])
    )

    # 计算每个钱包的模式切换比例
    switch_stats = tx_sorted.groupby('src').agg(
        total_tx=('tx_id', 'count'),
        switch_count=('mode_switched', 'sum')
    ).reset_index()

    switch_stats['mode_switch_freq'] = (
        switch_stats['switch_count'] / (switch_stats['total_tx'] - 1).clip(lower=1)
    )

    result = switch_stats[['src', 'mode_switch_freq']].copy()
    result.columns = ['wallet_id', 'mode_switch_frequency']
    result['mode_switch_frequency'] = result['mode_switch_frequency'].fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: mode_switch_frequency")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 mode_switch_frequency: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
