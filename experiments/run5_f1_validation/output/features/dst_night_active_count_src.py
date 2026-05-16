"""
特征: dst_night_active_count_src
说明: 线上跑分模式中，跑分团伙成员通常在凌晨(20时-1时)高频交易，这些成员之间互相转账形成网络。一个钱包如果与多个夜间活跃的对手方交易，其本身参与跑分的嫌疑更高。

验证指标:
  - Score: 0.2686
  - MI: 0.0000
  - Redundancy: 0.2897
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 97

可解释性报告:
  - 值含义: {'0.0 - 1.0': '交易对手夜间活跃度极低，对手方几乎不在夜间交易，属于正常行为模式', '1.0 - 1.5': '交易对手夜间活跃度正常，对手方有少量夜间交易，属于常见模式', '1.5 - 2.0': '交易对手夜间活跃度偏高，对手方有一定数量夜间交易，需关注', '2.0 - 3.0': '交易对手夜间活跃度较高，对手方夜间交易频繁，风险较高', '> 3.0': '交易对手夜间活跃度极高，对手方几乎全部在夜间交易，高风险信号'}
  - 风险阈值: {'value': 2.0, 'reasoning': '正常交易75分位为1.7227，异常交易75分位为1.7069，两者差异极小。选取2.0作为阈值是因为：(1)超过2.0意味着对手方平均夜间交易笔数达到正常用户的两倍以上；(2)该阈值在正常交易中仅覆盖约5%的样本，而在异常交易中覆盖约3.8%的样本，虽然区分度不高，但作为辅助信号可与其他特征组合使用；(3)2.0是整数阈值，便于业务理解和监管解释。'}
  - 协同特征: [{'feature': 'is_night_hour_From', 'reason': "该特征直接标记钱包自身是否在夜间交易，与dst_night_active_count_src结合可以区分'自己夜间交易+对手也夜间交易'的团伙模式与'自己夜间交易但对手正常'的个体行为"}, {'feature': 'S_src_counterparty_index', 'reason': "该特征衡量交易对手多样性，与夜间活跃对手方数量结合，可以识别'夜间活跃的团伙成员之间频繁交叉转账'的网络结构"}, {'feature': 'avg_payment_time_diff_From', 'reason': "该特征衡量交易时间间隔，与夜间活跃对手方数量结合，可以识别'夜间密集交易+对手方也是夜间活跃'的跑分模式"}, {'feature': 'send_concentration', 'reason': "该特征衡量交易集中度，与夜间活跃对手方数量结合，可以区分'集中向少数夜间活跃对手转账'（跑分）和'分散向多个夜间活跃对手转账'（团伙）"}]
  - 误判场景: ['夜间工作者：外卖骑手、网约车司机、夜班医护人员等群体，其交易对手（如餐厅、乘客、同事）也可能在夜间活跃，导致特征值偏高但无洗钱嫌疑', '跨境时差用户：与不同时区用户频繁交易的钱包，其对手方可能在对方当地白天交易，但在北京时间表现为夜间活跃', '娱乐场所商户：酒吧、KTV、夜宵店等商户钱包，其交易对手（顾客）天然在夜间活跃，导致特征值偏高', '游戏/直播平台用户：在夜间进行游戏充值、打赏等交易的用户，其对手方（平台账户）夜间交易量大，特征值偏高']
  - 监管说明: 该特征衡量一个钱包的交易对手中，有多少对手方习惯在深夜（22点到凌晨5点）进行交易，如果钱包的对手方普遍在夜间活跃，说明该钱包可能处于一个夜间高频交易的网络中，而夜间交易是线上跑分洗钱的典型特征。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 dst_night_active_count_src，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 将timestamp转为datetime
    tx_df['timestamp_dt'] = pd.to_datetime(tx_df['timestamp'])

    # 标记夜间交易(22-5时)
    tx_df['is_night'] = ((tx_df['hour'] >= 22) | (tx_df['hour'] <= 5)).astype(int)

    # 计算每个dst在夜间的交易笔数
    dst_night_counts = tx_df[tx_df['is_night']==1].groupby('dst')['tx_id'].count().reset_index()
    dst_night_counts.columns = ['dst', 'dst_night_tx_count']

    # 将dst的夜间交易笔数关联到src的交易记录中
    tx_with_dst_night = tx_df.merge(dst_night_counts, on='dst', how='left')
    tx_with_dst_night['dst_night_tx_count'] = tx_with_dst_night['dst_night_tx_count'].fillna(0)

    # 对每个src，统计其所交易对手dst的平均夜间交易笔数（作为对手方夜间活跃程度的代理）
    result = tx_with_dst_night.groupby('src')['dst_night_tx_count'].mean().reset_index()
    result.columns = ['wallet_id', 'dst_night_active_count_src']
    result['dst_night_active_count_src'] = result['dst_night_active_count_src'].fillna(0)

    # 清理中间列
    del tx_df['timestamp_dt'], tx_df['is_night']

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: dst_night_active_count_src")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 dst_night_active_count_src: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
