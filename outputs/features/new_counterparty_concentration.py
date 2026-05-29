"""KnowFeat #6: new_counterparty_concentration

维度: 主体关联
"""
def feature_new_counterparty_concentration(open_df, tx_df, close_df):
    """
    计算每个钱包的「新交易对手集中度」特征。
    
    核心指标：新交易对手比例 / (交易对手数量 + 1)
    - 新交易对手：只出现过1次的交易对手
    - 洗钱账户特征：交易对手少（6.91 vs 13.88）+ 新交易对手比例高（0.88 vs 0.75）
    - 正常账户：交易对手多 + 新交易对手比例适中
    
    数据发现：
    - 异常钱包新交易对手比例均值: 0.878（高）
    - 正常钱包新交易对手比例均值: 0.748（较低）
    - 异常钱包交易对手数均值: 6.91（少）
    - 正常钱包交易对手数均值: 13.88（多）
    
    领域知识依据：
    - 裁判文书：使用多个数字人民币钱包接收不明资金，分散收款账户
    - 邮储专家：跑分团伙使用大量钱包进行短期交易
    - 洗钱账户往往与大量"一次性"对手交易，且对手数量有限
    """
    import pandas as pd
    import numpy as np
    
    # 计算每个src的交易对手出现次数
    pair_counts = tx_df.groupby(['src', 'dst']).size().reset_index()
    pair_counts.columns = ['src', 'dst', 'pair_count']
    
    # 每个src的交易对手总数和只出现1次的对手数
    src_stats = pair_counts.groupby('src').agg(
        total_counterparties=('dst', 'count'),
        new_counterparties=('pair_count', lambda x: (x == 1).sum())
    ).reset_index()
    
    # 计算新交易对手比例
    src_stats['new_counterparty_ratio'] = src_stats['new_counterparties'] / (src_stats['total_counterparties'] + 1e-10)
    
    # 组合特征：新交易对手比例 / 交易对手数量
    # 当新交易对手比例高且交易对手数量少时，分数最高
    src_stats['new_counterparty_concentration'] = src_stats['new_counterparty_ratio'] / (src_stats['total_counterparties'] + 1.0)
    
    result = src_stats[['src', 'new_counterparty_concentration']].copy()
    result.columns = ['wallet_id', 'new_counterparty_concentration']
    
    return result
