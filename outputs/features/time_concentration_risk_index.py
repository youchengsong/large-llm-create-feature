"""KnowFeat #18: time_concentration_risk_index

维度: 时间地域
"""

def feature_time_concentration_risk_index(open_df, tx_df, close_df):
    """
    计算每个钱包的「时间集中度风险指数」特征。
    
    核心指标：最大小时交易占比 / 活跃小时数
    衡量交易是否集中在少数几个小时内完成。
    
    数据发现：
    - 异常钱包 time_concentration_risk 均值: 0.2710（高，交易高度集中）
    - 正常钱包 time_concentration_risk 均值: 0.0856（低，交易分散）
    - 差异倍数: 3.17x
    
    领域知识依据：
    - 线上跑分：持续10-30天，每日交易量达钱包限额上限，集中在特定时段
    - 线下跑分：跑分车队在短时间内集中取现
    - 商户跑分：24小时不间断交易，但集中在特定时段
    - RULE_006 [分散转入集中转出]：日交易笔数超过20笔，集中在短时间内
    - 裁判文书：跑分团伙在短时间内完成大量交易
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    
    # 每个src每个小时的交易数
    hour_counts = tx.groupby(['src', 'hour']).size().reset_index(name='count')
    
    # 每个src的总交易数
    src_total = tx.groupby('src').size().reset_index(name='total')
    
    # 合并计算每小时占比
    hour_pct = hour_counts.merge(src_total, on='src')
    hour_pct['pct'] = hour_pct['count'] / hour_pct['total']
    
    # 最大小时占比
    max_hour_pct = hour_pct.groupby('src')['pct'].max().reset_index()
    max_hour_pct.columns = ['src', 'max_hour_pct']
    
    # 活跃小时数
    active_hours = tx.groupby('src')['hour'].nunique().reset_index()
    active_hours.columns = ['src', 'active_hours']
    
    # 组合：最大小时占比 / 活跃小时数
    # 当交易集中在少数小时且最大小时占比高时，风险分数高
    result = max_hour_pct.merge(active_hours, on='src')
    result['time_concentration_risk_index'] = result['max_hour_pct'] / (result['active_hours'] + 1e-10)
    
    result = result[['src', 'time_concentration_risk_index']].copy()
    result.columns = ['wallet_id', 'time_concentration_risk_index']
    
    return result
