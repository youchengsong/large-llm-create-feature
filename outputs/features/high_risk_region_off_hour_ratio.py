"""KnowFeat #9: high_risk_region_off_hour_ratio

维度: 时间地域
"""
def feature_high_risk_region_off_hour_ratio(open_df, tx_df, close_df):
    """
    计算每个钱包的「高风险地区非工作时间交易占比」特征。
    
    核心指标：高风险地区(region_code=540000西藏/650000新疆) × 非工作时间(18:00-7:59)交易占比
    组合方式：高风险地区标志 × 非工作时间交易占比
    
    数据发现：
    - 异常钱包高风险地区占比: 69.9%（540000和650000的异常率高达65%）
    - 异常钱包非工作时间(18-8点)交易占比均值: 0.319
    - 正常钱包非工作时间交易占比均值: 0.258
    - 组合后差异放大：异常均值0.218 vs 正常均值0.005（43倍差异）
    
    领域知识依据：
    - RULE_014 [恐怖融资特征]: 来自高风险地区的客户，交易金额较小但频繁
    - 领域知识：新疆和西藏是涉恐高风险地区
    - 异常钱包69.9%来自高风险地区（540000, 650000）
    - 洗钱账户常在非工作时间交易以规避监控
    """
    import pandas as pd
    import numpy as np
    
    # 标记高风险地区
    high_risk_regions = [540000, 650000]
    open_df = open_df.copy()
    open_df['is_high_risk_region'] = open_df['region_code'].isin(high_risk_regions).astype(int)
    
    # 标记非工作时间 (18:00-7:59)
    tx = tx_df.copy()
    tx['is_off_hour'] = ((tx['hour'] >= 18) | (tx['hour'] < 8)).astype(int)
    
    # 关联src的地域信息
    tx_with_region = tx.merge(open_df[['wallet_id', 'is_high_risk_region']], 
                               left_on='src', right_on='wallet_id', how='left')
    
    # 按src聚合
    src_stats = tx_with_region.groupby('src').agg(
        total_tx=('tx_id', 'count'),
        off_hour_tx=('is_off_hour', 'sum'),
        high_risk_flag=('is_high_risk_region', 'first')
    ).reset_index()
    
    # 计算非工作时间占比
    src_stats['off_hour_ratio'] = src_stats['off_hour_tx'] / (src_stats['total_tx'] + 1e-10)
    
    # 组合特征：高风险地区 × 非工作时间占比
    # 当钱包来自高风险地区且主要在非工作时间交易时，信号最强
    src_stats['high_risk_region_off_hour_ratio'] = src_stats['high_risk_flag'] * src_stats['off_hour_ratio']
    
    result = src_stats[['src', 'high_risk_region_off_hour_ratio']].copy()
    result.columns = ['wallet_id', 'high_risk_region_off_hour_ratio']
    
    return result
