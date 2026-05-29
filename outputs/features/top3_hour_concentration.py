"""KnowFeat #3: top3_hour_concentration

维度: 时间地域
"""
def feature_top3_hour_concentration(open_df, tx_df, close_df):
    """
    计算每个钱包的「Top3小时交易集中度」特征。
    
    衡量交易在一天中最集中的3个小时的交易占比。
    洗钱账户（如跑分、线上跑分）往往在特定时段密集交易，
    导致交易时间高度集中在少数几个小时内。
    
    数据发现：
    - 异常钱包Top3小时占比均值: 0.7632
    - 正常钱包Top3小时占比均值: 0.6183
    - 差异显著（异常比正常高23%）
    
    领域知识依据：
    - 线上跑分常见凌晨12点后操作
    - 网络赌博资金深夜交易特征（RULE_015）
    - 毒品交易凌晨交易特征（RULE_011）
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    tx['hour'] = tx['hour'].astype(int)
    
    def calc_top3_hour_ratio(group):
        """计算交易最集中的3个小时的交易占比"""
        counts = group.value_counts()
        if len(counts) == 0:
            return 0.0
        return counts.head(3).sum() / len(group)
    
    # 按 src 聚合
    result = tx.groupby('src')['hour'].apply(calc_top3_hour_ratio).reset_index()
    result.columns = ['wallet_id', 'top3_hour_concentration']
    
    return result
