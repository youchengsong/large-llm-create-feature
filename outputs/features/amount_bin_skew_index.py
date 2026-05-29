"""KnowFeat #15: amount_bin_skew_index

维度: 金额频次
"""
def feature_amount_bin_skew_index(open_df, tx_df, close_df):
    """
    计算每个钱包的「金额区间偏态指数」特征。
    
    核心指标：极小额(<=1元)占比 / (中等金额(1000-50000)占比 + 大额(>=50000)占比 + 0.01)
    
    洗钱账户（跑分、薅羊毛）的金额分布呈"L型"偏态：
    1. 极小额(<=1元)占比极高（异常0.158 vs 正常0.009，差异18.49倍）
    2. 中等金额(1000-50000)占比低（异常0.562 vs 正常0.654，差异0.86倍）
    3. 大额(>=50000)占比极低（异常0.011 vs 正常0.053，差异0.20倍）
    
    组合后差异放大至15倍以上，且与已有特征（如amount_concentration_entropy、
    small_test_round_amount_ratio）角度不同，它衡量的是金额分布的"偏态"而非"集中度"。
    
    领域知识依据：
    - RULE_002 [休眠账户激活]：先进行小额测试交易（<1000元）
    - 邮储专家：跑分团伙先进行小额测试确认账户可用性
    - 裁判文书：使用多个钱包接收不明资金，分散收款账户
    - 交易模式知识：洗钱账户先进行小额测试交易，然后进行大额转移
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    
    # 极小额(<=1元)占比
    src_tiny = tx.groupby('src')['amount_abs'].apply(lambda x: (x <= 1).mean()).reset_index()
    src_tiny.columns = ['src', 'ultra_tiny_ratio']
    
    # 中等金额(1000-50000)占比
    src_mid = tx.groupby('src')['amount_abs'].apply(lambda x: ((x >= 1000) & (x <= 50000)).mean()).reset_index()
    src_mid.columns = ['src', 'mid_amount_ratio']
    
    # 大额(>=50000)占比
    src_large = tx.groupby('src')['amount_abs'].apply(lambda x: (x >= 50000).mean()).reset_index()
    src_large.columns = ['src', 'large_amount_ratio']
    
    # 组合：极小额占比 / (中等金额占比 + 大额占比 + 0.01)
    result = src_tiny.merge(src_mid, on='src').merge(src_large, on='src')
    result['amount_bin_skew_index'] = result['ultra_tiny_ratio'] / (result['mid_amount_ratio'] + result['large_amount_ratio'] + 0.01)
    
    result = result[['src', 'amount_bin_skew_index']].copy()
    result.columns = ['wallet_id', 'amount_bin_skew_index']
    
    return result
