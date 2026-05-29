"""KnowFeat #1: small_test_round_amount_ratio

维度: 金额频次
"""
def feature_small_test_round_amount_ratio(open_df, tx_df, close_df):
    """
    计算每个钱包的「小额测试交易比例」与「整数金额交易比例」的组合特征。
    小额测试交易：金额 <= 5 元的交易（用于测试账户可用性）
    整数金额交易：金额为100整数倍的交易（规避监管的典型模式）
    组合方式：小额测试比例 × (1 + 整数金额比例)，放大异常信号
    """
    import pandas as pd
    import numpy as np
    
    # 计算绝对金额
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    
    # 按src（发送方）聚合
    src_features = tx.groupby('src').agg(
        tx_count=('amount', 'count'),
        small_tx_count=('amount_abs', lambda x: (x <= 5).sum()),
        round_100_count=('amount_abs', lambda x: (x % 100 == 0).sum()),
    ).reset_index()
    
    # 计算比例
    src_features['small_tx_ratio'] = src_features['small_tx_count'] / (src_features['tx_count'] + 1e-10)
    src_features['round_100_ratio'] = src_features['round_100_count'] / (src_features['tx_count'] + 1e-10)
    
    # 组合特征：小额测试比例 × (1 + 整数金额比例)
    # 当两者同时高时，信号被放大
    src_features['small_test_round_amount_ratio'] = src_features['small_tx_ratio'] * (1 + src_features['round_100_ratio'])
    
    result = src_features[['src', 'small_test_round_amount_ratio']].copy()
    result.columns = ['wallet_id', 'small_test_round_amount_ratio']
    
    return result
