"""KnowFeat #2: amount_concentration_entropy

维度: 金额频次
"""
def feature_amount_concentration_entropy(open_df, tx_df, close_df):
    """
    计算每个钱包的「金额集中度」特征，捕捉洗钱账户金额模式单一的特点。
    洗钱账户（如跑分、商户跑分）往往在特定金额区间反复交易，金额多样性低。
    
    指标1: Top3金额占比 - 出现频率最高的3个金额占总交易的比例
    指标2: 金额熵 - 金额分布的多样性（越低越集中）
    
    组合方式：Top3占比 × (1 + (最大熵 - 实际熵)/最大熵)
    当Top3占比高且熵低时（金额高度集中），信号最强。
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    
    def calc_top3_ratio(x):
        """计算Top3金额占比"""
        counts = x.value_counts()
        if len(counts) == 0:
            return 0.0
        return counts.head(3).sum() / len(x)
    
    def calc_entropy(x):
        """计算金额分布的熵"""
        counts = x.value_counts()
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    # 按src聚合
    src_features = tx.groupby('src')['amount_abs'].agg([
        ('amount_top3_ratio', calc_top3_ratio),
        ('amount_entropy', calc_entropy),
        ('tx_count', 'count')
    ]).reset_index()
    
    # 归一化熵：熵值 / log2(交易数)，得到0-1之间的归一化值
    # 交易数少时熵自然低，需要校正
    max_entropy = np.log2(src_features['tx_count'].clip(lower=2))
    src_features['norm_entropy'] = src_features['amount_entropy'] / (max_entropy + 1e-10)
    
    # 组合特征：Top3占比高 且 归一化熵低 => 金额高度集中
    # 使用 (1 - norm_entropy) 将熵转为集中度指标
    src_features['amount_concentration_entropy'] = src_features['amount_top3_ratio'] * (1 - src_features['norm_entropy'].clip(0, 1))
    
    result = src_features[['src', 'amount_concentration_entropy']].copy()
    result.columns = ['wallet_id', 'amount_concentration_entropy']
    
    return result
