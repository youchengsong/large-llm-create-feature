"""KnowFeat #5: mode_transition_entropy

维度: 行为序列
"""
def feature_mode_transition_entropy(open_df, tx_df, close_df):
    """
    计算每个钱包的「交易模式转换熵」特征。
    
    核心指标：交易模式分布的熵（归一化后），衡量交易模式的多样性。
    熵越低，说明交易模式越单一集中。
    
    洗钱账户的交易模式往往非常单一：
    - 线上跑分：以fan_out（分散转出）为主
    - 线下跑分（ATM取现）：以single（单笔交易）为主
    - 商户跑分：以fan_in（收款）为主
    - 薅羊毛：以fan_out（分散消费）为主
    
    数据发现：
    - 异常src mode_nunique均值: 2.19（使用更少的交易模式种类）
    - 正常src mode_nunique均值: 2.99
    - 异常src mode_entropy均值: 0.8755（熵更低，模式更集中）
    - 正常src mode_entropy均值: 1.0112
    
    领域知识依据：
    - 裁判文书：使用多个钱包收款以规避冻结（fan_in模式）
    - 邮储专家：线下跑分ATM取现（single模式）
    - 邮储专家：线上跑分钱包互转（fan_out模式）
    - 邮储专家：商户收款码跑分（fan_in模式）
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    
    def calc_normalized_entropy(group):
        """计算交易模式分布的归一化熵"""
        counts = group.value_counts()
        probs = counts / len(group)
        # 计算熵
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        # 归一化：除以最大可能熵（log2(种类数)）
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    # 按src计算归一化熵
    src_entropy = tx.groupby('src')['transaction_mode'].apply(calc_normalized_entropy).reset_index()
    src_entropy.columns = ['wallet_id', 'mode_transition_entropy']
    
    # 熵越低越异常，但为了直观，使用 (1 - 熵) 作为异常分数
    # 这样分数越高表示越异常（模式越单一）
    src_entropy['mode_transition_entropy'] = 1.0 - src_entropy['mode_transition_entropy']
    
    return src_entropy
