"""KnowFeat #7: counterparty_level_diversity_risk

维度: 主体关联
"""
def feature_counterparty_level_diversity_risk(open_df, tx_df, close_df):
    """
    计算每个钱包的「交易对手等级多样性风险」特征。
    
    核心指标：交易对手钱包等级多样性（归一化后取反）
    - 洗钱账户（特别是四类匿名钱包跑分）往往只与特定等级的钱包交易
    - 正常账户的交易对手等级分布更广泛
    
    数据发现：
    - 异常钱包交易对手等级多样性均值: 2.61（低）
    - 正常钱包交易对手等级多样性均值: 3.25（高）
    - 异常钱包src_wallet_level均值: 1.91（更多低等级钱包）
    
    领域知识依据：
    - 邮储专家：线下跑分使用四类匿名钱包ATM取现
    - 邮储专家：线上跑分使用匿名→实名→银行账户链
    - RULE_001：四类账户短时间大额取现异常
    - 裁判文书：使用多个数字人民币钱包接收不明资金
    """
    import pandas as pd
    import numpy as np
    
    # 计算每个src的交易对手等级多样性
    level_diversity = tx_df.groupby('src')['dst_wallet_level'].nunique().reset_index()
    level_diversity.columns = ['wallet_id', 'dst_level_diversity']
    
    # 获取src自身的wallet_level
    src_self_level = tx_df.groupby('src')['src_wallet_level'].first().reset_index()
    src_self_level.columns = ['wallet_id', 'self_wallet_level']
    
    # 合并
    result = level_diversity.merge(src_self_level, on='wallet_id', how='left')
    
    # 组合特征：
    # 1. 等级多样性低（max=4，归一化后取反，即 1 - diversity/4）
    # 2. 自身等级低（低等级钱包更可疑）
    # 组合：低多样性 × 低等级
    diversity_risk = 1.0 - result['dst_level_diversity'] / 4.0
    level_risk = 1.0 - result['self_wallet_level'] / 4.0
    
    result['counterparty_level_diversity_risk'] = diversity_risk * (1.0 + level_risk)
    
    return result[['wallet_id', 'counterparty_level_diversity_risk']]
