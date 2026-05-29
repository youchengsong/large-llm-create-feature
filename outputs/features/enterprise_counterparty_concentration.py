"""KnowFeat #12: enterprise_counterparty_concentration

维度: 主体关联
"""
def feature_enterprise_counterparty_concentration(open_df, tx_df, close_df):
    """
    计算每个钱包的「企业钱包交易对手集中度」特征。
    
    核心指标：企业钱包(wallet_type=2)交易对手占比 × (1 - 交易对手类型熵归一化)
    当企业钱包交易对手占比高且交易对手类型集中（熵低）时，信号最强。
    
    数据发现：
    - 异常src企业钱包交易对手占比: 0.5427（更高）
    - 正常src企业钱包交易对手占比: 0.4328
    - 异常src交易对手类型熵: 1.0111（更低，类型更集中）
    - 正常src交易对手类型熵: 1.2509
    
    洗钱账户（特别是跑分账户）的特征：
    - 线上跑分：匿名钱包→实名钱包→银行账户的链式转账，涉及企业钱包
    - 商户跑分：与商户勾结，但企业钱包也是重要交易对手
    - 邮储专家：跑分金额300-5000元，持续10-30天
    
    领域知识依据：
    - 邮储专家[线上跑分]: 匿名钱包→实名钱包→银行账户的链式转账
    - 裁判文书: 利用数字人民币钱包和多个银行账户为上游犯罪提供资金转移通道
    - RULE_004 [商户账户异常交易]: 同一时间段内多个商户账户出现大额交易
    """
    import pandas as pd
    import numpy as np
    
    # 获取钱包类型映射
    wallet_type_map = open_df[['wallet_id', 'wallet_type']].set_index('wallet_id')['wallet_type'].to_dict()
    
    # 获取每个src的交易对手类型
    tx = tx_df[['src', 'dst']].copy()
    tx['dst_wtype'] = tx['dst'].map(wallet_type_map)
    
    # 去掉缺失值
    tx = tx.dropna(subset=['dst_wtype'])
    
    def calc_enterprise_concentration(group):
        """计算企业钱包占比和类型集中度的组合"""
        counts = group.value_counts()
        total = len(group)
        
        # 企业钱包(wtype=2)占比
        enterprise_ratio = counts.get(2, 0) / total
        
        # 类型熵
        probs = counts / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        norm_entropy = entropy / max_entropy
        
        # 组合：企业占比高 且 类型集中（熵低）
        return enterprise_ratio * (1 - norm_entropy)
    
    # 按src计算
    result = tx.groupby('src')['dst_wtype'].apply(calc_enterprise_concentration).reset_index()
    result.columns = ['wallet_id', 'enterprise_counterparty_concentration']
    
    return result
