"""KnowFeat #16: counterparty_type_amount_divergence

维度: 主体关联, 金额频次
"""
def feature_counterparty_type_amount_divergence(open_df, tx_df, close_df):
    """
    计算每个钱包的「交易对手类型金额背离度」特征。
    
    核心指标：向个人钱包(type=1)的金额占比 - 向商户钱包(type=3)的金额占比。
    洗钱账户倾向于向个人钱包转大额资金（跑分），而非商户钱包（正常消费），
    导致个人钱包金额占比显著高于商户钱包金额占比。
    
    数据发现：
    - 异常钱包 type1金额占比 - type3金额占比: 0.2097（高，偏向个人钱包）
    - 正常钱包 type1金额占比 - type3金额占比: 0.1344（低）
    - 差异倍数: 1.56x
    
    领域知识依据：
    - 线上跑分模式：匿名钱包→实名钱包（个人）→银行账户的链式转账
    - 邮储专家：线上跑分资金从匿名钱包转入实名钱包再转出到银行账户
    - 裁判文书：利用数字人民币钱包和多个银行账户为上游犯罪提供资金转移通道
    - 正常消费模式：个人钱包在商户消费，商户钱包收款
    """
    import pandas as pd
    import numpy as np
    
    # 获取dst的wallet_type
    wallet_type = open_df[['wallet_id', 'wallet_type']].set_index('wallet_id')
    tx = tx_df.copy()
    tx['dst_wallet_type'] = tx['dst'].map(wallet_type['wallet_type'])
    
    # 只考虑正金额（支出）
    tx_pos = tx[tx['amount'] > 0].copy()
    
    if len(tx_pos) == 0:
        result = pd.DataFrame({'wallet_id': tx_df['src'].unique(), 'counterparty_type_amount_divergence': 0.0})
        return result
    
    # 总金额
    total_amt = tx_pos.groupby('src')['amount'].sum().reset_index()
    total_amt.columns = ['wallet_id', 'total_amount']
    
    # 向个人钱包(type=1)的金额
    type1_tx = tx_pos[tx_pos['dst_wallet_type'] == 1]
    type1_amt = type1_tx.groupby('src')['amount'].sum().reset_index()
    type1_amt.columns = ['wallet_id', 'amount_to_type1']
    
    # 向商户钱包(type=3)的金额
    type3_tx = tx_pos[tx_pos['dst_wallet_type'] == 3]
    type3_amt = type3_tx.groupby('src')['amount'].sum().reset_index()
    type3_amt.columns = ['wallet_id', 'amount_to_type3']
    
    # 合并
    merged = pd.merge(total_amt, type1_amt, on='wallet_id', how='left').fillna(0)
    merged = pd.merge(merged, type3_amt, on='wallet_id', how='left').fillna(0)
    
    # 计算占比
    merged['type1_amount_ratio'] = merged['amount_to_type1'] / (merged['total_amount'] + 1e-10)
    merged['type3_amount_ratio'] = merged['amount_to_type3'] / (merged['total_amount'] + 1e-10)
    
    # 背离度：个人钱包金额占比 - 商户钱包金额占比
    # 正值越高，越偏向个人钱包（跑分特征）
    merged['counterparty_type_amount_divergence'] = merged['type1_amount_ratio'] - merged['type3_amount_ratio']
    
    result = merged[['wallet_id', 'counterparty_type_amount_divergence']].copy()
    
    return result
