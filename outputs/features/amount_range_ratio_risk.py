"""KnowFeat #8: amount_range_ratio_risk

维度: 金额频次
"""
def feature_amount_range_ratio_risk(open_df, tx_df, close_df):
    """
    计算每个钱包的「金额范围比风险」特征。
    
    核心指标：最大正金额 / 最小正金额，取 log 变换。
    衡量钱包交易的金额跨度。
    
    洗钱账户特征：
    - 同时有小额测试交易（<=5元）和大额交易（>10000元）
    - 导致金额范围比极大
    - 正常账户的金额范围相对稳定
    
    数据发现（有标签样本）：
    - 异常钱包 amount_range_ratio 均值: 10273.8
    - 正常钱包 amount_range_ratio 均值: 1691.5
    - 差异倍数: 6.07x
    
    领域知识依据：
    - RULE_002：休眠账户激活后先小额测试再大额交易
    - 裁判文书：跑分团伙使用验卡测试（小额）确认账户可用后再大额转账
    - 邮储专家：线上跑分单笔200-5000元，金额跨度大
    - 异常交易模式5（线下跑分）：休眠→突然激活，金额从0到20万
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    
    # 只考虑正金额
    tx_pos = tx[tx['amount'] > 0].copy()
    
    if len(tx_pos) == 0:
        result = pd.DataFrame({'wallet_id': tx_df['src'].unique(), 'amount_range_ratio_risk': 0.0})
        return result
    
    # 按src聚合
    range_stats = tx_pos.groupby('src').agg(
        max_amount=('amount', 'max'),
        min_positive=('amount', 'min'),
        tx_count=('amount', 'count'),
    ).reset_index()
    
    # 计算金额范围比
    range_stats['amount_range_ratio'] = range_stats['max_amount'] / (range_stats['min_positive'] + 1e-10)
    
    # log 变换处理偏态
    range_stats['amount_range_ratio_risk'] = np.log1p(range_stats['amount_range_ratio'])
    
    # 对于交易次数过少（<2）的钱包，降低分数
    range_stats.loc[range_stats['tx_count'] < 2, 'amount_range_ratio_risk'] = 0.0
    
    result = range_stats[['src', 'amount_range_ratio_risk']].copy()
    result.columns = ['wallet_id', 'amount_range_ratio_risk']
    
    return result
