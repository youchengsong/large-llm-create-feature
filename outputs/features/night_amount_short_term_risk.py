"""KnowFeat #14: night_amount_short_term_risk

维度: 时间地域
"""
def feature_night_amount_short_term_risk(open_df, tx_df, close_df):
    """
    计算每个钱包的「深夜交易金额短期风险」特征。
    
    核心指标：深夜交易金额占比 × 时间跨度风险(1/(天数+1))
    
    洗钱账户特征：
    1. 深夜交易金额占比高（异常0.118 vs 正常0.053，差异2.2倍）
    2. 时间跨度短（异常151.8天 vs 正常220.4天，差异0.69倍）
    3. 组合后差异放大至6.16倍
    
    领域知识依据：
    - 线上跑分：常见凌晨12点后操作，持续10-30天
    - 商户跑分：24小时不间断交易
    - RULE_015 [网络赌博]：深夜进行网银交易
    - 邮储专家：跑分车队凌晨操作，钱包存活时间短
    - 裁判文书：洗钱活动在夜间或非工作时间进行
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    
    # 深夜时段：22:00 - 5:59
    tx['is_deep_night'] = (tx['hour'] >= 22) | (tx['hour'] < 6)
    tx['amount_abs'] = tx['amount'].abs()
    
    # 深夜交易金额
    deep_night_amt = tx[tx['is_deep_night']].groupby('src')['amount_abs'].sum().reset_index()
    deep_night_amt.columns = ['src', 'deep_night_amount']
    
    # 总交易金额
    total_amt = tx.groupby('src')['amount_abs'].sum().reset_index()
    total_amt.columns = ['src', 'total_amount']
    
    # 深夜金额占比
    amt_ratio = total_amt.merge(deep_night_amt, on='src', how='left').fillna(0)
    amt_ratio['deep_night_amt_ratio'] = amt_ratio['deep_night_amount'] / (amt_ratio['total_amount'] + 1e-10)
    
    # 交易时间跨度（首笔到末笔的天数）
    time_span = tx.groupby('src')['timestamp'].agg(['min', 'max']).reset_index()
    time_span['time_span_days'] = (time_span['max'] - time_span['min']).dt.days
    
    # 组合特征
    result = amt_ratio.merge(time_span[['src', 'time_span_days']], on='src')
    result['night_amount_short_term_risk'] = result['deep_night_amt_ratio'] * (1.0 / (result['time_span_days'] + 1.0))
    
    result = result[['src', 'night_amount_short_term_risk']].copy()
    result.columns = ['wallet_id', 'night_amount_short_term_risk']
    
    return result
