"""KnowFeat #10: rhythm_stability_risk

维度: 行为序列
"""
def feature_rhythm_stability_risk(open_df, tx_df, close_df):
    """
    计算每个钱包的「交易节奏稳定性风险」特征。
    
    核心指标：长间隔比例(>300s) × 节奏稳定性(1/(1+CV))
    
    洗钱账户（跑分、商户跑分）的交易节奏特征：
    1. 长间隔比例高（0.47 vs 0.25）- 交易间隔更长
    2. 间隔变异系数低（1.08 vs 2.28）- 交易节奏更规律
    3. 组合后差异放大（0.23 vs 0.095，2.4倍）
    
    领域知识依据：
    - 线上跑分：持续10-30天，每日交易量达钱包限额上限，节奏规律
    - 商户跑分：24小时不间断交易，节奏稳定
    - 邮储专家：跑分车队有组织地进行取现，交易节奏规律
    - RULE_006 [分散转入集中转出]：日交易笔数超过20笔，节奏规律
    """
    import pandas as pd
    import numpy as np
    
    # 按 src 聚合计算交易间隔统计
    src_stats = tx_df.groupby('src').agg(
        interval_mean=('interval', 'mean'),
        interval_std=('interval', 'std'),
        long_ratio=('interval', lambda x: (x > 300).mean()),
    ).reset_index()
    
    # 计算变异系数（越低越规律）
    src_stats['interval_cv'] = src_stats['interval_std'] / (src_stats['interval_mean'] + 1e-10)
    
    # 节奏稳定性 = 1/(1+CV)，值越高越规律
    src_stats['rhythm_stability'] = 1.0 / (1.0 + src_stats['interval_cv'])
    
    # 组合特征：长间隔比例 × 节奏稳定性
    # 当交易间隔长且规律时，风险分数高
    src_stats['rhythm_stability_risk'] = src_stats['long_ratio'] * src_stats['rhythm_stability']
    
    result = src_stats[['src', 'rhythm_stability_risk']].copy()
    result.columns = ['wallet_id', 'rhythm_stability_risk']
    
    return result
