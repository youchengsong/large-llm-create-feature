"""KnowFeat #4: fan_in_fan_out_ratio

维度: 行为序列
"""
def feature_fan_in_fan_out_ratio(open_df, tx_df, close_df):
    """
    计算每个钱包的「资金归集-分散模式」特征。
    
    核心指标：fan_in占比 / (fan_out占比 + epsilon)
    - fan_in（多对一）：多个来源向同一账户转账，资金归集模式
    - fan_out（一对多）：一个账户向多个目标转账，资金分散模式
    
    洗钱账户（特别是跑分账户）的特征：
    1. 资金归集（fan_in）：多个上游账户向跑分钱包转入资金
    2. 资金分散（fan_out）：跑分钱包将资金分散转出到多个下游账户
    3. 两者同时出现形成"分散转入→集中转出"或"集中转入→分散转出"的洗钱模式
    
    领域知识依据：
    - RULE_006：地下钱庄"分散转入集中转出"模式
    - RULE_015：网络赌博"分散转入分散转出"模式
    - 裁判文书：使用多个钱包收款以规避冻结，资金经过多层账户后转移
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    
    # 按src统计各种交易模式占比
    def calc_mode_ratios(group):
        total = len(group)
        fan_in = (group == 'fan_in').sum()
        fan_out = (group == 'fan_out').sum()
        many_to_many = (group == 'many_to_many').sum()
        one_to_many = (group == 'one_to_many').sum()
        return pd.Series({
            'fan_in_ratio': fan_in / total,
            'fan_out_ratio': fan_out / total,
            'many_to_many_ratio': many_to_many / total,
            'one_to_many_ratio': one_to_many / total,
            'total_tx': total
        })
    
    mode_stats = tx.groupby('src')['transaction_mode'].apply(calc_mode_ratios).unstack()
    
    # 组合特征1: fan_in / fan_out 比值（资金归集vs分散的平衡）
    # 当两者都高时，说明既有归集又有分散，是典型的洗钱模式
    mode_stats['fan_in_fan_out_ratio'] = mode_stats['fan_in_ratio'] / (mode_stats['fan_out_ratio'] + 1e-10)
    
    # 组合特征2: 异常模式总占比（fan_in + many_to_many + one_to_many）
    # 这些模式在正常交易中极少出现
    mode_stats['abnormal_mode_ratio'] = (mode_stats['fan_in_ratio'] + 
                                          mode_stats['many_to_many_ratio'] + 
                                          mode_stats['one_to_many_ratio'])
    
    # 最终特征：fan_in_fan_out_ratio × (1 + abnormal_mode_ratio)
    # 当fan_in占比高且fan_out占比也高（比值适中），同时异常模式总占比高时信号最强
    result_val = mode_stats['fan_in_fan_out_ratio'] * (1 + mode_stats['abnormal_mode_ratio'])
    
    result = pd.DataFrame({
        'wallet_id': mode_stats.index,
        'fan_in_fan_out_ratio': result_val.values
    })
    
    return result
