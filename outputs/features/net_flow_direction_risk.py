"""KnowFeat #11: net_flow_direction_risk

维度: 行为序列
"""
def feature_net_flow_direction_risk(open_df, tx_df, close_df):
    """
    计算每个钱包的「净流量方向风险」特征。
    
    核心指标：净流量比例 × (1 - 方向切换比例)
    
    洗钱账户的资金流动特征：
    1. 净流量比例高（0.87 vs 0.57）- 资金单向流动（净流出或净流入）
    2. 方向切换比例低（0.21 vs 0.33）- 出入交替少
    3. 组合后：单向流动且少切换 = 高洗钱风险
    
    领域知识依据：
    - RULE_006 [分散转入集中转出]：资金分散转入后集中转出，单向流动
    - RULE_008 [诈骗资金转移]：收到资金后迅速转出，不留余额
    - 邮储专家：线上跑分使用匿名→实名→银行账户链，单向流动
    - 裁判文书：跑分团伙使用大量钱包进行单向资金转移
    """
    import pandas as pd
    import numpy as np
    
    # 从 src 角度（出金）
    tx_src = tx_df[['src', 'timestamp', 'amount']].copy()
    tx_src.columns = ['wallet_id', 'timestamp', 'amount']
    tx_src['direction'] = -1  # 出金
    
    # 从 dst 角度（入金）
    tx_dst = tx_df[['dst', 'timestamp', 'amount']].copy()
    tx_dst.columns = ['wallet_id', 'timestamp', 'amount']
    tx_dst['direction'] = 1  # 入金
    
    # 合并并排序
    all_tx = pd.concat([tx_src, tx_dst]).sort_values(['wallet_id', 'timestamp'])
    
    def calc_flow_pattern(group):
        if len(group) < 2:
            return pd.Series({'net_flow_ratio': 0.0, 'switch_ratio': 0.0})
        
        directions = group['direction'].values
        amounts = group['amount'].abs().values * directions
        
        # 净流量比例
        total_in = amounts[amounts > 0].sum()
        total_out = abs(amounts[amounts < 0].sum())
        total = total_in + total_out
        if total == 0:
            net_flow_ratio = 0.0
        else:
            net_flow_ratio = (total_in - total_out) / total
        
        # 方向切换比例
        switches = (directions[1:] != directions[:-1]).sum()
        switch_ratio = switches / (len(directions) - 1)
        
        return pd.Series({'net_flow_ratio': net_flow_ratio, 'switch_ratio': switch_ratio})
    
    # 只取有足够交易的钱包
    wallet_counts = pd.concat([
        tx_df.groupby('src').size(),
        tx_df.groupby('dst').size()
    ]).groupby(level=0).sum()
    
    active = wallet_counts[wallet_counts >= 2].index
    all_tx_active = all_tx[all_tx['wallet_id'].isin(active)]
    
    flow_pattern = all_tx_active.groupby('wallet_id').apply(calc_flow_pattern).reset_index()
    
    # 组合特征：净流量绝对值 × (1 - 方向切换比例)
    # 当资金单向流动（净流量绝对值大）且方向切换少时，风险高
    flow_pattern['net_flow_abs'] = flow_pattern['net_flow_ratio'].abs()
    flow_pattern['net_flow_direction_risk'] = flow_pattern['net_flow_abs'] * (1.0 - flow_pattern['switch_ratio'])
    
    result = flow_pattern[['wallet_id', 'net_flow_direction_risk']].copy()
    
    return result
