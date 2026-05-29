"""KnowFeat #17: tiny_amount_concentration

维度: 金额频次
"""
def feature_tiny_amount_concentration(open_df, tx_df, close_df):
    """
    计算每个钱包的「极小额金额集中度」特征。
    
    核心指标：金额<=1元的交易中，最常见金额值的出现占比。
    洗钱账户（特别是跑分、薅羊毛）的典型行为：
    1. 先进行极小额测试交易（0.1元、0.4元、0.5元、1元等）
    2. 测试金额高度集中在少数几个固定值上
    3. 正常账户的极小额交易分布更分散
    
    数据发现：
    - 异常钱包 tiny_amount_concentration 均值: 0.1885
    - 正常钱包 tiny_amount_concentration 均值: 0.0487
    - 差异倍数: 3.87x
    
    领域知识依据：
    - RULE_002 [休眠账户激活]：先进行小额测试交易（<1000元），随后发生大额交易
    - 邮储专家：跑分团伙先进行小额测试确认账户可用性
    - 裁判文书：上线通过小额测试交易验证账户可用性后再进行大额转账
    - 交易模式知识：洗钱账户先进行小额测试交易（验证账户可用性）
    """
    import pandas as pd
    import numpy as np
    
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    
    def calc_tiny_concentration(group):
        """计算金额<=1元的交易中，最常见金额值的出现占比"""
        tiny = group[group <= 1]
        if len(tiny) == 0:
            return 0.0
        # 最常见的金额值的出现次数占比
        value_counts = tiny.value_counts()
        if len(value_counts) == 0:
            return 0.0
        top_count = value_counts.iloc[0]
        return top_count / len(tiny)
    
    result = tx.groupby('src')['amount_abs'].apply(calc_tiny_concentration).reset_index()
    result.columns = ['wallet_id', 'tiny_amount_concentration']
    
    return result
