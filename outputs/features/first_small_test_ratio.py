"""KnowFeat #13: first_small_test_ratio

维度: 行为序列
"""
def feature_first_small_test_ratio(open_df, tx_df, close_df):
    """
    计算每个钱包的「首笔小额测试交易比例」特征。
    
    核心指标：钱包的首笔交易是否为小额测试（<=5元）。
    洗钱账户（特别是跑分、薅羊毛）在开始大额交易前，通常会先进行小额测试交易，
    以确认账户可用。
    
    数据发现：
    - 异常钱包首笔小额比例: 41.5%
    - 正常钱包首笔小额比例: 4.7%
    - 差异倍数: 8.8x
    
    领域知识依据：
    - RULE_002 [沉睡期异常]: 账户沉睡后先进行小额测试交易，随后发生大额频繁交易
    - 邮储专家：跑分团伙通过验卡测试确认账户可用性后再进行大额转账
    - 裁判文书：上线使用卡农手机注册数字人民币钱包后进行测试交易
    """
    import pandas as pd
    import numpy as np
    
    # 按 src 和时间排序
    tx_sorted = tx_df.sort_values(['src', 'timestamp'])
    
    # 取每个 src 的第一笔交易
    first_tx = tx_sorted.groupby('src').first().reset_index()
    
    # 判断首笔是否小额测试（<=5元）
    first_tx['amount_abs'] = first_tx['amount'].abs()
    first_tx['first_small_test'] = (first_tx['amount_abs'] <= 5).astype(int)
    
    result = first_tx[['src', 'first_small_test']].copy()
    result.columns = ['wallet_id', 'first_small_test_ratio']
    
    return result
