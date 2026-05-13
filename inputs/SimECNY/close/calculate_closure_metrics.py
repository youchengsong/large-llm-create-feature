import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data():
    """加载基础数据"""
    transactions_df = pd.read_csv('/home/liberce/miniconda3/lwy/E-CNY wallet closure/wallet_transaction.csv')
    accounts_df = pd.read_csv('/home/liberce/miniconda3/lwy/E-CNY wallet closure/accounts_distribution_based.csv')
    wallet_close_df = pd.read_csv('/home/liberce/miniconda3/lwy/E-CNY wallet closure/wallet_close_data.csv')
    
    # 转换时间戳
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    accounts_df['wallet_open_timestamp'] = pd.to_datetime(accounts_df['wallet_open_timestamp'])
    wallet_close_df['zs_timestamp'] = pd.to_datetime(wallet_close_df['zs_timestamp'])
    
    return transactions_df, accounts_df, wallet_close_df

def calculate_transaction_frequency_ratio(transactions_df, wallet_id, zs_timestamp):
    """计算注销前交易频率比例 = 注销前7天交易次数 / 钱包开立后平均每周交易次数（剔除最近一周）"""
    # 获取该钱包的所有交易
    wallet_txs = transactions_df[
        (transactions_df['src'] == wallet_id) | 
        (transactions_df['dst'] == wallet_id)
    ]
    
    if wallet_txs.empty:
        return 0
    
    # 注销前7天
    closure_7days_before = zs_timestamp - timedelta(days=7)
    recent_txs = wallet_txs[wallet_txs['timestamp'] >= closure_7days_before]
    recent_count = len(recent_txs)
    
    # 历史平均每周交易次数（剔除最近一周）
    historical_txs = wallet_txs[wallet_txs['timestamp'] < closure_7days_before]
    if historical_txs.empty:
        return recent_count
    
    # 计算历史平均每周交易次数
    time_span_weeks = (closure_7days_before - historical_txs['timestamp'].min()).days / 7
    if time_span_weeks <= 0:
        return recent_count
    
    historical_weekly_avg = len(historical_txs) / time_span_weeks
    
    return recent_count / historical_weekly_avg if historical_weekly_avg > 0 else recent_count

def calculate_transaction_amount_ratio(transactions_df, wallet_id, zs_timestamp):
    """计算注销前交易金额比例 = 注销前7天交易金额 / 钱包开立后平均每周交易金额（剔除最近一周）"""
    wallet_txs = transactions_df[
        (transactions_df['src'] == wallet_id) | 
        (transactions_df['dst'] == wallet_id)
    ]
    
    if wallet_txs.empty:
        return 0
    
    closure_7days_before = zs_timestamp - timedelta(days=7)
    recent_txs = wallet_txs[wallet_txs['timestamp'] >= closure_7days_before]
    recent_amount = recent_txs['amount'].sum()
    
    historical_txs = wallet_txs[wallet_txs['timestamp'] < closure_7days_before]
    if historical_txs.empty:
        return recent_amount
    
    time_span_weeks = (closure_7days_before - historical_txs['timestamp'].min()).days / 7
    if time_span_weeks <= 0:
        return recent_amount
    
    historical_weekly_avg_amount = historical_txs['amount'].sum() / time_span_weeks
    
    return recent_amount / historical_weekly_avg_amount if historical_weekly_avg_amount > 0 else recent_amount

def calculate_counterpart_diversity_ratio(transactions_df, wallet_id, zs_timestamp):
    """计算注销前交易对手多样性比例 = 注销前7天交易对手数量 / 钱包开立后平均每周交易对手数量（剔除最近一周）"""
    wallet_txs = transactions_df[
        (transactions_df['src'] == wallet_id) | 
        (transactions_df['dst'] == wallet_id)
    ]
    
    if wallet_txs.empty:
        return 0
    
    closure_7days_before = zs_timestamp - timedelta(days=7)
    recent_txs = wallet_txs[wallet_txs['timestamp'] >= closure_7days_before]
    
    # 计算最近7天的对手数量
    src_counterparts = set(recent_txs[recent_txs['src'] == wallet_id]['dst'].unique())
    dst_counterparts = set(recent_txs[recent_txs['dst'] == wallet_id]['src'].unique())
    recent_diversity = len(src_counterparts.union(dst_counterparts))
    
    historical_txs = wallet_txs[wallet_txs['timestamp'] < closure_7days_before]
    if historical_txs.empty:
        return recent_diversity
    
    # 计算历史平均每周对手数量
    time_span_weeks = (closure_7days_before - historical_txs['timestamp'].min()).days / 7
    if time_span_weeks <= 0:
        return recent_diversity
    
    # 按周计算对手数量
    historical_diversity_weekly = []
    for week_start in pd.date_range(historical_txs['timestamp'].min(), closure_7days_before, freq='W'):
        week_end = week_start + timedelta(days=7)
        week_txs = historical_txs[
            (historical_txs['timestamp'] >= week_start) & 
            (historical_txs['timestamp'] < week_end)
        ]
        if not week_txs.empty:
            week_src = set(week_txs[week_txs['src'] == wallet_id]['dst'].unique())
            week_dst = set(week_txs[week_txs['dst'] == wallet_id]['src'].unique())
            week_diversity = len(week_src.union(week_dst))
            historical_diversity_weekly.append(week_diversity)
    
    historical_weekly_avg_diversity = np.mean(historical_diversity_weekly) if historical_diversity_weekly else 0
    
    return recent_diversity / historical_weekly_avg_diversity if historical_weekly_avg_diversity > 0 else recent_diversity

def calculate_wallet_duration_ratio(accounts_df, wallet_id, wallet_type):
    """计算钱包存续时间比例 = 钱包存续天数 / 同类型钱包平均存续天数"""
    # 获取该钱包信息
    wallet_info = accounts_df[accounts_df['wallet_id'] == wallet_id]
    if wallet_info.empty:
        return 0
    
    wallet_open_time = wallet_info['wallet_open_timestamp'].iloc[0]
    
    # 计算该钱包的存续天数（到当前时间）
    current_time = datetime.now()
    wallet_duration_days = (current_time - wallet_open_time).days
    
    # 计算同类型钱包的平均存续天数
    same_type_wallets = accounts_df[accounts_df['wallet_type'] == wallet_type]
    if same_type_wallets.empty:
        return 1.0
    
    # 计算同类型钱包的平均存续天数
    same_type_durations = []
    for _, wallet in same_type_wallets.iterrows():
        duration = (current_time - wallet['wallet_open_timestamp']).days
        same_type_durations.append(duration)
    
    avg_duration = np.mean(same_type_durations) if same_type_durations else wallet_duration_days
    
    return wallet_duration_days / avg_duration if avg_duration > 0 else 1.0

def calculate_historical_closure_count(wallet_close_df, wallet_open_tel):
    """计算历史注销钱包数量 = 同一手机号注册的钱包的注销数量"""
    return len(wallet_close_df[wallet_close_df['wallet_open_tel'] == wallet_open_tel])

def calculate_closure_metrics():
    """计算钱包注销侧场景的5个指标"""
    print("加载数据...")
    transactions_df, accounts_df, wallet_close_df = load_data()
    
    print(f"交易数据: {len(transactions_df)} 条")
    print(f"账户数据: {len(accounts_df)} 条")
    print(f"注销数据: {len(wallet_close_df)} 条")
    
    # 只处理在账户数据中存在的钱包
    print("筛选在账户数据中存在的钱包...")
    valid_wallets = wallet_close_df[wallet_close_df['wallet_id'].isin(accounts_df['wallet_id'])]
    print(f"有效钱包数量: {len(valid_wallets)}")
    
    if len(valid_wallets) == 0:
        print("错误: 没有找到任何有效的钱包！")
        return pd.DataFrame()
    
    # 计算每个钱包的指标
    results = []
    total_wallets = len(valid_wallets)
    
    print("开始计算指标...")
    for i, (_, wallet) in enumerate(valid_wallets.iterrows()):
        if i % 50 == 0:
            print(f"处理进度: {i}/{total_wallets}")
        
        wallet_id = wallet['wallet_id']  # 保持原始数据类型
        zs_timestamp = wallet['zs_timestamp']
        wallet_open_tel = str(wallet['wallet_open_tel'])
        
        # 获取钱包信息
        wallet_info = accounts_df[accounts_df['wallet_id'] == wallet_id]
        if wallet_info.empty:
            print(f"警告: 钱包 {wallet_id} 在账户数据中未找到")
            continue
        wallet_type = wallet_info['wallet_type'].iloc[0]
        
        # 1. 计算注销前交易频率比例
        freq_ratio = calculate_transaction_frequency_ratio(transactions_df, wallet_id, zs_timestamp)
        
        # 2. 计算注销前交易金额比例
        amount_ratio = calculate_transaction_amount_ratio(transactions_df, wallet_id, zs_timestamp)
        
        # 3. 计算注销前交易对手多样性比例
        diversity_ratio = calculate_counterpart_diversity_ratio(transactions_df, wallet_id, zs_timestamp)
        
        # 4. 计算钱包存续时间比例
        duration_ratio = calculate_wallet_duration_ratio(accounts_df, wallet_id, wallet_type)
        
        # 5. 计算历史注销钱包数量
        closure_count = calculate_historical_closure_count(wallet_close_df, wallet_open_tel)
        
        results.append({
            'wallet_id': wallet_id,
            'zs_id': wallet['zs_id'],
            'zs_channel': wallet['zs_channel'],
            'zs_timestamp': wallet['zs_timestamp'],
            'wallet_open_tel': wallet_open_tel,
            'is_zs_laundering': wallet['is_zs_laundering'],
            'wallet_type': wallet_type,
            'closure_freq_ratio': freq_ratio,
            'closure_amount_ratio': amount_ratio,
            'closure_diversity_ratio': diversity_ratio,
            'wallet_duration_ratio': duration_ratio,
            'historical_closure_count': closure_count
        })
    
    return pd.DataFrame(results)

def main():
    """主函数"""
    print("=== 数字人民币钱包注销侧场景指标计算 ===")
    
    # 计算指标
    result_df = calculate_closure_metrics()
    
    # 保存结果
    output_file = '/home/liberce/miniconda3/lwy/E-CNY wallet closure/wallet_close_data_1.csv'
    result_df.to_csv(output_file, index=False)
    
    print(f"\n=== 计算完成 ===")
    print(f"处理钱包数量: {len(result_df)}")
    print(f"结果已保存到: {output_file}")
    
    # 显示统计信息
    print(f"\n=== 指标统计 ===")
    print(f"注销前交易频率比例 - 均值: {result_df['closure_freq_ratio'].mean():.2f}, 中位数: {result_df['closure_freq_ratio'].median():.2f}")
    print(f"注销前交易金额比例 - 均值: {result_df['closure_amount_ratio'].mean():.2f}, 中位数: {result_df['closure_amount_ratio'].median():.2f}")
    print(f"注销前交易对手多样性比例 - 均值: {result_df['closure_diversity_ratio'].mean():.2f}, 中位数: {result_df['closure_diversity_ratio'].median():.2f}")
    print(f"钱包存续时间比例 - 均值: {result_df['wallet_duration_ratio'].mean():.2f}, 中位数: {result_df['wallet_duration_ratio'].median():.2f}")
    print(f"历史注销钱包数量 - 均值: {result_df['historical_closure_count'].mean():.2f}, 中位数: {result_df['historical_closure_count'].median():.2f}")
    
    # 显示前几行
    print(f"\n=== 数据预览 ===")
    print(result_df.head(10))

if __name__ == "__main__":
    main()
