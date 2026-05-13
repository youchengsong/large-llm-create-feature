import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

def load_data():
    """加载基础数据"""
    accounts_df = pd.read_csv('/home/lwy/DIAN-DATA-Simulate-test-0114/transaction/accounts.csv')
    transactions_df = pd.read_csv('/home/lwy/DIAN-DATA-Simulate-test-0114/transaction/wallet_temporal_transactions_1105_1450.csv')
    # 转换时间戳
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    return accounts_df, transactions_df

def load_params():
    """加载参数配置"""
    with open('wallet_close_params.json', 'r') as f:
        return json.load(f)

def get_wallet_last_transaction_time(transactions_df, wallet_id):
    """获取钱包最后一次交易时间"""
    wallet_txs = transactions_df[
        (transactions_df['src'] == wallet_id)| 
        (transactions_df['dst'] == wallet_id)]
    if len(wallet_txs) > 0:
        return wallet_txs['timestamp'].max()
    return None

def calculate_wallet_features(transactions_df, wallet_id, accounts_df):
    """计算钱包特征用于风险判断"""
    wallet_txs = transactions_df[
        (transactions_df['src'] == wallet_id)| 
        (transactions_df['dst'] == wallet_id)]
    
    if len(wallet_txs) == 0:
        return {
            'transaction_frequency': 0,
            'counterpart_diversity': 0,
            'wallet_duration': 0,
            'phone_history': 0,
            'risk_transaction_count': 0,
            'risk_transaction_ratio': 0.0,
            'participated_in_risk': False
        }
    
    # 计算交易频率（按天）
    time_span = (wallet_txs['timestamp'].max() - wallet_txs['timestamp'].min()).days + 1
    frequency = len(wallet_txs) / time_span if time_span > 0 else 0
    
    # 计算交易对手多样性
    src_counterparts = set(wallet_txs[wallet_txs['src'] == wallet_id]['dst'].unique())
    dst_counterparts = set(wallet_txs[wallet_txs['dst'] == wallet_id]['src'].unique())
    all_counterparts = src_counterparts.union(dst_counterparts)
    counterpart_diversity = len(all_counterparts)
    
    # 钱包存续时间
    duration_days = time_span
    
    # 统计异常交易情况（is_risk=1）
    # 确保is_risk列可以被正确识别
    if 'is_risk' in wallet_txs.columns:
        # 转换为字符串并清理，然后筛选
        wallet_txs['is_risk_clean'] = wallet_txs['is_risk'].astype(str).str.strip()
        risk_txs = wallet_txs[wallet_txs['is_risk_clean'].isin(['1', 'True', 'true', 'TRUE', 'Yes', 'yes', 'YES'])]
        risk_transaction_count = len(risk_txs)
        risk_transaction_ratio = risk_transaction_count / len(wallet_txs) if len(wallet_txs) > 0 else 0.0
        participated_in_risk = risk_transaction_count > 0
    else:
        risk_transaction_count = 0
        risk_transaction_ratio = 0.0
        participated_in_risk = False
    
    return {
        'transaction_frequency': frequency,
        'counterpart_diversity': counterpart_diversity,
        'wallet_duration': duration_days,
        'phone_history': 0,  # 将在主函数中计算
        'risk_transaction_count': risk_transaction_count,
        'risk_transaction_ratio': risk_transaction_ratio,
        'participated_in_risk': participated_in_risk
    }

def calculate_phone_history(accounts_df, phone_number):
    """计算手机号的历史开户数量"""
    phone_accounts = accounts_df[accounts_df['wallet_open_tel'] == phone_number]
    return len(phone_accounts)

def calculate_risk_score(features):
    """计算风险分数（0-1之间）"""
    risk_score = 0
    
    # ⭐ 最重要：同时满足 is_abnormal=1 且参与异常交易的评分（最高优先级）
    is_abnormal = features.get('is_abnormal', False)
    participated_in_risk = features.get('participated_in_risk', False)
    
    if is_abnormal and participated_in_risk:
        # 同时满足两个条件，给予最高风险分数加成
        risk_transaction_ratio = features.get('risk_transaction_ratio', 0.0)
        risk_transaction_count = features.get('risk_transaction_count', 0)
        
        # 基础分（因为同时满足两个条件）
        risk_score += 0.7
        
        # 根据异常交易占比额外加分
        if risk_transaction_ratio > 0.5:  # 异常交易占比超过50%
            risk_score += 0.2
        elif risk_transaction_ratio > 0.2:  # 异常交易占比20-50%
            risk_score += 0.15
        elif risk_transaction_ratio > 0.1:  # 异常交易占比10-20%
            risk_score += 0.1
        
        # 如果参与多笔异常交易，额外加分
        if risk_transaction_count >= 5:
            risk_score += 0.15
        elif risk_transaction_count >= 3:
            risk_score += 0.1
        elif risk_transaction_count >= 1:
            risk_score += 0.05
    
    # 第二优先级：只参与异常交易（但没有is_abnormal=1）
    elif participated_in_risk:
        risk_transaction_ratio = features.get('risk_transaction_ratio', 0.0)
        risk_transaction_count = features.get('risk_transaction_count', 0)
        
        # 如果参与了异常交易，给予显著的风险分数加成
        if risk_transaction_ratio > 0.5:  # 异常交易占比超过50%
            risk_score += 0.5
        elif risk_transaction_ratio > 0.2:  # 异常交易占比20-50%
            risk_score += 0.4
        elif risk_transaction_ratio > 0.1:  # 异常交易占比10-20%
            risk_score += 0.3
        else:  # 异常交易占比低于10%
            risk_score += 0.2
        
        # 如果参与多笔异常交易，额外加分
        if risk_transaction_count >= 5:
            risk_score += 0.2
        elif risk_transaction_count >= 3:
            risk_score += 0.15
        elif risk_transaction_count >= 1:
            risk_score += 0.1
    
    # 第三优先级：只是 is_abnormal=1（但没有参与异常交易）
    elif is_abnormal:
        # 开立端标记为异常，给予中等风险分数加成
        risk_score += 0.3
    
    # 其他特征评分（降低权重，确保异常交易特征占主导）
    # 交易频率评分
    freq = features['transaction_frequency']
    if freq > 1:  # 每天超过1笔交易
        risk_score += 0.08
    elif freq > 0.3:
        risk_score += 0.04
    
    # 交易对手多样性评分
    diversity = features['counterpart_diversity']
    if diversity > 5:  # 超过5个不同对手
        risk_score += 0.08
    elif diversity > 2:
        risk_score += 0.04
    
    # 钱包存续时间评分
    duration = features['wallet_duration']
    if duration < 90:  # 少于90天
        risk_score += 0.05
    elif duration < 180:
        risk_score += 0.02
    
    # 手机号历史开户数量评分
    phone_history = features['phone_history']
    if phone_history > 3:  # 超过3个账户
        risk_score += 0.05
    elif phone_history > 1:
        risk_score += 0.02
    
    # 随机因素（进一步降低随机性，让异常交易特征更明显）
    risk_score += random.uniform(0, 0.1)
    
    # 确保分数在0-1之间
    return min(1.0, max(0.0, risk_score))


def generate_zs_timestamp(last_tx_time, wallet_open_time, transactions_df, wallet_id, params,
                          is_risk_wallet: bool = False):
    """生成注销时间
    参数:
    - last_tx_time: 最后一次交易时间（如果钱包有交易记录）
    - wallet_open_time: 账户开立时间（作为备选基准）
    - transactions_df: 交易数据（用于获取全局最晚交易时间）
    - wallet_id: 钱包ID（用于二次验证最后交易时间）
    - params: 参数配置
    返回值:
    - zs_time: 注销时间（保证晚于最后交易时间）
    """
    # ⭐ 关键保证：优先使用最后交易时间，确保注销时间晚于最后交易时间
    if last_tx_time is not None:
        base_time = last_tx_time
    else:
        # 如果没有获取到最后交易时间，再次尝试查找（防止遗漏）
        if wallet_id is not None and transactions_df is not None and len(transactions_df) > 0:
            wallet_txs = transactions_df[
                (transactions_df['src'].astype(str) == str(wallet_id)) | 
                (transactions_df['dst'].astype(str) == str(wallet_id))
            ]
            if len(wallet_txs) > 0:
                base_time = wallet_txs['timestamp'].max()
                last_tx_time = base_time  # 更新最后交易时间
            elif wallet_open_time is not None:
                # 如果确实没有交易记录，使用账户开立时间作为基准
                base_time = wallet_open_time
            else:
                # 如果既没有交易记录也没有开立时间，使用交易数据的最晚时间作为基准
                base_time = transactions_df['timestamp'].max()
        elif wallet_open_time is not None:
            # 如果钱包没有交易记录，使用账户开立时间作为基准
            base_time = wallet_open_time
        else:
            # 如果既没有交易记录也没有开立时间，使用交易数据的最晚时间作为基准
            if len(transactions_df) > 0:
                base_time = transactions_df['timestamp'].max()
            else:
                # 最后的备选方案：使用当前时间（但应该避免这种情况）
                base_time = datetime.now()
    
    # 使用相同的基础时间偏移范围
    min_hours = params['time_offsets']['min_hours_after_last_tx']
    max_hours = params['time_offsets']['max_hours_after_last_tx']
    
    # 对异常注销钱包（is_risk_wallet=True）加入“快速注销”机制：
    # 大约 60% 的异常注销钱包会在最后一笔交易后的 1-2 天内注销
    if is_risk_wallet and last_tx_time is not None:
        fast_close_prob = 0.6
        if random.random() < fast_close_prob:
            # 快速注销：1-2 天
            fast_min_hours = 24
            fast_max_hours = 48
            offset_hours = random.randint(fast_min_hours, fast_max_hours)
        else:
            # 其余仍按照全局配置的时间窗口
            offset_hours = random.randint(min_hours, max_hours)
    else:
        # 普通钱包或没有交易记录的钱包，使用统一时间窗口
        offset_hours = random.randint(min_hours, max_hours)
    
    # 随机生成注销时间（相对于 base_time）
    zs_time = base_time + timedelta(hours=offset_hours)
    
    # ⭐ 最终验证：如果找到了最后交易时间，确保注销时间晚于它
    if last_tx_time is not None and zs_time <= last_tx_time:
        # 如果生成的注销时间早于或等于最后交易时间，强制设置为最后交易时间之后
        zs_time = last_tx_time + timedelta(hours=min_hours)
    
    return zs_time

def generate_wallet_close_data():
    """生成钱包注销数据"""
    # 加载数据和参数
    accounts_df, transactions_df = load_data()
    params = load_params()
    
    # 随机选择要注销的钱包
    total_wallets = params['simulation_params']['total_wallets']
    risk_ratio = params['simulation_params']['risk_wallet_ratio']
    selected_wallets = accounts_df.sample(n=min(total_wallets, len(accounts_df)))
    
    wallet_close_data = []
    zs_id = 1
    
    # 计算需要多少个风险钱包
    num_risk_wallets = int(total_wallets * risk_ratio)
    num_normal_wallets = total_wallets - num_risk_wallets
    
    print(f"Target: {num_risk_wallets} risk wallets ({risk_ratio*100:.1f}%), {num_normal_wallets} normal wallets")
    
    # 首先识别参与异常交易的钱包
    print("\nIdentifying wallets that participated in risk transactions...")
    if 'is_risk' in transactions_df.columns:
        transactions_df['is_risk_clean'] = transactions_df['is_risk'].astype(str).str.strip()
        risk_txs = transactions_df[transactions_df['is_risk_clean'].isin(['1', 'True', 'true', 'TRUE', 'Yes', 'yes', 'YES'])]
        risk_src_wallets = set(risk_txs['src'].astype(str).unique())
        risk_dst_wallets = set(risk_txs['dst'].astype(str).unique())
        all_risk_wallets = risk_src_wallets.union(risk_dst_wallets)
        print(f"Found {len(all_risk_wallets)} wallets that participated in risk transactions")
    else:
        all_risk_wallets = set()
        print("Warning: 'is_risk' column not found in transactions")
    
    # 为每个钱包计算风险分数
    wallet_scores = []
    for _, wallet in selected_wallets.iterrows():
        wallet_id = str(wallet['wallet_id'])
        wallet_open_tel = str(wallet['wallet_open_tel'])
        
        # 检查开立端是否为异常账户（is_abnormal=True）
        is_abnormal = wallet.get('is_abnormal', False)
        if pd.isna(is_abnormal):
            is_abnormal = False
        elif isinstance(is_abnormal, str):
            is_abnormal = is_abnormal.lower() in ['true', '1', 'yes']
        else:
            is_abnormal = bool(is_abnormal)
        
        # 计算钱包特征
        features = calculate_wallet_features(transactions_df, wallet_id, accounts_df)
        
        # 计算手机号历史开户数量
        phone_history = calculate_phone_history(accounts_df, wallet_open_tel)
        features['phone_history'] = phone_history
        
        # 标记是否参与异常交易
        participated_in_risk = wallet_id in all_risk_wallets
        
        # 将is_abnormal和participated_in_risk添加到features中，用于风险分数计算
        features['is_abnormal'] = is_abnormal
        features['participated_in_risk'] = participated_in_risk
        
        # 计算风险分数（0-1之间）
        risk_score = calculate_risk_score(features)
        
        wallet_scores.append({
            'wallet_id': wallet_id,
            'wallet_open_tel': wallet_open_tel,
            'features': features,
            'risk_score': risk_score,
            'participated_in_risk': participated_in_risk,  # 标记是否参与异常交易
            'is_abnormal': is_abnormal  # 标记开立端是否为异常账户
        })
    
    # ⭐ 优先选择：同时满足 is_abnormal=1 且参与异常交易的钱包作为风险钱包
    # 分类钱包：1. 既是is_abnormal=1又参与异常交易（最高优先级）
    #           2. 只参与异常交易
    #           3. 只是is_abnormal=1
    #           4. 其他
    wallets_abnormal_and_risk = [w for w in wallet_scores if w['is_abnormal'] and w['participated_in_risk']]
    wallets_only_risk_tx = [w for w in wallet_scores if not w['is_abnormal'] and w['participated_in_risk']]
    wallets_only_abnormal = [w for w in wallet_scores if w['is_abnormal'] and not w['participated_in_risk']]
    wallets_other = [w for w in wallet_scores if not w['is_abnormal'] and not w['participated_in_risk']]
    
    print(f"\nSelected wallets breakdown:")
    print(f"  Wallets that are is_abnormal=1 AND participated in risk transactions: {len(wallets_abnormal_and_risk)}")
    print(f"  Wallets that only participated in risk transactions: {len(wallets_only_risk_tx)}")
    print(f"  Wallets that only are is_abnormal=1: {len(wallets_only_abnormal)}")
    print(f"  Other wallets: {len(wallets_other)}")
    
    # 按风险分数排序（优先级从高到低）
    wallets_abnormal_and_risk.sort(key=lambda x: x['risk_score'], reverse=True)
    wallets_only_risk_tx.sort(key=lambda x: x['risk_score'], reverse=True)
    wallets_only_abnormal.sort(key=lambda x: x['risk_score'], reverse=True)
    wallets_other.sort(key=lambda x: x['risk_score'], reverse=True)
    
    # 选择风险钱包：按优先级顺序选择
    risk_wallets_selected = []
    
    # 初始化计数器
    count_from_abnormal_and_risk = 0
    count_from_only_risk = 0
    count_from_only_abnormal = 0
    remaining_risk_needed = num_risk_wallets
    
    # 第一优先级：同时满足 is_abnormal=1 且参与异常交易的钱包
    count_from_abnormal_and_risk = min(remaining_risk_needed, len(wallets_abnormal_and_risk))
    if count_from_abnormal_and_risk > 0:
        risk_wallets_selected.extend(wallets_abnormal_and_risk[:count_from_abnormal_and_risk])
        for w in risk_wallets_selected:
            w['is_risk'] = 1
        remaining_risk_needed -= count_from_abnormal_and_risk
        print(f"  Selected {count_from_abnormal_and_risk} risk wallets from abnormal+risk_tx wallets (highest priority)")
    
    # 第二优先级：只参与异常交易的钱包
    if remaining_risk_needed > 0 and len(wallets_only_risk_tx) > 0:
        count_from_only_risk = min(remaining_risk_needed, len(wallets_only_risk_tx))
        additional_risk_wallets = wallets_only_risk_tx[:count_from_only_risk]
        risk_wallets_selected.extend(additional_risk_wallets)
        for w in additional_risk_wallets:
            w['is_risk'] = 1
        remaining_risk_needed -= count_from_only_risk
        print(f"  Selected {count_from_only_risk} risk wallets from wallets that only participated in risk transactions")
    
    # 第三优先级：只是 is_abnormal=1 的钱包
    if remaining_risk_needed > 0 and len(wallets_only_abnormal) > 0:
        count_from_only_abnormal = min(remaining_risk_needed, len(wallets_only_abnormal))
        additional_risk_wallets = wallets_only_abnormal[:count_from_only_abnormal]
        risk_wallets_selected.extend(additional_risk_wallets)
        for w in additional_risk_wallets:
            w['is_risk'] = 1
        remaining_risk_needed -= count_from_only_abnormal
        print(f"  Selected {count_from_only_abnormal} risk wallets from wallets that only are is_abnormal=1")
    
    # 第四优先级：其他钱包（按风险分数选择）
    if remaining_risk_needed > 0 and len(wallets_other) >= remaining_risk_needed:
        additional_risk_wallets = wallets_other[:remaining_risk_needed]
        risk_wallets_selected.extend(additional_risk_wallets)
        for w in additional_risk_wallets:
            w['is_risk'] = 1
        print(f"  Selected {remaining_risk_needed} additional risk wallets from other wallets")
        remaining_risk_needed = 0
    
    # 标记剩余钱包为正常
    # 计算每个类别中剩余的钱包数量
    remaining_abnormal_and_risk = wallets_abnormal_and_risk[count_from_abnormal_and_risk:]
    remaining_only_risk = wallets_only_risk_tx[count_from_only_risk:]
    remaining_only_abnormal = wallets_only_abnormal[count_from_only_abnormal:]
    remaining_other = wallets_other[remaining_risk_needed:] if remaining_risk_needed > 0 else wallets_other
    
    remaining_wallets = remaining_abnormal_and_risk + remaining_only_risk + remaining_only_abnormal + remaining_other
    for w in remaining_wallets:
        w['is_risk'] = 0
    
    # 合并所有钱包并打乱顺序
    wallet_scores = risk_wallets_selected + remaining_wallets
    random.shuffle(wallet_scores)
    
    print(f"\nFinal risk wallet selection:")
    risk_wallets_final = [w for w in wallet_scores if w.get('is_risk') == 1]
    risk_wallets_with_risk_tx = [w for w in risk_wallets_final if w.get('participated_in_risk', False)]
    risk_wallets_abnormal = [w for w in risk_wallets_final if w.get('is_abnormal', False)]
    risk_wallets_abnormal_and_risk = [w for w in risk_wallets_final if w.get('is_abnormal', False) and w.get('participated_in_risk', False)]
    
    print(f"  Total risk wallets: {len(risk_wallets_final)}")
    print(f"  Risk wallets that participated in risk transactions: {len(risk_wallets_with_risk_tx)} ({len(risk_wallets_with_risk_tx)/len(risk_wallets_final)*100:.1f}%)")
    print(f"  Risk wallets that are is_abnormal=1: {len(risk_wallets_abnormal)} ({len(risk_wallets_abnormal)/len(risk_wallets_final)*100:.1f}%)")
    print(f"  Risk wallets that are is_abnormal=1 AND participated in risk transactions: {len(risk_wallets_abnormal_and_risk)} ({len(risk_wallets_abnormal_and_risk)/len(risk_wallets_final)*100:.1f}%)")
    
    for wallet_info in wallet_scores:
        wallet_id = wallet_info['wallet_id']
        wallet_open_tel = wallet_info['wallet_open_tel']
        features = wallet_info['features']
        is_zs_laundering = wallet_info['is_risk']
        
        # 获取最后一次交易时间
        last_tx_time = get_wallet_last_transaction_time(transactions_df, wallet_id)
        
        # 获取账户开立时间（作为备选基准）
        wallet_account = accounts_df[accounts_df['wallet_id'].astype(str) == wallet_id]
        wallet_open_time = None
        if len(wallet_account) > 0:
            if 'wallet_open_timestamp' in wallet_account.columns:
                try:
                    wallet_open_time = pd.to_datetime(wallet_account.iloc[0]['wallet_open_timestamp'])
                except:
                    pass
        
        # 生成注销时间（确保在最后一次交易之后）
        zs_timestamp = generate_zs_timestamp(
            last_tx_time,
            wallet_open_time,
            transactions_df,
            wallet_id,
            params['simulation_params'],
            is_risk_wallet=bool(is_zs_laundering)
        )
        
        # 选择注销渠道
        channel_weights = list(params['simulation_params']['zs_channel_distribution'].values())
        zs_channel = np.random.choice(
            list(params['simulation_params']['zs_channel_distribution'].keys()),
            p=channel_weights
        )
        
        # 添加到结果
        wallet_close_data.append({
            'wallet_id': wallet_id,
            'zs_id': f"ZS{zs_id:06d}",
            'zs_channel': zs_channel,
            'zs_timestamp': zs_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'wallet_open_tel': wallet_open_tel,
            'is_zs_laundering': is_zs_laundering
        })
        
        zs_id += 1
    
    return pd.DataFrame(wallet_close_data)

def main():
    """主函数"""
    print("Generating wallet close data...")
    
    # 加载数据
    accounts_df, transactions_df = load_data()
    
    # 生成数据
    wallet_close_df = generate_wallet_close_data()
    
    # 按注销时间排序并保存数据
    wallet_close_df['zs_timestamp'] = pd.to_datetime(wallet_close_df['zs_timestamp'])
    wallet_close_df = wallet_close_df.sort_values('zs_timestamp')
    wallet_close_df['zs_timestamp'] = wallet_close_df['zs_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    output_file = 'new_close_3.csv'
    wallet_close_df.to_csv(output_file, index=False)
    
    print(f"Generated {len(wallet_close_df)} wallet close records")
    print(f"Risk wallets: {wallet_close_df['is_zs_laundering'].sum()}")
    print(f"Normal wallets: {(wallet_close_df['is_zs_laundering'] == 0).sum()}")
    print(f"Data saved to: {output_file}")
    
    # 分析参与异常交易的钱包被标记为异常注销的情况
    print("\n=== 参与异常交易的钱包注销分析 ===")
    # 使用已加载的交易数据进行分析
    transactions_df_for_analysis = transactions_df.copy()
    
    # 识别参与异常交易的钱包
    if 'is_risk' in transactions_df_for_analysis.columns:
        transactions_df_for_analysis['is_risk_clean'] = transactions_df_for_analysis['is_risk'].astype(str).str.strip()
        risk_txs = transactions_df_for_analysis[transactions_df_for_analysis['is_risk_clean'].isin(['1', 'True', 'true', 'TRUE', 'Yes', 'yes', 'YES'])]
        
        # 找出参与异常交易的钱包（作为src或dst）
        risk_src_wallets = set(risk_txs['src'].astype(str).unique())
        risk_dst_wallets = set(risk_txs['dst'].astype(str).unique())
        all_risk_wallets = risk_src_wallets.union(risk_dst_wallets)
        
        print(f"参与异常交易的钱包总数: {len(all_risk_wallets)}")
        
        # 检查这些钱包有多少被标记为异常注销
        wallet_close_df['wallet_id_str'] = wallet_close_df['wallet_id'].astype(str)
        risk_wallets_in_close = wallet_close_df[wallet_close_df['wallet_id_str'].isin(all_risk_wallets)]
        
        if len(risk_wallets_in_close) > 0:
            risk_marked_as_zs_laundering = risk_wallets_in_close[risk_wallets_in_close['is_zs_laundering'] == 1]
            print(f"参与异常交易且在注销数据中的钱包: {len(risk_wallets_in_close)}")
            print(f"其中被标记为异常注销(is_zs_laundering=1): {len(risk_marked_as_zs_laundering)}")
            if len(risk_wallets_in_close) > 0:
                print(f"⭐ 参与异常交易钱包的异常注销比例: {len(risk_marked_as_zs_laundering)/len(risk_wallets_in_close):.2%}")
            
            # 对比所有注销钱包的异常注销比例
            all_zs_laundering = wallet_close_df[wallet_close_df['is_zs_laundering'] == 1]
            print(f"\n对比:")
            print(f"  所有注销钱包中异常注销比例: {len(all_zs_laundering)/len(wallet_close_df):.2%}")
            print(f"  参与异常交易钱包的异常注销比例: {len(risk_marked_as_zs_laundering)/len(risk_wallets_in_close):.2%}")
            if len(risk_marked_as_zs_laundering)/len(risk_wallets_in_close) > len(all_zs_laundering)/len(wallet_close_df):
                print(f"  ✅ 参与异常交易的钱包更倾向于被标记为异常注销")
        else:
            print("  警告：没有参与异常交易的钱包在注销数据中")
    else:
        print("  警告：交易数据中没有找到is_risk列")
    
    # 显示统计信息
    print("\n=== Channel Distribution ===")
    print(wallet_close_df['zs_channel'].value_counts())
    
    print("\n=== Risk Distribution ===")
    print(wallet_close_df['is_zs_laundering'].value_counts())
    
    # 显示手机号历史统计
    print("\n=== Phone History Statistics ===")
    phone_history_stats = accounts_df['wallet_open_tel'].value_counts()
    print(f"Phone numbers with 1 account: {(phone_history_stats == 1).sum()}")
    print(f"Phone numbers with 2-3 accounts: {((phone_history_stats >= 2) & (phone_history_stats <= 3)).sum()}")
    print(f"Phone numbers with 4-5 accounts: {((phone_history_stats >= 4) & (phone_history_stats <= 5)).sum()}")
    print(f"Phone numbers with 6+ accounts: {(phone_history_stats > 5).sum()}")
    print(f"Max accounts per phone: {phone_history_stats.max()}")
    
    # 显示前几行
    print("\n=== Sample Data ===")
    print(wallet_close_df.head(10))

if __name__ == "__main__":
    main()
