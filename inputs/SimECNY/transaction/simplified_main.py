#!/usr/bin/env python3
"""
简化的统一交易模式主程序
基于TransactionManager和UnifiedTransactionGenerator生成交易
支持真实的交易时间分布
"""

import json
import csv
import numpy as np
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict
import time
import os
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from transaction_manager import TransactionManager
from unified_transaction_generator4 import UnifiedTransactionGenerator

def get_process_info() -> str:
    """获取进程信息（内存使用等）"""
    info = []
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        info.append(f"内存: {mem_mb:.1f} MB")
    return ", ".join(info) if info else ""

def load_data(edges_file: str,  accounts_file: str) -> Tuple:
    """加载所有必要的数据"""
    # 加载图数据
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始加载图数据... {get_process_info()}")
    edges_start = time.time()
    g = nx.DiGraph()
    with open(edges_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            if len(row) >= 2:
                g.add_edge(row[0], row[1])
    edges_elapsed = time.time() - edges_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 图数据加载完成: {g.number_of_nodes()} 个节点, {g.number_of_edges()} 条边, 耗时 {edges_elapsed:.2f} 秒 {get_process_info()}")
    
    # 加载账户数据
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始加载账户数据... {get_process_info()}")
    accounts_start = time.time()
    balances = {}
    wallet_to_attrs = {}
    wallet_open_timestamps = {}  # 新增：存储钱包开立时间
    abnormal_wallets = set()  # 新增：存储异常账户集合
    attr_headers = []
    with open(accounts_file, 'r') as f:
        reader = csv.reader(f)
        attr_headers = next(reader)
        for row in reader:
            if len(row) >= 14:  # 确保有足够的列（包含银行和地区信息）
                wallet_id = row[0]
                # init_balance 在第12列（索引11）
                balance = float(row[11]) if row[11] else 0.0
                balances[wallet_id] = balance
                wallet_to_attrs[wallet_id] = row[1:]  # 除了wallet_id之外的所有属性
                
                # 解析钱包开立时间（第7列，索引6）
                wallet_open_timestamp_str = row[6] if len(row) > 6 else None
                if wallet_open_timestamp_str:
                    try:
                        wallet_open_timestamps[wallet_id] = datetime.strptime(
                            wallet_open_timestamp_str, "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        # 如果解析失败，使用默认时间
                        wallet_open_timestamps[wallet_id] = datetime(2024, 1, 1, 0, 0, 0)
                else:
                    wallet_open_timestamps[wallet_id] = datetime(2024, 1, 1, 0, 0, 0)
                
                # 检查is_abnormal字段（第14列，索引13，需要根据实际CSV列位置调整）
                # 查找is_abnormal列的索引
                try:
                    is_abnormal_idx = attr_headers.index('is_abnormal')
                    if len(row) > is_abnormal_idx:
                        is_abnormal_value = row[is_abnormal_idx].strip()
                        # 判断是否为异常账户（支持 '1', 'True', 'true', 'TRUE' 等）
                        if is_abnormal_value in ['1', 'True', 'true', 'TRUE', 'Yes', 'yes', 'YES']:
                            abnormal_wallets.add(wallet_id)
                except ValueError:
                    # 如果找不到is_abnormal列，尝试通过常见位置查找（通常是倒数第二列）
                    if len(row) >= 14:
                        is_abnormal_value = row[13].strip()  # 第14列（索引13）
                        if is_abnormal_value in ['1', 'True', 'true', 'TRUE', 'Yes', 'yes', 'YES']:
                            abnormal_wallets.add(wallet_id)
    
    accounts_elapsed = time.time() - accounts_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 账户数据加载完成: {len(balances)} 个账户, {len(abnormal_wallets)} 个异常账户, 耗时 {accounts_elapsed:.2f} 秒 {get_process_info()}")
    
    return g, balances, wallet_to_attrs, attr_headers, wallet_open_timestamps, abnormal_wallets

def generate_daily_transaction_schedule(config: Dict, rng) -> List[datetime]:
    """基于配置生成每日交易时间表"""
    daily_config = config.get("daily_transaction_config", {})
    base_transactions = daily_config.get("base_transactions_per_day", 500)
    variance = daily_config.get("transaction_variance", 0.3)
    hourly_dist = daily_config.get("hourly_distribution", {})
    
    # 计算实际每日交易数（添加随机变化）
    actual_transactions = int(base_transactions * (1 + rng.normal(0, variance)))
    actual_transactions = max(1, actual_transactions)  # 至少1笔交易
    
    # 生成交易时间
    transaction_times = []
    
    for _ in range(actual_transactions):
        # 随机选择小时段
        hour_weights = []
        hour_ranges = []
        
        for hour_range, weight in hourly_dist.items():
            hour_weights.append(weight)
            # 解析时间范围，例如 "09:00-10:00" -> (9, 10)
            start_str, end_str = hour_range.split('-')
            start_hour = int(start_str.split(':')[0])
            end_hour = int(end_str.split(':')[0])
            hour_ranges.append((start_hour, end_hour))
        
        # 归一化权重
        total_weight = sum(hour_weights)
        if total_weight > 0:
            hour_weights = [w / total_weight for w in hour_weights]
        
        # 根据权重选择小时段
        selected_hour_range = rng.choice(len(hour_ranges), p=hour_weights)
        start_hour, end_hour = hour_ranges[selected_hour_range]
        
        # 在该小时段内随机选择时间
        if start_hour == end_hour:
            hour = start_hour
        else:
            hour = rng.integers(start_hour, end_hour)
        minute = rng.integers(0, 60)
        second = rng.integers(0, 60)
        
        # 创建时间（使用当前日期，后续会调整到具体日期）
        transaction_time = datetime(2024, 1, 1, hour, minute, second)
        transaction_times.append(transaction_time)
    
    # 按时间排序
    transaction_times.sort()
    return transaction_times

def fix_merchant_laundering_many_to_many(transactions: List[Dict], merchant_wallet: str, 
                                         base_time: datetime, rng, 
                                         wallet_open_timestamps: Dict[str, datetime] = None) -> List[Dict]:
    """
    修复 merchant_laundering 的所有模式（many_to_many, one_to_many, many_to_one）：
    1. 确保转入交易时间早于转出交易时间
    2. 确保转入总额 = 转出总额 / (1 - 佣金率)，其中佣金率在 0.8%-1.8% 之间
    """
    if not transactions:
        return transactions
    
    # 检查是否是 merchant_laundering 支持的模式
    first_tx = transactions[0]
    transaction_mode = first_tx.get("transaction_mode")
    supported_modes = ["many_to_many", "one_to_many", "many_to_one"]
    
    if transaction_mode not in supported_modes:
        return transactions
    
    # 识别商户账户（主钱包）
    # 对于 many_to_many 和 many_to_one：商户账户是 dst 出现次数最多的账户
    # 对于 one_to_many：商户账户是 src（source_wallet）
    if transaction_mode == "one_to_many":
        main_wallet = merchant_wallet  # one_to_many 中，source_wallet 就是商户账户
    else:
        # many_to_many 或 many_to_one：商户账户是 dst 出现次数最多的账户
        dst_counts = {}
        for tx in transactions:
            dst = tx.get("dst")
            if dst:
                dst_counts[dst] = dst_counts.get(dst, 0) + 1
        
        if dst_counts:
            main_wallet = max(dst_counts, key=dst_counts.get)
        else:
            main_wallet = merchant_wallet
    
    # 分离转入和转出交易
    incoming_txs = []  # 转入商户账户的交易 (dst == main_wallet)
    outgoing_txs = []   # 从商户账户转出的交易 (src == main_wallet)
    
    for tx in transactions:
        if tx.get("dst") == main_wallet:
            incoming_txs.append(tx)
        elif tx.get("src") == main_wallet:
            outgoing_txs.append(tx)
    
    # 如果没有转入或转出交易，返回原交易列表
    if not incoming_txs or not outgoing_txs:
        return transactions
    
    # 计算当前转入和转出总额
    total_incoming = sum(float(tx.get("amount", 0)) for tx in incoming_txs)
    total_outgoing = sum(float(tx.get("amount", 0)) for tx in outgoing_txs)
    
    # 如果转入或转出总额为0，无法修复，返回原交易列表
    if total_incoming <= 0 or total_outgoing <= 0:
        return transactions
    
    # 确定佣金率（0.8%-1.8%之间）
    commission_rate = rng.uniform(0.008, 0.018)
    
    # 智能金额调整策略：
    # 如果转入总额 < 转出总额，说明转入不足，应该根据转出总额计算目标转入总额
    # 如果转入总额 >= 转出总额，说明转入充足，应该根据转入总额计算目标转出总额
    # 目标关系：转出总额 = 转入总额 * (1 - 佣金率)
    if total_incoming < total_outgoing:
        # 情况1：转入总额 < 转出总额，根据转出总额计算目标转入总额
        target_incoming_total = total_outgoing / (1 - commission_rate)
        scale_factor = target_incoming_total / total_incoming
        # 调整转入交易的金额（增加）
        for tx in incoming_txs:
            original_amount = float(tx.get("amount", 0))
            new_amount = original_amount * scale_factor
            tx["amount"] = round(new_amount, 2)
    else:
        # 情况2：转入总额 >= 转出总额，根据转入总额计算目标转出总额
        target_outgoing_total = total_incoming * (1 - commission_rate)
        scale_factor = target_outgoing_total / total_outgoing
        # 调整转出交易的金额（减少）
        for tx in outgoing_txs:
            original_amount = float(tx.get("amount", 0))
            new_amount = original_amount * scale_factor
            tx["amount"] = round(new_amount, 2)
    
    # 重新分配时间戳：转入交易在前，转出交易在后
    # 转入交易从 base_time 开始，每笔间隔 1-5 分钟
    time_offset = 0
    for tx in incoming_txs:
        tx_time = base_time + timedelta(seconds=int(time_offset))
        tx["timestamp"] = tx_time.strftime("%Y-%m-%d %H:%M:%S")
        time_offset += int(rng.integers(60, 300))  # 1-5 分钟间隔
    
    # 转出交易在转入交易之后，每笔间隔 1-5 分钟
    # 确保转出交易的时间晚于所有转入交易
    base_outgoing_time = base_time + timedelta(seconds=int(time_offset + 60))  # 至少间隔1分钟
    time_offset = 0
    for tx in outgoing_txs:
        tx_time = base_outgoing_time + timedelta(seconds=int(time_offset))
        tx["timestamp"] = tx_time.strftime("%Y-%m-%d %H:%M:%S")
        time_offset += int(rng.integers(60, 300))  # 1-5 分钟间隔
    
    # 合并并排序：先转入后转出
    fixed_transactions = incoming_txs + outgoing_txs
    
    # 按时间戳排序（确保顺序正确）
    fixed_transactions.sort(key=lambda x: x.get("timestamp", ""))
    
    return fixed_transactions

def adjust_motif_balance_after_limit_check(motif_id: str, pending_motifs: Dict, 
                                           saved_incoming: float, saved_outgoing: float,
                                           main_wallet: str, rng) -> None:
    """
    在限额检查之后，动态调整 pending_motifs 中剩余交易的金额，确保转入总额 > 转出总额
    这个函数会在每次从 pending_motifs 取出交易并通过限额检查后调用
    """
    if motif_id not in pending_motifs:
        return
    
    motif_data = pending_motifs[motif_id]
    remaining_txs = motif_data['transactions'][motif_data['current_index']:]
    
    if not remaining_txs:
        return
    
    # 计算剩余交易中的转入和转出总额
    remaining_incoming = 0.0
    remaining_outgoing = 0.0
    remaining_incoming_txs = []
    remaining_outgoing_txs = []
    
    for tx in remaining_txs:
        if tx.get("dst") == main_wallet:
            remaining_incoming += float(tx.get("amount", 0))
            remaining_incoming_txs.append(tx)
        elif tx.get("src") == main_wallet:
            remaining_outgoing += float(tx.get("amount", 0))
            remaining_outgoing_txs.append(tx)
    
    # 计算总转入和总转出
    total_incoming = saved_incoming + remaining_incoming
    total_outgoing = saved_outgoing + remaining_outgoing
    
    
    # 如果总转入 < 总转出，需要调整
    # 注意：diff=0.00（即 total_incoming == total_outgoing）是可接受的，不需要调整
    # 情况1：如果还有剩余的转入交易，调整转入交易金额
    if remaining_incoming > 0 and total_incoming < total_outgoing:
        # 目标：总转入 = 总转出 * (1 + margin)，margin 在 5%-15% 之间
        margin = rng.uniform(0.05, 0.15)
        target_total_incoming = (saved_outgoing + remaining_outgoing) * (1 + margin)
        target_remaining_incoming = target_total_incoming - saved_incoming
        
        if target_remaining_incoming > 0:
            scale_factor = target_remaining_incoming / remaining_incoming
            
            
            # 调整剩余的转入交易金额
            for tx in remaining_incoming_txs:
                original_amount = float(tx.get("amount", 0))
                new_amount = original_amount * scale_factor
                tx["amount"] = round(new_amount, 2)
            
            # 验证调整后的总额
            new_remaining_incoming = sum(float(tx.get("amount", 0)) for tx in remaining_incoming_txs)
            new_total_incoming = saved_incoming + new_remaining_incoming
            new_total_outgoing = saved_outgoing + remaining_outgoing
            
            
            # 如果调整后仍然 <= 转出总额（由于四舍五入），再次调整最后一笔转入交易
            if new_total_incoming <= new_total_outgoing and remaining_incoming_txs:
                diff = new_total_outgoing - new_total_incoming
                last_tx = remaining_incoming_txs[-1]
                old_last_amount = float(last_tx.get("amount", 0))
                last_tx["amount"] = round(old_last_amount + diff + 0.01, 2)
                
                # 最终验证
                final_remaining_incoming = sum(float(tx.get("amount", 0)) for tx in remaining_incoming_txs)
                final_total_incoming = saved_incoming + final_remaining_incoming
    
    # 情况2：如果没有剩余的转入交易（所有转入交易都被拒绝），调整转出交易金额
    elif remaining_incoming == 0 and remaining_outgoing > 0 and saved_incoming > 0 and total_incoming < total_outgoing:
        # 目标：转出总额 = 已保存转入总额 * (1 - margin)，margin 在 5%-15% 之间
        # 这样确保转入总额 > 转出总额
        margin = rng.uniform(0.05, 0.15)
        target_total_outgoing = saved_incoming * (1 - margin)
        target_remaining_outgoing = target_total_outgoing - saved_outgoing
        
        if target_remaining_outgoing > 0 and target_remaining_outgoing < remaining_outgoing:
            scale_factor = target_remaining_outgoing / remaining_outgoing
            
            
            # 调整剩余的转出交易金额
            for tx in remaining_outgoing_txs:
                original_amount = float(tx.get("amount", 0))
                new_amount = original_amount * scale_factor
                tx["amount"] = round(new_amount, 2)
            
            # 验证调整后的总额
            new_remaining_outgoing = sum(float(tx.get("amount", 0)) for tx in remaining_outgoing_txs)
            new_total_incoming = saved_incoming
            new_total_outgoing = saved_outgoing + new_remaining_outgoing
            
            
            # 如果调整后仍然 >= 转入总额（由于四舍五入），再次调整最后一笔转出交易
            if new_total_incoming <= new_total_outgoing and remaining_outgoing_txs:
                diff = new_total_outgoing - new_total_incoming
                last_tx = remaining_outgoing_txs[-1]
                old_last_amount = float(last_tx.get("amount", 0))
                last_tx["amount"] = round(old_last_amount - diff - 0.01, 2)
                
                # 最终验证
                final_remaining_outgoing = sum(float(tx.get("amount", 0)) for tx in remaining_outgoing_txs)
                final_total_outgoing = saved_outgoing + final_remaining_outgoing
        else:
            pass  # Cannot adjust outgoing

def fix_transaction_balance(transactions: List[Dict], source_wallet: str, 
                            base_time: datetime, rng,
                            wallet_open_timestamps: Dict[str, datetime] = None) -> List[Dict]:
    """
    通用的交易平衡修复函数（适用于所有交易类型，除了 merchant_laundering）：
    对于 many_to_many, one_to_many, many_to_one 模式：
    1. 确保转入交易时间早于转出交易时间
    2. 确保转入总额 > 转出总额（转入必须高于转出，无佣金）
    """
    if not transactions:
        return transactions
    
    # 检查是否是支持的模式
    first_tx = transactions[0]
    transaction_mode = first_tx.get("transaction_mode")
    transaction_motif = first_tx.get("risk_type", "unknown")
    motif_id = first_tx.get("motif_id", "unknown")
    supported_modes = ["many_to_many", "one_to_many", "many_to_one"]
    
    # DEBUG: 记录函数调用信息
    
    if transaction_mode not in supported_modes:
        return transactions
    
    # 识别主钱包（central wallet）
    # 对于 many_to_many 和 many_to_one：主钱包是 dst 出现次数最多的账户
    # 对于 one_to_many：主钱包是 src（source_wallet）
    if transaction_mode == "one_to_many":
        main_wallet = source_wallet  # one_to_many 中，source_wallet 就是主钱包
    else:
        # many_to_many 或 many_to_one：主钱包是 dst 出现次数最多的账户
        dst_counts = {}
        for tx in transactions:
            dst = tx.get("dst")
            if dst:
                dst_counts[dst] = dst_counts.get(dst, 0) + 1
        
        if dst_counts:
            main_wallet = max(dst_counts, key=dst_counts.get)
        else:
            main_wallet = source_wallet
    
    # 分离转入和转出交易
    incoming_txs = []  # 转入主钱包的交易 (dst == main_wallet)
    outgoing_txs = []   # 从主钱包转出的交易 (src == main_wallet)
    
    for tx in transactions:
        if tx.get("dst") == main_wallet:
            incoming_txs.append(tx)
        elif tx.get("src") == main_wallet:
            outgoing_txs.append(tx)
    
    # 如果没有转入或转出交易，返回原交易列表
    if not incoming_txs or not outgoing_txs:
        return transactions
    
    # 计算当前转入和转出总额
    total_incoming = sum(float(tx.get("amount", 0)) for tx in incoming_txs)
    total_outgoing = sum(float(tx.get("amount", 0)) for tx in outgoing_txs)
    
    # 如果转入或转出总额为0，无法修复，返回原交易列表
    if total_incoming <= 0 or total_outgoing <= 0:
        return transactions
    
    # 通用要求：转入总额 >= 转出总额（无佣金）
    # 注意：diff=0.00（即 total_incoming == total_outgoing）是可接受的
    # 如果转入总额 < 转出总额，需要调整转入交易的金额，使得转入总额 >= 转出总额
    if total_incoming < total_outgoing:
        # 目标转入总额 = 转出总额 * (1 + margin)，margin 在 5%-15% 之间
        margin = rng.uniform(0.05, 0.15)
        target_incoming_total = total_outgoing * (1 + margin)
        scale_factor = target_incoming_total / total_incoming
        
        
        # 调整转入交易的金额
        adjusted_details = []
        for tx in incoming_txs:
            original_amount = float(tx.get("amount", 0))
            new_amount = original_amount * scale_factor
            tx["amount"] = round(new_amount, 2)
            adjusted_details.append((tx.get("tx_id"), original_amount, tx.get("amount")))
        
        
        # 验证修复后的转入总额（处理四舍五入误差）
        new_total_incoming = sum(float(tx.get("amount", 0)) for tx in incoming_txs)
        
        if new_total_incoming < total_outgoing:
            # 如果修复后仍然 < 转出总额（由于四舍五入），再次调整最后一笔转入交易
            diff = total_outgoing - new_total_incoming
            if incoming_txs and diff > 0:
                # 增加最后一笔转入交易的金额，确保转入总额 >= 转出总额（至少相等）
                last_tx = incoming_txs[-1]
                old_last_amount = float(last_tx.get("amount", 0))
                last_tx["amount"] = round(old_last_amount + diff + 0.01, 2)
    
    # 重新分配时间戳：转入交易在前，转出交易在后
    # 确保 base_time 不早于所有相关钱包的开户时间
    if wallet_open_timestamps:
        min_open_time = None
        for tx in transactions:
            src = tx.get("src")
            dst = tx.get("dst")
            if src and src in wallet_open_timestamps:
                open_time = wallet_open_timestamps[src]
                if min_open_time is None or open_time > min_open_time:
                    min_open_time = open_time
            if dst and dst in wallet_open_timestamps:
                open_time = wallet_open_timestamps[dst]
                if min_open_time is None or open_time > min_open_time:
                    min_open_time = open_time
        if min_open_time and base_time < min_open_time:
            base_time = min_open_time
    
    # 转入交易从 base_time 开始，每笔间隔 1-5 分钟
    time_offset = 0
    for tx in incoming_txs:
        tx_time = base_time + timedelta(seconds=int(time_offset))
        # 确保时间戳不早于相关钱包的开户时间
        if wallet_open_timestamps:
            src = tx.get("src")
            dst = tx.get("dst")
            if src and src in wallet_open_timestamps:
                src_open_time = wallet_open_timestamps[src]
                if tx_time < src_open_time:
                    tx_time = src_open_time
            if dst and dst in wallet_open_timestamps:
                dst_open_time = wallet_open_timestamps[dst]
                if tx_time < dst_open_time:
                    tx_time = dst_open_time
        tx["timestamp"] = tx_time.strftime("%Y-%m-%d %H:%M:%S")
        time_offset += int(rng.integers(60, 300))  # 1-5 分钟间隔
    
    # 转出交易在转入交易之后，每笔间隔 1-5 分钟
    # 确保转出交易的时间晚于所有转入交易
    base_outgoing_time = base_time + timedelta(seconds=int(time_offset + 60))  # 至少间隔1分钟
    time_offset = 0
    for tx in outgoing_txs:
        tx_time = base_outgoing_time + timedelta(seconds=int(time_offset))
        # 确保时间戳不早于相关钱包的开户时间
        if wallet_open_timestamps:
            src = tx.get("src")
            dst = tx.get("dst")
            if src and src in wallet_open_timestamps:
                src_open_time = wallet_open_timestamps[src]
                if tx_time < src_open_time:
                    tx_time = src_open_time
            if dst and dst in wallet_open_timestamps:
                dst_open_time = wallet_open_timestamps[dst]
                if tx_time < dst_open_time:
                    tx_time = dst_open_time
        tx["timestamp"] = tx_time.strftime("%Y-%m-%d %H:%M:%S")
        time_offset += int(rng.integers(60, 300))  # 1-5 分钟间隔
    
    # 最终验证：确保修复后的转入总额 > 转出总额
    final_total_incoming = sum(float(tx.get("amount", 0)) for tx in incoming_txs)
    final_total_outgoing = sum(float(tx.get("amount", 0)) for tx in outgoing_txs)
    
    
    if final_total_incoming < final_total_outgoing:
        # 如果最终验证失败（total_incoming < total_outgoing），强制调整最后一笔转入交易
        # 注意：diff=0.00（即 total_incoming == total_outgoing）是可接受的，不需要强制调整
        if incoming_txs:
            diff = final_total_outgoing - final_total_incoming
            last_tx = incoming_txs[-1]
            old_last_amount = float(last_tx.get("amount", 0))
            # 确保至少增加 0.01，使得转入总额 >= 转出总额
            last_tx["amount"] = round(old_last_amount + diff + 0.01, 2)
            
            # 再次验证
            final_total_incoming = sum(float(tx.get("amount", 0)) for tx in incoming_txs)
    elif final_total_incoming > final_total_outgoing and final_total_outgoing > 0:
        # 如果 diff > 0 且 diff 相对于转出总额的比例超过 1%，强制调整使转入总额等于转出总额
        diff = final_total_incoming - final_total_outgoing
        diff_ratio = diff / final_total_outgoing
        if diff_ratio > 0.01:  # 超过 1%
            if incoming_txs:
                # 计算需要减少的金额，使得转入总额等于转出总额
                target_total_incoming = final_total_outgoing
                current_total_incoming = sum(float(tx.get("amount", 0)) for tx in incoming_txs)
                reduction_needed = current_total_incoming - target_total_incoming
                
                if reduction_needed > 0:
                    # 按比例减少所有转入交易的金额
                    scale_factor = target_total_incoming / current_total_incoming
                    
                    for tx in incoming_txs:
                        old_amount = float(tx.get("amount", 0))
                        tx["amount"] = round(old_amount * scale_factor, 2)
                    
                    # 验证调整后的总额（处理四舍五入误差）
                    new_total_incoming = sum(float(tx.get("amount", 0)) for tx in incoming_txs)
                    if new_total_incoming != final_total_outgoing:
                        # 如果由于四舍五入导致不完全相等，调整最后一笔交易
                        diff_remaining = new_total_incoming - final_total_outgoing
                        last_tx = incoming_txs[-1]
                        old_last_amount = float(last_tx.get("amount", 0))
                        last_tx["amount"] = round(old_last_amount - diff_remaining, 2)
                    
                    # 最终验证
                    final_total_incoming = sum(float(tx.get("amount", 0)) for tx in incoming_txs)
    
    # 合并并排序：先转入后转出
    fixed_transactions = incoming_txs + outgoing_txs
    
    # 按时间戳排序（确保顺序正确）
    fixed_transactions.sort(key=lambda x: x.get("timestamp", ""))
    
    # DEBUG: 记录修复后的完整交易列表
    final_incoming_details = [(tx.get("tx_id"), tx.get("src"), tx.get("dst"), tx.get("amount"), tx.get("timestamp")) for tx in incoming_txs]
    final_outgoing_details = [(tx.get("tx_id"), tx.get("src"), tx.get("dst"), tx.get("amount"), tx.get("timestamp")) for tx in outgoing_txs]
    
    if final_total_incoming < final_total_outgoing:
        pass  # total_incoming < total_outgoing 是不允许的，应该在之前已经被修复
    else:
        # diff >= 0 视为成功
        diff_value = final_total_incoming - final_total_outgoing
        if diff_value == 0.0:
            pass  # total_incoming == total_outgoing 是可接受的
        elif final_total_outgoing > 0 and diff_value / final_total_outgoing > 0.01:
            # 如果 diff 超过 1%，应该在之前已经被调整为相等，这里不应该出现
            pass
        else:
            pass  # total_incoming > total_outgoing 且 diff <= 1% 是可接受的
    
    return fixed_transactions

def generate_transactions(g: nx.DiGraph, balances: Dict[str, float],
                        wallet_to_attrs: Dict[str, List[str]], attr_headers: List[str],
                        wallet_open_timestamps: Dict[str, datetime], abnormal_wallets: set,
                        config: Dict, output_file: str = None) -> List[Dict]:
    """生成交易的主函数"""
    
    # 记录开始时间（用于进度估算）
    generation_start_time = time.time()
    
    # 初始化交易管理器
    manager = TransactionManager("transaction_config.json")
    generator = UnifiedTransactionGenerator("transaction_config.json")
    
    # 配置参数
    start_time = datetime.strptime(config.get("start_time", "2024-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
    duration_days = int(config.get("duration_days", 7))
    max_frac_per_tx = float(config.get("max_frac_per_tx", 0.2))
    min_abs_amount = float(config.get("min_abs_amount", 0.01))
    seed = int(config.get("random_seed", 44))
    
    rng = np.random.default_rng(seed)
    transactions = []
    current_balances = balances.copy()
    
    # 初始化批量保存CSV（每1万条保存一次）
    csv_file = None
    csv_writer = None
    transaction_buffer = []  # 交易缓冲区
    buffer_size = 10000  # 每10000笔交易写入一次
    safe_motifs = {'single_transaction', 'normal_small_high_freq', 'regular_large_low_freq'}  # 安全交易类型
    wallet_info_cache = {}  # 缓存钱包信息，避免重复计算
    
    # 跟踪每个 motif 的已保存转入和转出总额（用于限额检查后的动态调整）
    motif_saved_amounts = {}  # {motif_id: {'main_wallet': str, 'saved_incoming': float, 'saved_outgoing': float}}
    
    # 优化：记录每个钱包的第一次交易时间，避免重复验证开立时间
    wallet_first_tx_time = {}  # {wallet_id: datetime} - 记录每个钱包的第一次交易时间
    abandoned_wallets = set()  # 记录废弃的钱包（第一次交易时间早于开户时间的钱包）
    
    if output_file:
        csv_file = open(output_file, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        # 写入表头
        csv_writer.writerow(["tx_id", "timestamp", "src", "dst", "amount", "transaction_motif",
                            "motif_id", "transaction_mode", "is_risk", "src_bank_account_number",
                            "dst_bank_account_number", "src_wallet_level", "dst_wallet_level"])
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已创建CSV文件: {output_file}，将每 {buffer_size} 笔交易批量写入 {get_process_info()}")
    
    # 为每个钱包创建交易计划，并预先计算 wallet_level 和 limits（优化）
    # 允许 wallet_level="1", "2", "3", "4" 的钱包（放宽限制以增加交易生成）
    wallet_plans = {}
    allowed_levels = ["1", "2", "3", "4"]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在初始化钱包计划并预计算属性（wallet_level='1', '2', '3', '4'）... {get_process_info()}")
    init_start = time.time()
    level_1_count = 0
    level_2_count = 0
    level_3_count = 0
    level_4_count = 0
    skipped_count = 0
    for wallet_id in g.nodes():
        wallet_attrs = wallet_to_attrs.get(wallet_id, [])
        wallet_type = None
        wallet_level = None
        
        # 解析钱包属性
        for i, h in enumerate(attr_headers):
            if h in ['wallet_type', 'type'] and i > 0 and i-1 < len(wallet_attrs):
                wallet_type = wallet_attrs[i-1]  # wallet_attrs排除了wallet_id，所以索引要减1
            elif h in ['wallet_level', 'level'] and i > 0 and i-1 < len(wallet_attrs):
                wallet_level = wallet_attrs[i-1]
        
        # 如果通过列名没找到，直接通过索引获取
        if wallet_type is None and len(wallet_attrs) > 0:
            wallet_type = wallet_attrs[0]  # wallet_type在第0列（因为wallet_id被排除了）
        if wallet_level is None and len(wallet_attrs) > 1:
            wallet_level = wallet_attrs[1]  # wallet_level在第1列
        
        # 确保类型转换为字符串（用于配置匹配）
        if wallet_type is not None:
            wallet_type = str(wallet_type)
        if wallet_level is not None:
            wallet_level = str(wallet_level)
        
        # 只处理 wallet_level="1", "2", "3", "4" 的钱包（放宽限制）
        if wallet_level not in allowed_levels:
            skipped_count += 1
            continue
        
        # 统计各等级钱包数量
        if wallet_level == "1":
            level_1_count += 1
        elif wallet_level == "2":
            level_2_count += 1
        elif wallet_level == "3":
            level_3_count += 1
        elif wallet_level == "4":
            level_4_count += 1
        
        # 预先计算并缓存 wallet_level 和 limits（优化：避免重复调用）
        cached_wallet_level = manager.get_wallet_level(wallet_id, wallet_to_attrs, attr_headers)
        cached_limits = manager.get_wallet_level_limits(cached_wallet_level)
        
        wallet_plans[wallet_id] = {
            'neighbors': list(g.successors(wallet_id)),
            'last_tx_time': start_time,
            'tx_count': 0,
            'risk_tx_count': 0,
            'wallet_type': wallet_type,
            'wallet_level': wallet_level,
            'cached_wallet_level': cached_wallet_level,  # 缓存的计算结果
            'cached_limits': cached_limits,  # 缓存的限额信息
            'current_daily_amount': 0.0,  # 优化：直接存储当前日期的累计金额，避免字典查找
            'daily_limit_reached': False,  # 是否达到日交易额度限制
            'reservation_transactions': [],  # 新增：存储预留交易
            'last_salary_time': None,
            'next_salary_time': None,
            'salary_interval_days': None,
            'salary_employees': [],
            'base_salary': None,
            'can_generate_salary': True
        }
    init_elapsed = time.time() - init_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 完成初始化 {len(wallet_plans)} 个钱包计划 (level 1: {level_1_count}, level 2: {level_2_count}, level 3: {level_3_count}, level 4: {level_4_count}), 跳过 {skipped_count} 个其他等级钱包, 耗时 {init_elapsed:.2f} 秒 {get_process_info()}")
    
    # 优化：日交易额度跟踪 - 直接在 wallet_plans 中存储当前日期的累计金额
    # 这样访问时直接 wallet_plans[wallet_id]['current_daily_amount']，比字典查找更快
    current_date = None
    
    # 优化：缓存可用钱包列表（只在日期变化时更新）
    cached_available_wallets = None
    cached_available_wallets_date = None
    
    # 优化：预先缓存所有钱包列表和集合（包含 wallet_level="1", "2", "3", "4" 的钱包）
    all_wallets_list = list(wallet_plans.keys())  # 包含 level 1, 2, 3, 4 钱包
    all_wallets_set = set(all_wallets_list)

    # 优化：缓存 motif_id 的统计信息，避免重复遍历所有交易
    motif_stats_cache = {}  # {motif_id: (risk_count, normal_count)}
    
    # 优化：缓存混合钱包数量，避免每次都遍历
    current_mixed_count = 0  # 全局混合钱包计数器

    # 统计信息
    total_transactions = 0
    risk_transactions = 0
    normal_transactions = 0
    
    # 跟踪参与风险交易的钱包集合（用于限制参与风险交易的钱包总数）
    risk_wallets_set = set()
    
    # 跟踪钱包的交易类型偏好（用于确保钱包主要只参与一种交易类型）
    # wallet_tx_type_preference: {wallet_id: 'risk' or 'normal' or 'mixed'}
    # 初始为None，第一次参与交易时设置为对应类型
    # 如果钱包既参与风险又参与正常交易，则标记为'mixed'
    wallet_tx_type_preference = {}  # 钱包交易类型偏好字典
    max_mixed_wallets_ratio = 0.1  # 最多允许10%的钱包混合参与两种交易类型
    max_mixed_wallets_count = int(len(g.nodes()) * max_mixed_wallets_ratio)
    
    # 强制参与异常交易的钱包列表（原本is_abnormal=0，但被选中参与异常交易后，只能参与异常交易）
    forced_risk_wallets = set()
    
    # 获取钱包选择策略配置
    wallet_policy = manager.global_settings.get("wallet_selection_policy", {})
    abnormal_wallet_risk_ratio = wallet_policy.get("abnormal_wallet_risk_ratio", 0.95)
    normal_wallet_risk_ratio = wallet_policy.get("normal_wallet_risk_ratio", 0.05)
    abnormal_wallet_normal_ratio = wallet_policy.get("abnormal_wallet_normal_ratio", 0.1)
    normal_wallet_normal_ratio = wallet_policy.get("normal_wallet_normal_ratio", 0.95)
    max_risk_wallets_ratio = wallet_policy.get("max_risk_wallets_ratio", 0.15)
    
    # 计算最大风险钱包数量
    # 策略：优先使用异常账户，如果异常账户不足，允许少量正常账户补充
    # 但一旦确定了参与异常交易的钱包集合，就固定使用这些钱包，不再扩展
    max_risk_wallets_count_by_ratio = int(len(g.nodes()) * max_risk_wallets_ratio)
    
    # 允许异常账户 + 少量正常账户（最多为异常账户数量的20%）参与异常交易
    max_risk_wallets_count_by_abnormal = int(len(abnormal_wallets) * 1.2)  # 异常账户 + 20%正常账户
    
    # 取两者中的较小值
    max_risk_wallets_count = min(max_risk_wallets_count_by_ratio, max_risk_wallets_count_by_abnormal)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始生成交易，风险交易比例目标: {manager.get_risk_ratio():.1%} {get_process_info()}")
    print(f"每个钱包最大风险交易次数: {manager.get_max_risk_transactions_per_wallet()}")
    print(f"异常账户数量: {len(abnormal_wallets):,}")
    print(f"最多参与风险交易的钱包数: {max_risk_wallets_count:,} (基于比例: {max_risk_wallets_ratio*100:.1f}% = {max_risk_wallets_count_by_ratio:,}, 基于异常账户1.2倍: {max_risk_wallets_count_by_abnormal:,})")
    print(f"策略：优先使用异常账户（is_abnormal=1），如果异常账户不足，允许少量正常账户补充")
    print(f"      一旦正常账户被选中参与异常交易，将加入forced_risk_wallets，以后只能参与异常交易")
    print(f"      一旦达到限制，只从已参与风险交易的钱包中选择（不再扩展）")
    print(f"异常账户参与风险交易比例: {abnormal_wallet_risk_ratio*100:.1f}%")
    print(f"正常账户参与风险交易比例: {normal_wallet_risk_ratio*100:.1f}%")
    print(f"最多混合钱包数量: {max_mixed_wallets_count:,} ({max_mixed_wallets_ratio*100:.1f}%)")
    
    # 为每一天生成交易时间表
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始生成交易时间表... {get_process_info()}")
    schedule_start = time.time()
    all_transaction_times = []
    for day in range(duration_days):
        current_date = start_time + timedelta(days=day)
        daily_times = generate_daily_transaction_schedule(config, rng)
        
        # 将时间调整到具体日期
        for tx_time in daily_times:
            actual_time = current_date.replace(hour=tx_time.hour, minute=tx_time.minute, second=tx_time.second)
            all_transaction_times.append(actual_time)
    
    # 按时间排序所有交易时间
    all_transaction_times.sort()
    schedule_elapsed = time.time() - schedule_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 生成了 {len(all_transaction_times):,} 个交易时间点, 耗时 {schedule_elapsed:.2f} 秒 {get_process_info()}")
    
    # 生成交易
    tx_id = 1
    processed_count = 0  # 用于进度输出
    
    # 初始化 motif 穿插机制：待处理的 motif 队列
    # 格式：{motif_id: {'transactions': [...], 'current_index': 0, 'transaction_type': ..., 'source_wallet': ..., 'plan': ...}}
    pending_motifs = {}
    # 从配置文件读取 motif 穿插概率，如果没有配置则使用默认值 0.3
    motif_interleaving_config = config.get("motif_interleaving", {})
    continue_motif_probability = float(motif_interleaving_config.get("continue_motif_probability", 0.3))  # 30%概率继续现有motif，70%概率生成新motif
    
    for current_time in all_transaction_times:
        processed_count += 1
        # 每处理5000个时间点输出一次进度（减少输出频率，提升性能）
        if processed_count % 5000 == 0:
            progress = processed_count / len(all_transaction_times) * 100
            elapsed = time.time() - generation_start_time
            if processed_count > 0 and elapsed > 0:
                rate = processed_count / elapsed
                remaining = (len(all_transaction_times) - processed_count) / rate if rate > 0 else 0
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 进度: {processed_count:,}/{len(all_transaction_times):,} ({progress:.1f}%) | "
                      f"已用时: {elapsed/60:.1f}分钟 | 预计剩余: {remaining/60:.1f}分钟 | "
                      f"已生成交易: {total_transactions:,} 笔 (风险: {risk_transactions:,}, 正常: {normal_transactions:,}) {get_process_info()}")
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 进度: {processed_count:,}/{len(all_transaction_times):,} ({progress:.1f}%) {get_process_info()}")
        
        # 优化：检查日期变化，重置日交易额度跟踪和限额标志
        new_date = current_time.date()
        if current_date != new_date:
            current_date = new_date
            
            # 优化：重置所有钱包的当前日累计金额和限额标志（新的一天）
            # 使用批量操作，比逐个访问更快
            for wallet_id in wallet_plans:
                wallet_plans[wallet_id]['current_daily_amount'] = 0.0
                wallet_plans[wallet_id]['daily_limit_reached'] = False
            
            # 优化：在日期变化时，预先过滤并缓存可用钱包列表
            # 获取所有有余额的钱包（使用集合操作优化），考虑 wallet_level="1", "2", "3", "4" 的钱包
            wallets_with_balance = [w for w in all_wallets_list 
                                   if w in wallet_plans and  # 只考虑在 wallet_plans 中的钱包（即 level 1, 2, 3, 4）
                                   current_balances.get(w, 0) > 0]
            
            # 预先过滤：排除已达到限额的钱包（提前检查 daily_limit_reached）
            # 优化：使用列表推导式，减少函数调用
            cached_available_wallets = []
            for wallet in wallets_with_balance:
                plan = wallet_plans[wallet]
                # 提前检查：如果已达到日交易额度限制，直接跳过
                if plan.get('daily_limit_reached', False):
                    continue
                
                # 检查钱包是否有邻居（必须条件）- 提前检查，避免后续计算
                if not plan['neighbors']:
                    continue
                
                # 使用缓存的 limits（优化：避免重复调用）
                limits = plan['cached_limits']
                daily_limit = limits.get('daily_limit')
                
                # 优化：直接从 wallet_plans 中获取当前日累计金额，避免字典查找
                # 检查当前日累计金额（虽然刚重置，但保留检查逻辑）
                if daily_limit is not None:
                    current_daily_amount = plan['current_daily_amount']
                    if current_daily_amount >= daily_limit:
                        plan['daily_limit_reached'] = True
                        continue
                
                # 如果钱包通过所有检查，添加到缓存列表
                cached_available_wallets.append(wallet)
            
            cached_available_wallets_date = current_date
            # 如果正常交易没有可用钱包，则允许异常交易使用余额 <= 0 的钱包
            # 考虑 wallet_level="1", "2", "3", "4" 的钱包
            if not cached_available_wallets:
                # 优化：使用集合查找，更快，考虑 level 1, 2, 3, 4 钱包
                for wallet in all_wallets_list:
                    if wallet not in wallet_plans:  # 跳过不在 wallet_plans 中的钱包
                        continue
                    if wallet in current_balances:
                        plan = wallet_plans[wallet]
                        if plan.get('daily_limit_reached', False):
                            continue
                        if not plan['neighbors']:
                            continue
                        cached_available_wallets.append(wallet)
        
        # 优化：使用集合和预计算，减少重复检查
        # 优化：只在限额变化时才重新过滤（使用集合差集快速移除达到限额的钱包）
        if cached_available_wallets:
            # 优化：使用集合操作，更快地过滤
            # 先快速检查 daily_limit_reached 标志（O(1)检查）
            available_wallets_set = set(cached_available_wallets)
            
            # 优化：批量检查限额，使用列表推导式一次性过滤
            # 只检查那些可能达到限额的钱包（有daily_limit的钱包）
            wallets_to_check = []
            for wallet in cached_available_wallets:
                plan = wallet_plans[wallet]
                if plan.get('daily_limit_reached', False):
                    available_wallets_set.discard(wallet)
                    continue
                
                limits = plan['cached_limits']
                daily_limit = limits.get('daily_limit')
                if daily_limit is not None:
                    wallets_to_check.append((wallet, daily_limit))
            
            # 优化：批量检查限额（直接从 wallet_plans 中获取，避免字典查找）
            for wallet, daily_limit in wallets_to_check:
                # 优化：直接从 wallet_plans 中获取当前日累计金额
                plan = wallet_plans[wallet]
                current_daily_amount = plan['current_daily_amount']
                if current_daily_amount >= daily_limit:
                    plan['daily_limit_reached'] = True
                    available_wallets_set.discard(wallet)
            
            # 转换为列表（保持原有接口）
            available_wallets_filtered = list(available_wallets_set)
        else:
            available_wallets_filtered = []
        
        # 优化：排除废弃的钱包（第一次交易时间早于开户时间的钱包）
        available_wallets_filtered = [w for w in available_wallets_filtered if w not in abandoned_wallets]
        
        if not available_wallets_filtered:
            # 如果所有钱包都达到限额，跳过当前时间点，等待下一天重置限额
            if processed_count % 10000 == 0:
                print(f"当前时间点所有钱包都达到日交易额度限制，跳过（已处理 {processed_count:,}/{len(all_transaction_times):,} 个时间点，等待下一天重置限额）")
            continue
        
        # ========== Motif 穿插机制：决定是继续现有motif还是生成新motif ==========
        new_transactions = []
        selected_transaction_type = None
        source_wallet = None
        plan = None
        is_continue_motif = False
        is_continued_motif_processed = False  # 标记是否已处理继续的motif
        
        # 决定是继续现有motif还是生成新motif
        if pending_motifs and rng.random() < continue_motif_probability:
            is_continue_motif = True
        
        if is_continue_motif and pending_motifs:
            # 从待处理队列中随机选择一个motif
            motif_id = rng.choice(list(pending_motifs.keys()))
            motif_data = pending_motifs[motif_id]
            
            # 获取下一个交易
            if motif_data['current_index'] < len(motif_data['transactions']):
                next_tx = motif_data['transactions'][motif_data['current_index']].copy()
                next_tx_amount_before = next_tx.get("amount")
                next_tx_timestamp_before = next_tx.get("timestamp")
                # 对于 merchant_laundering，使用修复后的时间戳，不要覆盖
                if motif_data['transaction_type'] == 'merchant_laundering':
                    pass  # 使用修复后的时间戳
                # 对于其他交易类型（many_to_many, many_to_one, one_to_many），也使用修复后的时间戳，不要覆盖
                elif next_tx.get("transaction_mode") in ["many_to_many", "many_to_one", "one_to_many"]:
                    pass  # 使用修复后的时间戳
                else:
                    next_tx['timestamp'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                new_transactions = [next_tx]
                motif_data['current_index'] += 1
                
                # 如果该motif的所有交易都已处理，从队列中移除
                if motif_data['current_index'] >= len(motif_data['transactions']):
                    del pending_motifs[motif_id]
                
                selected_transaction_type = motif_data['transaction_type']
                source_wallet = motif_data['source_wallet']
                plan = motif_data.get('plan')
                
                # 标记为已处理继续的motif
                is_continued_motif_processed = True
                
                # 对于继续的motif，需要检查限额和余额（实时更新）
                if new_transactions:
                    # 检查限额（只检查当前这笔交易，因为余额已经实时更新）
                    tx = new_transactions[0]
                    src = tx['src']
                    dst = tx.get('dst')
                    amount = float(tx['amount'])
                    motif_id_check = tx.get('motif_id')
                    tx_id_check = tx.get('tx_id')
                    tx_timestamp_str = tx.get("timestamp")
                    
                    # 优化：检查钱包是否已被废弃
                    if src in abandoned_wallets or (dst and dst in abandoned_wallets):
                        new_transactions = []
                        continue
                    
                    # 优化：如果是第一次交易，验证开立时间
                    if tx_timestamp_str:
                        try:
                            tx_timestamp = datetime.strptime(tx_timestamp_str, "%Y-%m-%d %H:%M:%S")
                            
                            # 检查 src 钱包是否是第一次交易
                            if src not in wallet_first_tx_time:
                                src_open_time = wallet_open_timestamps.get(src)
                                if src_open_time and tx_timestamp < src_open_time:
                                    # 交易时间早于开户时间，废弃该钱包
                                    abandoned_wallets.add(src)
                                    new_transactions = []
                                    continue
                                else:
                                    # 通过验证，记录第一次交易时间
                                    wallet_first_tx_time[src] = tx_timestamp
                            
                            # 检查 dst 钱包是否是第一次交易
                            if dst and dst not in wallet_first_tx_time:
                                dst_open_time = wallet_open_timestamps.get(dst)
                                if dst_open_time and tx_timestamp < dst_open_time:
                                    # 交易时间早于开户时间，废弃该钱包
                                    abandoned_wallets.add(dst)
                                    new_transactions = []
                                    continue
                                else:
                                    # 通过验证，记录第一次交易时间
                                    wallet_first_tx_time[dst] = tx_timestamp
                        except:
                            pass  # 时间戳格式错误，跳过验证
                    
                    # 如果验证失败导致 new_transactions 被清空，跳过后续处理
                    if not new_transactions:
                        continue
                    
                    if src in wallet_plans:
                        plan_wallet = wallet_plans[src]
                        limits = plan_wallet['cached_limits']
                        single_limit = limits.get('single_transaction_limit')
                        daily_limit = limits.get('daily_limit')
                        current_daily_amount = plan_wallet['current_daily_amount']
                        current_balance = current_balances.get(src, 0)
                        
                        
                        # 限额检查后，动态调整剩余交易金额（仅对 many_to_many, many_to_one, one_to_many 模式）
                        # 无论交易是否通过限额检查，都需要调整剩余金额
                        transaction_mode = tx.get("transaction_mode")
                        should_adjust = (transaction_mode in ["many_to_many", "many_to_one", "one_to_many"] and motif_id_check)
                        
                        if should_adjust:
                            # 识别主钱包
                            if transaction_mode == "one_to_many":
                                main_wallet = source_wallet
                            else:
                                # many_to_many 或 many_to_one：主钱包是 dst 出现次数最多的账户
                                if motif_id_check in motif_saved_amounts:
                                    main_wallet = motif_saved_amounts[motif_id_check].get('main_wallet', tx.get('dst'))
                                else:
                                    # 从 pending_motifs 中找出主钱包
                                    if motif_id_check in pending_motifs:
                                        motif_data_temp = pending_motifs[motif_id_check]
                                        dst_counts = {}
                                        for temp_tx in motif_data_temp['transactions']:
                                            temp_dst = temp_tx.get("dst")
                                            if temp_dst:
                                                dst_counts[temp_dst] = dst_counts.get(temp_dst, 0) + 1
                                        main_wallet = max(dst_counts, key=dst_counts.get) if dst_counts else tx.get('dst')
                                    else:
                                        main_wallet = tx.get('dst')
                            
                            # 初始化或获取已保存的转入和转出总额
                            if motif_id_check not in motif_saved_amounts:
                                motif_saved_amounts[motif_id_check] = {
                                    'main_wallet': main_wallet,
                                    'saved_incoming': 0.0,
                                    'saved_outgoing': 0.0
                                }
                        
                        # 检查单笔限额
                        if single_limit is not None and amount > single_limit:
                            new_transactions = []
                            plan_wallet['daily_limit_reached'] = True
                        # 检查日累计限额
                        elif daily_limit is not None and (current_daily_amount + amount) > daily_limit:
                            new_transactions = []
                            plan_wallet['daily_limit_reached'] = True
                        # 检查余额
                        elif current_balance < amount:
                            new_transactions = []
                        else:
                            # 通过检查，更新余额和限额
                            current_balances[src] -= amount
                            current_balances[tx['dst']] = current_balances.get(tx['dst'], 0) + amount
                            plan_wallet['current_daily_amount'] += amount
                            # 注意：不在限额检查时更新 motif_saved_amounts，而是在保存到CSV时更新
                        
                        # 无论交易是否通过限额检查，都需要调整剩余交易金额
                        # （如果交易被拒绝，saved_incoming/saved_outgoing 不会被更新，所以剩余交易需要被调整以补偿）
                        if should_adjust:
                            saved_incoming = motif_saved_amounts[motif_id_check]['saved_incoming']
                            saved_outgoing = motif_saved_amounts[motif_id_check]['saved_outgoing']
                            adjust_motif_balance_after_limit_check(
                                motif_id_check, pending_motifs, saved_incoming, saved_outgoing, main_wallet, rng
                            )
                    else:
                        new_transactions = []
                
                # 如果继续的motif交易被拒绝，跳过后续处理
                if not new_transactions:
                    # motif_id_check 和 tx_id_check 已在上面定义（如果有 new_transactions）
                    if 'motif_id_check' in locals():
                        pass  # 交易被拒绝，不会保存到 CSV
                    else:
                        pass  # 交易被拒绝，不会保存到 CSV
                    continue
        
        # ========== 新逻辑：先决定交易类型，再根据类型选择钱包 ==========
        # 只有在没有继续现有motif时才生成新motif
        if not new_transactions:
            # 优化：缓存风险比例计算，避免重复计算
            if total_transactions > 0:
                current_risk_ratio = risk_transactions / total_transactions
            else:
                current_risk_ratio = 0.0
            target_risk_ratio = manager.get_risk_ratio()
            
            # 根据当前风险比例决定生成风险交易还是正常交易
            # 但如果已经达到最大风险钱包数量限制，且没有可用的风险钱包，则强制生成正常交易
            is_risk_tx_decision = False
            
            # 优化：使用集合操作，预计算可用风险钱包
            available_wallets_set = set(available_wallets_filtered)
            
            # 检查是否已达到风险钱包数量限制且没有可用风险钱包
            # 优化：使用集合交集操作，更快
            if len(risk_wallets_set) >= max_risk_wallets_count:
                # 优化：使用集合交集，O(min(len(available_wallets_set), len(risk_wallets_set)))
                risk_available_set = available_wallets_set & risk_wallets_set
                # 快速检查是否有邻居（使用预缓存的neighbors信息）
                risk_available_check = [w for w in risk_available_set if wallet_plans[w]['neighbors']]
                if not risk_available_check:
                    # 已达到限制且没有可用风险钱包，强制生成正常交易
                    is_risk_tx_decision = False
                else:
                    # 虽然达到限制，但还有可用风险钱包，可以继续生成风险交易
                    if current_risk_ratio < target_risk_ratio:
                        # 更保守的概率计算：目标比例很小（1%），所以概率也应该很小
                        risk_probability = target_risk_ratio + (target_risk_ratio - current_risk_ratio) * 0.5
                        is_risk_tx_decision = rng.random() < min(risk_probability, target_risk_ratio * 2)
                    else:
                        # 当前比例高于目标，大幅降低风险交易概率
                        risk_probability = max(0.0, target_risk_ratio * 0.1)
                        is_risk_tx_decision = rng.random() < risk_probability
            else:
                # 未达到限制，正常决策
                if current_risk_ratio < target_risk_ratio:
                    # 如果当前风险比例低于目标，倾向于生成风险交易，但概率要小
                    risk_probability = target_risk_ratio + (target_risk_ratio - current_risk_ratio) * 0.5
                    is_risk_tx_decision = rng.random() < min(risk_probability, target_risk_ratio * 2)
                else:
                    # 如果当前风险比例高于目标，大幅降低风险交易概率
                    risk_probability = max(0.0, target_risk_ratio * 0.1)
                    is_risk_tx_decision = rng.random() < risk_probability
            
            # 2. 根据交易类型选择钱包
            if is_risk_tx_decision:
                # 风险交易策略：
                # 1. 优先使用异常账户
                # 2. 如果异常账户不足，允许少量正常账户补充
                # 3. 一旦达到 max_risk_wallets_count，只从已参与风险交易的钱包中选择（不再扩展）
                # 4. 这样可以确保异常账户和正常账户分得开，且参与异常交易的钱包集合相对固定
                
                if len(risk_wallets_set) >= max_risk_wallets_count:
                    # 已达到最大限制，只从已参与风险交易的钱包中选择（不再扩展）
                    # 优化：使用集合交集操作，更快（复用已计算的available_wallets_set）
                    risk_candidates_set = available_wallets_set & risk_wallets_set
                    risk_available = [w for w in risk_candidates_set if wallet_plans[w]['neighbors']]
                    if risk_available:
                        # 优先选择偏好为'risk'的钱包，避免选择偏好为'normal'的钱包
                        preferred_risk = [w for w in risk_available 
                                         if wallet_tx_type_preference.get(w) in ['risk', None]]
                        non_preferred_risk = [w for w in risk_available 
                                             if wallet_tx_type_preference.get(w) == 'normal']
                        
                        if preferred_risk:
                            # 优先选择异常账户（偏好为'risk'的异常账户）
                            abnormal_preferred = [w for w in preferred_risk if w in abnormal_wallets]
                            if abnormal_preferred and rng.random() < abnormal_wallet_risk_ratio:
                                source_wallet = rng.choice(abnormal_preferred)
                            else:
                                source_wallet = rng.choice(preferred_risk)
                        elif non_preferred_risk:
                            # 如果没有偏好为'risk'的钱包，才选择偏好为'normal'的钱包（混合）
                            source_wallet = rng.choice(non_preferred_risk)
                        else:
                            source_wallet = rng.choice(risk_available)
                    else:
                        # 如果达到限制且没有可用风险钱包，强制转为正常交易
                        is_risk_tx_decision = False
                        source_wallet = None  # 重置，让后续正常交易逻辑重新选择
                else:
                    # 未达到限制，优先从异常账户中选择，如果异常账户不足则允许正常账户补充
                    # 优化：使用集合交集操作，更快（复用已计算的available_wallets_set）
                    abnormal_available_set = abnormal_wallets & available_wallets_set - risk_wallets_set
                    abnormal_available = [w for w in abnormal_available_set if wallet_plans[w]['neighbors']]
                    
                    # 如果异常账户没有余额，放宽限制（包括已参与的异常账户）
                    if not abnormal_available:
                        # 优化：使用集合查找
                        abnormal_available_set = abnormal_wallets & all_wallets_set
                        abnormal_available = [w for w in abnormal_available_set
                                             if w in current_balances and
                                             not wallet_plans[w].get('daily_limit_reached', False) and
                                             wallet_plans[w]['neighbors']]
                    
                    # 计算还需要补充多少正常账户才能达到限制
                    remaining_slots = max_risk_wallets_count - len(risk_wallets_set)
                    abnormal_not_in_set = len([w for w in abnormal_wallets if w not in risk_wallets_set])
                    
                    if abnormal_available and rng.random() < abnormal_wallet_risk_ratio:
                        # 95%概率选择异常账户（is_abnormal=1）
                        source_wallet = rng.choice(abnormal_available)
                        risk_wallets_set.add(source_wallet)
                    elif abnormal_not_in_set < remaining_slots:
                        # 如果剩余的异常账户数量 < 还需要补充的数量，允许正常账户补充
                        # 优化：使用集合差集操作，更快（复用已计算的available_wallets_set）
                        normal_candidates = available_wallets_set - abnormal_wallets - risk_wallets_set - forced_risk_wallets
                        normal_available = [w for w in normal_candidates if wallet_plans[w]['neighbors']]
                        if normal_available and len(risk_wallets_set) < max_risk_wallets_count:
                            source_wallet = rng.choice(normal_available)
                            risk_wallets_set.add(source_wallet)
                            # 如果选择了is_abnormal=0的钱包参与异常交易，将其加入forced_risk_wallets
                            forced_risk_wallets.add(source_wallet)
                        elif abnormal_available:
                            # 如果正常账户不可用，回退到异常账户
                            source_wallet = rng.choice(abnormal_available)
                            risk_wallets_set.add(source_wallet)
                    else:
                        # 如果异常账户还足够，优先使用异常账户
                        if abnormal_available:
                            source_wallet = rng.choice(abnormal_available)
                            risk_wallets_set.add(source_wallet)
                        else:
                            # 如果异常账户不可用，允许正常账户补充（但限制数量）
                            # 优化：使用集合差集操作，更快（复用已计算的available_wallets_set）
                            normal_candidates = available_wallets_set - abnormal_wallets - risk_wallets_set - forced_risk_wallets
                            normal_available = [w for w in normal_candidates if wallet_plans[w]['neighbors']]
                            if normal_available and len(risk_wallets_set) < max_risk_wallets_count:
                                source_wallet = rng.choice(normal_available)
                                risk_wallets_set.add(source_wallet)
                                # 如果选择了is_abnormal=0的钱包参与异常交易，将其加入forced_risk_wallets
                                forced_risk_wallets.add(source_wallet)
                
                if not source_wallet:
                    # 如果没有选择到钱包，且是因为达到限制且没有可用风险钱包，强制转为正常交易
                    if len(risk_wallets_set) >= max_risk_wallets_count:
                        is_risk_tx_decision = False
                        # 继续执行正常交易的逻辑（不continue，让代码继续执行正常交易部分）
                    else:
                        continue
                
                if source_wallet and is_risk_tx_decision:
                    plan = wallet_plans[source_wallet]
                    
                    # 根据钱包类型选择具体的风险交易类型
                    wallet_type = plan['wallet_type']
                    wallet_level = plan['wallet_level']
                    current_risk_ratio = risk_transactions / total_transactions if total_transactions > 0 else 0
                    weights = manager.get_wallet_weights(wallet_type, wallet_level, current_risk_ratio)
                    
                    # 只保留风险交易类型
                    risk_types = [tx for tx in weights.keys() if manager.is_risk_transaction(tx)]
                    if not risk_types:
                        continue
                    
                    risk_weights = [weights[tx] for tx in risk_types]
                    # 过滤无效值
                    valid_risk_types = []
                    valid_risk_weights = []
                    for tx_type, weight in zip(risk_types, risk_weights):
                        if weight is not None and not np.isnan(weight) and not np.isinf(weight) and weight >= 0:
                            valid_risk_types.append(tx_type)
                            valid_risk_weights.append(weight)
                    
                    if not valid_risk_types or sum(valid_risk_weights) == 0:
                        continue
                    
                    # 归一化权重
                    total_weight = sum(valid_risk_weights)
                    if total_weight > 0:
                        transaction_weights = [w / total_weight for w in valid_risk_weights]
                        if any(np.isnan(w) or np.isinf(w) for w in transaction_weights):
                            continue
                        selected_transaction_type = rng.choice(valid_risk_types, p=transaction_weights)
                    else:
                        continue
            else:
                # 正常交易：优先从正常账户中选择
                # 优先选择偏好为'normal'的钱包，避免选择偏好为'risk'的钱包（除非达到混合限制）
                # 排除forced_risk_wallets中的钱包（这些钱包只能参与异常交易）
                # 优化：使用集合差集操作，更快（复用已计算的available_wallets_set）
                normal_candidates_set = available_wallets_set - abnormal_wallets - forced_risk_wallets
                normal_available_all = [w for w in normal_candidates_set if wallet_plans[w]['neighbors']]
                
                abnormal_normal_candidates_set = abnormal_wallets & available_wallets_set
                abnormal_normal_available_all = [w for w in abnormal_normal_candidates_set if wallet_plans[w]['neighbors']]
                
                # 按照偏好分类
                normal_preferred = [w for w in normal_available_all 
                                   if wallet_tx_type_preference.get(w) in ['normal', None]]
                normal_non_preferred = [w for w in normal_available_all 
                                       if wallet_tx_type_preference.get(w) == 'risk']
                
                abnormal_normal_preferred = [w for w in abnormal_normal_available_all 
                                            if wallet_tx_type_preference.get(w) in ['normal', None]]
                abnormal_normal_non_preferred = [w for w in abnormal_normal_available_all 
                                                if wallet_tx_type_preference.get(w) == 'risk']
                
                # 优化：使用缓存的混合钱包数量（已在函数开始时定义）
                
                # 根据配置选择钱包，优先选择偏好匹配的钱包
                if normal_preferred and rng.random() < normal_wallet_normal_ratio:
                    # 95%概率选择正常账户（偏好为'normal'）
                    source_wallet = rng.choice(normal_preferred)
                elif abnormal_normal_preferred and rng.random() < abnormal_wallet_normal_ratio:
                    # 10%概率选择异常账户参与正常交易（偏好为'normal'）
                    source_wallet = rng.choice(abnormal_normal_preferred)
                elif normal_preferred:
                    source_wallet = rng.choice(normal_preferred)
                elif abnormal_normal_preferred:
                    source_wallet = rng.choice(abnormal_normal_preferred)
                elif current_mixed_count < max_mixed_wallets_count:
                    # 如果偏好匹配的钱包不可用，且未达到混合限制，允许选择偏好不匹配的钱包
                    if normal_non_preferred:
                        source_wallet = rng.choice(normal_non_preferred)
                    elif abnormal_normal_non_preferred:
                        source_wallet = rng.choice(abnormal_normal_non_preferred)
                    elif normal_available_all:
                        source_wallet = rng.choice(normal_available_all)
                    elif abnormal_normal_available_all:
                        source_wallet = rng.choice(abnormal_normal_available_all)
                
                if not source_wallet:
                    continue
                    
                plan = wallet_plans[source_wallet]
                
                # 检查是否到了发工资时间（只有对公钱包wallet_type=='1'才能发工资）
                wallet_type = plan['wallet_type']
                if (str(wallet_type) == '1' and 
                    plan['next_salary_time'] and 
                    current_time >= plan['next_salary_time'] and 
                    plan['can_generate_salary']):
                    selected_transaction_type = 'regular_large_low_freq'
                else:
                    # 根据钱包类型选择具体的正常交易类型
                    wallet_level = plan['wallet_level']
                    current_risk_ratio = risk_transactions / total_transactions if total_transactions > 0 else 0
                    weights = manager.get_wallet_weights(wallet_type, wallet_level, current_risk_ratio)
                    
                    # 只保留正常交易类型
                    normal_types = [tx for tx in weights.keys() if not manager.is_risk_transaction(tx)]
                    
                    # 对于regular_large_low_freq，只有对公钱包（wallet_type=='1'）才能生成
                    # 如果钱包不是对公钱包，从normal_types中移除regular_large_low_freq
                    if str(wallet_type) != '1':
                        normal_types = [tx for tx in normal_types if tx != 'regular_large_low_freq']
                    
                    # 对于regular_large_low_freq，如果还没有设置next_salary_time，增加初始触发概率
                    if 'regular_large_low_freq' in weights and not plan.get('next_salary_time'):
                        original_weight = weights.get('regular_large_low_freq', 0)
                        if original_weight > 0 and not np.isnan(original_weight) and not np.isinf(original_weight):
                            weights['regular_large_low_freq'] = original_weight * 1.2
                    
                    if not normal_types:
                        continue
                    
                    normal_weights = [weights[tx] for tx in normal_types]
                    # 过滤无效值
                    valid_normal_types = []
                    valid_normal_weights = []
                    for tx_type, weight in zip(normal_types, normal_weights):
                        if weight is not None and not np.isnan(weight) and not np.isinf(weight) and weight >= 0:
                            valid_normal_types.append(tx_type)
                            valid_normal_weights.append(weight)
                    
                    if not valid_normal_types or sum(valid_normal_weights) == 0:
                        continue
                    
                    # 归一化权重
                    total_weight = sum(valid_normal_weights)
                    if total_weight > 0:
                        transaction_weights = [w / total_weight for w in valid_normal_weights]
                        if any(np.isnan(w) or np.isinf(w) for w in transaction_weights):
                            continue
                        selected_transaction_type = rng.choice(valid_normal_types, p=transaction_weights)
                    else:
                        continue
        
        # 只有在没有继续现有motif时才进行这些检查
        if not is_continued_motif_processed:
            # 优化：再次检查钱包限额（在生成交易前）- 直接从 wallet_plans 中获取
            if not source_wallet:
                continue
            # 检查钱包是否已被废弃
            if source_wallet in abandoned_wallets:
                continue
            plan = wallet_plans[source_wallet]
            limits = plan['cached_limits']  # 使用缓存的 limits
            daily_limit = limits.get('daily_limit')
            single_limit = limits.get('single_transaction_limit')
            
            # 优化：直接从 wallet_plans 中获取当前日累计金额，最快的方式
            current_daily_amount = plan['current_daily_amount']
            
            # 如果已达到每日限额，标记钱包并跳过
            if daily_limit is not None and current_daily_amount >= daily_limit:
                plan['daily_limit_reached'] = True
                continue
            
            # 检查钱包是否有邻居
            if not plan['neighbors']:
                continue

            
            # 检查风险交易限制
            if manager.is_risk_transaction(selected_transaction_type):
                if plan['risk_tx_count'] >= manager.get_max_risk_transactions_per_wallet():
                    continue
        else:
            # 对于继续的motif，plan已经在上面设置
            if not source_wallet or not plan:
                continue
        
        if selected_transaction_type == 'regular_large_low_freq':
            # 只有对公钱包（wallet_type=='1'）才能生成regular_large_low_freq交易
            if str(plan.get('wallet_type')) != '1':
                # 如果不是对公钱包，跳过此交易类型
                continue
            
            # 设置预留模式
            generator._reservation_mode = True
            
            # 生成预留交易
            result = generator.generate_transaction(
                selected_transaction_type, g, source_wallet, current_balances,
                current_time, tx_id, max_frac_per_tx, min_abs_amount, rng,
                wallet_to_attrs, attr_headers, None, wallet_tx_type_preference
            )
            
            # 重置预留模式
            generator._reservation_mode = False
            
            if result is None:
                # 生成失败，尝试其他正常交易类型
                if source_wallet and plan:
                    wallet_type = plan['wallet_type']
                    wallet_level = plan['wallet_level']
                    current_risk_ratio = risk_transactions / total_transactions if total_transactions > 0 else 0
                    weights = manager.get_wallet_weights(wallet_type, wallet_level, current_risk_ratio)
                    
                    # 获取所有可用的正常交易类型（排除已失败的 regular_large_low_freq）
                    normal_types = [tx for tx in weights.keys() 
                                   if not manager.is_risk_transaction(tx) and tx != 'regular_large_low_freq']
                    
                    # 按权重排序，尝试其他类型
                    available_types_with_weights = [(tx_type, weights.get(tx_type, 0)) 
                                                   for tx_type in normal_types 
                                                   if weights.get(tx_type, 0) > 0]
                    available_types_with_weights.sort(key=lambda x: x[1], reverse=True)
                    
                    # 尝试最多3个其他交易类型
                    max_retries = min(3, len(available_types_with_weights))
                    for i in range(max_retries):
                        retry_type, retry_weight = available_types_with_weights[i]
                        retry_result = generator.generate_transaction(
                            retry_type, g, source_wallet, current_balances,
                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng,
                            wallet_to_attrs, attr_headers, None, wallet_tx_type_preference
                        )
                        if retry_result is not None:
                            result = retry_result
                            selected_transaction_type = retry_type
                            # 注意：这里不设置预留模式，因为其他类型不需要预留
                            break
                
                # 如果所有尝试都失败，才跳过
                if result is None:
                    continue
                
            new_transactions, tx_id, current_time = result
            
            # 将预留交易存储到钱包计划中
            plan['reservation_transactions'].extend(new_transactions)
            
            # 设置下次发工资时间
            if new_transactions:
                first_tx = new_transactions[0]
                salary_interval_days = first_tx.get('reservation_data', {}).get('salary_interval_days')
                if salary_interval_days:
                    plan['last_salary_time'] = current_time
                    plan['next_salary_time'] = current_time + timedelta(days=int(salary_interval_days))
                    plan['salary_interval_days'] = salary_interval_days
                    plan['base_salary'] = first_tx.get('reservation_data', {}).get('base_salary')
                    plan['can_generate_salary'] = False  # 暂时不能再次生成发工资交易
            
            continue
        
        # 其他交易类型的正常处理（只有在没有继续现有motif时才生成新motif）
        if not new_transactions:
            result = generator.generate_transaction(
                selected_transaction_type, g, source_wallet, current_balances,
                current_time, tx_id, max_frac_per_tx, min_abs_amount, rng,
                wallet_to_attrs, attr_headers, None, wallet_tx_type_preference
            )
            
            if result is None:
                # 生成失败，尝试其他交易类型
                if source_wallet and plan:
                    wallet_type = plan['wallet_type']
                    wallet_level = plan['wallet_level']
                    current_risk_ratio = risk_transactions / total_transactions if total_transactions > 0 else 0
                    weights = manager.get_wallet_weights(wallet_type, wallet_level, current_risk_ratio)
                    
                    # 判断当前是风险交易还是正常交易
                    is_risk = manager.is_risk_transaction(selected_transaction_type)
                    
                    # 获取所有可用的交易类型（排除已失败的）
                    all_available_types = list(weights.keys())
                    if selected_transaction_type in all_available_types:
                        all_available_types.remove(selected_transaction_type)
                    
                    # 根据当前决策类型，选择相同类型的其他交易
                    if is_risk:
                        # 如果是风险交易失败，尝试其他风险交易类型
                        retry_types = [tx for tx in all_available_types if manager.is_risk_transaction(tx)]
                    else:
                        # 如果是正常交易失败，尝试其他正常交易类型
                        retry_types = [tx for tx in all_available_types if not manager.is_risk_transaction(tx)]
                    
                    # 按权重排序，尝试其他类型
                    available_types_with_weights = [(tx_type, weights.get(tx_type, 0)) 
                                                   for tx_type in retry_types 
                                                   if weights.get(tx_type, 0) > 0]
                    available_types_with_weights.sort(key=lambda x: x[1], reverse=True)
                    
                    # 尝试最多3个其他交易类型
                    max_retries = min(3, len(available_types_with_weights))
                    for i in range(max_retries):
                        retry_type, retry_weight = available_types_with_weights[i]
                        retry_result = generator.generate_transaction(
                            retry_type, g, source_wallet, current_balances,
                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng,
                            wallet_to_attrs, attr_headers, None, wallet_tx_type_preference
                        )
                        if retry_result is not None:
                            result = retry_result
                            selected_transaction_type = retry_type
                            break
                
                # 如果所有尝试都失败，才跳过
                if result is None:
                    continue
                
            all_motif_transactions, tx_id, current_time = result
            
            # DEBUG: 记录生成器返回的交易
            if all_motif_transactions:
                motif_id_gen = all_motif_transactions[0].get("motif_id") if all_motif_transactions else None
                transaction_mode_gen = all_motif_transactions[0].get("transaction_mode") if all_motif_transactions else None
                transaction_motif_gen = all_motif_transactions[0].get("risk_type", "unknown") if all_motif_transactions else None
                amounts_before_fix = [tx.get("amount") for tx in all_motif_transactions]
            
            # 优化：验证每个钱包的第一次交易时间是否晚于开户时间
            valid_transactions = []
            for tx in all_motif_transactions:
                src = tx.get("src")
                dst = tx.get("dst")
                tx_timestamp_str = tx.get("timestamp")
                
                if not tx_timestamp_str:
                    continue  # 如果没有时间戳，跳过
                
                try:
                    tx_timestamp = datetime.strptime(tx_timestamp_str, "%Y-%m-%d %H:%M:%S")
                except:
                    continue  # 如果时间戳格式错误，跳过
                
                # 检查 src 钱包是否是第一次交易
                if src not in wallet_first_tx_time:
                    # 第一次交易，验证是否晚于开户时间
                    src_open_time = wallet_open_timestamps.get(src)
                    if src_open_time and tx_timestamp < src_open_time:
                        # 交易时间早于开户时间，废弃该钱包
                        abandoned_wallets.add(src)
                        continue  # 跳过这笔交易
                    else:
                        # 通过验证，记录第一次交易时间
                        wallet_first_tx_time[src] = tx_timestamp
                # 如果 src 钱包已被废弃，跳过
                elif src in abandoned_wallets:
                    continue
                
                # 检查 dst 钱包是否是第一次交易
                if dst and dst not in wallet_first_tx_time:
                    # 第一次交易，验证是否晚于开户时间
                    dst_open_time = wallet_open_timestamps.get(dst)
                    if dst_open_time and tx_timestamp < dst_open_time:
                        # 交易时间早于开户时间，废弃该钱包
                        abandoned_wallets.add(dst)
                        continue  # 跳过这笔交易
                    else:
                        # 通过验证，记录第一次交易时间
                        wallet_first_tx_time[dst] = tx_timestamp
                # 如果 dst 钱包已被废弃，跳过
                elif dst and dst in abandoned_wallets:
                    continue
                
                # 通过验证，保留交易
                valid_transactions.append(tx)
            
            all_motif_transactions = valid_transactions
            
            # 修复 merchant_laundering 的所有模式：时间顺序和金额匹配
            if all_motif_transactions and selected_transaction_type == 'merchant_laundering':
                all_motif_transactions = fix_merchant_laundering_many_to_many(
                    all_motif_transactions, source_wallet, current_time, rng, wallet_open_timestamps
                )
            
            # 通用修复：对于所有其他交易类型，在 many_to_many, many_to_one, one_to_many 模式下确保转入 > 转出
            if all_motif_transactions and selected_transaction_type != 'merchant_laundering':
                motif_id_before = all_motif_transactions[0].get("motif_id") if all_motif_transactions else None
                transaction_count_before = len(all_motif_transactions)
                amounts_before = [tx.get("amount") for tx in all_motif_transactions]
                all_motif_transactions = fix_transaction_balance(
                    all_motif_transactions, source_wallet, current_time, rng, wallet_open_timestamps
                )
                motif_id_after = all_motif_transactions[0].get("motif_id") if all_motif_transactions else None
                transaction_count_after = len(all_motif_transactions)
                amounts_after = [tx.get("amount") for tx in all_motif_transactions]
            
            # 在修复时间戳之后，再次验证所有交易的时间戳是否晚于开户时间
            if all_motif_transactions:
                valid_transactions_after_fix = []
                for tx in all_motif_transactions:
                    src = tx.get("src")
                    dst = tx.get("dst")
                    tx_timestamp_str = tx.get("timestamp")
                    
                    if not tx_timestamp_str:
                        continue  # 如果没有时间戳，跳过
                    
                    try:
                        tx_timestamp = datetime.strptime(tx_timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except:
                        continue  # 如果时间戳格式错误，跳过
                    
                    # 检查 src 钱包
                    if src in abandoned_wallets:
                        continue  # 如果已被废弃，跳过
                    
                    if src not in wallet_first_tx_time:
                        # 第一次交易，验证是否晚于开户时间
                        src_open_time = wallet_open_timestamps.get(src)
                        if src_open_time and tx_timestamp < src_open_time:
                            # 交易时间早于开户时间，废弃该钱包
                            abandoned_wallets.add(src)
                            continue  # 跳过这笔交易
                        else:
                            # 通过验证，记录第一次交易时间
                            wallet_first_tx_time[src] = tx_timestamp
                    elif tx_timestamp < wallet_first_tx_time[src]:
                        # 如果新时间戳早于已记录的第一次交易时间，更新为更早的时间
                        # 但需要再次验证是否晚于开户时间
                        src_open_time = wallet_open_timestamps.get(src)
                        if src_open_time and tx_timestamp < src_open_time:
                            # 交易时间早于开户时间，废弃该钱包
                            abandoned_wallets.add(src)
                            continue
                        else:
                            wallet_first_tx_time[src] = tx_timestamp
                    
                    # 检查 dst 钱包
                    if dst:
                        if dst in abandoned_wallets:
                            continue  # 如果已被废弃，跳过
                        
                        if dst not in wallet_first_tx_time:
                            # 第一次交易，验证是否晚于开户时间
                            dst_open_time = wallet_open_timestamps.get(dst)
                            if dst_open_time and tx_timestamp < dst_open_time:
                                # 交易时间早于开户时间，废弃该钱包
                                abandoned_wallets.add(dst)
                                continue  # 跳过这笔交易
                            else:
                                # 通过验证，记录第一次交易时间
                                wallet_first_tx_time[dst] = tx_timestamp
                        elif tx_timestamp < wallet_first_tx_time[dst]:
                            # 如果新时间戳早于已记录的第一次交易时间，更新为更早的时间
                            # 但需要再次验证是否晚于开户时间
                            dst_open_time = wallet_open_timestamps.get(dst)
                            if dst_open_time and tx_timestamp < dst_open_time:
                                # 交易时间早于开户时间，废弃该钱包
                                abandoned_wallets.add(dst)
                                continue
                            else:
                                wallet_first_tx_time[dst] = tx_timestamp
                    
                    # 通过验证，保留交易
                    valid_transactions_after_fix.append(tx)
                
                all_motif_transactions = valid_transactions_after_fix
            
            if all_motif_transactions:
                # 第一笔交易立即使用
                first_tx = all_motif_transactions[0].copy()
                motif_id_first = first_tx.get("motif_id")
                # 对于 merchant_laundering，使用修复后的时间戳，不要覆盖
                if selected_transaction_type == 'merchant_laundering':
                    pass  # 使用修复后的时间戳
                # 对于其他交易类型（many_to_many, many_to_one, one_to_many），也使用修复后的时间戳，不要覆盖
                elif first_tx.get("transaction_mode") in ["many_to_many", "many_to_one", "one_to_many"]:
                    pass  # 使用修复后的时间戳
                else:
                    first_tx['timestamp'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                new_transactions = [first_tx]
                
                # 剩余交易存入队列（如果有）
                if len(all_motif_transactions) > 1:
                    motif_id = first_tx.get("motif_id")
                    if motif_id:
                        remaining_txs = [tx.copy() for tx in all_motif_transactions[1:]]
                        remaining_amounts = [tx.get("amount") for tx in remaining_txs]
                        remaining_details = [(tx.get("tx_id"), tx.get("src"), tx.get("dst"), tx.get("amount"), tx.get("timestamp")) for tx in remaining_txs]
                        pending_motifs[motif_id] = {
                            'transactions': remaining_txs,
                            'current_index': 0,
                            'transaction_type': selected_transaction_type,
                            'source_wallet': source_wallet,
                            'plan': plan
                        }
                        
                        # 初始化 motif 的跟踪信息（仅对 many_to_many, many_to_one, one_to_many 模式）
                        first_tx_mode = first_tx.get("transaction_mode")
                        if first_tx_mode in ["many_to_many", "many_to_one", "one_to_many"] and selected_transaction_type != 'merchant_laundering':
                            # 识别主钱包
                            if first_tx_mode == "one_to_many":
                                main_wallet = source_wallet
                            else:
                                # many_to_many 或 many_to_one：主钱包是 dst 出现次数最多的账户
                                dst_counts = {}
                                for temp_tx in all_motif_transactions:
                                    temp_dst = temp_tx.get("dst")
                                    if temp_dst:
                                        dst_counts[temp_dst] = dst_counts.get(temp_dst, 0) + 1
                                main_wallet = max(dst_counts, key=dst_counts.get) if dst_counts else source_wallet
                            
                            motif_saved_amounts[motif_id] = {
                                'main_wallet': main_wallet,
                                'saved_incoming': 0.0,
                                'saved_outgoing': 0.0
                            }
                            # 注意：不在初始化时设置第一笔交易的金额，而是在保存到CSV时更新（避免重复计算）
        
        # 检查每笔交易的限额（只检查转出限额）
        # 注意：对于继续的motif，限额检查已经在上面完成，这里只检查新生成的motif
        if new_transactions and not is_continued_motif_processed:
            # 获取motif_id（如果存在）
            motif_id = new_transactions[0].get("motif_id") if new_transactions else None
            
            # 对于新生成的motif，需要检查第一笔交易和队列中剩余交易的限额
            # 但为了保持motif完整性，如果第一笔交易超出限额，则拒绝整个motif（包括队列中的交易）
            all_transactions_valid = True
            wallets_exceeded_limit = set()
            current_date = current_time.date()
            
            # 检查第一笔交易的限额
            if new_transactions:
                tx = new_transactions[0]
                src = tx['src']
                dst = tx.get('dst')
                amount = float(tx['amount'])
                
                # 优化：检查钱包是否已被废弃
                if src in abandoned_wallets or (dst and dst in abandoned_wallets):
                    all_transactions_valid = False
                    if motif_id and motif_id in pending_motifs:
                        del pending_motifs[motif_id]
                    new_transactions = []
                elif src in wallet_plans:
                    plan_wallet = wallet_plans[src]
                    limits = plan_wallet['cached_limits']
                    single_limit = limits.get('single_transaction_limit')
                    daily_limit = limits.get('daily_limit')
                    current_daily_amount = plan_wallet['current_daily_amount']
                    
                    # 检查单笔限额
                    if single_limit is not None and amount > single_limit:
                        all_transactions_valid = False
                        wallets_exceeded_limit.add(src)
                        plan_wallet['daily_limit_reached'] = True
                    # 检查日累计限额（只检查第一笔，队列中的交易会在后续时间点检查）
                    elif daily_limit is not None and (current_daily_amount + amount) > daily_limit:
                        all_transactions_valid = False
                        wallets_exceeded_limit.add(src)
                        plan_wallet['daily_limit_reached'] = True
                    # 检查余额
                    elif current_balances.get(src, 0) < amount:
                        all_transactions_valid = False
                    
                    # 如果第一笔交易通过检查，更新余额和限额
                    if all_transactions_valid:
                        current_balances[src] -= amount
                        current_balances[tx['dst']] = current_balances.get(tx['dst'], 0) + amount
                        plan_wallet['current_daily_amount'] += amount
                else:
                    all_transactions_valid = False
            
            # 如果第一笔交易被拒绝，需要从队列中移除该motif的剩余交易
            if not all_transactions_valid:
                if motif_id and motif_id in pending_motifs:
                    del pending_motifs[motif_id]
                new_transactions = []
                if source_wallet and source_wallet in wallets_exceeded_limit:
                    plan['daily_limit_reached'] = True
        
        # 如果所有交易都被拒绝（因为限额），跳过后续处理
        if not new_transactions:
            continue
        
        if new_transactions:
            # 对于风险交易类型的motif，确保异常交易笔数多于正常交易
            # 计算当前motif_id中的风险交易和正常交易数量
            if manager.is_risk_transaction(selected_transaction_type):
                # 获取当前生成交易的motif_id
                motif_id = new_transactions[0].get("motif_id") if new_transactions else None
                if motif_id:
                    # 优化：使用字典缓存 motif_id 的统计信息，避免重复遍历
                    # 统计该motif_id中已有的风险交易和正常交易数量
                    # 优化：只在第一次遇到该 motif_id 时统计，后续使用缓存
                    if motif_id not in motif_stats_cache:
                        existing_risk_count = sum(1 for tx in transactions 
                                                if tx.get("motif_id") == motif_id 
                                                and manager.is_risk_transaction(tx.get("risk_type", "unknown")))
                        existing_normal_count = sum(1 for tx in transactions 
                                                  if tx.get("motif_id") == motif_id 
                                                  and not manager.is_risk_transaction(tx.get("risk_type", "unknown")))
                        motif_stats_cache[motif_id] = (existing_risk_count, existing_normal_count)
                    else:
                        existing_risk_count, existing_normal_count = motif_stats_cache[motif_id]
                        # 更新缓存（因为刚添加了新交易）
                        if manager.is_risk_transaction(selected_transaction_type):
                            existing_risk_count += len(new_transactions)
                        else:
                            existing_normal_count += len(new_transactions)
                        motif_stats_cache[motif_id] = (existing_risk_count, existing_normal_count)
                    
                    # 如果正常交易已经多于或等于风险交易，且当前生成的是风险交易，则优先添加
                    # 这样可以确保风险交易笔数最终多于正常交易
                    if existing_normal_count >= existing_risk_count:
                        # 风险交易motif，确保风险交易笔数更多
                        pass  # 当前生成的就是风险交易，直接添加
            
            transactions.extend(new_transactions)
            total_transactions += len(new_transactions)
            
            # 批量保存到CSV（每1万条保存一次）
            if csv_writer and new_transactions:
                for tx in new_transactions:
                    motif_id_save = tx.get("motif_id")
                    tx_id_save = tx.get("tx_id")
                    tx_amount_save = tx.get("amount")
                    tx_timestamp_save = tx.get("timestamp")
                    tx_mode_save = tx.get("transaction_mode")
                    
                    transaction_motif = tx.get("risk_type", "unknown")
                    is_risk = '0' if transaction_motif in safe_motifs else '1'
                    
                    src_id = tx["src"]
                    dst_id = tx["dst"]
                    # 使用缓存获取钱包信息
                    if src_id not in wallet_info_cache:
                        wallet_info_cache[src_id] = get_wallet_info(src_id, wallet_to_attrs, attr_headers)
                    if dst_id not in wallet_info_cache:
                        wallet_info_cache[dst_id] = get_wallet_info(dst_id, wallet_to_attrs, attr_headers)
                    
                    src_info = wallet_info_cache[src_id]
                    dst_info = wallet_info_cache[dst_id]
                    
                    transaction_buffer.append([
                        tx["tx_id"], tx["timestamp"], tx["src"], tx["dst"], tx["amount"],
                        transaction_motif, tx["motif_id"], tx.get("transaction_mode", ""), is_risk,
                        src_info['bank_account_number'], dst_info['bank_account_number'],
                        src_info['wallet_level'], dst_info['wallet_level']
                    ])
                    
                    # 更新 motif 的已保存金额（仅对 many_to_many, many_to_one, one_to_many 模式，且非 merchant_laundering）
                    motif_id_for_tracking = tx.get("motif_id")
                    transaction_mode_for_tracking = tx.get("transaction_mode")
                    if motif_id_for_tracking and transaction_mode_for_tracking in ["many_to_many", "many_to_one", "one_to_many"]:
                        if motif_id_for_tracking in motif_saved_amounts:
                            main_wallet_tracking = motif_saved_amounts[motif_id_for_tracking].get('main_wallet')
                            if main_wallet_tracking:
                                if tx.get("dst") == main_wallet_tracking:
                                    # 转入交易
                                    motif_saved_amounts[motif_id_for_tracking]['saved_incoming'] += float(tx.get("amount", 0))
                                elif tx.get("src") == main_wallet_tracking:
                                    # 转出交易
                                    motif_saved_amounts[motif_id_for_tracking]['saved_outgoing'] += float(tx.get("amount", 0))
                
                # 当缓冲区达到1万条时，批量写入
                if len(transaction_buffer) >= buffer_size:
                    csv_writer.writerows(transaction_buffer)
                    csv_file.flush()
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已批量写入 {len(transaction_buffer):,} 笔交易到CSV文件 (累计: {total_transactions:,} 笔, 风险: {risk_transactions:,}, 正常: {normal_transactions:,}) {get_process_info()}")
                    transaction_buffer = []
            
            # 更新风险/正常交易计数，并更新钱包交易类型偏好
            is_risk_tx = manager.is_risk_transaction(selected_transaction_type)
            
            # 收集所有参与交易的钱包（src和dst）
            wallets_in_tx = set()
            for tx in new_transactions:
                wallets_in_tx.add(str(tx.get("src", "")))
                wallets_in_tx.add(str(tx.get("dst", "")))
                
                if manager.is_risk_transaction(tx.get("risk_type", "unknown")):
                    risk_transactions += 1
                    plan['risk_tx_count'] += 1
                else:
                    normal_transactions += 1
            
            # 更新所有参与交易的钱包（src和dst）的类型偏好
            for wallet_id in wallets_in_tx:
                if not wallet_id or wallet_id not in g.nodes():
                    continue
                
                # 如果钱包在forced_risk_wallets中，强制设置为'risk'偏好，不允许参与正常交易
                if wallet_id in forced_risk_wallets:
                    wallet_tx_type_preference[wallet_id] = 'risk'
                    continue
                    
                if wallet_id not in wallet_tx_type_preference:
                    # 第一次参与交易，设置偏好
                    wallet_tx_type_preference[wallet_id] = 'risk' if is_risk_tx else 'normal'
                else:
                    # 已参与过交易，检查是否需要更新为'mixed'
                    current_pref = wallet_tx_type_preference[wallet_id]
                    if current_pref == 'risk' and not is_risk_tx:
                        # 从风险转为正常，标记为混合
                        # 优化：使用缓存的混合钱包数量
                        if current_mixed_count < max_mixed_wallets_count:
                            wallet_tx_type_preference[wallet_id] = 'mixed'
                            current_mixed_count += 1
                        # 如果已达到混合限制，保持原偏好
                    elif current_pref == 'normal' and is_risk_tx:
                        # 从正常转为风险，标记为混合
                        # 优化：使用缓存的混合钱包数量
                        if current_mixed_count < max_mixed_wallets_count:
                            wallet_tx_type_preference[wallet_id] = 'mixed'
                            current_mixed_count += 1
                        # 如果已达到混合限制，保持原偏好
                    # 如果已经是'mixed'，保持不变
            
            plan['last_tx_time'] = current_time
            plan['tx_count'] += len(new_transactions)
        
        # 检查是否有到期的预留交易需要处理
        if plan['reservation_transactions']:
            # 按时间排序预留交易
            plan['reservation_transactions'].sort(key=lambda x: x['timestamp'])
            
            # 处理到期的预留交易
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
            expired_reservations = []
            
            for i, tx in enumerate(plan['reservation_transactions']):
                if tx['timestamp'] <= current_time_str:
                    expired_reservations.append(i)
            
            if expired_reservations:
                # 处理到期的预留交易
                expired_txs = [plan['reservation_transactions'][i] for i in expired_reservations]
                processed_transactions = process_reservation_transactions(
                    expired_txs, current_balances, rng, manager
                )
                
                if processed_transactions:
                    transactions.extend(processed_transactions)
                    total_transactions += len(processed_transactions)
                    
                    # 批量保存预留交易到CSV
                    if csv_writer and processed_transactions:
                        for tx in processed_transactions:
                            transaction_motif = tx.get("risk_type", "unknown")
                            is_risk = '0' if transaction_motif in safe_motifs else '1'
                            
                            src_id = tx["src"]
                            dst_id = tx["dst"]
                            # 使用缓存获取钱包信息
                            if src_id not in wallet_info_cache:
                                wallet_info_cache[src_id] = get_wallet_info(src_id, wallet_to_attrs, attr_headers)
                            if dst_id not in wallet_info_cache:
                                wallet_info_cache[dst_id] = get_wallet_info(dst_id, wallet_to_attrs, attr_headers)
                            
                            src_info = wallet_info_cache[src_id]
                            dst_info = wallet_info_cache[dst_id]
                            
                            transaction_buffer.append([
                                tx["tx_id"], tx["timestamp"], tx["src"], tx["dst"], tx["amount"],
                                transaction_motif, tx["motif_id"], tx.get("transaction_mode", ""), is_risk,
                                src_info['bank_account_number'], dst_info['bank_account_number'],
                                src_info['wallet_level'], dst_info['wallet_level']
                            ])
                        
                        # 当缓冲区达到1万条时，批量写入
                        if len(transaction_buffer) >= buffer_size:
                            csv_writer.writerows(transaction_buffer)
                            csv_file.flush()
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已批量写入 {len(transaction_buffer):,} 笔交易到CSV文件 (累计: {total_transactions:,} 笔, 风险: {risk_transactions:,}, 正常: {normal_transactions:,}) {get_process_info()}")
                            transaction_buffer = []
                    
                    # 从预留列表中移除已处理的交易
                    for i in sorted(expired_reservations, reverse=True):
                        del plan['reservation_transactions'][i]
    
    # 保存剩余的缓冲区交易
    if csv_writer and transaction_buffer:
        csv_writer.writerows(transaction_buffer)
        csv_file.flush()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已批量写入剩余 {len(transaction_buffer):,} 笔交易到CSV文件 (累计: {total_transactions:,} 笔) {get_process_info()}")
        transaction_buffer = []
    
    # 关闭CSV文件
    if csv_file:
        csv_file.close()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CSV文件已关闭: {output_file} {get_process_info()}")
    
    
    # 输出统计信息
    generation_elapsed = time.time() - generation_start_time
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 交易生成完成, 耗时 {generation_elapsed:.2f} 秒 ({generation_elapsed/60:.2f} 分钟) {get_process_info()}")
    print(f"\n=== 交易生成统计 ===")
    print(f"总交易数: {total_transactions}")
    print(f"正常交易数: {normal_transactions}")
    print(f"风险交易数: {risk_transactions}")
    
    if total_transactions > 0:
        actual_risk_ratio = risk_transactions / total_transactions
        print(f"实际风险交易比例: {actual_risk_ratio:.1%}")
        print(f"目标风险交易比例: {manager.get_risk_ratio():.1%}")
    
    # 统计异常账户参与风险交易的情况
    print(f"\n=== 异常账户参与情况统计 ===")
    risk_txs_with_abnormal = 0
    risk_txs_with_normal = 0
    for tx in transactions:
        if manager.is_risk_transaction(tx.get("risk_type", "unknown")):
            src = tx.get("src", "")
            if src in abnormal_wallets:
                risk_txs_with_abnormal += 1
            else:
                risk_txs_with_normal += 1
    
    if risk_transactions > 0:
        abnormal_participation_ratio = risk_txs_with_abnormal / risk_transactions
        print(f"风险交易总数: {risk_transactions}")
        print(f"异常账户参与的风险交易: {risk_txs_with_abnormal} ({abnormal_participation_ratio:.1%})")
        print(f"正常账户参与的风险交易: {risk_txs_with_normal} ({1-abnormal_participation_ratio:.1%})")
    
    # 统计异常账户参与交易的总体情况
    abnormal_total_txs = sum(1 for tx in transactions if tx.get("src", "") in abnormal_wallets)
    if abnormal_total_txs > 0:
        abnormal_risk_in_abnormal = sum(1 for tx in transactions 
                                       if tx.get("src", "") in abnormal_wallets and 
                                       manager.is_risk_transaction(tx.get("risk_type", "unknown")))
        print(f"\n异常账户总交易数: {abnormal_total_txs}")
        print(f"异常账户风险交易数: {abnormal_risk_in_abnormal} ({abnormal_risk_in_abnormal/abnormal_total_txs:.1%})")
    
    # 统计参与风险交易的钱包情况
    risk_wallets_abnormal = len([w for w in risk_wallets_set if w in abnormal_wallets])
    risk_wallets_normal = len([w for w in risk_wallets_set if w not in abnormal_wallets])
    print(f"\n=== 参与风险交易的钱包统计 ===")
    print(f"总参与风险交易的钱包数: {len(risk_wallets_set):,} (限制: {max_risk_wallets_count:,})")
    print(f"  - 异常账户（is_abnormal=1）: {risk_wallets_abnormal:,} 个 ({risk_wallets_abnormal/len(risk_wallets_set)*100:.1f}%)")
    print(f"  - 正常账户（is_abnormal=0）: {risk_wallets_normal:,} 个 ({risk_wallets_normal/len(risk_wallets_set)*100:.1f}%)")
    print(f"异常账户参与率: {risk_wallets_abnormal/len(abnormal_wallets)*100:.1f}% ({risk_wallets_abnormal}/{len(abnormal_wallets)})")
    if len(risk_wallets_set) > 0:
        print(f"参与风险交易的钱包占比: {len(risk_wallets_set)/len(g.nodes())*100:.2f}% ({len(risk_wallets_set)}/{len(g.nodes())})")
    
    # 统计forced_risk_wallets（原本is_abnormal=0，但被选中参与异常交易的钱包）
    print(f"\n=== forced_risk_wallets 统计 ===")
    print(f"forced_risk_wallets数量: {len(forced_risk_wallets):,}")
    print(f"  - 这些钱包原本是is_abnormal=0，但被选中参与异常交易后，只能参与异常交易")
    print(f"  - 这些钱包已被排除在正常交易选择之外")
    
    # 统计混合钱包情况
    mixed_wallets_count = len([w for w in wallet_tx_type_preference.values() if w == 'mixed'])
    print(f"\n=== 混合钱包统计 ===")
    print(f"混合钱包数量: {mixed_wallets_count:,} (限制: {max_mixed_wallets_count:,}, {max_mixed_wallets_ratio*100:.1f}%)")
    print(f"  - 这些钱包既参与异常交易又参与正常交易")
    if len(wallet_tx_type_preference) > 0:
        risk_only_count = len([w for w in wallet_tx_type_preference.values() if w == 'risk'])
        normal_only_count = len([w for w in wallet_tx_type_preference.values() if w == 'normal'])
        print(f"只参与异常交易的钱包: {risk_only_count:,}")
        print(f"只参与正常交易的钱包: {normal_only_count:,}")
    
    # 按交易类型统计
    print(f"\n=== 各类型交易统计 ===")
    type_counts = {}
    for tx in transactions:
        transaction_type = tx.get("risk_type", "unknown")
        type_counts[transaction_type] = type_counts.get(transaction_type, 0) + 1
    
    for transaction_type, count in type_counts.items():
        percentage = (count / total_transactions * 100) if total_transactions > 0 else 0
        print(f"{transaction_type}: {count}笔 ({percentage:.1f}%)")
    
    # 按交易模式统计
    print(f"\n=== 各交易模式统计 ===")
    mode_counts = {}
    for tx in transactions:
        transaction_mode = tx.get("transaction_mode", "unknown")
        mode_counts[transaction_mode] = mode_counts.get(transaction_mode, 0) + 1
    
    for transaction_mode, count in mode_counts.items():
        percentage = (count / total_transactions * 100) if total_transactions > 0 else 0
        print(f"{transaction_mode}: {count}笔 ({percentage:.1f}%)")
    
    # 按风险类型和交易模式组合统计
    print(f"\n=== 风险类型+交易模式组合统计 ===")
    risk_mode_counts = {}
    for tx in transactions:
        risk_type = tx.get("risk_type", "unknown")
        transaction_mode = tx.get("transaction_mode", "unknown")
        key = f"{risk_type}_{transaction_mode}"
        risk_mode_counts[key] = risk_mode_counts.get(key, 0) + 1
    
    for key, count in risk_mode_counts.items():
        percentage = (count / total_transactions * 100) if total_transactions > 0 else 0
        print(f"{key}: {count}笔 ({percentage:.1f}%)")
    
    # 按motif_id为单位计算异常占比（判断每个motif_id是异常的还是正常的）
    print(f"\n=== 按motif_id为单位的异常/正常占比统计 ===")
    motif_classification = {}  # {motif_id: 'risk' or 'normal'}
    
    for tx in transactions:
        motif_id = tx.get("motif_id", "unknown")
        is_risk_tx = manager.is_risk_transaction(tx.get("risk_type", "unknown"))
        
        if motif_id not in motif_classification:
            # 根据该motif_id的第一笔交易类型判断该motif是否为异常
            motif_classification[motif_id] = 'risk' if is_risk_tx else 'normal'
        else:
            # 如果同一个motif_id中既有异常又有正常交易，以主要类型为准
            # 如果当前交易是异常类型，且之前标记为正常，则改为异常（优先异常）
            if is_risk_tx and motif_classification[motif_id] == 'normal':
                motif_classification[motif_id] = 'risk'
    
    # 统计信息
    total_motifs = len(motif_classification)
    risk_motifs = sum(1 for classification in motif_classification.values() if classification == 'risk')
    normal_motifs = total_motifs - risk_motifs
    
    print(f"总motif数: {total_motifs}")
    print(f"异常motif数: {risk_motifs} ({risk_motifs/total_motifs*100:.1f}%)")
    print(f"正常motif数: {normal_motifs} ({normal_motifs/total_motifs*100:.1f}%)")
    
    # 显示前10个motif的详细统计
    print(f"\n前10个motif的详细统计:")
    sorted_motifs = sorted(motif_classification.items(), key=lambda x: x[0])[:10]
    for motif_id, classification in sorted_motifs:
        # 统计该motif_id的交易笔数
        motif_tx_count = sum(1 for tx in transactions if tx.get("motif_id") == motif_id)
        motif_tx_types = set(tx.get("risk_type", "unknown") for tx in transactions if tx.get("motif_id") == motif_id)
        print(f"  motif_id={motif_id}: {classification} motif, {motif_tx_count}笔交易, 类型={', '.join(list(motif_tx_types)[:3])}")
    
    return transactions

def process_reservation_transactions(transactions: List[Dict], current_balances: Dict[str, float], 
                                   rng, manager) -> List[Dict]:
    """处理预留交易，填充金额信息"""
    processed_transactions = []
    
    for tx in transactions:
        if tx.get("is_reservation", False):
            # 获取预留数据
            reservation_data = tx.get("reservation_data", {})
            base_salary = reservation_data.get("base_salary", 50)
            amount_range = reservation_data.get("amount_range", {"min": 10, "max": 100})
            min_abs_amount = reservation_data.get("min_abs_amount", 0.01)
            salary_variation = reservation_data.get("salary_variation", 0.1)
            
            # 生成工资金额（±10%变化）
            variation = rng.uniform(1 - salary_variation, 1 + salary_variation)
            salary_amount = base_salary * variation
            salary_amount = max(salary_amount, min_abs_amount)
            
            # 检查余额
            if current_balances[tx["src"]] >= salary_amount:
                current_balances[tx["src"]] -= salary_amount
                current_balances[tx["dst"]] += salary_amount
                
                # 更新交易记录
                tx["amount"] = round(salary_amount, 2)
                del tx["is_reservation"]
                del tx["reservation_data"]
                processed_transactions.append(tx)
            else:
                # 余额不足，跳过这笔交易
                # print(f"警告：钱包 {tx['src']} 余额不足，跳过交易 {tx['motif_id']}")
                continue
        else:
            # 非预留交易，直接添加
            processed_transactions.append(tx)
    
    return processed_transactions

def get_wallet_info(wallet_id: str, wallet_to_attrs: Dict[str, List[str]], attr_headers: List[str]) -> Dict[str, str]:
    """获取钱包的银行账户号和等级信息"""
    wallet_attrs = wallet_to_attrs.get(wallet_id, [])
    
    # 初始化返回字典
    wallet_info = {
        'bank_account_number': '',
        'wallet_level': ''
    }
    
    # 如果没有找到该钱包的信息，返回空值
    if not wallet_attrs:
        return wallet_info
    
    # 确保有足够的列（注意：由于去掉了wallet_id，所以只有13列而不是14列）
    if len(wallet_attrs) < 13:
        return wallet_info
    
    # 查找银行账户号（在原CSV中是第11列索引10，去掉wallet_id后变成索引9）
    bank_account_value = wallet_attrs[9] if len(wallet_attrs) > 9 else ''
    wallet_info['bank_account_number'] = bank_account_value if bank_account_value else ''
    
    # 查找钱包等级（在原CSV中是第3列索引2，去掉wallet_id后变成索引1）
    wallet_level_value = wallet_attrs[1] if len(wallet_attrs) > 1 else ''
    wallet_info['wallet_level'] = wallet_level_value if wallet_level_value else ''
    
    return wallet_info

def generate_random_mac() -> str:
    """生成随机MAC地址"""
    return ":".join(f"{np.random.randint(0, 256):02X}" for _ in range(6))

def generate_random_ip() -> str:
    """生成随机IP地址"""
    return f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 254)}"

def build_account_device_ip_map(accounts_df: pd.DataFrame) -> dict:
    """从账户数据构建设备/IP映射"""
    mapping = {}
    for _, row in accounts_df.iterrows():
        mapping[row["wallet_id"]] = {
            "device": row.get("open_device", None),
            "ip": row.get("open_ip", None),
        }
    return mapping

def add_device_ip_columns(transactions_df: pd.DataFrame, account_map: dict) -> pd.DataFrame:
    """为交易数据添加设备/IP列（优化：使用向量化操作）"""
    print(f"  开始添加设备/IP列，共 {len(transactions_df):,} 行...")
    step_start = time.time()
    
    # 准备列
    if "src_device" not in transactions_df.columns:
        transactions_df["src_device"] = None
    if "src_ip" not in transactions_df.columns:
        transactions_df["src_ip"] = None
    if "dst_device" not in transactions_df.columns:
        transactions_df["dst_device"] = None
    if "dst_ip" not in transactions_df.columns:
        transactions_df["dst_ip"] = None

    # 优化：使用 map 操作，比 apply 更快
    # 先创建映射字典（只包含有值的）
    device_map = {k: v.get("device") for k, v in account_map.items() if v.get("device") and not pd.isna(v.get("device"))}
    ip_map = {k: v.get("ip") for k, v in account_map.items() if v.get("ip") and not pd.isna(v.get("ip"))}
    
    # 使用 map 填充已有的值（比 apply 快很多）
    transactions_df["src_device"] = transactions_df["src"].map(device_map)
    transactions_df["src_ip"] = transactions_df["src"].map(ip_map)
    transactions_df["dst_device"] = transactions_df["dst"].map(device_map)
    transactions_df["dst_ip"] = transactions_df["dst"].map(ip_map)
    
    # 为缺失的值生成随机值（优化：批量生成）
    missing_src_device_mask = transactions_df["src_device"].isna()
    missing_src_ip_mask = transactions_df["src_ip"].isna()
    missing_dst_device_mask = transactions_df["dst_device"].isna()
    missing_dst_ip_mask = transactions_df["dst_ip"].isna()
    
    missing_src_device_count = missing_src_device_mask.sum()
    missing_src_ip_count = missing_src_ip_mask.sum()
    missing_dst_device_count = missing_dst_device_mask.sum()
    missing_dst_ip_count = missing_dst_ip_mask.sum()
    
    if missing_src_device_count > 0:
        transactions_df.loc[missing_src_device_mask, "src_device"] = [
            generate_random_mac() for _ in range(missing_src_device_count)
        ]
    if missing_src_ip_count > 0:
        transactions_df.loc[missing_src_ip_mask, "src_ip"] = [
            generate_random_ip() for _ in range(missing_src_ip_count)
        ]
    if missing_dst_device_count > 0:
        transactions_df.loc[missing_dst_device_mask, "dst_device"] = [
            generate_random_mac() for _ in range(missing_dst_device_count)
        ]
    if missing_dst_ip_count > 0:
        transactions_df.loc[missing_dst_ip_mask, "dst_ip"] = [
            generate_random_ip() for _ in range(missing_dst_ip_count)
        ]
    
    step_elapsed = time.time() - step_start
    print(f"  设备/IP列添加完成，耗时 {step_elapsed:.2f} 秒")

    return transactions_df

def apply_victim_pattern(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """应用受害者模式：在异常交易中随机选择上游账户作为受害者，新增 is_src_victim 列"""
    import numpy as np
    
    # 初始化 is_src_victim 列
    if "is_src_victim" not in transactions_df.columns:
        transactions_df["is_src_victim"] = 0
    
    # 确保is_risk列是数值类型
    if transactions_df["is_risk"].dtype == 'object':
        transactions_df["is_risk"] = transactions_df["is_risk"].astype(str).str.strip()
        risk_df = transactions_df[(transactions_df["is_risk"] == '1') | (transactions_df["is_risk"] == 1)]
    else:
        risk_df = transactions_df[transactions_df["is_risk"] == 1]
    print(f"Total risk transactions: {len(risk_df)}")
    
    # 排除 merchant_laundering 类型的交易
    risk_df = risk_df[risk_df["transaction_motif"] != "merchant_laundering"]
    print(f"Risk transactions after excluding merchant_laundering: {len(risk_df)}")
    
    # 筛选支持的 transaction_mode: fan_out, one_to_many, many_to_many, many_to_one
    supported_modes = ['fan_out', 'one_to_many', 'many_to_many', 'many_to_one']
    risk_df = risk_df[risk_df['transaction_mode'].isin(supported_modes)]
    print(f"Risk transactions with supported modes ({supported_modes}): {len(risk_df)}")
    
    # 筛选金额 >= 100 的风险交易
    MIN_AMOUNT = 100
    large_risk = risk_df[risk_df["amount"] >= MIN_AMOUNT]
    print(f"Risk transactions with amount >= {MIN_AMOUNT}: {len(large_risk)}")
    
    if len(large_risk) < 10:
        print(f"Not enough large-amount risk transactions, using all risk transactions")
        large_risk = risk_df
    
    # 按 motif_id 分组，选择每个 motif_id 中的上游账户作为受害者
    selected_txs = []
    selected_accounts_per_motif = {}
    
    # 设置随机种子以保持一致性
    np.random.seed(1243)
    
    for motif_id in large_risk['motif_id'].unique():
        motif_txs = large_risk[large_risk['motif_id'] == motif_id]
        
        if len(motif_txs) == 0:
            continue
        
        # 检查是否为 many_to_many 模式
        transaction_mode = motif_txs['transaction_mode'].iloc[0] if len(motif_txs) > 0 else None
        
        if transaction_mode == 'many_to_many':
            # 对于 many_to_many 模式，需要识别主钱包并排除它
            # 主钱包是 dst 出现次数最多的钱包
            dst_counts = motif_txs['dst'].value_counts()
            if len(dst_counts) > 0:
                main_wallet = dst_counts.index[0]  # 出现次数最多的 dst 就是主钱包
                
                # 只选择转入主钱包的交易中的 src 账户（排除主钱包本身）
                incoming_txs = motif_txs[motif_txs['dst'] == main_wallet]
                upstream_accounts = incoming_txs['src'].unique()
                
                # 排除主钱包
                upstream_accounts = [acc for acc in upstream_accounts if acc != main_wallet]
            else:
                upstream_accounts = []
        else:
            # 对于其他模式（fan_out, one_to_many, many_to_one），直接使用所有 src
            upstream_accounts = motif_txs['src'].unique()
            dst_counts = None  # 其他模式不需要 dst_counts
        
        if len(upstream_accounts) > 0:
            # 随机选择 1-2 个上游账户作为受害者（根据 motif 大小决定）
            num_victims = min(2, len(upstream_accounts))
            if len(upstream_accounts) > 5:
                num_victims = min(3, len(upstream_accounts))
            
            selected_accounts = np.random.choice(upstream_accounts, size=num_victims, replace=False)
            
            # 为每个选中的账户选择一笔交易（金额最大的）
            # 对于 many_to_many 模式，只从转入主钱包的交易中选择
            if transaction_mode == 'many_to_many' and dst_counts is not None and len(dst_counts) > 0:
                main_wallet = dst_counts.index[0]
                incoming_txs = motif_txs[motif_txs['dst'] == main_wallet]
                for account in selected_accounts:
                    account_txs = incoming_txs[incoming_txs['src'] == account]
                    if len(account_txs) > 0:
                        best_tx = account_txs.sort_values('amount', ascending=False).iloc[0]
                        selected_txs.append(best_tx.name)
            else:
                for account in selected_accounts:
                    account_txs = motif_txs[motif_txs['src'] == account]
                    if len(account_txs) > 0:
                        best_tx = account_txs.sort_values('amount', ascending=False).iloc[0]
                        selected_txs.append(best_tx.name)
            
            if len(selected_accounts) > 0:
                selected_accounts_per_motif[motif_id] = selected_accounts.tolist()
    
    print(f"Found {len(selected_txs)} upstream victim transactions from {len(selected_accounts_per_motif)} unique motifs")
    
    # 如果还不够，从剩余的 motif 中继续选择
    target_count = 50  # 目标受害者交易数量
    if len(selected_txs) < target_count:
        processed_motifs = set(selected_accounts_per_motif.keys())
        remaining_motif_ids = [mid for mid in large_risk['motif_id'].unique() if mid not in processed_motifs]
        np.random.shuffle(remaining_motif_ids)  # 随机打乱
        
        for motif_id in remaining_motif_ids:
            if len(selected_txs) >= target_count:
                break
            
            motif_txs = large_risk[large_risk['motif_id'] == motif_id]
            
            if len(motif_txs) == 0:
                continue
            
            # 检查是否为 many_to_many 模式
            transaction_mode = motif_txs['transaction_mode'].iloc[0] if len(motif_txs) > 0 else None
            
            if transaction_mode == 'many_to_many':
                # 对于 many_to_many 模式，需要识别主钱包并排除它
                dst_counts = motif_txs['dst'].value_counts()
                if len(dst_counts) > 0:
                    main_wallet = dst_counts.index[0]
                    incoming_txs = motif_txs[motif_txs['dst'] == main_wallet]
                    upstream_accounts = [acc for acc in incoming_txs['src'].unique() if acc != main_wallet]
                else:
                    upstream_accounts = []
            else:
                upstream_accounts = motif_txs['src'].unique()
                dst_counts = None  # 其他模式不需要 dst_counts
            
            if len(upstream_accounts) > 0:
                num_victims = min(1, len(upstream_accounts))
                selected_accounts = np.random.choice(upstream_accounts, size=num_victims, replace=False)
                
                # 对于 many_to_many 模式，只从转入主钱包的交易中选择
                if transaction_mode == 'many_to_many' and dst_counts is not None and len(dst_counts) > 0:
                    main_wallet = dst_counts.index[0]
                    incoming_txs = motif_txs[motif_txs['dst'] == main_wallet]
                    for account in selected_accounts:
                        account_txs = incoming_txs[incoming_txs['src'] == account]
                        if len(account_txs) > 0:
                            best_tx = account_txs.sort_values('amount', ascending=False).iloc[0]
                            selected_txs.append(best_tx.name)
                else:
                    for account in selected_accounts:
                        account_txs = motif_txs[motif_txs['src'] == account]
                        if len(account_txs) > 0:
                            best_tx = account_txs.sort_values('amount', ascending=False).iloc[0]
                            selected_txs.append(best_tx.name)
    
    victim_idx = pd.Index(selected_txs)
    print(f"Selected {len(victim_idx)} risk transactions as victims")
    
    # 获取所有受害者账户（src账户）
    victim_accounts = set()
    for idx in victim_idx:
        victim_accounts.add(transactions_df.at[idx, 'src'])
    
    print(f"Victim accounts (src): {len(victim_accounts)}")
    
    # 标记所有涉及这些受害者账户的交易（is_src_victim = 1）
    # 对于 fan_out 等模式，同一个 src 的所有交易都应该标记为 1
    # 对于 many_to_many 模式，只标记转入主钱包的交易（排除主钱包转出的交易）
    for account in victim_accounts:
        # 找到所有以该账户为 src 的异常交易
        account_txs = transactions_df[
            (transactions_df['src'] == account) & 
            (transactions_df['is_risk'] == 1) &
            (transactions_df['transaction_mode'].isin(supported_modes)) &
            (transactions_df['transaction_motif'] != 'merchant_laundering')
        ]
        
        if len(account_txs) > 0:
            # 对于 many_to_many 模式，需要特殊处理
            many_to_many_txs = account_txs[account_txs['transaction_mode'] == 'many_to_many']
            other_txs = account_txs[account_txs['transaction_mode'] != 'many_to_many']
            
            # 对于其他模式，标记所有交易
            if len(other_txs) > 0:
                transactions_df.loc[other_txs.index, 'is_src_victim'] = 1
            
            # 对于 many_to_many 模式，只标记转入主钱包的交易
            if len(many_to_many_txs) > 0:
                # 按 motif_id 分组处理
                for motif_id in many_to_many_txs['motif_id'].unique():
                    motif_m2m_txs = many_to_many_txs[many_to_many_txs['motif_id'] == motif_id]
                    
                    # 识别主钱包（dst 出现次数最多的）
                    dst_counts = motif_m2m_txs['dst'].value_counts()
                    if len(dst_counts) > 0:
                        main_wallet = dst_counts.index[0]
                        
                        # 只标记转入主钱包的交易（dst == main_wallet）
                        incoming_tx_indices = motif_m2m_txs[motif_m2m_txs['dst'] == main_wallet].index
                        if len(incoming_tx_indices) > 0:
                            transactions_df.loc[incoming_tx_indices, 'is_src_victim'] = 1
    
    # 20% 诱导（不变），80% 盗刷（更换设备/IP）
    # 对于同一个受害者账户的所有交易，使用相同的随机决定（要么都改，要么都不改）
    np.random.seed(1243)  # 保持一致性
    for account in victim_accounts:
        # 决定该账户是否更换设备/IP（20%概率不换，80%概率换）
        should_change = np.random.random() >= 0.2
        
        if should_change:
            # 为该账户生成新的设备和IP
            new_device = generate_random_mac()
            new_ip = generate_random_ip()
            
            # 更新该账户所有标记为受害者的交易的设备/IP
            account_victim_txs = transactions_df[
                (transactions_df['src'] == account) & 
                (transactions_df['is_src_victim'] == 1)
            ]
            transactions_df.loc[account_victim_txs.index, 'src_device'] = new_device
            transactions_df.loc[account_victim_txs.index, 'src_ip'] = new_ip
    
    print(f"Victim statistics:")
    print(f"  Total transactions with is_src_victim=1: {(transactions_df['is_src_victim'] == 1).sum()}")
    print(f"  Unique victim accounts: {len(victim_accounts)}")

    return transactions_df

def fix_duplicate_motif_ids(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """修复重复使用的motif_id，确保每个motif_id只对应一个transaction_motif"""
    print("\n=== Fixing duplicate motif_id usage ===")
    
    # 步骤1：合并motif_id后缀（如"123-1", "123-2" -> "123"）
    print("Step 1: Merging motif_id suffixes (e.g., '123-1', '123-2' -> '123')...")
    
    def merge_motif_id_suffix(motif_id):
        """移除motif_id的后缀部分（如'-1', '-2'等）"""
        if pd.isna(motif_id):
            return motif_id
        motif_str = str(motif_id)
        # 如果包含"-数字"后缀，则移除
        if '-' in motif_str:
            # 尝试分割，只保留主部分
            parts = motif_str.split('-')
            # 检查最后一部分是否为数字
            if len(parts) > 1 and parts[-1].isdigit():
                return '-'.join(parts[:-1])
        return motif_str
    
    transactions_df['motif_id_original'] = transactions_df['motif_id'].copy()
    transactions_df['motif_id'] = transactions_df['motif_id'].apply(merge_motif_id_suffix)
    
    # 统计合并情况
    merged_count = (transactions_df['motif_id_original'] != transactions_df['motif_id']).sum()
    unique_original = transactions_df['motif_id_original'].nunique()
    unique_merged = transactions_df['motif_id'].nunique()
    print(f"  Merged {merged_count} motif_id entries")
    print(f"  Unique motif_ids: {unique_original} -> {unique_merged}")
    
    # 步骤2：检查并修复重复的motif_id
    print("\nStep 2: Checking for duplicate motif_id usage across different transaction_motif...")
    
    # 找出每个motif_id对应的transaction_motif
    motif_motif_analysis = transactions_df.groupby(['motif_id', 'transaction_motif']).size().reset_index(name='count')
    motif_id_motif_counts = motif_motif_analysis.groupby('motif_id')['transaction_motif'].nunique().reset_index(name='unique_motifs')
    duplicate_motifs = motif_id_motif_counts[motif_id_motif_counts['unique_motifs'] > 1]
    
    print(f"  Found {len(duplicate_motifs)} motif_ids used in multiple transaction_motif")
    
    if len(duplicate_motifs) > 0:
        # 创建新的motif_id映射
        new_motif_id_counter = int(transactions_df['motif_id'].max()) + 1 if transactions_df['motif_id'].dtype in [np.int64, np.float64] else 100000
        motif_id_mapping = {}  # {(old_motif_id, transaction_motif): new_motif_id}
        
        # 为每个重复的motif_id分配新的motif_id
        for old_motif_id in duplicate_motifs['motif_id'].values:
            motif_transactions = transactions_df[transactions_df['motif_id'] == old_motif_id]
            unique_motifs = motif_transactions['transaction_motif'].unique()
            
            # 为主motif（交易数量最多的）保留原motif_id
            motif_counts = motif_transactions.groupby('transaction_motif').size()
            main_motif = motif_counts.idxmax()
            
            # 为其他motif分配新的motif_id
            for motif_type in unique_motifs:
                if motif_type == main_motif:
                    # 保留原motif_id
                    continue
                else:
                    # 分配新的motif_id
                    new_motif_id = new_motif_id_counter
                    motif_id_mapping[(old_motif_id, motif_type)] = new_motif_id
                    new_motif_id_counter += 1
                    # print(f"  Reassigning motif_id {old_motif_id} (transaction_motif: {motif_type}) -> {new_motif_id}")
        
        # 应用映射
        def apply_motif_id_mapping(row):
            old_motif_id = row['motif_id']
            transaction_motif = row['transaction_motif']
            key = (old_motif_id, transaction_motif)
            if key in motif_id_mapping:
                return motif_id_mapping[key]
            return old_motif_id
        
        transactions_df['motif_id'] = transactions_df.apply(apply_motif_id_mapping, axis=1)
        
        # 验证修复结果
        motif_id_motif_counts_after = transactions_df.groupby('motif_id')['transaction_motif'].nunique()
        still_duplicate = motif_id_motif_counts_after[motif_id_motif_counts_after > 1]
        if len(still_duplicate) > 0:
            print(f"  Warning: Still found {len(still_duplicate)} duplicate motif_ids after fix")
        else:
            print(f"  ✅ All duplicate motif_ids have been fixed")
        
        print(f"  Final unique motif_ids: {transactions_df['motif_id'].nunique()}")
    else:
        print("  ✅ No duplicate motif_ids found")
    
    # 删除临时列
    if 'motif_id_original' in transactions_df.columns:
        transactions_df = transactions_df.drop(columns=['motif_id_original'])
    
    return transactions_df

def add_interval_and_hour_columns(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """添加interval和hour列（在motif_id修复之后）"""
    print("\n=== Adding interval and hour columns ===")
    
    # 确保timestamp列是datetime类型
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    
    # 添加hour列 - 提取小时
    transactions_df['hour'] = transactions_df['timestamp'].dt.hour
    print(f"  Added hour column")
    
    # 按motif_id分组，然后按timestamp排序
    transactions_df = transactions_df.sort_values(['motif_id', 'timestamp']).reset_index(drop=True)
    
    # 初始化interval列
    transactions_df['interval'] = 0.0
    
    # 计算每个motif_id内的时间间隔
    motif_groups = transactions_df.groupby('motif_id')
    
    processed_groups = 0
    for motif_id, group in motif_groups:
        if len(group) > 1:
            # 计算时间间隔（秒）
            timestamps = group['timestamp'].values
            intervals = [0.0]  # 第一行间隔为0
            
            for i in range(1, len(timestamps)):
                # 计算时间差（秒）- 处理numpy.timedelta64类型
                time_diff = timestamps[i] - timestamps[i-1]
                # 转换为秒数（处理numpy.timedelta64）
                if isinstance(time_diff, np.timedelta64):
                    time_diff_seconds = time_diff / np.timedelta64(1, 's')
                elif hasattr(time_diff, 'total_seconds'):
                    time_diff_seconds = time_diff.total_seconds()
                else:
                    time_diff_seconds = float(time_diff) / 1e9  # 纳秒转秒（numpy默认）
                intervals.append(time_diff_seconds)
            
            # 更新interval列
            transactions_df.loc[group.index, 'interval'] = intervals
            processed_groups += 1
    
    print(f"  Calculated intervals for {processed_groups} motif groups")
    
    return transactions_df

def main():
    """主函数"""
    # 记录总开始时间
    total_start_time = time.time()
    
    print("=" * 70)
    print("=== 交易生成程序 ===")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载配置
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载配置文件... {get_process_info()}")
    config_start = time.time()
    with open("conf_temporal_with_risk.json", 'r') as f:
        config = json.load(f)
    config_elapsed = time.time() - config_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置文件加载完成, 耗时 {config_elapsed:.2f} 秒 {get_process_info()}")
    
    # 加载数据
    g,  balances, wallet_to_attrs, attr_headers, wallet_open_timestamps, abnormal_wallets = load_data(
        config["input"]["edges"],
        config["input"]["accounts"])
    
    # 显示钱包开立时间统计
    print(f"\n=== 钱包开立时间统计 ===")
    open_times = list(wallet_open_timestamps.values())
    if open_times:
        earliest_open = min(open_times)
        latest_open = max(open_times)
        print(f"最早开立时间: {earliest_open}")
        print(f"最晚开立时间: {latest_open}")
        print(f"开立时间范围: {(latest_open - earliest_open).days} 天")
    
    # 设计层面的取舍：完全排除“开户时间晚于模拟开始时间”的钱包
    # 这样可以在不破坏已生成交易链条/金额的前提下，保证所有参与模拟的钱包在模拟期一开始就已经存在
    sim_start_time = datetime.strptime(config["simulation"]["start_time"], "%Y-%m-%d %H:%M:%S")
    original_wallet_count = len(wallet_open_timestamps)
    allowed_wallets = {wid for wid, t in wallet_open_timestamps.items() if t <= sim_start_time}
    removed_wallets = set(wallet_open_timestamps.keys()) - allowed_wallets
    
    if removed_wallets:
        # 从图中移除这些钱包节点（既不会作为 src 也不会作为 dst 出现）
        for wid in list(g.nodes()):
            if wid in removed_wallets:
                g.remove_node(wid)
        
        # 从余额、属性、开立时间和异常账户集合中移除
        for wid in list(balances.keys()):
            if wid in removed_wallets:
                del balances[wid]
        for wid in list(wallet_to_attrs.keys()):
            if wid in removed_wallets:
                del wallet_to_attrs[wid]
        for wid in list(wallet_open_timestamps.keys()):
            if wid in removed_wallets:
                del wallet_open_timestamps[wid]
        
        abnormal_wallets -= removed_wallets
        
        print(f"\n=== 钱包过滤（开户时间晚于模拟开始时间） ===")
        print(f"模拟开始时间: {sim_start_time}")
        print(f"原始钱包数: {original_wallet_count}")
        print(f"移除钱包数: {len(removed_wallets)}")
        print(f"参与模拟的钱包数: {len(allowed_wallets)}")
    
    # 生成交易（批量保存到临时文件，每1万条保存一次）
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始生成交易... {get_process_info()}")
    temp_output_file = config["output"]["transactions"].replace('.csv', '_temp.csv')
    transactions = generate_transactions(g,  balances, wallet_to_attrs, attr_headers, 
                                       wallet_open_timestamps, abnormal_wallets, config["simulation"],
                                       output_file=temp_output_file)
    
    # 输出交易类型下 transaction mode 的占比统计
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === 输出交易类型下 Transaction Mode 占比统计 ===")
    from transaction_manager import TransactionManager
    manager = TransactionManager("transaction_config.json")
    manager.output_transaction_mode_distribution()

    # 定义安全的motif类型
    safe_motifs = ['single_transaction', 'normal_small_high_freq', 'regular_large_low_freq']
    
    # 过滤掉单笔的forward交易
    def filter_single_transactions(transactions):
        """过滤单笔的forward和many_to_one交易，并验证many_to_one的完整性"""
        
        def is_many_to_one_complete(many_to_one_txs):
            """检查many_to_one交易是否完整（有转入和转出）"""
            if len(many_to_one_txs) < 3:
                return False
            
            # 按主钱包分组（出现次数最多的dst）
            dst_counts = {}
            for tx in many_to_one_txs:
                dst = tx.get("dst")
                dst_counts[dst] = dst_counts.get(dst, 0) + 1
            
            # 找出主钱包（出现次数最多的dst）
            main_wallet = max(dst_counts, key=dst_counts.get)
            
            # 检查是否有从主钱包转出的交易
            has_outgoing = any(tx.get("src") == main_wallet for tx in many_to_one_txs)
            
            # 检查转入交易数量是否至少2笔
            incoming_count = sum(1 for tx in many_to_one_txs if tx.get("dst") == main_wallet)
            
            return has_outgoing and incoming_count >= 2

        def is_one_to_many_complete(one_to_many_txs):
            """检查one_to_many交易是否完整（有转入和转出）"""
            if len(one_to_many_txs) < 3:
                return False
            dst_counts = {}
            for tx in one_to_many_txs:
                dst = tx.get("dst")
                dst_counts[dst] = dst_counts.get(dst, 0) + 1
            
            # 找出主钱包（出现次数最多的dst）
            main_wallet = max(dst_counts, key=dst_counts.get)
            
            # 检查是否有转入主钱包的交易（src != main_wallet and dst == main_wallet）
            has_incoming = any(tx.get("src") != main_wallet and tx.get("dst") == main_wallet for tx in one_to_many_txs)
            
            # 检查转出交易数量是否至少2笔（src == main_wallet）
            outgoing_count = sum(1 for tx in one_to_many_txs if tx.get("src") == main_wallet)
            
            return has_incoming and outgoing_count >= 2

        def is_many_to_many_complete(many_to_many_txs):
            """检查many_to_many交易是否完整（有转入和转出）- 必须验证主钱包模式"""
            if len(many_to_many_txs) < 4:
                return False
            
            # 按主钱包分组（出现次数最多的dst和src）
            dst_counts = {}
            src_counts = {}
            for tx in many_to_many_txs:
                dst = tx.get("dst")
                src = tx.get("src")
                dst_counts[dst] = dst_counts.get(dst, 0) + 1
                src_counts[src] = src_counts.get(src, 0) + 1
            
            # 找出主钱包（出现次数最多的dst和src）
            main_wallet_by_dst = max(dst_counts, key=dst_counts.get)
            main_wallet_by_src = max(src_counts, key=src_counts.get)
            
            # 主钱包应该是同一个（这是many_to_many模式的基本要求）
            if main_wallet_by_dst != main_wallet_by_src:
                return False
            
            main_wallet = main_wallet_by_dst
            
            # 检查转入和转出交易数量
            incoming_count = sum(1 for tx in many_to_many_txs if tx.get("dst") == main_wallet)
            outgoing_count = sum(1 for tx in many_to_many_txs if tx.get("src") == main_wallet)
            
            # 判断是否为异常交易（根据 transaction_motif 判断）
            first_tx = many_to_many_txs[0]
            transaction_motif = first_tx.get("risk_type", "")
            is_risk = transaction_motif in ["class4_laundering", "merchant_laundering", "online_laundering", "small_amount_testing"] or \
                      first_tx.get("is_risk") == 1 or first_tx.get("is_risk") == '1'
            
            # 根据是否为异常交易设置不同的最小要求
            if is_risk:
                # merchant_laundering 使用更宽松的要求（因为金额较大，交易数可能较少）
                if transaction_motif == "merchant_laundering":
                    # merchant_laundering: 至少2笔转入和2笔转出
                    return incoming_count >= 2 and outgoing_count >= 2
                else:
                    # 其他异常交易：至少3笔转入和4笔转出
                    return incoming_count >= 3 and outgoing_count >= 4
            else:
                # 正常交易：至少2笔转入和2笔转出
                return incoming_count >= 2 and outgoing_count >= 2
        
        # 按motif_id分组
        motif_groups = {}
        for tx in transactions:
            motif_id = tx.get("motif_id")
            if motif_id not in motif_groups:
                motif_groups[motif_id] = []
            motif_groups[motif_id].append(tx)
        
        filtered_transactions = []
        deleted_single_forward_count = 0
        deleted_incomplete_many_to_one_count = 0
        deleted_incomplete_one_to_many_count = 0
        deleted_incomplete_many_to_many_count = 0
        deleted_incomplete_fan_out_count = 0
        deleted_incomplete_fan_in_count = 0
        
        for motif_id, group in motif_groups.items():
            # 检查这个motif_id组中是否有forward、many_to_one、one_to_many、many_to_many、fan_out或fan_in交易
            forward_txs = [tx for tx in group if tx.get("transaction_mode") == "forward"]
            many_to_one_txs = [tx for tx in group if tx.get("transaction_mode") == "many_to_one"]
            one_to_many_txs = [tx for tx in group if tx.get("transaction_mode") == "one_to_many"]
            many_to_many_txs = [tx for tx in group if tx.get("transaction_mode") == "many_to_many"]
            fan_out_txs = [tx for tx in group if tx.get("transaction_mode") == "fan_out"]
            fan_in_txs = [tx for tx in group if tx.get("transaction_mode") == "fan_in"]
            
            # 处理forward交易
            if forward_txs:
                if len(forward_txs) == 2:
                    # forward交易应该是成对的，保留
                    filtered_transactions.extend(forward_txs)
                else:
                    # 单笔forward交易，删除
                    deleted_single_forward_count += len(forward_txs)
            
            # 处理many_to_one交易
            if many_to_one_txs:
                # 检查many_to_one交易是否完整
                if is_many_to_one_complete(many_to_one_txs):
                    # 完整的many_to_one交易，保留
                    filtered_transactions.extend(many_to_one_txs)
                else:
                    # 不完整的many_to_one交易，删除
                    deleted_incomplete_many_to_one_count += len(many_to_one_txs)
            
            # 处理one_to_many交易
            if one_to_many_txs:
                # 检查one_to_many交易是否完整
                if is_one_to_many_complete(one_to_many_txs):
                    # 完整的one_to_many交易，保留
                    filtered_transactions.extend(one_to_many_txs)
                else:
                    # 不完整的one_to_many交易，删除
                    deleted_incomplete_one_to_many_count += len(one_to_many_txs)
            
            # 处理many_to_many交易
            if many_to_many_txs:
                # 检查many_to_many交易是否完整
                if is_many_to_many_complete(many_to_many_txs):
                    # 完整的many_to_many交易，保留
                    filtered_transactions.extend(many_to_many_txs)
                else:
                    # 不完整的many_to_many交易，删除
                    deleted_incomplete_many_to_many_count += len(many_to_many_txs)
            
            # 处理fan_out交易
            if fan_out_txs:
                # 检查fan_out交易是否完整（至少2笔）
                if len(fan_out_txs) >= 2:
                    # 完整的fan_out交易，保留
                    filtered_transactions.extend(fan_out_txs)
                else:
                    # 不完整的fan_out交易，删除
                    deleted_incomplete_fan_out_count += len(fan_out_txs)
            
            # 处理fan_in交易
            if fan_in_txs:
                # 检查fan_in交易是否完整（至少2笔）
                if len(fan_in_txs) >= 2:
                    # 完整的fan_in交易，保留
                    filtered_transactions.extend(fan_in_txs)
                else:
                    # 不完整的fan_in交易，删除
                    deleted_incomplete_fan_in_count += len(fan_in_txs)
            
            # 处理其他类型的交易
            other_txs = [tx for tx in group if tx.get("transaction_mode") not in ["forward", "many_to_one", "one_to_many", "many_to_many", "fan_out", "fan_in"]]
            filtered_transactions.extend(other_txs)
        
        return filtered_transactions, deleted_single_forward_count, deleted_incomplete_many_to_one_count, deleted_incomplete_one_to_many_count, deleted_incomplete_many_to_many_count, deleted_incomplete_fan_out_count, deleted_incomplete_fan_in_count


    # 先保存过滤前的中间版本（用于调试）
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 保存过滤前的中间版本... {get_process_info()}")
    intermediate_output_file = config["output"]["transactions"].replace('.csv', '_before_filter.csv')
    intermediate_start = time.time()
    with open(intermediate_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["tx_id", "timestamp", "src", "dst", "amount", "transaction_motif", "motif_id", "transaction_mode", "is_risk", "src_bank_account_number", "dst_bank_account_number", 
    "src_wallet_level", "dst_wallet_level"])
        for tx in transactions:
            # 计算is_risk值
            transaction_motif = tx.get("risk_type", "unknown")
            is_risk = '0' if transaction_motif in safe_motifs else '1'
            src_info = get_wallet_info(tx["src"], wallet_to_attrs, attr_headers)
            dst_info = get_wallet_info(tx["dst"], wallet_to_attrs, attr_headers)
            writer.writerow([
                tx["tx_id"],
                tx["timestamp"],
                tx["src"],
                tx["dst"],
                tx["amount"],
                transaction_motif,
                tx["motif_id"],
                tx.get("transaction_mode", ""),
                is_risk,
                src_info['bank_account_number'],
                dst_info['bank_account_number'],
                src_info['wallet_level'],
                dst_info['wallet_level']
            ])
    intermediate_elapsed = time.time() - intermediate_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 中间版本保存完成, 耗时 {intermediate_elapsed:.2f} 秒 {get_process_info()}")
    print(f"文件: {intermediate_output_file}")
    
    # 过滤交易
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始过滤交易... {get_process_info()}")
    filter_start = time.time()
    filtered_transactions, deleted_forward, deleted_incomplete_many_to_one, deleted_incomplete_one_to_many, deleted_incomplete_many_to_many, deleted_incomplete_fan_out, deleted_incomplete_fan_in = filter_single_transactions(transactions)
    filter_elapsed = time.time() - filter_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 交易过滤完成, 耗时 {filter_elapsed:.2f} 秒 {get_process_info()}")
    
    print(f"原始交易数: {len(transactions)}")
    print(f"过滤后交易数: {len(filtered_transactions)}")
    print(f"删除了 {deleted_forward} 笔单笔forward交易")
    print(f"删除了 {deleted_incomplete_many_to_one} 笔不完整many_to_one交易")
    print(f"删除了 {deleted_incomplete_one_to_many} 笔不完整one_to_many交易")
    print(f"删除了 {deleted_incomplete_many_to_many} 笔不完整many_to_many交易")
    print(f"删除了 {deleted_incomplete_fan_out} 笔不完整fan_out交易")
    print(f"删除了 {deleted_incomplete_fan_in} 笔不完整fan_in交易")
    
    # 先保存初始交易数据（不包含设备/IP等信息）
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 保存临时交易文件... {get_process_info()}")
    temp_output_file = config["output"]["transactions"].replace('.csv', '_temp.csv')
    temp_start = time.time()
    with open(temp_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["tx_id", "timestamp", "src", "dst", "amount", "transaction_motif", "motif_id", "transaction_mode", "is_risk", "src_bank_account_number", "dst_bank_account_number", 
    "src_wallet_level", "dst_wallet_level"])
        for tx in filtered_transactions:
            # 计算is_risk值
            transaction_motif = tx.get("risk_type", "unknown")
            is_risk = '0' if transaction_motif in safe_motifs else '1'
            src_info = get_wallet_info(tx["src"], wallet_to_attrs, attr_headers)
            dst_info = get_wallet_info(tx["dst"], wallet_to_attrs, attr_headers)
            writer.writerow([
                tx["tx_id"],
                tx["timestamp"],
                tx["src"],
                tx["dst"],
                tx["amount"],
                transaction_motif,  # 这里作为transaction_motif输出
                tx["motif_id"],
                tx.get("transaction_mode", ""),
                is_risk,
                src_info['bank_account_number'],  # 新增
                dst_info['bank_account_number'],  # 新增
                src_info['wallet_level'],         # 新增
                dst_info['wallet_level']          # 新增
            ])
    temp_elapsed = time.time() - temp_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 临时文件保存完成, 耗时 {temp_elapsed:.2f} 秒 {get_process_info()}")
    print(f"文件: {temp_output_file}")
    
    # 步骤1：读取临时文件并添加设备/IP列
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step 1: Adding device/IP columns === {get_process_info()}")
    step1_start = time.time()
    transactions_df = pd.read_csv(temp_output_file)
    accounts_df = pd.read_csv(config["input"]["accounts"])
    account_map = build_account_device_ip_map(accounts_df)
    transactions_df = add_device_ip_columns(transactions_df, account_map)
    step1_elapsed = time.time() - step1_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 1 完成, 耗时 {step1_elapsed:.2f} 秒 {get_process_info()}")
    
    # 步骤2：修复重复的motif_id（确保每个motif_id只对应一个transaction_motif）
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step 2: Fixing duplicate motif_id usage === {get_process_info()}")
    step2_start = time.time()
    transactions_df = fix_duplicate_motif_ids(transactions_df)
    step2_elapsed = time.time() - step2_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 2 完成, 耗时 {step2_elapsed:.2f} 秒 {get_process_info()}")
    
    # 步骤3：添加interval和hour列（在motif_id修复之后）
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step 3: Adding interval and hour columns === {get_process_info()}")
    step3_start = time.time()
    transactions_df = add_interval_and_hour_columns(transactions_df)
    step3_elapsed = time.time() - step3_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 3 完成, 耗时 {step3_elapsed:.2f} 秒 {get_process_info()}")
    
    # 步骤4：应用受害者模式（在motif_id修复之后，避免冲突）
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step 4: Applying victim pattern === {get_process_info()}")
    step4_start = time.time()
    transactions_df = apply_victim_pattern(transactions_df)
    step4_elapsed = time.time() - step4_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 4 完成, 耗时 {step4_elapsed:.2f} 秒 {get_process_info()}")
    
    # 步骤5：保存最终文件（包含所有列）
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Step 5: Saving final transactions === {get_process_info()}")
    step5_start = time.time()
    # 重新排列列顺序
    final_columns = ["tx_id", "timestamp", "src", "dst", "amount", "transaction_motif", "motif_id", 
                     "transaction_mode", "is_risk", "is_src_victim", "src_bank_account_number", "dst_bank_account_number",
                     "src_wallet_level", "dst_wallet_level", "src_device", "src_ip", "dst_device", "dst_ip",
                     "interval", "hour"]
    # 只保留存在的列
    available_columns = [col for col in final_columns if col in transactions_df.columns]
    transactions_df[available_columns].to_csv(config["output"]["transactions"], index=False)
    step5_elapsed = time.time() - step5_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 5 完成, 耗时 {step5_elapsed:.2f} 秒 {get_process_info()}")
    print(f"最终文件: {config['output']['transactions']}")
    print(f"总列数: {len(available_columns)}")
    print(f"列: {', '.join(available_columns)}")
    
    # 计算总运行时间
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    # 输出总运行时间
    print("\n" + "=" * 70)
    print("=== 程序执行完成 ===")
    print("=" * 70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总运行时间: {total_elapsed_time:.2f} 秒 ({total_elapsed_time/60:.2f} 分钟)")
    print(f"进程信息: {get_process_info()}")
    print("=" * 70)
    



if __name__ == "__main__":
    main()
