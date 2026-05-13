#!/usr/bin/env python3
"""
基于度数频率数据生成静态边分布
"""

import csv
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import time
import os
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_process_info() -> str:
    """获取进程信息（内存使用等）"""
    info = []
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        info.append(f"内存: {mem_mb:.1f} MB")
    return ", ".join(info) if info else ""

def load_degree_freq_data(freq_file: str) -> List[Dict]:
    """加载度数频率数据"""
    start_time = time.time()
    freq_data = []
    with open(freq_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq_data.append({
                'in_degree': int(row['in_degree']),
                'out_degree': int(row['out_degree']),
                'count': int(row['count'])
            })
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载度数频率数据完成: {len(freq_data)} 条记录, 耗时 {elapsed:.2f} 秒 {get_process_info()}")
    return freq_data

def load_wallet_data(accounts_file: str) -> Tuple[List[str], List[str], Dict[str, datetime]]:
    """加载钱包数据，包括开立时间"""
    start_time = time.time()
    wallet_ids = []
    wallet_open_timestamps = {}
    attr_headers = []
    with open(accounts_file, 'r') as f:
        reader = csv.reader(f)
        attr_headers = next(reader)
        # 查找开立时间列的索引
        try:
            timestamp_idx = attr_headers.index('wallet_open_timestamp')
        except ValueError:
            # 如果找不到wallet_open_timestamp，尝试wallet_open_date
            try:
                timestamp_idx = attr_headers.index('wallet_open_date')
            except ValueError:
                timestamp_idx = 6  # 默认第7列（索引6）
        
        for row in reader:
            if len(row) >= 12:  # 确保有足够的列
                wallet_id = row[0]
                wallet_ids.append(wallet_id)
                
                # 解析开立时间
                if len(row) > timestamp_idx:
                    timestamp_str = row[timestamp_idx].strip()
                    if timestamp_str:
                        try:
                            # 尝试解析完整时间戳格式
                            wallet_open_timestamps[wallet_id] = datetime.strptime(
                                timestamp_str, "%Y-%m-%d %H:%M:%S"
                            )
                        except ValueError:
                            try:
                                # 尝试解析日期格式
                                wallet_open_timestamps[wallet_id] = datetime.strptime(
                                    timestamp_str, "%Y-%m-%d"
                                )
                            except ValueError:
                                # 如果解析失败，使用默认时间
                                wallet_open_timestamps[wallet_id] = datetime(2024, 1, 1, 0, 0, 0)
                    else:
                        wallet_open_timestamps[wallet_id] = datetime(2024, 1, 1, 0, 0, 0)
                else:
                    wallet_open_timestamps[wallet_id] = datetime(2024, 1, 1, 0, 0, 0)
    
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载钱包数据完成: {len(wallet_ids)} 个钱包, {len(wallet_open_timestamps)} 个时间戳, 耗时 {elapsed:.2f} 秒 {get_process_info()}")
    return wallet_ids, attr_headers, wallet_open_timestamps

def load_abnormal_accounts(accounts_file: str, attr_headers: List[str]) -> set:
    """加载异常账户集合"""
    start_time = time.time()
    abnormal_accounts = set()
    
    # 查找is_abnormal列的索引
    try:
        is_abnormal_idx = attr_headers.index('is_abnormal')
    except ValueError:
        # 如果找不到，尝试通过位置查找（通常是倒数第二列）
        is_abnormal_idx = len(attr_headers) - 2 if len(attr_headers) >= 2 else -1
    
    with open(accounts_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        
        for row in reader:
            if len(row) > is_abnormal_idx and is_abnormal_idx >= 0:
                wallet_id = row[0]
                is_abnormal_value = row[is_abnormal_idx].strip() if is_abnormal_idx < len(row) else ''
                # 判断是否为异常账户
                if is_abnormal_value in ['1', 'True', 'true', 'TRUE', 'Yes', 'yes', 'YES']:
                    abnormal_accounts.add(wallet_id)
    
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载异常账户完成: {len(abnormal_accounts)} 个异常账户, 耗时 {elapsed:.2f} 秒 {get_process_info()}")
    return abnormal_accounts

def generate_static_edges_distribution_based(freq_data: List[Dict], all_wallet_ids: List[str], 
                                           abnormal_accounts: set, output_file: str, rng,
                                           abnormal_high_degree_ratio: float = 0.7,
                                           high_degree_threshold: int = 10,
                                           wallet_open_timestamps: Dict[str, datetime] = None,
                                           time_window_days: int = 30,
                                           time_filter_ratio: float = 0.6) -> None:
    """
    基于度数频率数据生成静态边分布，按比例分配高度数给异常账户
    Args:
        freq_data: 度数频率数据
        all_wallet_ids: 所有钱包ID列表
        abnormal_accounts: 异常账户集合
        output_file: 输出文件路径
        rng: 随机数生成器
        abnormal_high_degree_ratio: 高度数组合中分配给异常账户的比例（默认0.7，即70%）
        high_degree_threshold: 高度数的阈值（总度数>=该值的视为高度数，默认10）
        wallet_open_timestamps: 钱包开立时间字典
        time_window_days: 时间窗口（天数），用于筛选开立时间相近的账户（默认30天）
        time_filter_ratio: 使用时间筛选的边的比例（默认0.6，即60%的边优先选择时间相近的账户）
    """
    # 获取所有钱包ID
    all_wallets = all_wallet_ids
    all_wallets_set = set(all_wallets)  # 优化：预先创建集合，用于快速查找
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 总共有 {len(all_wallets)} 个钱包 {get_process_info()}")
    
    # 为每个钱包分配度数
    wallet_degrees = {}
    for wallet_id in all_wallets:
        wallet_degrees[wallet_id] = {'in_degree': 0, 'out_degree': 0}
    
    # 优化：使用集合维护可用钱包列表（还未分配度数的），动态更新
    available_wallets_set = all_wallets_set.copy()
    
    # 优化：预先分离异常和正常账户到集合（用于快速查找）
    abnormal_set = abnormal_accounts  # 已经是集合
    normal_set = all_wallets_set - abnormal_set
    
    # 按总度数排序频率数据（度数高的优先）
    freq_data_sorted = sorted(freq_data, key=lambda x: x['in_degree'] + x['out_degree'], reverse=True)
    
    # 根据频率数据分配度数
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始分配度数... {get_process_info()}")
    degree_allocation_start = time.time()
    total_allocated = 0
    for freq_item in freq_data_sorted:
        in_deg = freq_item['in_degree']
        out_deg = freq_item['out_degree']
        count = freq_item['count']
        total_deg = in_deg + out_deg
        
        # 优化：使用集合维护的可用钱包列表（还未分配度数的）
        # 转换为列表用于随机选择
        available_wallets = list(available_wallets_set)
        
        if len(available_wallets) >= count:
            # 优化：使用集合交集操作，更快
            abnormal_available = list(available_wallets_set & abnormal_set)
            normal_available = list(available_wallets_set & normal_set)
            
            selected_wallets = []
            
            # 判断是否为高度数组合
            is_high_degree = total_deg >= high_degree_threshold
            
            if is_high_degree:
                # 高度数组合：按比例分配
                abnormal_count = max(1, int(count * abnormal_high_degree_ratio))
                normal_count = count - abnormal_count
                
                # 确保不超过可用数量
                actual_abnormal_count = min(abnormal_count, len(abnormal_available))
                actual_normal_count = min(normal_count, len(normal_available))
                
                # 如果总数不足，调整分配
                if actual_abnormal_count + actual_normal_count < count:
                    # 优先确保异常账户的比例
                    if len(abnormal_available) > actual_abnormal_count:
                        actual_abnormal_count = min(abnormal_count, len(abnormal_available))
                    if len(normal_available) > actual_normal_count:
                        actual_normal_count = min(count - actual_abnormal_count, len(normal_available))
                
                # 选择异常账户
                if actual_abnormal_count > 0 and len(abnormal_available) > 0:
                    if len(abnormal_available) > actual_abnormal_count:
                        selected_abnormal = list(rng.choice(abnormal_available, size=actual_abnormal_count, replace=False))
                    else:
                        selected_abnormal = abnormal_available.copy()
                    selected_wallets.extend(selected_abnormal)
                
                # 选择正常账户
                if actual_normal_count > 0 and len(normal_available) > 0:
                    if len(normal_available) > actual_normal_count:
                        selected_normal = list(rng.choice(normal_available, size=actual_normal_count, replace=False))
                    else:
                        selected_normal = normal_available.copy()
                    selected_wallets.extend(selected_normal)
            else:
                # 低度数组合：随机分配，不特别偏好异常账户
                # 按照账户比例随机选择
                if len(abnormal_available) > 0 and len(normal_available) > 0:
                    # 根据账户比例分配
                    abnormal_ratio = len(abnormal_available) / len(available_wallets)
                    abnormal_count = max(0, int(count * abnormal_ratio))
                    normal_count = count - abnormal_count
                elif len(abnormal_available) > 0:
                    abnormal_count = min(count, len(abnormal_available))
                    normal_count = 0
                else:
                    abnormal_count = 0
                    normal_count = min(count, len(normal_available))
                
                # 选择异常账户
                if abnormal_count > 0 and len(abnormal_available) > 0:
                    if len(abnormal_available) > abnormal_count:
                        selected_abnormal = list(rng.choice(abnormal_available, size=abnormal_count, replace=False))
                    else:
                        selected_abnormal = abnormal_available.copy()
                    selected_wallets.extend(selected_abnormal)
                
                # 选择正常账户
                if normal_count > 0 and len(normal_available) > 0:
                    if len(normal_available) > normal_count:
                        selected_normal = list(rng.choice(normal_available, size=normal_count, replace=False))
                    else:
                        selected_normal = normal_available.copy()
                    selected_wallets.extend(selected_normal)
            
            # 如果选择的账户数不足，从剩余账户中补充
            # 优化：使用集合操作，更快
            if len(selected_wallets) < count:
                remaining_needed = count - len(selected_wallets)
                selected_set = set(selected_wallets)
                remaining_available = list(available_wallets_set - selected_set)
                if len(remaining_available) >= remaining_needed:
                    additional = list(rng.choice(remaining_available, size=remaining_needed, replace=False))
                    selected_wallets.extend(additional)
                else:
                    selected_wallets.extend(remaining_available)
            
            # 分配度数
            selected_count = min(count, len(selected_wallets))
            for wallet_id in selected_wallets[:selected_count]:
                wallet_degrees[wallet_id]['in_degree'] = in_deg
                wallet_degrees[wallet_id]['out_degree'] = out_deg
                # 优化：从可用集合中移除已分配的钱包
                available_wallets_set.discard(wallet_id)
            
            actual_count = selected_count
            total_allocated += actual_count
        else:
            # 分配所有可用的钱包，按比例分配
            # 优化：使用集合操作
            abnormal_available = list(available_wallets_set & abnormal_set)
            normal_available = list(available_wallets_set & normal_set)
            
            selected_wallets = []
            
            # 如果没有可用钱包，跳过此频率项
            if len(available_wallets) == 0:
                continue
            
            # 按比例分配
            is_high_degree = total_deg >= high_degree_threshold
            if is_high_degree:
                abnormal_count = max(1, int(len(available_wallets) * abnormal_high_degree_ratio))
            else:
                # 计算异常账户在可用钱包中的比例，然后按这个比例分配
                abnormal_ratio = len(abnormal_available) / len(available_wallets) if len(available_wallets) > 0 else 0
                abnormal_count = max(0, int(len(available_wallets) * abnormal_ratio))
            
            abnormal_count = min(abnormal_count, len(abnormal_available))
            normal_count = min(len(available_wallets) - abnormal_count, len(normal_available))
            
            if abnormal_count > 0:
                selected_wallets.extend(abnormal_available[:abnormal_count])
            if normal_count > 0:
                selected_wallets.extend(normal_available[:normal_count])
            
            for wallet_id in selected_wallets:
                wallet_degrees[wallet_id]['in_degree'] = in_deg
                wallet_degrees[wallet_id]['out_degree'] = out_deg
                # 优化：从可用集合中移除已分配的钱包
                available_wallets_set.discard(wallet_id)
            total_allocated += len(selected_wallets)
    
    degree_allocation_elapsed = time.time() - degree_allocation_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 度数分配完成: 总共分配了 {total_allocated} 个钱包的度数, 耗时 {degree_allocation_elapsed:.2f} 秒 {get_process_info()}")
    
    # 优化：预先构建时间索引（按时间排序的钱包列表），加速时间筛选
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 构建时间索引... {get_process_info()}")
    time_index_start = time.time()
    time_index = {}  # {wallet_id: (timestamp, index_in_sorted_list)}
    sorted_wallets_by_time = None
    if wallet_open_timestamps:
        sorted_wallets_by_time = sorted(wallet_open_timestamps.items(), key=lambda x: x[1])
        for idx, (wallet_id, timestamp) in enumerate(sorted_wallets_by_time):
            time_index[wallet_id] = (timestamp, idx)
    time_window_seconds = time_window_days * 86400  # 转换为秒，避免重复计算
    time_index_elapsed = time.time() - time_index_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 时间索引构建完成, 耗时 {time_index_elapsed:.2f} 秒 {get_process_info()}")
    
    # 生成边
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始生成边... {get_process_info()}")
    edge_generation_start = time.time()
    edges_set = set()  # 优化：使用集合直接去重，避免后续转换
    edge_count = 0
    
    def filter_wallets_by_time(wallet_id: str, candidate_wallets: List[str], 
                               wallet_open_timestamps: Dict[str, datetime],
                               time_index: Dict,
                               sorted_wallets_by_time: List,
                               time_window_seconds: float) -> List[str]:
        """根据开立时间筛选相近的钱包（优化版本）"""
        if wallet_open_timestamps is None or wallet_id not in wallet_open_timestamps:
            return candidate_wallets
        
        if wallet_id not in time_index or sorted_wallets_by_time is None:
            return candidate_wallets
        
        source_time, source_idx = time_index[wallet_id]
        candidate_set = set(candidate_wallets)  # 优化：使用集合快速查找
        
        nearby_wallets = []
        # 向前查找（从source_idx向前）
        for i in range(source_idx, -1, -1):
            w_id, w_time = sorted_wallets_by_time[i]
            if w_id in candidate_set:
                time_diff = abs((w_time - source_time).total_seconds())
                if time_diff <= time_window_seconds:
                    nearby_wallets.append(w_id)
                else:
                    break  # 由于已排序，更早的时间肯定超出窗口
        
        # 向后查找（从source_idx+1向后）
        for i in range(source_idx + 1, len(sorted_wallets_by_time)):
            w_id, w_time = sorted_wallets_by_time[i]
            if w_id in candidate_set:
                time_diff = abs((w_time - source_time).total_seconds())
                if time_diff <= time_window_seconds:
                    nearby_wallets.append(w_id)
                else:
                    break  # 由于已排序，更晚的时间肯定超出窗口
        
        # 如果找到时间相近的钱包，优先使用它们；否则使用所有候选钱包
        if len(nearby_wallets) > 0:
            return nearby_wallets
        else:
            return candidate_wallets
    
    # 优化：不需要预计算完整的other_wallets列表（节省大量内存和时间）
    # 直接使用all_wallets，在需要时排除自己即可
    # 使用集合操作可以更高效地排除单个元素
    all_wallets_set_for_exclusion = all_wallets_set  # 用于快速排除操作
    
    processed_wallets = 0
    for wallet_id, degrees in wallet_degrees.items():
        if degrees['in_degree'] > 0 or degrees['out_degree'] > 0:
            processed_wallets += 1
            if processed_wallets % 10000 == 0:
                elapsed = time.time() - edge_generation_start
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已处理 {processed_wallets:,} 个钱包，已生成 {len(edges_set):,} 条边，耗时 {elapsed:.2f} 秒 {get_process_info()}")
            
            # 生成出边
            if degrees['out_degree'] > 0:
                # 优化：按需生成other_wallets，使用集合差集操作（O(1)排除单个元素）
                # 只在需要时转换为列表，避免存储大量重复数据
                other_wallets_set = all_wallets_set_for_exclusion - {wallet_id}
                other_wallets = list(other_wallets_set)
                time_filtered_targets = filter_wallets_by_time(
                    wallet_id, other_wallets, wallet_open_timestamps, time_index, sorted_wallets_by_time, time_window_seconds
                )
                
                # 计算需要从时间相近钱包中选择的数量（60%）
                time_filtered_count = max(0, int(degrees['out_degree'] * time_filter_ratio))
                remaining_count = degrees['out_degree'] - time_filtered_count
                
                selected_targets = []
                
                # 优先从时间相近的钱包中选择
                if time_filtered_count > 0 and len(time_filtered_targets) > 0:
                    actual_time_count = min(time_filtered_count, len(time_filtered_targets))
                    time_selected = list(rng.choice(time_filtered_targets, size=actual_time_count, replace=False))
                    selected_targets.extend(time_selected)
                    remaining_count = degrees['out_degree'] - len(selected_targets)
                
                # 从剩余候选钱包中补充（包括时间相近但未选中的，以及其他所有钱包）
                if remaining_count > 0:
                    remaining_candidates = [w for w in other_wallets if w not in selected_targets]
                    if len(remaining_candidates) >= remaining_count:
                        additional_targets = list(rng.choice(remaining_candidates, size=remaining_count, replace=False))
                        selected_targets.extend(additional_targets)
                    else:
                        # 如果还不够，允许重复
                        additional_targets = list(rng.choice(remaining_candidates, size=remaining_count, replace=True))
                        selected_targets.extend(additional_targets)
                
                # 添加边（优化：直接添加到集合，自动去重）
                for target in selected_targets:
                    edges_set.add((wallet_id, target))
                    edge_count += 1
            
            # 生成入边
            if degrees['in_degree'] > 0:
                # 优化：按需生成other_wallets，使用集合差集操作
                other_wallets_set = all_wallets_set_for_exclusion - {wallet_id}
                other_wallets = list(other_wallets_set)
                time_filtered_sources = filter_wallets_by_time(
                    wallet_id, other_wallets, wallet_open_timestamps, time_index, sorted_wallets_by_time, time_window_seconds
                )
                
                # 计算需要从时间相近钱包中选择的数量（60%）
                time_filtered_count = max(0, int(degrees['in_degree'] * time_filter_ratio))
                remaining_count = degrees['in_degree'] - time_filtered_count
                
                selected_sources = []
                
                # 优先从时间相近的钱包中选择
                if time_filtered_count > 0 and len(time_filtered_sources) > 0:
                    actual_time_count = min(time_filtered_count, len(time_filtered_sources))
                    time_selected = list(rng.choice(time_filtered_sources, size=actual_time_count, replace=False))
                    selected_sources.extend(time_selected)
                    remaining_count = degrees['in_degree'] - len(selected_sources)
                
                # 从剩余候选钱包中补充（包括时间相近但未选中的，以及其他所有钱包）
                if remaining_count > 0:
                    remaining_candidates = [w for w in other_wallets if w not in selected_sources]
                    if len(remaining_candidates) >= remaining_count:
                        additional_sources = list(rng.choice(remaining_candidates, size=remaining_count, replace=False))
                        selected_sources.extend(additional_sources)
                    else:
                        # 如果还不够，允许重复
                        additional_sources = list(rng.choice(remaining_candidates, size=remaining_count, replace=True))
                        selected_sources.extend(additional_sources)
                
                # 添加边（优化：直接添加到集合，自动去重）
                for source in selected_sources:
                    edges_set.add((source, wallet_id))
                    edge_count += 1
    
    # 优化：edges_set 已经是去重后的集合，直接转换为列表
    edge_generation_elapsed = time.time() - edge_generation_start
    edges = list(edges_set)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 边生成完成: {len(edges)} 条静态边（去重前: {edge_count} 条），耗时 {edge_generation_elapsed:.2f} 秒 {get_process_info()}")
    
    # 写入文件
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始写入文件... {get_process_info()}")
    write_start = time.time()
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst'])
        for edge in edges:
            writer.writerow(edge)
    write_elapsed = time.time() - write_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 静态边已保存到 {output_file}，耗时 {write_elapsed:.2f} 秒 {get_process_info()}")
    
    # 统计信息
    print("\n=== 度数分布统计 ===")
    abnormal_out_degree_dist = defaultdict(int)
    normal_out_degree_dist = defaultdict(int)
    
    for wallet_id, degrees in wallet_degrees.items():
        if degrees['out_degree'] > 0:
            if wallet_id in abnormal_accounts:
                abnormal_out_degree_dist[degrees['out_degree']] += 1
            else:
                normal_out_degree_dist[degrees['out_degree']] += 1
    
    # 计算平均度数
    abnormal_total_deg = sum(deg * count for deg, count in abnormal_out_degree_dist.items())
    abnormal_count_with_out = sum(abnormal_out_degree_dist.values())
    normal_total_deg = sum(deg * count for deg, count in normal_out_degree_dist.items())
    normal_count_with_out = sum(normal_out_degree_dist.values())
    
    print(f"\n=== 平均度数对比 ===")
    abnormal_avg_out = 0.0
    normal_avg_out = 0.0
    if abnormal_count_with_out > 0:
        abnormal_avg_out = abnormal_total_deg / abnormal_count_with_out
        print(f"异常账户平均出度: {abnormal_avg_out:.2f}")
    if normal_count_with_out > 0:
        normal_avg_out = normal_total_deg / normal_count_with_out
        print(f"正常账户平均出度: {normal_avg_out:.2f}")
    
    # 计算比例
    if normal_avg_out > 0:
        degree_ratio = abnormal_avg_out / normal_avg_out
        print(f"异常账户/正常账户平均出度比例: {degree_ratio:.2f}:1")

def main():
    """主函数"""
    main_start_time = time.time()
    
    # 文件路径
    freq_file = "wallet_static_edges_in_out_freq.csv"
    accounts_file = "accounts.csv"  # 使用当前的开立数据文件
    output_file = "wallet_static_edges_distribution_based.csv"
    
    # 配置参数
    # abnormal_high_degree_ratio: 高度数组合中分配给异常账户的比例（0.0-1.0）
    # 例如：0.7 表示高度数中70%给异常账户，30%给正常账户
    abnormal_high_degree_ratio = 0.6  # 可调整：控制异常账户在高度数中的占比
    
    # high_degree_threshold: 高度数的阈值（总度数>=该值的视为高度数）
    high_degree_threshold = 10  # 可调整：总度数>=10的视为高度数
    
    # time_window_days: 时间窗口（天数），用于筛选开立时间相近的账户
    time_window_days = 30  # 可调整：开立时间相差在30天内的账户视为时间相近
    
    # time_filter_ratio: 使用时间筛选的边的比例（0.0-1.0）
    # 例如：0.6 表示60%的边优先选择时间相近的账户，40%从所有候选账户中选择
    time_filter_ratio = 0.6  # 可调整：控制时间筛选的影响程度
    
    # 设置随机种子确保可重现性
    rng = np.random.default_rng(42)
    
    print("=" * 70)
    print("=== 基于度数分布的静态边生成器（按比例分配高度数）===")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"度数频率文件: {freq_file}")
    print(f"钱包数据文件: {accounts_file}")
    print(f"输出文件: {output_file}")
    print(f"\n配置参数:")
    print(f"  高度数中异常账户占比: {abnormal_high_degree_ratio:.1%}")
    print(f"  高度数阈值（总度数）: >= {high_degree_threshold}")
    print(f"  时间窗口（天数）: {time_window_days} 天")
    print(f"  时间筛选比例: {time_filter_ratio:.1%} (优先选择时间相近的账户)")
    print()
    
    try:
        # 加载数据
        freq_data = load_degree_freq_data(freq_file)
        all_wallet_ids, attr_headers, wallet_open_timestamps = load_wallet_data(accounts_file)
        abnormal_accounts = load_abnormal_accounts(accounts_file, attr_headers)
        
        # 生成静态边
        generate_static_edges_distribution_based(
            freq_data, all_wallet_ids, abnormal_accounts, output_file, rng,
            abnormal_high_degree_ratio=abnormal_high_degree_ratio,
            high_degree_threshold=high_degree_threshold,
            wallet_open_timestamps=wallet_open_timestamps,
            time_window_days=time_window_days,
            time_filter_ratio=time_filter_ratio
        )
        
        total_elapsed = time.time() - main_start_time
        print("\n" + "=" * 70)
        print("=== 生成完成 ===")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)")
        print(f"进程信息: {get_process_info()}")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 找不到文件 {e.filename}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()