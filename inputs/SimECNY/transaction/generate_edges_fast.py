#!/usr/bin/env python3
"""
Fast edge generation using integer indices and bisect for time windows.
Drop-in replacement for generate_static_edges.py with much lower memory usage.
"""
import csv
import bisect
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
import sys

def main():
    main_start = time.time()
    rng = np.random.default_rng(42)

    # Config
    freq_file = "wallet_static_edges_in_out_freq.csv"
    accounts_file = "accounts.csv"
    output_file = "wallet_static_edges_distribution_based.csv"
    abnormal_high_degree_ratio = 0.6
    high_degree_threshold = 10
    time_window_days = 30
    time_filter_ratio = 0.6
    time_window_seconds = time_window_days * 86400

    print("=" * 70)
    print("=== 快速边生成器 (优化版) ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load accounts
    print("加载钱包数据...")
    wallet_ids = []
    is_abnormal = []
    open_timestamps = []
    with open(accounts_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wallet_ids.append(row['wallet_id'])
            is_abnormal.append(row.get('is_abnormal', 'False').strip().lower() in ('true', '1', 'yes'))
            ts_str = row.get('wallet_open_timestamp', '')
            try:
                open_timestamps.append(datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S'))
            except (ValueError, TypeError):
                open_timestamps.append(datetime(2024, 1, 1))

    n_wallets = len(wallet_ids)
    print(f"加载 {n_wallets} 个钱包")

    # Build integer index sets
    abnormal_set = set(i for i, a in enumerate(is_abnormal) if a)
    normal_set = set(i for i, a in enumerate(is_abnormal) if not a)
    print(f"异常钱包: {len(abnormal_set)}, 正常钱包: {len(normal_set)}")

    # Sort by time for bisect
    sorted_by_time = sorted(range(n_wallets), key=lambda i: open_timestamps[i])
    epoch = open_timestamps[sorted_by_time[0]]
    sorted_ts_sec = [((open_timestamps[i] - epoch).total_seconds()) for i in sorted_by_time]
    int_to_sorted_pos = {}
    for pos, idx in enumerate(sorted_by_time):
        int_to_sorted_pos[idx] = pos

    # Load frequency data
    print("加载度数频率数据...")
    freq_data = []
    with open(freq_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq_data.append({
                'in_degree': int(row['in_degree']),
                'out_degree': int(row['out_degree']),
                'count': int(row['count'])
            })

    # Allocate degrees
    print("分配度数...")
    wallet_in_deg = np.zeros(n_wallets, dtype=np.int32)
    wallet_out_deg = np.zeros(n_wallets, dtype=np.int32)
    available = set(range(n_wallets))

    freq_data.sort(key=lambda x: x['in_degree'] + x['out_degree'], reverse=True)

    for item in freq_data:
        in_deg = item['in_degree']
        out_deg = item['out_degree']
        count = item['count']
        total_deg = in_deg + out_deg

        if len(available) == 0:
            break

        actual_count = min(count, len(available))
        avail_abnormal = list(available & abnormal_set)
        avail_normal = list(available & normal_set)

        if total_deg >= high_degree_threshold:
            ab_count = max(1, int(actual_count * abnormal_high_degree_ratio))
            ab_count = min(ab_count, len(avail_abnormal))
            no_count = min(actual_count - ab_count, len(avail_normal))
        else:
            ab_ratio = len(avail_abnormal) / len(available) if len(available) > 0 else 0
            ab_count = min(int(actual_count * ab_ratio), len(avail_abnormal))
            no_count = min(actual_count - ab_count, len(avail_normal))

        selected = []
        if ab_count > 0 and len(avail_abnormal) >= ab_count:
            sel = list(rng.choice(avail_abnormal, size=ab_count, replace=False))
            selected.extend(sel)
        elif ab_count > 0:
            selected.extend(avail_abnormal)

        if no_count > 0 and len(avail_normal) >= no_count:
            sel = list(rng.choice(avail_normal, size=no_count, replace=False))
            selected.extend(sel)
        elif no_count > 0:
            selected.extend(avail_normal)

        if len(selected) < actual_count:
            remaining = list(available - set(selected))
            need = actual_count - len(selected)
            if len(remaining) >= need:
                sel = list(rng.choice(remaining, size=need, replace=False))
                selected.extend(sel)
            else:
                selected.extend(remaining)

        for idx in selected[:actual_count]:
            wallet_in_deg[idx] = in_deg
            wallet_out_deg[idx] = out_deg
            available.discard(idx)

    total_allocated = n_wallets - len(available)
    print(f"度数分配完成: {total_allocated} 个钱包")

    # Generate edges
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始生成边...")
    edge_start = time.time()
    edges_set = set()
    edge_count = 0

    def select_peers(src, degree):
        if degree <= 0:
            return []
        time_count = max(0, int(degree * time_filter_ratio))
        selected = set()

        # Time window sampling (bisect)
        if time_count > 0 and src in int_to_sorted_pos:
            src_pos = int_to_sorted_pos[src]
            src_ts = sorted_ts_sec[src_pos]
            lo = bisect.bisect_left(sorted_ts_sec, src_ts - time_window_seconds)
            hi = bisect.bisect_right(sorted_ts_sec, src_ts + time_window_seconds)
            window_size = hi - lo
            if window_size > 1:
                need = min(time_count, window_size - 1)
                offsets = rng.choice(window_size, size=min(need * 3, window_size), replace=False)
                for off in offsets:
                    peer = sorted_by_time[lo + int(off)]
                    if peer != src:
                        selected.add(peer)
                        if len(selected) >= time_count:
                            break

        # Random global sampling
        remaining = degree - len(selected)
        if remaining > 0:
            sample_size = min(remaining + len(selected) + 20, n_wallets)
            idxs = rng.choice(n_wallets, size=sample_size, replace=False)
            for idx in idxs:
                idx_int = int(idx)
                if idx_int != src and idx_int not in selected:
                    selected.add(idx_int)
                    if len(selected) >= degree:
                        break

        result = list(selected)
        return result[:degree]

    processed = 0
    for i in range(n_wallets):
        out_d = wallet_out_deg[i]
        in_d = wallet_in_deg[i]
        if out_d > 0 or in_d > 0:
            processed += 1
            if processed % 10000 == 0:
                elapsed = time.time() - edge_start
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已处理 {processed:,} 个钱包, {len(edges_set):,} 条边, {elapsed:.1f}s")
                sys.stdout.flush()

            if out_d > 0:
                targets = select_peers(i, out_d)
                for t in targets:
                    edges_set.add((i, t))
                    edge_count += 1

            if in_d > 0:
                sources = select_peers(i, in_d)
                for s in sources:
                    edges_set.add((s, i))
                    edge_count += 1

    edge_elapsed = time.time() - edge_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 边生成完成: {len(edges_set)} 条静态边 (去重前 {edge_count}), 耗时 {edge_elapsed:.1f}s")

    # Write output (convert int IDs back to wallet_id strings)
    print("写入文件...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst'])
        for src_i, dst_i in edges_set:
            writer.writerow([wallet_ids[src_i], wallet_ids[dst_i]])

    # Stats
    ab_out = defaultdict(int)
    no_out = defaultdict(int)
    for i in range(n_wallets):
        if wallet_out_deg[i] > 0:
            if i in abnormal_set:
                ab_out[wallet_out_deg[i]] += 1
            else:
                no_out[wallet_out_deg[i]] += 1

    ab_total = sum(d * c for d, c in ab_out.items())
    ab_n = sum(ab_out.values())
    no_total = sum(d * c for d, c in no_out.items())
    no_n = sum(no_out.values())

    print(f"\n=== 平均度数对比 ===")
    if ab_n > 0:
        print(f"异常账户平均出度: {ab_total / ab_n:.2f}")
    if no_n > 0:
        print(f"正常账户平均出度: {no_total / no_n:.2f}")
    if no_n > 0 and ab_n > 0:
        print(f"异常/正常度数比: {(ab_total / ab_n) / (no_total / no_n):.2f}")

    total_elapsed = time.time() - main_start
    print(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} 分钟)")
    print(f"输出文件: {output_file}")
    print(f"总边数: {len(edges_set)}")

if __name__ == "__main__":
    main()
