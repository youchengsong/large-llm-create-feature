#!/usr/bin/env python3
"""Patch simplified_main.py to optimize the hot loop performance.

Key bottleneck: every time point (90K total) rebuilds a set from 43K wallets
and does multiple O(N) list comprehensions for wallet selection.

Optimizations:
1. Build available wallet set once per day, maintain incrementally
2. Remove redundant neighbor checks (already filtered at day boundary)
3. Replace per-time-point O(N) filtering with O(1) incremental updates
4. Pre-categorize wallets at day boundary instead of every time point
"""

import sys

filepath = '/home/ubuntu/Projects/大模型提取特征/inputs/SimECNY/transaction/simplified_main.py'

with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

original_code = code

# ============================================================
# PATCH 1: Initialize incremental sets before the main loop
# ============================================================
old1 = '    for current_time in all_transaction_times:\n        processed_count += 1'
new1 = """    day_available_set = set()
    _wallets_reached_limit = set()
    for current_time in all_transaction_times:
        processed_count += 1"""
code = code.replace(old1, new1, 1)
if old1 not in original_code:
    print("WARN: Patch 1 target not found")
else:
    print("Patch 1 applied: initialized incremental sets before loop")

# ============================================================
# PATCH 2: Build day_available_set at end of day-change block
# ============================================================
old2 = """                        cached_available_wallets.append(wallet)

        # 优化：使用集合和预计算，减少重复检查
        # 优化：只在限额变化时才重新过滤"""
new2 = """                        cached_available_wallets.append(wallet)
            day_available_set = set(cached_available_wallets)
            _wallets_reached_limit.clear()

        # 增量更新可用钱包集合"""
code = code.replace(old2, new2, 1)
print("Patch 2 applied: build day_available_set at day change")

# ============================================================
# PATCH 3: Replace per-time-point O(N) filtering with incremental
# ============================================================
# Find the block from "优化：使用集合和预计算" to "abandoned_wallets]"
old3_marker_start = "        # 优化：使用集合和预计算，减少重复检查\n"
old3_marker_end = "        available_wallets_filtered = [w for w in available_wallets_filtered if w not in abandoned_wallets]\n"

# Also try the new marker from patch 2
old3_marker_start_v2 = "        # 增量更新可用钱包集合\n"

start_marker = old3_marker_start_v2 if old3_marker_start_v2 in code else old3_marker_start
start_pos = code.find(start_marker)
end_pos = code.find(old3_marker_end)

if start_pos != -1 and end_pos != -1:
    end_pos += len(old3_marker_end)
    new3 = """        # 增量更新：移除已达限额和废弃的钱包
        if _wallets_reached_limit:
            day_available_set -= _wallets_reached_limit
            _wallets_reached_limit.clear()
        day_available_set -= abandoned_wallets
        available_wallets_filtered = list(day_available_set) if day_available_set else []
"""
    code = code[:start_pos] + new3 + code[end_pos:]
    print("Patch 3 applied: replaced per-time-point O(N) filtering with incremental update")
else:
    print(f"WARN: Patch 3 targets not found. start={start_pos}, end={end_pos}")

# ============================================================
# PATCH 4: Replace available_wallets_set rebuild in wallet selection
# ============================================================
old4 = "            available_wallets_set = set(available_wallets_filtered)\n"
new4 = "            available_wallets_set = day_available_set.copy()\n"
# Only replace the FIRST occurrence (in the risk decision section)
code = code.replace(old4, new4, 1)
print("Patch 4 applied: reuse day_available_set instead of rebuilding from list")

# ============================================================
# PATCH 5: Remove redundant neighbor checks in wallet selection
# These wallets already passed neighbor check at day boundary
# ============================================================
redundant_neighbor_checks = [
    ("                    risk_available_check = [w for w in risk_available_set if wallet_plans[w]['neighbors']]",
     "                    risk_available_check = list(risk_available_set)"),
    ("                    risk_available = [w for w in risk_candidates_set if wallet_plans[w]['neighbors']]",
     "                    risk_available = list(risk_candidates_set)"),
    ("                    abnormal_available = [w for w in abnormal_available_set if wallet_plans[w]['neighbors']]",
     "                    abnormal_available = list(abnormal_available_set)"),
    ("                    normal_available = [w for w in normal_candidates if wallet_plans[w]['neighbors']]",
     "                    normal_available = list(normal_candidates)"),
    ("                normal_available_all = [w for w in normal_candidates_set if wallet_plans[w]['neighbors']]",
     "                normal_available_all = list(normal_candidates_set)"),
    ("                abnormal_normal_available_all = [w for w in abnormal_normal_candidates_set if wallet_plans[w]['neighbors']]",
     "                abnormal_normal_available_all = list(abnormal_normal_candidates_set)"),
]

patch5_count = 0
for old_line, new_line in redundant_neighbor_checks:
    if old_line in code:
        code = code.replace(old_line, new_line, 1)
        patch5_count += 1
print(f"Patch 5 applied: removed {patch5_count} redundant neighbor checks")

# ============================================================
# PATCH 6: Track wallets that reach daily limits for incremental removal
# Find all places where daily_limit_reached = True and add tracking
# Must keep _wallets_reached_limit.add() INSIDE the if/elif blocks (same indent as the True assignment)
# ============================================================
import re

# Replace pattern: "daily_limit_reached'] = True\n" with "daily_limit_reached'] = True\n<same_indent>_wallets_reached_limit.add(src)\n"
# Use regex to capture the indentation and add at the same level
def add_limit_tracking(match):
    indent = match.group(1)
    var_name = match.group(2)
    return f"{indent}{var_name}['daily_limit_reached'] = True\n{indent}_wallets_reached_limit.add(src)\n"

code = re.sub(
    r"^( +)(plan_wallet|plan)\['daily_limit_reached'\] = True\n",
    add_limit_tracking,
    code,
    flags=re.MULTILINE
)

patch6_count = code.count('_wallets_reached_limit.add(')
print(f"Patch 6 applied: added {patch6_count} limit tracking points")

# ============================================================
# PATCH 7: Reduce progress output frequency for less I/O overhead
# ============================================================
old7 = "        if processed_count % 5000 == 0:"
new7 = "        if processed_count % 10000 == 0:"
code = code.replace(old7, new7, 1)
print("Patch 7 applied: reduced progress output frequency")

# ============================================================
# Write the optimized file
# ============================================================
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(code)

print(f"\nAll patches applied. File saved to {filepath}")
print(f"Original size: {len(original_code)} chars, New size: {len(code)} chars")
