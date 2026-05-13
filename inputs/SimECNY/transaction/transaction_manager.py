#!/usr/bin/env python3
"""
统一的交易模式管理器
负责管理所有交易类型的配置、生成和验证
支持所有钱包类型进行所有交易类型
"""

import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
import random

class TransactionManager:
    """统一的交易模式管理器"""
    
    def __init__(self, config_path: str = "transaction_config.json"):
        """初始化交易管理器"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.transaction_types = self.config["transaction_types"]
        self.wallet_weights = self.config["wallet_type_weights"]
        self.global_settings = self.config.get("global_settings", {})
        
        # 风险交易类型列表
        self.risk_transaction_types = [
            "small_amount_testing", 
            "merchant_laundering", 
            "class4_laundering", 
            "online_laundering"
        ]
        
        # 正常交易类型列表
        self.normal_transaction_types = [
            "single_transaction",
            "normal_small_high_freq", 
            "regular_large_low_freq"
        ]
    
    def get_wallet_weights(self, wallet_type: str, wallet_level: str, 
                          current_risk_ratio: float = 0.0) -> Dict[str, float]:
        """根据钱包类型获取交易权重，考虑全局风险比例和当前风险比例
        规则：
        - 如果 wallet_type=='1'（对公钱包），只使用 corporate_wallet 的权重
        - 如果 wallet_type=='3'（无论 wallet_level），只使用 merchant_wallet 的权重
        - 否则，如果 wallet_level=='4'（无论 wallet_type），只使用 class4_wallet 的权重
        - 其他情况，只使用 other_wallet 的权重
        """
        # 根据优先级选择权重配置
        if str(wallet_type) == '1':
            # 对公钱包：只使用 corporate_wallet 的权重
            base_weights = self.wallet_weights.get("corporate_wallet", {}).copy()
            # 如果 corporate_wallet 不存在，回退到 other_wallet
            if not base_weights:
                base_weights = self.wallet_weights.get("other_wallet", {}).copy()
        elif str(wallet_type) == '3':
            # 商户钱包：只使用 merchant_wallet 的权重
            base_weights = self.wallet_weights.get("merchant_wallet", {}).copy()
        elif str(wallet_level) == '4':
            # Level 4 钱包：只使用 class4_wallet 的权重
            base_weights = self.wallet_weights.get("class4_wallet", {}).copy()
        else:
            # 其他情况：只使用 other_wallet 的权重
            base_weights = self.wallet_weights.get("other_wallet", {}).copy()
        
        # 应用全局风险比例控制
        return self._apply_global_risk_ratio(base_weights, current_risk_ratio)
    
    def _apply_global_risk_ratio(self, weights: Dict[str, float], 
                                current_risk_ratio: float = 0.0) -> Dict[str, float]:
        """应用全局风险比例控制，支持动态调整"""
        target_risk_ratio = self.global_settings.get("risk_transaction_ratio", 0.1)
        force_risk_ratio = self.global_settings.get("force_risk_ratio", False)
        
        # 如果 force_risk_ratio 为 False，直接返回原始权重，不进行压缩
        if not force_risk_ratio:
            return weights
        
        # 分离风险交易和正常交易
        risk_weights = {}
        normal_weights = {}
        
        for tx_type, weight in weights.items():
            if tx_type in self.risk_transaction_types:
                risk_weights[tx_type] = weight
            elif tx_type in self.normal_transaction_types:
                normal_weights[tx_type] = weight
        
        # 如果强制风险比例，根据当前比例与目标比例的差距调整权重
        if current_risk_ratio < target_risk_ratio:
            # 当前比例低于目标，增加风险交易权重
            risk_boost = 2.0  # 风险交易权重提升倍数
            for tx_type in risk_weights:
                risk_weights[tx_type] *= risk_boost
        elif current_risk_ratio > target_risk_ratio:
            # 当前比例高于目标，降低风险交易权重
            # 根据差距动态调整压缩比例
            ratio_diff = current_risk_ratio / target_risk_ratio if target_risk_ratio > 0 else 1.0
            if ratio_diff > 2.0:
                risk_reduction = 0.2  # 如果超过目标2倍以上，大幅压缩
            elif ratio_diff > 1.5:
                risk_reduction = 0.3  # 如果超过目标1.5倍，中等压缩
            else:
                risk_reduction = 0.5  # 轻微超过目标，轻微压缩
            for tx_type in risk_weights:
                risk_weights[tx_type] *= risk_reduction
        
        # 重新分配权重
        adjusted_weights = {}
        
        # 正常交易权重：直接使用原始权重值，不进行归一化或比例调整
        if normal_weights:
            # 过滤掉无效权重
            valid_normal_weights = {k: v for k, v in normal_weights.items() 
                                   if v is not None and not np.isnan(v) and not np.isinf(v) and v >= 0}
            
            # 直接使用原始权重值（从原始weights中获取）
            for tx_type in valid_normal_weights.keys():
                original_weight = weights.get(tx_type, 0)
                if original_weight is None or np.isnan(original_weight) or np.isinf(original_weight) or original_weight < 0:
                    adjusted_weights[tx_type] = 0.0
                else:
                    # 直接使用原始权重值，不进行任何归一化或比例调整
                    adjusted_weights[tx_type] = original_weight
        
        # 风险交易权重：直接使用原始权重值，不进行归一化或比例调整
        if risk_weights:
            # 过滤掉无效权重
            valid_risk_weights = {k: v for k, v in risk_weights.items() 
                                if v is not None and not np.isnan(v) and not np.isinf(v) and v >= 0}
            
            # 直接使用原始权重值（从原始weights中获取，不经过boost/reduction）
            for tx_type in valid_risk_weights.keys():
                original_weight = weights.get(tx_type, 0)
                if original_weight is None or np.isnan(original_weight) or np.isinf(original_weight) or original_weight < 0:
                    adjusted_weights[tx_type] = 0.0
                else:
                    # 直接使用原始权重值，不进行任何归一化或比例调整
                    adjusted_weights[tx_type] = original_weight
        
        return adjusted_weights
    
    def get_transaction_config(self, transaction_type: str) -> Dict:
        """获取特定交易类型的配置"""
        return self.transaction_types.get(transaction_type, {})
    
    def is_risk_transaction(self, transaction_type: str) -> bool:
        """判断是否为风险交易"""
        return transaction_type in self.risk_transaction_types
    
    def get_risk_ratio(self) -> float:
        """获取全局风险交易比例"""
        return self.global_settings.get("risk_transaction_ratio", 0.1)
    
    def get_max_risk_transactions_per_wallet(self) -> int:
        """获取每个钱包最大风险交易次数"""
        return self.global_settings.get("max_risk_transactions_per_wallet", 3)
    
    def validate_transaction_conditions(self, transaction_type: str, wallet_id: str, 
                                      wallet_to_attrs: Dict, attr_headers: List[str], 
                                      current_time: datetime) -> bool:
        """验证交易条件 - 所有钱包类型都可以进行所有交易类型"""
        # 对于特定风险交易，仍然需要验证main钱包类型
        if transaction_type == "merchant_laundering":
            return self._validate_merchant_wallet_for_merchant_laundering(wallet_id, wallet_to_attrs, attr_headers)
        elif transaction_type == "class4_laundering":
            return self._validate_class4_wallet_for_class4_laundering(wallet_id, wallet_to_attrs, attr_headers)
        elif transaction_type == "small_amount_testing":
            return self._validate_small_amount_testing(wallet_id, wallet_to_attrs, attr_headers, current_time)
        elif transaction_type == "online_laundering":
            return self._validate_online_laundering(wallet_id, wallet_to_attrs, attr_headers, current_time)
        else:
            # 所有其他交易类型，所有钱包都可以进行
            return True
    
    
    def _validate_merchant_wallet_for_merchant_laundering(self, wallet_id: str, wallet_to_attrs: Dict, 
                                                        attr_headers: List[str]) -> bool:
        """验证商户跑分：main钱包必须是商户钱包（wallet_type='3'）"""
        if wallet_id not in wallet_to_attrs:
            return False
        
        wallet_attrs = wallet_to_attrs[wallet_id]
        wallet_type = None
        wallet_level = None
        
        for i, header in enumerate(attr_headers):
            if header in ['wallet_type', 'type'] and i > 0 and i-1 < len(wallet_attrs):
                wallet_type = wallet_attrs[i-1]  # wallet_attrs排除了wallet_id，所以索引要减1
            elif header in ['wallet_level', 'level'] and i > 0 and i-1 < len(wallet_attrs):
                wallet_level = wallet_attrs[i-1]
        
        return wallet_type == '3'
    
    def _validate_class4_wallet_for_class4_laundering(self, wallet_id: str, wallet_to_attrs: Dict, 
                                                    attr_headers: List[str]) -> bool:
        """验证四类钱包跑分：main钱包必须是四类钱包"""
        if wallet_id not in wallet_to_attrs:
            return False
        
        wallet_attrs = wallet_to_attrs[wallet_id]
        
        for i, header in enumerate(attr_headers):
            if header in ['wallet_level', 'level'] and i > 0 and i-1 < len(wallet_attrs):
                wallet_level = wallet_attrs[i-1]
                # 支持字符串"4"和数字4两种类型
                return str(wallet_level) == '4' or wallet_level == 4
        
        return False
    
    def _validate_small_amount_testing(self, wallet_id: str, wallet_to_attrs: Dict, 
                                     attr_headers: List[str], current_time: datetime) -> bool:
        """验证小额试探条件"""
        if wallet_id not in wallet_to_attrs:
            return False
        
        wallet_attrs = wallet_to_attrs[wallet_id]
        
        # 检查开卡时间
        try:
            wallet_open_date_idx = None
            for i, header in enumerate(attr_headers):
                if header in ["wallet_open_date", "open_date"]:
                    wallet_open_date_idx = i
                    break
            
            if wallet_open_date_idx is not None and wallet_open_date_idx < len(wallet_attrs):
                open_date_str = wallet_attrs[wallet_open_date_idx]
                open_date = datetime.strptime(open_date_str, "%Y-%m-%d")
                
                days_since_opening = (current_time.date() - open_date.date()).days
                min_days_threshold = 90  # 至少30天
                
                if days_since_opening < min_days_threshold:
                    return False
        except (ValueError, IndexError, AttributeError):
            pass
        
        return True
    
    def _validate_online_laundering(self, wallet_id: str, wallet_to_attrs: Dict, 
                                  attr_headers: List[str], current_time: datetime) -> bool:
        """验证线上跑分条件"""
        if wallet_id not in wallet_to_attrs:
            return False
        
        wallet_attrs = wallet_to_attrs[wallet_id]
        
        # 检查开卡时间（60%概率选择早期钱包）
        try:
            wallet_open_date_idx = None
            for i, header in enumerate(attr_headers):
                if header in ["wallet_open_date", "open_date"]:
                    wallet_open_date_idx = i
                    break
            
            if wallet_open_date_idx is not None and wallet_open_date_idx < len(wallet_attrs):
                open_date_str = wallet_attrs[wallet_open_date_idx]
                open_date = datetime.strptime(open_date_str, "%Y-%m-%d")
                
                # 60%概率选择2021年及以前的钱包，40%概率选择其他年份
                
                if open_date.year <= 2021:
                    return random.random() < 0.6  # 60%概率通过
                else:
                    return random.random() < 0.4  # 40%概率通过
        except (ValueError, IndexError, AttributeError):
            pass
        
        return True
    
    def get_wallet_level_limits(self, wallet_level: str) -> dict:
        """获取指定钱包等级的限额配置
        只有 wallet_level="4" 的钱包有交易限制，其他等级钱包无限制（返回 None）
        """
        # 只有 level 4 有配置的限制
        if str(wallet_level) == '4':
            return self.config.get('wallet_level_limits', {}).get('4', {
                'single_transaction_limit': 2000,
                'daily_limit': 5000
        })
        else:
            # Level 1, 2, 3 钱包无限制
            return {
                'single_transaction_limit': None,
                'daily_limit': None
            }
    
    def check_single_transaction_limit(self, wallet_level: str, amount: float) -> bool:
        """检查单笔交易是否超过限额
        如果限额为 None（非 level 4 钱包），返回 True（无限制）
        """
        limits = self.get_wallet_level_limits(wallet_level)
        single_limit = limits.get('single_transaction_limit')
        if single_limit is None:
            return True  # 无限制，通过检查
        return amount <= single_limit
    
    def check_daily_limit(self, wallet_level: str, daily_amount: float) -> bool:
        """检查日累计金额是否超过限额
        如果限额为 None（非 level 4 钱包），返回 True（无限制）
        """
        limits = self.get_wallet_level_limits(wallet_level)
        daily_limit = limits.get('daily_limit')
        if daily_limit is None:
            return True  # 无限制，通过检查
        return daily_amount <= daily_limit
    
    def get_wallet_level(self, wallet_id: str, wallet_to_attrs: dict, attr_headers: list) -> str:
        """获取钱包等级"""
        wallet_attrs = wallet_to_attrs.get(wallet_id, [])
        for j, header in enumerate(attr_headers):
            if header in ['wallet_level', 'level']:
                # wallet_attrs排除了wallet_id，所以索引要减1
                if j > 0 and j-1 < len(wallet_attrs):
                    return str(wallet_attrs[j-1])
        return '2'  # 默认二类钱包


    def _check_wallet_limits(self, wallet_id: str, transaction_amount: float, 
                            current_daily_total: float, wallet_to_attrs: dict, 
                            attr_headers: list) -> Tuple[bool, str]:
        """检查单笔交易和日累计交易是否超过钱包等级限额"""
        wallet_attrs = wallet_to_attrs.get(wallet_id)
        if not wallet_attrs:
            return False, f"钱包ID {wallet_id} 未找到属性"
        
        wallet_level_idx = attr_headers.index('wallet_level')
        wallet_level_str = wallet_attrs[wallet_level_idx]
        if not wallet_level_str or wallet_level_str.strip() == '':
            wallet_level = 2  # 默认为2级钱包
        else:
            # 处理 -99 或 -99.0 的情况（空值被替换为 -99）
            wallet_level_str_clean = str(wallet_level_str).strip()
            if wallet_level_str_clean in ['-99', '-99.0', '-99.00']:
                wallet_level = 2  # 默认为2级钱包
            else:
                # 先转换为 float 再转换为 int，以处理 "1.0" 这种情况
                try:
                    wallet_level = int(float(wallet_level_str_clean))
                except (ValueError, TypeError):
                    wallet_level = 2  # 转换失败时默认为2级钱包
        
        limits = self.get_wallet_level_limits(wallet_level)
        
        single_limit = limits.get("single_transaction_limit")
        daily_limit = limits.get("daily_limit")
        
        if single_limit is not None and transaction_amount > single_limit:
            return False, f"单笔交易{transaction_amount:.2f}元超过{wallet_level}类钱包限额{single_limit}元"
        
        if daily_limit is not None and current_daily_total > daily_limit:
            return False, f"日累计{current_daily_total:.2f}元超过{wallet_level}类钱包限额{daily_limit}元"
        
        return True, ""
    
    def is_preferred_time(self, current_time: datetime, transaction_type: str = None) -> bool:
        """检查当前时间是否适合生成交易"""
        if not transaction_type:
            return True  # 如果没有指定类型，默认允许
        
        config = self.get_transaction_config(transaction_type)
        time_preference = config.get("time_preference", None)
        
        if not time_preference:
            return True  # 如果没有时间偏好配置，默认允许
        
        hour = current_time.hour
        peak_hours = time_preference.get("peak_hours", [])
        avoid_hours = time_preference.get("avoid_hours", [])
        peak_weight = time_preference.get("peak_weight", 0.7)
        normal_weight = time_preference.get("normal_weight", 0.3)
        
        # 完全避开指定时段
        if hour in avoid_hours:
            return False
        
        # 根据时段权重决定
        if hour in peak_hours:
            return random.random() < peak_weight
        else:
            return random.random() < normal_weight

    def get_time_preference_weight(self, current_time: datetime, transaction_type: str) -> float:
        """获取当前时间的时间偏好权重"""
        config = self.get_transaction_config(transaction_type)
        time_preference = config.get("time_preference", None)
        
        if not time_preference:
            return 1.0  # 默认权重
        
        hour = current_time.hour
        peak_hours = time_preference.get("peak_hours", [])
        avoid_hours = time_preference.get("avoid_hours", [])
        peak_weight = time_preference.get("peak_weight", 0.7)
        normal_weight = time_preference.get("normal_weight", 0.3)
        
        if hour in avoid_hours:
            return 0.0
        elif hour in peak_hours:
            return peak_weight
        else:
            return normal_weight
    # 在 transaction_manager.py 中添加以下方法

    def get_wallet_bank_code(self, wallet_id: str, wallet_to_attrs: Dict, attr_headers: List[str]) -> str:
        """获取钱包的银行代码"""
        if wallet_id not in wallet_to_attrs:
            return None
        
        wallet_attrs = wallet_to_attrs[wallet_id]
        for i, header in enumerate(attr_headers):
            if header == 'bank_account_number' and i < len(wallet_attrs):
                bank_account = wallet_attrs[i]
                if bank_account and len(bank_account) >= 4:
                    return bank_account[:4]  # 返回前4位银行代码
        return None

    def get_wallet_region_code(self, wallet_id: str, wallet_to_attrs: Dict, attr_headers: List[str]) -> str:
        """获取钱包的地区代码"""
        if wallet_id not in wallet_to_attrs:
            return None
        
        wallet_attrs = wallet_to_attrs[wallet_id]
        for i, header in enumerate(attr_headers):
            if header == 'region_code' and i < len(wallet_attrs):
                region_code = wallet_attrs[i]
                if region_code and len(region_code) >= 4:
                    return region_code[:4]  # 返回前4位地区代码
        return None

    def is_cross_bank_transaction(self, source_wallet: str, target_wallet: str, 
                                wallet_to_attrs: Dict, attr_headers: List[str]) -> bool:
        """判断是否为跨行交易"""
        source_bank = self.get_wallet_bank_code(source_wallet, wallet_to_attrs, attr_headers)
        target_bank = self.get_wallet_bank_code(target_wallet, wallet_to_attrs, attr_headers)
        
        if not source_bank or not target_bank:
            return False
        
        return source_bank != target_bank

    def is_same_region_transaction(self, source_wallet: str, target_wallet: str, 
                                wallet_to_attrs: Dict, attr_headers: List[str]) -> bool:
        """判断是否为同地区交易"""
        source_region = self.get_wallet_region_code(source_wallet, wallet_to_attrs, attr_headers)
        target_region = self.get_wallet_region_code(target_wallet, wallet_to_attrs, attr_headers)
        
        if not source_region or not target_region:
            return False
        
        return source_region == target_region

    def apply_cross_bank_preference(self, available_wallets: List[str], source_wallet: str,
                                   wallet_to_attrs: Dict, attr_headers: List[str], 
                                   transaction_type: str, rng) -> List[str]:
        """应用跨行偏好（通用方法）"""
        config = self.get_transaction_config(transaction_type)
        cross_bank_preference = config.get("cross_bank_preference", 0.5)
        
        if not available_wallets:
            return available_wallets
        
        # 计算每个钱包的偏好权重
        wallet_weights = []
        for wallet in available_wallets:
            weight = 1.0
            
            # 跨行偏好：根据配置的概率选择跨行交易
            if self.is_cross_bank_transaction(source_wallet, wallet, wallet_to_attrs, attr_headers):
                weight *= cross_bank_preference
            else:
                weight *= (1 - cross_bank_preference)
            
            wallet_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(wallet_weights)
        if total_weight > 0:
            wallet_weights = [w / total_weight for w in wallet_weights]
        else:
            wallet_weights = [1.0 / len(available_wallets)] * len(available_wallets)
        
        # 按权重选择钱包
        selected_wallets = []
        available_wallets_copy = available_wallets.copy()
        wallet_weights_copy = wallet_weights.copy()
        
        for _ in range(min(len(available_wallets_copy), 10)):  # 最多选择10个钱包
            if not available_wallets_copy:
                break
            
            # 按权重随机选择
            selected_idx = rng.choice(len(available_wallets_copy), p=wallet_weights_copy)
            selected_wallet = available_wallets_copy.pop(selected_idx)
            selected_wallets.append(selected_wallet)
            
            # 重新归一化剩余钱包的权重
            if available_wallets_copy:
                remaining_weights = [wallet_weights_copy[i] for i in range(len(wallet_weights_copy)) if i != selected_idx]
                total_weight = sum(remaining_weights)
                if total_weight > 0:
                    wallet_weights_copy = [w / total_weight for w in remaining_weights]
                else:
                    wallet_weights_copy = [1.0 / len(available_wallets_copy)] * len(available_wallets_copy)
        
        return selected_wallets
    
    def apply_merchant_laundering_preferences(self, available_wallets: List[str], source_wallet: str,
                                            wallet_to_attrs: Dict, attr_headers: List[str], 
                                            rng) -> List[str]:
        """应用商户跑分的跨行和同地区偏好"""
        config = self.get_transaction_config("merchant_laundering")
        cross_bank_preference = config.get("cross_bank_preference", 0.9)
        same_region_preference = config.get("same_region_preference", 0.8)
        
        if not available_wallets:
            return available_wallets
        
        # 计算每个钱包的偏好权重
        wallet_weights = []
        for wallet in available_wallets:
            weight = 1.0
            
            # 跨行偏好：90%概率选择跨行交易
            if self.is_cross_bank_transaction(source_wallet, wallet, wallet_to_attrs, attr_headers):
                weight *= cross_bank_preference
            else:
                weight *= (1 - cross_bank_preference)
            
            # 同地区偏好：80%概率选择同地区交易
            if self.is_same_region_transaction(source_wallet, wallet, wallet_to_attrs, attr_headers):
                weight *= same_region_preference
            else:
                weight *= (1 - same_region_preference)
            
            wallet_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(wallet_weights)
        if total_weight > 0:
            wallet_weights = [w / total_weight for w in wallet_weights]
        else:
            wallet_weights = [1.0 / len(available_wallets)] * len(available_wallets)
        
        # 按权重选择钱包
        selected_wallets = []
        for _ in range(min(len(available_wallets), 10)):  # 最多选择10个钱包
            if not available_wallets:
                break
            
            # 按权重随机选择
            selected_idx = rng.choice(len(available_wallets), p=wallet_weights)
            selected_wallet = available_wallets.pop(selected_idx)
            selected_wallets.append(selected_wallet)
            
            # 重新归一化剩余钱包的权重
            if available_wallets:
                remaining_weights = [wallet_weights[i] for i in range(len(wallet_weights)) if i != selected_idx]
                total_weight = sum(remaining_weights)
                if total_weight > 0:
                    wallet_weights = [w / total_weight for w in remaining_weights]
                else:
                    wallet_weights = [1.0 / len(available_wallets)] * len(available_wallets)
        
        return selected_wallets
    
    def output_transaction_mode_distribution(self):
        """输出每个交易类型下 transaction mode 的占比"""
        print("\n" + "=" * 80)
        print("=== 交易类型下 Transaction Mode 占比统计 ===")
        print("=" * 80)
        
        # 按交易类型分类（正常交易和风险交易）
        normal_types = []
        risk_types = []
        
        for tx_type in self.transaction_types.keys():
            if tx_type in self.normal_transaction_types:
                normal_types.append(tx_type)
            elif tx_type in self.risk_transaction_types:
                risk_types.append(tx_type)
        
        # 输出正常交易类型
        if normal_types:
            print("\n【正常交易类型】")
            print("-" * 80)
            for tx_type in sorted(normal_types):
                config = self.get_transaction_config(tx_type)
                name = config.get("name", tx_type)
                mode_dist = config.get("transaction_mode_distribution", {})
                
                print(f"\n交易类型: {tx_type} ({name})")
                if mode_dist:
                    # 计算总权重（用于验证是否归一化）
                    total_weight = sum(mode_dist.values())
                    
                    # 按占比排序输出
                    sorted_modes = sorted(mode_dist.items(), key=lambda x: x[1], reverse=True)
                    for mode, weight in sorted_modes:
                        percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                        print(f"  - {mode:20s}: {weight:.4f} ({percentage:6.2f}%)")
                    
                    # 如果权重未归一化，给出提示
                    if abs(total_weight - 1.0) > 0.01:
                        print(f"  注意: 总权重为 {total_weight:.4f}，未归一化")
                else:
                    print("  - 未配置 transaction_mode_distribution")
        
        # 输出风险交易类型
        if risk_types:
            print("\n【风险交易类型】")
            print("-" * 80)
            for tx_type in sorted(risk_types):
                config = self.get_transaction_config(tx_type)
                name = config.get("name", tx_type)
                mode_dist = config.get("transaction_mode_distribution", {})
                
                print(f"\n交易类型: {tx_type} ({name})")
                if mode_dist:
                    # 计算总权重（用于验证是否归一化）
                    total_weight = sum(mode_dist.values())
                    
                    # 按占比排序输出
                    sorted_modes = sorted(mode_dist.items(), key=lambda x: x[1], reverse=True)
                    for mode, weight in sorted_modes:
                        percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                        print(f"  - {mode:20s}: {weight:.4f} ({percentage:6.2f}%)")
                    
                    # 如果权重未归一化，给出提示
                    if abs(total_weight - 1.0) > 0.01:
                        print(f"  注意: 总权重为 {total_weight:.4f}，未归一化")
                else:
                    print("  - 未配置 transaction_mode_distribution")
        
        # 统计所有 transaction mode 的使用情况
        print("\n" + "-" * 80)
        print("【Transaction Mode 使用情况汇总】")
        print("-" * 80)
        
        all_modes = set()
        mode_usage = {}  # {mode: [list of transaction types]}
        
        for tx_type in self.transaction_types.keys():
            config = self.get_transaction_config(tx_type)
            mode_dist = config.get("transaction_mode_distribution", {})
            for mode in mode_dist.keys():
                all_modes.add(mode)
                if mode not in mode_usage:
                    mode_usage[mode] = []
                mode_usage[mode].append(tx_type)
        
        for mode in sorted(all_modes):
            tx_types = mode_usage[mode]
            print(f"  {mode:20s}: 被 {len(tx_types)} 个交易类型使用 - {', '.join(tx_types)}")
        
        print("=" * 80 + "\n")