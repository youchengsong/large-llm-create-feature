#!/usr/bin/env python3
"""
统一的交易生成器
基于TransactionManager生成各种类型的交易
支持所有钱包类型进行所有交易类型
"""

from transaction_manager import TransactionManager
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import networkx as nx

class UnifiedTransactionGenerator:
    """统一的交易生成器"""
    
    def __init__(self, config_path: str = "transaction_config.json"):
            """初始化交易生成器"""
            self.manager = TransactionManager(config_path)
            self.single_tx_counter = 1000
            self.risk_pattern_counter = 2000
    
    def _filter_wallets_by_preference(self, wallets: List[str], is_risk: bool) -> Tuple[List[str], List[str]]:
        """
        根据交易类型和钱包偏好过滤钱包
        返回: (preferred_wallets, non_preferred_wallets)
        """
        preferred = []
        non_preferred = []
        
        for wallet in wallets:
            pref = self.wallet_tx_type_preference.get(wallet, None)
            if is_risk:
                # 风险交易：优先选择偏好为'risk'或None的钱包
                if pref in ['risk', None]:
                    preferred.append(wallet)
                elif pref == 'normal':
                    non_preferred.append(wallet)
                # 忽略'mixed'钱包（因为已达到混合限制）
            else:
                # 正常交易：优先选择偏好为'normal'或None的钱包
                if pref in ['normal', None]:
                    preferred.append(wallet)
                elif pref == 'risk':
                    non_preferred.append(wallet)
                # 忽略'mixed'钱包（因为已达到混合限制）
        
        return preferred, non_preferred
    
    def generate_transaction(self, transaction_type: str, g: nx.DiGraph, 
                           source_wallet: str, current_balances: Dict[str, float],
                           current_time: datetime, tx_id: int, max_frac_per_tx: float,
                           min_abs_amount: float, rng, wallet_to_attrs: Dict = None,
                           attr_headers: List[str] = None, motifs: List[Dict] = None,
                           wallet_tx_type_preference: Dict = None) -> Tuple[List[Dict], int, datetime]:
        """统一生成交易的主入口"""
        # 设置钱包属性以便在生成过程中使用
        self.wallet_to_attrs = wallet_to_attrs or {}
        self.attr_headers = attr_headers or []
        self.wallet_tx_type_preference = wallet_tx_type_preference or {}
        
        # 验证交易条件
        if not self.manager.validate_transaction_conditions(transaction_type, source_wallet, 
                                                           wallet_to_attrs or {}, attr_headers or [], current_time):
            return [], tx_id, current_time
        # 选择transaction_mode
        # transaction_mode = self.manager.select_transaction_mode(transaction_type, rng)
        available_modes = self._get_available_modes(g, source_wallet, transaction_type)
        if not available_modes:
            return [], tx_id, current_time  
        config = self.manager.get_transaction_config(transaction_type)
        mode_distribution = config.get("transaction_mode_distribution", {})
        
        # 过滤出可用的模式及其权重
        available_weights = {}
        for mode in available_modes:
            if mode in mode_distribution:
                available_weights[mode] = mode_distribution[mode]
        
        if not available_weights:
            return [], tx_id, current_time
        
        # 按权重选择模式
        modes = list(available_weights.keys())
        weights = list(available_weights.values())
        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        transaction_mode = rng.choice(modes, p=normalized_weights)
       
        # 根据交易类型和模式选择生成策略
        try:
            if transaction_type == "single_transaction":
                return self._generate_single_transaction_by_mode(transaction_type, transaction_mode, g, source_wallet, 
                                                           current_balances, current_time, tx_id, 
                                                           max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
            elif transaction_type == "normal_small_high_freq":
                return self._generate_normal_small_high_freq_by_mode(transaction_type, transaction_mode, g, source_wallet, 
                                                               current_balances, current_time, tx_id, 
                                                               max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
            elif transaction_type == "regular_large_low_freq":
                return self._generate_regular_large_low_freq(transaction_type, g, source_wallet,
                                                           current_balances, current_time, tx_id,
                                                           max_frac_per_tx, min_abs_amount, rng,
                                                           wallet_to_attrs, attr_headers)
            
            elif transaction_type == "small_amount_testing":
                return self._generate_small_amount_testing(transaction_type, g, source_wallet,
                                                         current_balances, current_time, tx_id,
                                                         max_frac_per_tx, min_abs_amount, rng,
                                                         wallet_to_attrs, attr_headers, motifs)
            
            elif transaction_type == "merchant_laundering":
                return self._generate_merchant_laundering(transaction_type, g, source_wallet,
                                                        current_balances, current_time, tx_id,
                                                        max_frac_per_tx, min_abs_amount, rng,
                                                        wallet_to_attrs, attr_headers, str(tx_id), 1)
            
            elif transaction_type == "class4_laundering":
                return self._generate_class4_laundering(transaction_type, g, source_wallet,
                                                      current_balances, current_time, tx_id,
                                                      max_frac_per_tx, min_abs_amount, rng,
                                                      wallet_to_attrs, attr_headers)
            
            elif transaction_type == "online_laundering":
                return self._generate_online_laundering(transaction_type, g, source_wallet,
                                                      current_balances, current_time, tx_id,
                                                      max_frac_per_tx, min_abs_amount, rng,
                                                      wallet_to_attrs, attr_headers)
            
            else:
                return [], tx_id, current_time
        except Exception as e:
            return [], tx_id, current_time
        
    def _get_available_modes(self, g: nx.DiGraph, source_wallet: str, transaction_type: str) -> List[str]:
        """获取可用的交易模式 - 基于图结构预检查"""
        predecessors = list(g.predecessors(source_wallet))
        successors = list(g.successors(source_wallet))
        
        available_modes = []
        
        # 检查各种模式的可行性
        if len(successors) >= 1:
            available_modes.append('single')
        
        if len(successors) >= 2:
            available_modes.append('fan_out')
        
        # one_to_many需要：至少1个上游钱包 + 至少2个下游钱包
        if len(predecessors) >= 1 and len(successors) >= 2:
            available_modes.append('one_to_many')
        
        if len(predecessors) >= 2:
            available_modes.append('fan_in')
        
        # many_to_one需要：至少2个上游钱包 + 至少1个下游钱包
        if len(predecessors) >= 2 and len(successors) >= 1:
            available_modes.append('many_to_one')
        
        if len(predecessors) >= 1 and len(successors) >= 1:
            available_modes.append('forward')
        
        if len(predecessors) >= 2 and len(successors) >= 2:
            available_modes.append('many_to_many')

        # 根据交易类型过滤模式
        if transaction_type == "single_transaction":
            # single_transaction只支持single和forward模式
            return [mode for mode in available_modes if mode in ['single', 'forward']]
        elif transaction_type == "normal_small_high_freq":
            # normal_small_high_freq支持所有5个模式
            return [mode for mode in available_modes if mode in ['fan_out', 'fan_in', 'one_to_many', 'many_to_one', 'many_to_many']]
        else:
            # 其他类型支持所有可用模式
            return available_modes

    def _select_transaction_mode_with_weights(self, g: nx.DiGraph, source_wallet: str, 
                                            transaction_type: str, rng) -> str:
        """通用的交易模式选择方法 - 结合图结构约束和配置权重"""
        
        # 步骤1：获取配置
        config = self.manager.get_transaction_config(transaction_type)
        mode_distribution = config.get("transaction_mode_distribution", {})
        
        # 步骤2：根据图结构获取可用模式
        available_modes = self._get_available_modes(g, source_wallet, transaction_type)
        
        if not available_modes:
            return None  # 没有可用模式
        
        # 步骤3：过滤出可用的模式及其权重
        available_weights = {}
        for mode in available_modes:
            if mode in mode_distribution:
                available_weights[mode] = mode_distribution[mode]
        
        # 步骤4：如果没有配置权重，使用均等权重
        if not available_weights:
            # 使用均等权重
            modes = available_modes
            weights = [1.0 / len(available_modes)] * len(available_modes)
        else:
            # 步骤5：归一化权重
            modes = list(available_weights.keys())
            weights = list(available_weights.values())
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(modes)] * len(modes)
        
        # 步骤6：按权重随机选择
        return rng.choice(modes, p=weights)
    def _generate_single_transaction_by_mode(self, risk_type: str, transaction_mode: str, g: nx.DiGraph,
                                        source_wallet: str, current_balances: Dict[str, float],
                                        current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                        min_abs_amount: float, rng, wallet_to_attrs: Dict = None,
                                        attr_headers: List[str] = None) -> Tuple[List[Dict], int, datetime]:
        """根据transaction_mode生成single_transaction交易（只支持single和forward）"""
        config = self.manager.get_transaction_config(risk_type)
        time_preference = config.get("time_preference", None)
        if transaction_mode == 'single':
            return self._generate_single_mode(risk_type, g, source_wallet, current_balances, 
                                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, time_preference)
        elif transaction_mode == 'forward':
            # 尝试生成forward模式交易
            forward_transactions, new_tx_id, new_current_time = self._generate_forward_mode(risk_type, g, source_wallet, current_balances, 
                                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, time_preference)
            
            # 如果forward模式无法生成完整交易，回退到single模式
            if not forward_transactions:
                return self._generate_single_mode(risk_type, g, source_wallet, current_balances, 
                                                current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, time_preference)
            else:
                return forward_transactions, new_tx_id, new_current_time
        else:
            # 默认使用single模式
            return self._generate_single_mode(risk_type, g, source_wallet, current_balances, 
                                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, time_preference)

    def _generate_normal_small_high_freq_by_mode(self, risk_type: str, transaction_mode: str, g: nx.DiGraph,
                                            source_wallet: str, current_balances: Dict[str, float],
                                            current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                            min_abs_amount: float, rng, wallet_to_attrs: Dict = None,
                                            attr_headers: List[str] = None) -> Tuple[List[Dict], int, datetime]:
        """根据transaction_mode生成normal_small_high_freq交易（支持所有5个模式）"""
        config = self.manager.get_transaction_config(risk_type)
        time_preference = config.get("time_preference", None)
    
        if transaction_mode == 'fan_out':
            return self._generate_fan_out_mode(risk_type, g, source_wallet, current_balances, 
                                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, time_preference)
        elif transaction_mode == 'fan_in':
            return self._generate_fan_in_mode(risk_type, g, source_wallet, current_balances, 
                                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, time_preference)
        elif transaction_mode == 'one_to_many':
            return self._generate_one_to_many_mode(risk_type, g, source_wallet, current_balances, 
                                                current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers, time_preference)
        elif transaction_mode == 'many_to_one':
            return self._generate_many_to_one_mode(risk_type, g, source_wallet, current_balances, 
                                                current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers, time_preference)
        elif transaction_mode == 'many_to_many':
            return self._generate_many_to_many_mode(risk_type, g, source_wallet, current_balances, 
                                                current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers, time_preference)
    
    def _generate_regular_large_low_freq(self, transaction_type: str, g: nx.DiGraph,
                                   source_wallet: str, current_balances: Dict[str, float],
                                   current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                   min_abs_amount: float, rng, wallet_to_attrs: Dict = None,
                                   attr_headers: List = None) -> Tuple[List[Dict], int, datetime]:
        """生成定期发工资交易 - 支持预留模式和正常模式"""
        
        # 检查时间偏好
        if not self.manager.is_preferred_time(current_time, transaction_type):
            return [], tx_id, current_time
            
        config = self.manager.get_transaction_config(transaction_type)
        time_preference = config.get("time_preference", None)
        # 检查是否是预留模式
        is_reservation_mode = getattr(self, '_reservation_mode', False)
        
        if is_reservation_mode:
            return self._generate_reservation_transactions(transaction_type, g, source_wallet, 
                                                        current_time, tx_id, rng)
        
        # 原有的正常生成逻辑（简化版，只生成当前月份）
        transactions = []
        motif_id = str(self.single_tx_counter)
        self.single_tx_counter += 1
        
        # 获取配置
        config = self.manager.get_transaction_config(transaction_type)
        amount_range = config.get("amount_range", {"min": 10, "max": 100})
        salary_config = config.get("salary_config", {})
        
        # 获取发工资配置
        intervals = salary_config.get("intervals", [10, 30])  # 发工资间隔（天）
        interval_weights = salary_config.get("interval_weights", [0.6, 0.4])
        same_day_delay = salary_config.get("same_day_delay", {"min": 1, "max": 10})
        salary_variation = salary_config.get("salary_amount_variation", 0.1)
        
        # 选择发工资间隔
        salary_interval_days = rng.choice(intervals, p=interval_weights)
        
        # 生成多批交易（-1, -2, -3, -4, -5）
        num_batches = rng.integers(2, 6)  # 生成2-5批交易
        
        # 获取所有员工钱包（下游钱包）
        successors = list(g.successors(source_wallet))
        if len(successors) < 1:
            return transactions, tx_id, current_time
        
        # 选择交易模式
        selected_mode = self._select_transaction_mode_with_weights(g, source_wallet, transaction_type, rng)
        
        if not selected_mode:
            return transactions, tx_id, current_time
        
        # 基础工资金额（所有员工使用相同的基础金额）
        base_salary = rng.uniform(amount_range["min"], amount_range["max"])
        
        if selected_mode == "single":
            # 单笔发工资模式
            if successors:
                target_wallet = rng.choice(successors)
                
                # 工资金额微调（±10%变化）
                variation = rng.uniform(1 - salary_variation, 1 + salary_variation)
                salary_amount = base_salary * variation
                salary_amount = max(salary_amount, min_abs_amount)
                
                # 检查余额
                if current_balances[source_wallet] >= salary_amount:
                    current_balances[source_wallet] -= salary_amount
                    current_balances[target_wallet] += salary_amount
                    
                    transactions.append({
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet,
                        "dst": target_wallet,
                        "amount": round(salary_amount, 2),
                        "risk_type": transaction_type,
                        "motif_id": f"{motif_id}-1",  # 当前月份标记为-1
                        "transaction_mode": "single",
                        "salary_interval_days": salary_interval_days,
                        "base_salary": base_salary,
                        "target_wallet": target_wallet
                    })
                    tx_id += 1
        
        elif selected_mode == "fan_out":
            # 一对多发工资模式
            if len(successors) >= 2:
                # 选择员工数量（3-8人，但不超过可用钱包数）
                num_employees = min(len(successors), rng.integers(3, max(3 + 1, 9)))
                selected_employees = rng.choice(successors, size=num_employees, replace=False)
                
                # 为每个员工生成工资（同一天内快速完成）
                for i, employee_wallet in enumerate(selected_employees):
                    # 工资金额微调（±10%变化，但基础金额相同）
                    variation = rng.uniform(1 - salary_variation, 1 + salary_variation)
                    salary_amount = base_salary * variation
                    salary_amount = max(salary_amount, min_abs_amount)
                    
                    # 检查余额
                    if current_balances[source_wallet] >= salary_amount:
                        current_balances[source_wallet] -= salary_amount
                        current_balances[employee_wallet] += salary_amount
                        
                        transactions.append({
                            "tx_id": tx_id,
                            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "src": source_wallet,
                            "dst": employee_wallet,
                            "amount": round(salary_amount, 2),
                            "risk_type": transaction_type,
                            "motif_id": f"{motif_id}-1",  # 当前月份标记为-1
                            "transaction_mode": "one_to_many",
                            "salary_interval_days": salary_interval_days,
                            "base_salary": base_salary,
                            "employee_list": selected_employees.tolist()
                        })
                        tx_id += 1
                        
                        # 同一天内的微小延迟（1-10秒）
                        current_time += timedelta(seconds=int(rng.integers(
                            same_day_delay["min"], 
                            same_day_delay["max"] + 1
                        )))
                    else:
                        # 余额不足，回滚之前的交易
                        for j in range(i):
                            prev_tx = transactions[-1-j]
                            current_balances[source_wallet] += prev_tx["amount"]
                            current_balances[prev_tx["dst"]] -= prev_tx["amount"]
                        transactions = transactions[:-i]  # 移除已添加的交易
                        break
        
        return transactions, tx_id, current_time
    def _generate_reservation_transactions(self, transaction_type: str, g: nx.DiGraph,
                                    source_wallet: str, current_time: datetime, 
                                    tx_id: int, rng) -> Tuple[List[Dict], int, datetime]:
        """生成预留交易（只包含时间和地址信息，不包含金额）- 生成多批交易"""
        transactions = []
        motif_id = str(tx_id)
        
        # 获取配置
        config = self.manager.get_transaction_config(transaction_type)
        amount_range = config.get("amount_range", {"min": 10, "max": 100})
        salary_config = config.get("salary_config", {})
        
        # 获取发工资配置
        intervals = salary_config.get("intervals", [10, 30])  # 发工资间隔（天）
        interval_weights = salary_config.get("interval_weights", [0.6, 0.4])
        same_day_delay = salary_config.get("same_day_delay", {"min": 1, "max": 10})
        salary_variation = salary_config.get("salary_amount_variation", 0.1)
        
        # 选择发工资间隔
        salary_interval_days = rng.choice(intervals, p=interval_weights)
        
        # 生成多批交易（-1, -2, -3, -4, -5）
        num_batches = rng.integers(2, 6)  # 生成2-5批交易
        
        # 获取所有员工钱包（下游钱包）
        successors = list(g.successors(source_wallet))
        if len(successors) < 1:
            return transactions, tx_id, current_time
        
        # 选择交易模式
        selected_mode = self._select_transaction_mode_with_weights(g, source_wallet, transaction_type, rng)
        
        if not selected_mode:
            return transactions, tx_id, current_time
        
        # 基础工资金额（所有员工使用相同的基础金额）
        base_salary = rng.uniform(amount_range["min"], amount_range["max"])
        
        if selected_mode == "single":
            # 单笔发工资预留 - 生成多批交易
            if successors:
                target_wallet = rng.choice(successors)
                
                # 生成多批交易
                for batch_num in range(1, num_batches + 1):
                    # 计算当前批次的时间（日期）
                    batch_time = current_time + timedelta(days=int(salary_interval_days * (batch_num - 1)))
                    
                    # 为每个批次添加随机的时间偏移（时、分、秒），避免批次间时间戳相同
                    batch_time = batch_time.replace(
                        hour=int(rng.integers(9, 18)),  # 9:00-17:59 办公时间
                        minute=int(rng.integers(0, 60)),
                        second=int(rng.integers(0, 60))
                    )
                    
                    transactions.append({
                        "tx_id": tx_id,
                        "timestamp": batch_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet,
                        "dst": target_wallet,
                        "amount": 0,  # 预留，稍后填充
                        "risk_type": transaction_type,
                        "motif_id": f"{motif_id}-{batch_num}",  # 批次标记
                        "transaction_mode": "single",
                        "is_reservation": True,  # 标记为预留交易
                        "reservation_data": {
                            "base_salary": base_salary,
                            "amount_range": amount_range,
                            "target_wallet": target_wallet,
                            "min_abs_amount": 0.01,
                            "salary_interval_days": salary_interval_days,
                            "salary_variation": salary_variation,
                            "batch_number": batch_num,
                            "total_batches": num_batches
                        }
                    })
                    tx_id += 1
        
        elif selected_mode == "fan_out":
            # 一对多发工资预留 - 生成多批交易
            if len(successors) >= 2:
                # 选择员工数量（3-8人，但不超过可用钱包数）
                num_employees = min(len(successors), rng.integers(3, max(3 + 1, 9)))
                selected_employees = rng.choice(successors, size=num_employees, replace=False)
                
                # 生成多批交易
                for batch_num in range(1, num_batches + 1):
                    # 计算当前批次的时间（日期）
                    batch_time = current_time + timedelta(days=int(salary_interval_days * (batch_num - 1)))
                    
                    # 为每个批次添加随机的时间偏移（时、分、秒），避免批次间时间戳相同
                    batch_time = batch_time.replace(
                        hour=int(rng.integers(9, 18)),  # 9:00-17:59 办公时间
                        minute=int(rng.integers(0, 60)),
                        second=int(rng.integers(0, 60))
                    )
                    
                    # 为每个员工生成当前批次的交易
                    for i, employee_wallet in enumerate(selected_employees):
                        transactions.append({
                            "tx_id": tx_id,
                            "timestamp": batch_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "src": source_wallet,
                            "dst": employee_wallet,
                            "amount": 0,  # 预留，稍后填充
                            "risk_type": transaction_type,
                            "motif_id": f"{motif_id}-{batch_num}",  # 批次标记
                            "transaction_mode": "fan_out",
                            "is_reservation": True,  # 标记为预留交易
                            "reservation_data": {
                                "base_salary": base_salary,
                                "amount_range": amount_range,
                                "employee_list": selected_employees.tolist(),
                                "min_abs_amount": 0.01,
                                "salary_interval_days": salary_interval_days,
                                "salary_variation": salary_variation,
                                "batch_number": batch_num,
                                "total_batches": num_batches
                            }
                        })
                        tx_id += 1
                        
                        # 同一天内的微小延迟（1-10秒）
                        batch_time += timedelta(seconds=int(rng.integers(
                            same_day_delay["min"], 
                            same_day_delay["max"] + 1
                        )))
        
        return transactions, tx_id, current_time
    
    def _generate_small_amount_testing(self, transaction_type: str, g: nx.DiGraph,
                                source_wallet: str, current_balances: Dict[str, float],
                                current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                min_abs_amount: float, rng, wallet_to_attrs: Dict,
                                attr_headers: List[str], motifs: List[Dict]) -> Tuple[List[Dict], int, datetime]:
        """生成小额试探交易 - 长期休眠账户突然进行小额试探"""
        transactions = []
        motif_id = str(self.risk_pattern_counter)
        self.risk_pattern_counter += 1
        
        # 检查时间偏好
        if not self.manager.is_preferred_time(current_time, transaction_type):
            return [], tx_id, current_time
        
        # 获取配置
        config = self.manager.get_transaction_config(transaction_type)
        amount_range = config.get("amount_range", {"min": 0.01, "max": 0.1})
        time_interval = config.get("time_interval", {"min": 60, "max": 300})
        
        # 固定的小额金额列表（0.1, 0.4, 0.5, 1.0）
        small_amount_options = [0.1, 0.4, 0.5, 1.0]
        
        # 使用配置文件中的交易数量（如果存在）
        transaction_count = config.get("transaction_count", {"min": 3, "max": 8})
        num_transactions = rng.integers(transaction_count["min"], max(transaction_count["min"] + 1, transaction_count["max"] + 1))
        
        # 检查是否是休眠钱包（完善验证逻辑）
        if wallet_to_attrs and attr_headers:
            wallet_attrs = wallet_to_attrs.get(source_wallet, [])
            wallet_open_date = None
            
            for j, header in enumerate(attr_headers):
                if header in ["wallet_open_date", "open_date"] and j < len(wallet_attrs):
                    try:
                        wallet_open_date = datetime.strptime(wallet_attrs[j], "%Y-%m-%d")
                        break
                    except:
                        pass
            
            # 如果钱包开立时间太近，不生成小额试探交易
            if wallet_open_date:
                days_since_open = (current_time.date() - wallet_open_date.date()).days
                min_dormant_days = config.get("min_dormant_days", 90)  # 默认90天休眠期
                
                if days_since_open < min_dormant_days:
                    return transactions, tx_id, current_time
                dormant_wallet_ratio = config.get("dormant_wallet_ratio", 0.6)
                if rng.random() > dormant_wallet_ratio:
                    return transactions, tx_id, current_time  # 60%概率生成，40%概率跳过
        
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if len(available_wallets) < 2:
            return transactions, tx_id, current_time
        
        # 选择交易模式
        selected_mode = self._select_transaction_mode_with_weights(g, source_wallet, transaction_type, rng)
        
        if not selected_mode:
            return transactions, tx_id, current_time
        
        # 根据选择的模式生成交易
        if selected_mode == "fan_out":
            # 扇出模式：一个钱包向多个钱包转账
            successors = list(g.successors(source_wallet))
            if len(successors) >= 2:
                # 应用跨行偏好
                if wallet_to_attrs and attr_headers:
                    config = self.manager.get_transaction_config(transaction_type)
                    if config.get("cross_bank_preference") is not None:
                        # 使用跨行偏好选择钱包
                        preferred_wallets = self.manager.apply_cross_bank_preference(
                            successors, source_wallet, wallet_to_attrs, attr_headers, transaction_type, rng
                        )
                        # 如果偏好选择返回的钱包不够，从原始列表补充
                        if len(preferred_wallets) >= 2:
                            num_targets = rng.integers(2, min(5, len(preferred_wallets) + 1))
                            target_wallets = preferred_wallets[:num_targets]
                        else:
                            num_targets = rng.integers(2, min(5, len(successors) + 1))
                            target_wallets = rng.choice(successors, size=num_targets, replace=False)
                    else:
                        num_targets = rng.integers(2, min(5, len(successors) + 1))
                        target_wallets = rng.choice(successors, size=num_targets, replace=False)
                else:
                    num_targets = rng.integers(2, min(5, len(successors) + 1))
                    target_wallets = rng.choice(successors, size=num_targets, replace=False)
                
                for i, target_wallet in enumerate(target_wallets):
                    # 从固定的小额金额列表中选择
                    amount = rng.choice(small_amount_options)
                    
                    if current_balances[source_wallet] >= amount:
                        current_balances[source_wallet] -= amount
                        current_balances[target_wallet] += amount
                        
                        transactions.append({
                            "tx_id": tx_id,
                            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "src": source_wallet,
                            "dst": target_wallet,
                            "amount": round(amount, 4),
                            "risk_type": transaction_type,
                            "motif_id": motif_id,
                            "transaction_mode": "fan_out"
                        })
                        tx_id += 1
                        current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                    else:
                        # 如果余额不足，回滚之前的所有交易
                        for j in range(i):
                            prev_tx = transactions[j]
                            current_balances[source_wallet] += prev_tx["amount"]
                            current_balances[prev_tx["dst"]] -= prev_tx["amount"]
                        transactions = []  # 清空交易列表
                        break
                
                # 验证完整性 - 确保生成了至少2笔交易
                if len(transactions) < 2:
                    # 如果交易数量少于2，回滚所有操作并返回空列表
                    for tx in transactions:
                        current_balances[source_wallet] += tx["amount"]
                        current_balances[tx["dst"]] -= tx["amount"]
                    return [], tx_id - len(transactions), current_time
        
        elif selected_mode == "fan_in":
            # 扇入模式：多个钱包向一个钱包转账
            predecessors = list(g.predecessors(source_wallet))
            if len(predecessors) >= 2:
                # 应用跨行偏好（注意：fan_in模式中，source_wallet是目标，predecessors是源）
                if wallet_to_attrs and attr_headers:
                    config = self.manager.get_transaction_config(transaction_type)
                    if config.get("cross_bank_preference") is not None:
                        # 对于fan_in，我们需要反向应用跨行偏好（从predecessors到source_wallet）
                        preferred_wallets = self.manager.apply_cross_bank_preference(
                            predecessors, source_wallet, wallet_to_attrs, attr_headers, transaction_type, rng
                        )
                        if len(preferred_wallets) >= 2:
                            num_sources = rng.integers(2, min(5, len(preferred_wallets) + 1))
                            source_wallets = preferred_wallets[:num_sources]
                        else:
                            num_sources = rng.integers(2, min(5, len(predecessors) + 1))
                            source_wallets = rng.choice(predecessors, size=num_sources, replace=False)
                    else:
                        num_sources = rng.integers(2, min(5, len(predecessors) + 1))
                        source_wallets = rng.choice(predecessors, size=num_sources, replace=False)
                else:
                    num_sources = rng.integers(2, min(5, len(predecessors) + 1))
                    source_wallets = rng.choice(predecessors, size=num_sources, replace=False)
                
                for i, src_wallet in enumerate(source_wallets):
                    # 从固定的小额金额列表中选择
                    amount = rng.choice(small_amount_options)
                    
                    if current_balances[src_wallet] >= amount:
                        current_balances[src_wallet] -= amount
                        current_balances[source_wallet] += amount
                        
                        transactions.append({
                            "tx_id": tx_id,
                            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "src": src_wallet,
                            "dst": source_wallet,
                            "amount": round(amount, 4),
                            "risk_type": transaction_type,
                            "motif_id": motif_id,
                            "transaction_mode": "fan_in"
                        })
                        tx_id += 1
                        current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                    else:
                        # 如果余额不足，回滚之前的所有交易
                        for j in range(i):
                            prev_tx = transactions[j]
                            current_balances[prev_tx["src"]] += prev_tx["amount"]
                            current_balances[source_wallet] -= prev_tx["amount"]
                        transactions = []  # 清空交易列表
                        break
                
                # 验证完整性 - 确保生成了至少2笔交易
                if len(transactions) < 2:
                    # 如果交易数量少于2，回滚所有操作并返回空列表
                    for tx in transactions:
                        current_balances[tx["src"]] += tx["amount"]
                        current_balances[source_wallet] -= tx["amount"]
                    return [], tx_id - len(transactions), current_time
        
        elif selected_mode == "one_to_many":
            # 一进多出：一个钱包向多个钱包转账
            # 应用跨行偏好
            if wallet_to_attrs and attr_headers:
                config = self.manager.get_transaction_config(transaction_type)
                if config.get("cross_bank_preference") is not None:
                    preferred_wallets = self.manager.apply_cross_bank_preference(
                        available_wallets, source_wallet, wallet_to_attrs, attr_headers, transaction_type, rng
                    )
                    if len(preferred_wallets) > 0:
                        available_wallets = preferred_wallets
            
            for i in range(num_transactions):
                if len(available_wallets) == 0:
                    break
                target_wallet = rng.choice(available_wallets)
                # 从固定的小额金额列表中选择
                amount = rng.choice(small_amount_options)
                
                if current_balances[source_wallet] >= amount:
                    current_balances[source_wallet] -= amount
                    current_balances[target_wallet] += amount
                    
                    transactions.append({
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet,
                        "dst": target_wallet,
                        "amount": round(amount, 4),
                        "risk_type": transaction_type,
                        "motif_id": motif_id,
                        "transaction_mode": "one_to_many"
                    })
                    tx_id += 1
                    current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                else:
                    # 如果余额不足，回滚之前的所有交易
                    for j in range(i):
                        prev_tx = transactions[j]
                        current_balances[source_wallet] += prev_tx["amount"]
                        current_balances[prev_tx["dst"]] -= prev_tx["amount"]
                    transactions = []  # 清空交易列表
                    break
            
            # 验证完整性 - 确保生成了至少2笔交易
            if len(transactions) < 2:
                # 如果交易数量少于2，回滚所有操作并返回空列表
                for tx in transactions:
                    current_balances[source_wallet] += tx["amount"]
                    current_balances[tx["dst"]] -= tx["amount"]
                return [], tx_id - len(transactions), current_time
        
        elif selected_mode == "many_to_one":
            # 多进一出：多个钱包向一个钱包转账，然后主钱包转出
            # 应用跨行偏好
            if wallet_to_attrs and attr_headers:
                config = self.manager.get_transaction_config(transaction_type)
                if config.get("cross_bank_preference") is not None:
                    preferred_wallets = self.manager.apply_cross_bank_preference(
                        available_wallets, source_wallet, wallet_to_attrs, attr_headers, transaction_type, rng
                    )
                    if len(preferred_wallets) > 0:
                        available_wallets = preferred_wallets
            
            if len(available_wallets) == 0:
                return [], tx_id, current_time
            target_wallet = rng.choice(available_wallets)
            
            # 先生成转入交易（至少2笔）
            incoming_count = max(2, num_transactions // 2)  # 至少2笔转入
            for i in range(incoming_count):
                if len(available_wallets) == 0:
                    break
                source_wallet_for_tx = rng.choice(available_wallets)
                # 从固定的小额金额列表中选择
                amount = rng.choice(small_amount_options)
                
                if current_balances[source_wallet_for_tx] >= amount:
                    current_balances[source_wallet_for_tx] -= amount
                    current_balances[target_wallet] += amount
                    
                    transactions.append({
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet_for_tx,
                        "dst": target_wallet,
                        "amount": round(amount, 4),
                        "risk_type": transaction_type,
                        "motif_id": motif_id,
                        "transaction_mode": "many_to_one"
                    })
                    tx_id += 1
                    current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                else:
                    # 如果余额不足，回滚之前的所有交易
                    for j in range(i):
                        prev_tx = transactions[j]
                        current_balances[prev_tx["src"]] += prev_tx["amount"]
                        current_balances[target_wallet] -= prev_tx["amount"]
                    transactions = []  # 清空交易列表
                    break
            
            # 再生成转出交易（至少1笔）
            if len(transactions) >= 2:  # 确保有足够的转入交易
                outgoing_count = max(1, num_transactions - incoming_count)  # 至少1笔转出
                for i in range(outgoing_count):
                    target_wallet_for_out = rng.choice(available_wallets)
                    # 从固定的小额金额列表中选择
                    amount = rng.choice(small_amount_options)
                    
                    if current_balances[target_wallet] >= amount:
                        current_balances[target_wallet] -= amount
                        current_balances[target_wallet_for_out] += amount
                        
                        transactions.append({
                            "tx_id": tx_id,
                            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "src": target_wallet,
                            "dst": target_wallet_for_out,
                            "amount": round(amount, 4),
                            "risk_type": transaction_type,
                            "motif_id": motif_id,
                            "transaction_mode": "many_to_one"
                        })
                        tx_id += 1
                        current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                    else:
                        # 如果余额不足，回滚之前的所有交易
                        for j in range(i):
                            prev_tx = transactions[incoming_count + j]
                            current_balances[target_wallet] += prev_tx["amount"]
                            current_balances[prev_tx["dst"]] -= prev_tx["amount"]
                        # 回滚转入交易
                        for j in range(incoming_count):
                            prev_tx = transactions[j]
                            current_balances[prev_tx["src"]] += prev_tx["amount"]
                            current_balances[target_wallet] -= prev_tx["amount"]
                        transactions = []  # 清空交易列表
                        break
            
            # 验证完整性 - 确保生成了至少3笔交易（至少2笔转入+1笔转出）
            if len(transactions) < 3:
                # 如果交易数量少于3，回滚所有操作并返回空列表
                for tx in transactions:
                    if tx["src"] == target_wallet:  # 转出交易
                        current_balances[target_wallet] += tx["amount"]
                        current_balances[tx["dst"]] -= tx["amount"]
                    else:  # 转入交易
                        current_balances[tx["src"]] += tx["amount"]
                        current_balances[target_wallet] -= tx["amount"]
                return [], tx_id - len(transactions), current_time
        
        elif selected_mode == "many_to_many":
            # 多进多出：多个钱包之间的复杂转账
            # 应用跨行偏好
            if wallet_to_attrs and attr_headers:
                config = self.manager.get_transaction_config(transaction_type)
                if config.get("cross_bank_preference") is not None:
                    preferred_wallets = self.manager.apply_cross_bank_preference(
                        available_wallets, source_wallet, wallet_to_attrs, attr_headers, transaction_type, rng
                    )
                    if len(preferred_wallets) > 0:
                        available_wallets = preferred_wallets
            
            for i in range(num_transactions):
                if len(available_wallets) < 2:
                    break
                source_wallet_for_tx = rng.choice(available_wallets)
                target_wallet = rng.choice(available_wallets)
                
                if source_wallet_for_tx != target_wallet:
                    # 从固定的小额金额列表中选择
                    amount = rng.choice(small_amount_options)
                    
                    if current_balances[source_wallet_for_tx] >= amount:
                        current_balances[source_wallet_for_tx] -= amount
                        current_balances[target_wallet] += amount
                        
                        transactions.append({
                            "tx_id": tx_id,
                            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "src": source_wallet_for_tx,
                            "dst": target_wallet,
                            "amount": round(amount, 4),
                            "risk_type": transaction_type,
                            "motif_id": motif_id,
                            "transaction_mode": "many_to_many"
                        })
                        tx_id += 1
                        current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                    else:
                        # 如果余额不足，回滚之前的所有交易
                        for j in range(i):
                            prev_tx = transactions[j]
                            current_balances[prev_tx["src"]] += prev_tx["amount"]
                            current_balances[prev_tx["dst"]] -= prev_tx["amount"]
                        transactions = []  # 清空交易列表
                        break
            
            # 验证完整性 - 确保生成了至少2笔交易
            if len(transactions) < 2:
                # 如果交易数量少于2，回滚所有操作并返回空列表
                for tx in transactions:
                    current_balances[tx["src"]] += tx["amount"]
                    current_balances[tx["dst"]] -= tx["amount"]
                return [], tx_id - len(transactions), current_time
        
        return transactions, tx_id, current_time
    def _generate_merchant_laundering(self, transaction_type: str, g: nx.DiGraph,
                               source_wallet: str, current_balances: Dict[str, float],
                               current_time: datetime, tx_id: int, max_frac_per_tx: float,
                               min_abs_amount: float, rng, wallet_to_attrs: Dict,
                               attr_headers: List[str], motif_id: str, num_transactions: int) -> Tuple[List[Dict], int, datetime]:
        """生成商户跑分交易 - 使用新的transaction_mode系统"""
        # 检查时间偏好
        is_preferred_time = self.manager.is_preferred_time(current_time, transaction_type)
       
        if not is_preferred_time:
            return [], tx_id, current_time
        
        # 选择交易模式
        transaction_mode = self._select_transaction_mode_with_weights(g, source_wallet, transaction_type, rng)
        
        if not transaction_mode:
            return [], tx_id, current_time
        
        # 根据选择的模式生成交易，并应用跨行和同地区偏好
        if transaction_mode == 'one_to_many':
            return self._generate_one_to_many_mode_with_preferences(transaction_type, g, source_wallet, 
                                                                current_balances, current_time, tx_id, 
                                                                max_frac_per_tx, min_abs_amount, rng, 
                                                                wallet_to_attrs, attr_headers)
        elif transaction_mode == 'many_to_many':
            return self._generate_many_to_many_mode_with_preferences(transaction_type, g, source_wallet, 
                                                                current_balances, current_time, tx_id, 
                                                                max_frac_per_tx, min_abs_amount, rng, 
                                                                wallet_to_attrs, attr_headers)
        else:
            return self._generate_many_to_one_mode_with_preferences(transaction_type, g, source_wallet, 
                                                                current_balances, current_time, tx_id, 
                                                                max_frac_per_tx, min_abs_amount, rng, 
                                                                wallet_to_attrs, attr_headers)
    
    def _generate_online_laundering(self, transaction_type: str, g: nx.DiGraph,
                                source_wallet: str, current_balances: Dict[str, float],
                                current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                min_abs_amount: float, rng, wallet_to_attrs: Dict = None,
                                attr_headers: List = None) -> Tuple[List[Dict], int, datetime]:
        """生成线上跑分交易 - 支持多种模式：扇出、扇入、一进多出、多进一出、多进多出"""
        # 检查时间偏好
        if not self.manager.is_preferred_time(current_time, transaction_type):
            return [], tx_id, current_time
            
        transactions = []
        motif_id = str(self.risk_pattern_counter)
        self.risk_pattern_counter += 1
        
        config = self.manager.get_transaction_config(transaction_type)
        transaction_count = config.get("transaction_count", {"min": 5, "max": 15})
        num_transactions = rng.integers(transaction_count["min"], max(transaction_count["min"] + 1, transaction_count["max"] + 1))
        
        # 获取配置参数
        amount_range = config.get("amount_range", {"min": 10, "max": 1000})
        time_interval = config.get("time_interval", {"min": 10, "max": 300})
        
        # 获取可用的交易模式
        available_modes = self._get_available_modes(g, source_wallet, transaction_type)

        if not available_modes:
            return [], tx_id, current_time

        # 使用配置文件中的交易模式分布
        mode_distribution = config.get("transaction_mode_distribution", {})
        
        # 过滤出可用的模式及其权重
        available_weights = {}
        for mode in available_modes:
            if mode in mode_distribution:
                available_weights[mode] = mode_distribution[mode]
        
        if not available_weights:
            return [], tx_id, current_time
        
        # 按权重选择模式
        modes = list(available_weights.keys())
        weights = list(available_weights.values())
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        selected_mode = rng.choice(modes, p=normalized_weights)
        
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if not available_wallets:
            return [], tx_id, current_time
        
        # 应用跨行偏好（如果配置中有）
        if wallet_to_attrs and attr_headers:
            config = self.manager.get_transaction_config(transaction_type)
            if config.get("cross_bank_preference") is not None:
                preferred_wallets = self.manager.apply_cross_bank_preference(
                    available_wallets, source_wallet, wallet_to_attrs, attr_headers, transaction_type, rng
                )
                if len(preferred_wallets) > 0:
                    available_wallets = preferred_wallets
        
        # 根据选择的模式生成指定数量的交易
        if selected_mode == "fan_out":
            # 扇出模式：一个钱包向多个钱包转账
            for i in range(num_transactions):
                if len(available_wallets) == 0:
                    break
                target_wallet = rng.choice(available_wallets)
                amount = rng.uniform(amount_range["min"], amount_range["max"])
                
                if current_balances[source_wallet] >= amount:
                    current_balances[source_wallet] -= amount
                    current_balances[target_wallet] += amount
                    
                    transaction = {
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet,
                        "dst": target_wallet,
                        "amount": round(amount, 2),
                        "risk_type": transaction_type,
                        "motif_id": motif_id,
                        "transaction_mode": "fan_out"
                    }
                    transactions.append(transaction)
                    tx_id += 1
                    current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                else:
                    break  # 余额不足，停止生成
            
        elif selected_mode == "fan_in":
            # 扇入模式：多个钱包向一个钱包转账
            if len(available_wallets) == 0:
                return [], tx_id, current_time
            target_wallet = rng.choice(available_wallets)
            for i in range(num_transactions):
                if len(available_wallets) == 0:
                    break
                source_wallet_for_tx = rng.choice(available_wallets)
                amount = rng.uniform(amount_range["min"], amount_range["max"])
                
                if current_balances[source_wallet_for_tx] >= amount:
                    current_balances[source_wallet_for_tx] -= amount
                    current_balances[target_wallet] += amount
                    
                    transaction = {
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet_for_tx,
                        "dst": target_wallet,
                        "amount": round(amount, 2),
                        "risk_type": transaction_type,
                        "motif_id": motif_id,
                        "transaction_mode": "fan_in"
                    }
                    transactions.append(transaction)
                    tx_id += 1
                    current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                else:
                    break  # 余额不足，停止生成
                    
        elif selected_mode == "one_to_many":
            # 一进多出：一个钱包向多个钱包转账
            for i in range(num_transactions):
                if len(available_wallets) == 0:
                    break
                target_wallet = rng.choice(available_wallets)
                amount = rng.uniform(amount_range["min"], amount_range["max"])
                
                if current_balances[source_wallet] >= amount:
                    current_balances[source_wallet] -= amount
                    current_balances[target_wallet] += amount
                    
                    transaction = {
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet,
                        "dst": target_wallet,
                        "amount": round(amount, 2),
                        "risk_type": transaction_type,
                        "motif_id": motif_id,
                        "transaction_mode": "one_to_many"
                    }
                    transactions.append(transaction)
                    tx_id += 1
                    current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                else:
                    break  # 余额不足，停止生成
                    
        elif selected_mode == "many_to_one":
            # 多进一出：多个钱包向一个钱包转账
            if len(available_wallets) == 0:
                return [], tx_id, current_time
            target_wallet = rng.choice(available_wallets)
            for i in range(num_transactions):
                if len(available_wallets) == 0:
                    break
                source_wallet_for_tx = rng.choice(available_wallets)
                amount = rng.uniform(amount_range["min"], amount_range["max"])
                
                if current_balances[source_wallet_for_tx] >= amount:
                    current_balances[source_wallet_for_tx] -= amount
                    current_balances[target_wallet] += amount
                    
                    transaction = {
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet_for_tx,
                        "dst": target_wallet,
                        "amount": round(amount, 2),
                        "risk_type": transaction_type,
                        "motif_id": motif_id,
                        "transaction_mode": "many_to_one"
                    }
                    transactions.append(transaction)
                    tx_id += 1
                    current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                else:
                    break  # 余额不足，停止生成
                    
        elif selected_mode == "many_to_many":
            # 多进多出：多个钱包之间的复杂转账
            for i in range(num_transactions):
                if len(available_wallets) < 2:
                    break
                source_wallet_for_tx = rng.choice(available_wallets)
                target_wallet = rng.choice(available_wallets)
                
                if source_wallet_for_tx != target_wallet:
                    amount = rng.uniform(amount_range["min"], amount_range["max"])
                    
                    if current_balances[source_wallet_for_tx] >= amount:
                        current_balances[source_wallet_for_tx] -= amount
                        current_balances[target_wallet] += amount
                        
                        transaction = {
                            "tx_id": tx_id,
                            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "src": source_wallet_for_tx,
                            "dst": target_wallet,
                            "amount": round(amount, 2),
                            "risk_type": transaction_type,
                            "motif_id": motif_id,
                            "transaction_mode": "many_to_many"
                        }
                        transactions.append(transaction)
                        tx_id += 1
                        current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
                    else:
                        break  # 余额不足，停止生成
                else:
                    # 如果源钱包和目标钱包相同，跳过这次循环
                    continue
        
        return transactions, tx_id, current_time
    
    def _is_class4_wallet(self, wallet_id: str, wallet_to_attrs: Dict, attr_headers: List[str]) -> bool:
        """检查是否为四类钱包"""
        if wallet_id not in wallet_to_attrs:
            return False
        
        wallet_attrs = wallet_to_attrs[wallet_id]
        
        for i, header in enumerate(attr_headers):
            if header in ['wallet_level', 'level'] and i > 0 and i-1 < len(wallet_attrs):
                wallet_level = wallet_attrs[i-1]
                # 支持字符串"4"和数字4两种类型
                return str(wallet_level) == '4' or wallet_level == 4
        
        return False
    
    def _generate_amount_near_limit(self, min_amount: float, max_amount: float, rng) -> float:
        """
        生成靠近上限的金额（用于异常四类钱包）
        使用beta分布，让金额更可能接近上限
        """
        if max_amount <= min_amount:
            return min_amount
        
        # 使用beta分布，alpha=1, beta=5，让分布偏向最大值
        # 归一化到[min_amount, max_amount]范围
        beta_value = rng.beta(1, 5)  # beta分布，偏向1（即上限）
        amount = min_amount + (max_amount - min_amount) * beta_value
        
        # 确保在范围内
        amount = max(min_amount, min(max_amount, amount))
        return amount
    def _generate_class4_laundering(self, transaction_type: str, g: nx.DiGraph,
                                   source_wallet: str, current_balances: Dict[str, float],
                                   current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                   min_abs_amount: float, rng, wallet_to_attrs: Dict = None,
                                   attr_headers: List[str] = None) -> Tuple[List[Dict], int, datetime]:
        """生成四类钱包洗钱交易
        单笔交易尽量接近2000元（第四类钱包单笔限额），日累计尽量接近5000元（第四类钱包日限额）
        """
        # 检查时间偏好
        is_preferred_time = self.manager.is_preferred_time(current_time, transaction_type)
       
        if not is_preferred_time:
            # 调试信息：记录被时间偏好过滤的情况
            if hasattr(self, '_class4_time_filter_count'):
                self._class4_time_filter_count += 1
            else:
                self._class4_time_filter_count = 1
            return [], tx_id, current_time
            
        transactions = []
        motif_id = str(self.risk_pattern_counter)
        self.risk_pattern_counter += 1
        
        # 验证是否是四类钱包
        if wallet_to_attrs and attr_headers:
            is_class4 = self._is_class4_wallet(source_wallet, wallet_to_attrs, attr_headers)
           
            if not is_class4:
                return transactions, tx_id, current_time
        
        # 获取第四类钱包的限额配置
        limits = self.manager.get_wallet_level_limits('4')
        single_limit = limits.get('single_transaction_limit', 2000.0)  # 第四类钱包单笔限额2000元
        daily_limit = limits.get('daily_limit', 5000.0)  # 第四类钱包日限额5000元
        
        # 获取配置
        config = self.manager.get_transaction_config(transaction_type)
        original_amount_range = config.get("amount_range", {"min": 100, "max": 2000})
        time_interval = config.get("time_interval", {"min": 60, "max": 1800})
        
        # 修改金额范围：让单笔交易尽量接近单笔限额（2000元）
        # 使用80%-100%的单笔限额范围，让金额更可能接近上限
        amount_range = {
            "min": max(original_amount_range["min"], single_limit * 0.8),
            "max": min(original_amount_range["max"], single_limit)  # 不超过单笔限额
        }
        
        # 临时修改配置，让生成方法使用新的金额范围
        original_config_amount_range = config.get("amount_range")
        config["amount_range"] = amount_range
        
        try:
            # 选择交易模式
            transaction_mode = self._select_transaction_mode_with_weights(g, source_wallet, transaction_type, rng)
            
            if not transaction_mode:
                return [], tx_id, current_time
            
            # 根据模式生成交易
            if transaction_mode == 'fan_out':
                result = self._generate_fan_out_mode(transaction_type, g, source_wallet, 
                                                current_balances, current_time, tx_id, 
                                                max_frac_per_tx, min_abs_amount, rng)
            elif transaction_mode == 'fan_in':
                result = self._generate_fan_in_mode(transaction_type, g, source_wallet, 
                                                current_balances, current_time, tx_id, 
                                                max_frac_per_tx, min_abs_amount, rng)
            elif transaction_mode == 'one_to_many':
                result = self._generate_one_to_many_mode(transaction_type, g, source_wallet, 
                                                    current_balances, current_time, tx_id, 
                                                    max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
            elif transaction_mode == 'many_to_one':
                result = self._generate_many_to_one_mode(transaction_type, g, source_wallet, 
                                                    current_balances, current_time, tx_id, 
                                                    max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
            elif transaction_mode == 'many_to_many':
                # 对于class4_laundering，使用简化版的many_to_many生成方法（参考normal_small_high_freq）
                result = self._generate_simplified_many_to_many_mode(transaction_type, g, source_wallet, 
                                                                  current_balances, current_time, tx_id, 
                                                                  max_frac_per_tx, min_abs_amount, rng, 
                                                                  motif_id)
            else:
                # 默认使用fan_out模式
                result = self._generate_fan_out_mode(transaction_type, g, source_wallet, 
                                                current_balances, current_time, tx_id, 
                                                max_frac_per_tx, min_abs_amount, rng)
            
            return result
        finally:
            # 恢复原始配置
            if original_config_amount_range is not None:
                config["amount_range"] = original_config_amount_range
            
    def _generate_online_laundering(self, transaction_type: str, g: nx.DiGraph,
                              source_wallet: str, current_balances: Dict[str, float],
                              current_time: datetime, tx_id: int, max_frac_per_tx: float,
                              min_abs_amount: float, rng, wallet_to_attrs: Dict = None,
                              attr_headers: List = None) -> Tuple[List[Dict], int, datetime]:
        """生成线上跑分交易 - 支持多种模式：扇出、扇入、一进多出、多进一出、多进多出"""
        # 检查时间偏好
        if not self.manager.is_preferred_time(current_time, transaction_type):
            return [], tx_id, current_time
            
        transactions = []
        motif_id = str(self.risk_pattern_counter)
        self.risk_pattern_counter += 1
        
        # 选择交易模式（使用权重选择）
        transaction_mode = self._select_transaction_mode_with_weights(g, source_wallet, transaction_type, rng)
        
        if not transaction_mode:
            return [], tx_id, current_time
        
        # 根据模式生成交易
        if transaction_mode == 'fan_out':
            return self._generate_fan_out_mode(transaction_type, g, source_wallet, 
                                            current_balances, current_time, tx_id, 
                                            max_frac_per_tx, min_abs_amount, rng)
        elif transaction_mode == 'fan_in':
            return self._generate_fan_in_mode(transaction_type, g, source_wallet, 
                                            current_balances, current_time, tx_id, 
                                            max_frac_per_tx, min_abs_amount, rng)
        elif transaction_mode == 'one_to_many':
            return self._generate_one_to_many_mode(transaction_type, g, source_wallet, 
                                                current_balances, current_time, tx_id, 
                                                max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
        elif transaction_mode == 'many_to_one':
            return self._generate_many_to_one_mode(transaction_type, g, source_wallet, 
                                                current_balances, current_time, tx_id, 
                                                max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
        elif transaction_mode == 'many_to_many':
            # 对于online_laundering，使用简化版的many_to_many生成方法（参考normal_small_high_freq）
            return self._generate_simplified_many_to_many_mode(transaction_type, g, source_wallet, 
                                                              current_balances, current_time, tx_id, 
                                                              max_frac_per_tx, min_abs_amount, rng, 
                                                              motif_id)
        else:
            # 默认使用fan_out模式
            return self._generate_fan_out_mode(transaction_type, g, source_wallet, 
                                            current_balances, current_time, tx_id, 
                                            max_frac_per_tx, min_abs_amount, rng)
            
    
    def _check_wallet_limits(self, wallet_id: str, amount: float, daily_amount: float, 
                           wallet_to_attrs: dict, attr_headers: list) -> Tuple[bool, str]:
        """检查钱包限额
        返回: (是否通过检查, 错误信息)
        """
        try:
            # 获取钱包等级
            wallet_level = self.manager.get_wallet_level(wallet_id, wallet_to_attrs, attr_headers)
            
            # 检查单笔限额
            if not self.manager.check_single_transaction_limit(wallet_level, amount):
                limits = self.manager.get_wallet_level_limits(wallet_level)
                return False, f'单笔交易{amount}元超过{wallet_level}类钱包限额{limits["single_transaction_limit"]}元'
            
            # 检查日累计限额
            if not self.manager.check_daily_limit(wallet_level, daily_amount):
                limits = self.manager.get_wallet_level_limits(wallet_level)
                return False, f'日累计{daily_amount}元超过{wallet_level}类钱包限额{limits["daily_limit"]}元'
            
            return True, ''
        except Exception as e:
            return False, f'限额检查出错: {e}'
    
    def _validate_transaction_amount(self, amount: float, wallet_id: str, 
                                   wallet_to_attrs: dict, attr_headers: list) -> bool:
        """验证交易金额是否满足钱包限额（仅检查单笔限额，不检查日累计）"""
        try:
            wallet_level = self.manager.get_wallet_level(wallet_id, wallet_to_attrs, attr_headers)
            return self.manager.check_single_transaction_limit(wallet_level, amount)
        except Exception:
            return False

    def _generate_transaction_by_mode(self, risk_type: str, transaction_mode: str, g: nx.DiGraph,
                                    source_wallet: str, current_balances: Dict[str, float],
                                    current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                    min_abs_amount: float, rng, wallet_to_attrs: Dict = None,
                                    attr_headers: List[str] = None) -> Tuple[List[Dict], int, datetime]:
        """根据transaction_mode生成交易"""
        
        if transaction_mode == 'single':
            return self._generate_single_mode(risk_type, g, source_wallet, current_balances, 
                                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng)
        elif transaction_mode == 'fan_out':
            return self._generate_fan_out_mode(risk_type, g, source_wallet, current_balances, 
                                             current_time, tx_id, max_frac_per_tx, min_abs_amount, rng)
        elif transaction_mode == 'fan_in':
            return self._generate_fan_in_mode(risk_type, g, source_wallet, current_balances, 
                                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng)
        elif transaction_mode == 'one_to_many':
            return self._generate_one_to_many_mode(risk_type, g, source_wallet, current_balances, 
                                                 current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
        elif transaction_mode == 'many_to_one':
            return self._generate_many_to_one_mode(risk_type, g, source_wallet, current_balances, 
                                                 current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
        elif transaction_mode == 'many_to_many':
            return self._generate_many_to_many_mode(risk_type, g, source_wallet, current_balances, 
                                                  current_time, tx_id, max_frac_per_tx, min_abs_amount, rng, wallet_to_attrs, attr_headers)
    
        elif transaction_mode == 'forward':
            # 尝试生成forward模式交易
            forward_transactions, new_tx_id, new_current_time = self._generate_forward_mode(risk_type, g, source_wallet, current_balances, 
                                             current_time, tx_id, max_frac_per_tx, min_abs_amount, rng)
            
            # 如果forward模式无法生成完整交易，回退到single模式
            if not forward_transactions:
                return self._generate_single_mode(risk_type, g, source_wallet, current_balances, 
                                                current_time, tx_id, max_frac_per_tx, min_abs_amount, rng)
            else:
                return forward_transactions, new_tx_id, new_current_time
        else:
            # 默认使用single模式
            return self._generate_single_mode(risk_type, g, source_wallet, current_balances, 
                                            current_time, tx_id, max_frac_per_tx, min_abs_amount, rng)
    
    def _generate_single_mode(self, risk_type: str, g: nx.DiGraph, source_wallet: str, 
                            current_balances: Dict[str, float], current_time: datetime, tx_id: int, 
                            max_frac_per_tx: float, min_abs_amount: float, rng, time_preference=None) -> Tuple[List[Dict], int, datetime]:
        """生成单笔交易模式"""
        transactions = []
        successors = list(g.successors(source_wallet))
        
        if not self.manager.is_preferred_time(current_time, risk_type):
            return [], tx_id, current_time
        
        if not successors:
            return transactions, tx_id, current_time
        
        # 根据交易类型和钱包偏好过滤目标钱包
        is_risk = self.manager.is_risk_transaction(risk_type)
        preferred_targets, non_preferred_targets = self._filter_wallets_by_preference(successors, is_risk)
        
        # 优先从preferred_targets中选择，如果不够再从non_preferred_targets补充
        if preferred_targets:
            target_wallet = rng.choice(preferred_targets)
        elif non_preferred_targets:
            target_wallet = rng.choice(non_preferred_targets)
        else:
            return [], tx_id, current_time
        
        # 获取配置
        config = self.manager.get_transaction_config(risk_type)
        amount_range = config.get("amount_range", {"min": 1, "max": 10000})
        time_interval = config.get("time_interval", {"min": 60, "max": 1800})
        
        # 生成交易金额，考虑钱包限额
        max_amount = amount_range["max"]
        
        # 获取源钱包等级限制
        if self.wallet_to_attrs and self.attr_headers:
            wallet_level = self.manager.get_wallet_level(source_wallet, self.wallet_to_attrs, self.attr_headers)
            limits = self.manager.get_wallet_level_limits(wallet_level)
            max_amount = min(max_amount, limits.get("single_transaction_limit", max_amount))
        
        # 确保不超过余额
        max_amount = min(max_amount, current_balances[source_wallet])
        
        if max_amount < amount_range["min"]:
            return [], tx_id, current_time
        
        # 如果是class4_laundering或异常的四类钱包，让金额靠近单笔限额
        if (is_risk and risk_type == 'class4_laundering') or \
           (is_risk and wallet_level == 4 and self.wallet_to_attrs and self.attr_headers):
            limits = self.manager.get_wallet_level_limits(4)
            limit_max = limits.get("single_transaction_limit", max_amount)
            amount = self._generate_amount_near_limit(amount_range["min"], min(limit_max, max_amount), rng)
        else:
            amount = rng.uniform(amount_range["min"], max_amount)
        
        # 检查余额
        if current_balances[source_wallet] >= amount:
            # 更新余额
            current_balances[source_wallet] -= amount
            current_balances[target_wallet] += amount
            
            # 生成交易记录
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": source_wallet,
                "dst": target_wallet,
                "amount": round(amount, 2),
                "risk_type": risk_type,
                "motif_id": str(tx_id),
                "transaction_mode": "single"
            })
            
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
      
        
        return transactions, tx_id, current_time

    def _generate_fan_out_mode(self, risk_type: str, g: nx.DiGraph, source_wallet: str, 
                             current_balances: Dict[str, float], current_time: datetime, tx_id: int, 
                             max_frac_per_tx: float, min_abs_amount: float, rng, time_preference=None) -> Tuple[List[Dict], int, datetime]:
        """生成扇出模式：一个钱包（主钱包）→多个钱包（大于等于2个）"""
        # 判断是否为异常交易
        is_risk = self.manager.is_risk_transaction(risk_type)
        transactions = []
        if not self.manager.is_preferred_time(current_time, risk_type):
            return [], tx_id, current_time
        # 第一步：检查是否有下游钱包
        successors = list(g.successors(source_wallet))
      
        if len(successors) < 2:  # 需要至少2个下游钱包
            return transactions, tx_id, current_time
        # 获取配置
        config = self.manager.get_transaction_config(risk_type)
        amount_range = config.get("amount_range", {"min": 1, "max": 10000})
        time_interval = config.get("time_interval", {"min": 60, "max": 1800})
        
        # 第二步：选择目标钱包数量
        # 如果是异常交易，生成更多的下游账户；如果是正常交易，生成较少的下游账户
        if is_risk:
            # 异常交易：至少3个下游账户，不设上限
            max_targets = len(successors)  # 不设上限，使用所有可用钱包
            min_targets = 3
        else:
            # 正常交易：生成2-4个下游账户
            max_targets = min(4, len(successors))
            min_targets = 2
        
        if len(successors) < min_targets:
            return [], tx_id, current_time
        
        # 根据交易类型和钱包偏好过滤目标钱包
        preferred_targets, non_preferred_targets = self._filter_wallets_by_preference(successors, is_risk)
        
        # 优先从preferred_targets中选择，如果不够再从non_preferred_targets补充
        if len(preferred_targets) >= min_targets:
            available_targets = preferred_targets
        else:
            # 如果preferred不够，从preferred + non_preferred中选择
            available_targets = preferred_targets + non_preferred_targets
        
        if len(available_targets) < min_targets:
            return [], tx_id, current_time
        
        # 应用跨行偏好（如果配置中有）
        if self.wallet_to_attrs and self.attr_headers:
            config = self.manager.get_transaction_config(risk_type)
            if config.get("cross_bank_preference") is not None:
                preferred_targets_cross_bank = self.manager.apply_cross_bank_preference(
                    available_targets, source_wallet, self.wallet_to_attrs, self.attr_headers, risk_type, rng
                )
                if len(preferred_targets_cross_bank) >= min_targets:
                    available_targets = preferred_targets_cross_bank
        
        num_targets = rng.integers(min_targets, min(max_targets + 1, len(available_targets) + 1))
        target_wallets = rng.choice(available_targets, size=num_targets, replace=False)
        # 第四步：生成每笔交易
        motif_id = str(tx_id)  # 所有fan_out交易使用相同的motif_id
        
        for i, target_wallet in enumerate(target_wallets):
            # 生成交易金额，考虑钱包限额
            max_amount = amount_range["max"]
            wallet_level = None
            
            # 获取源钱包等级限制
            if self.wallet_to_attrs and self.attr_headers:
                wallet_level = self.manager.get_wallet_level(source_wallet, self.wallet_to_attrs, self.attr_headers)
                limits = self.manager.get_wallet_level_limits(wallet_level)
                max_amount = min(max_amount, limits.get("single_transaction_limit", max_amount))
            
            # 对于异常交易，放宽余额检查
            if is_risk:
                # 异常交易：允许余额不足，直接使用配置的最大金额
                max_amount = amount_range["max"]
                
                # 如果是class4_laundering或异常的四类钱包，让金额靠近单笔限额
                if (risk_type == 'class4_laundering' or 
                    (wallet_level == 4 and self.wallet_to_attrs and self.attr_headers)):
                    limits = self.manager.get_wallet_level_limits(4)
                    limit_max = limits.get("single_transaction_limit", max_amount)
                    # 使用限额作为上限，生成靠近限额的金额
                    amount = self._generate_amount_near_limit(amount_range["min"], min(limit_max, max_amount), rng)
                else:
                    amount = rng.uniform(amount_range["min"], max_amount)
                
                # 直接执行交易，不检查余额（余额会在后续账户开立数据中修正）
                current_balances[source_wallet] = current_balances.get(source_wallet, 0) - amount
                current_balances[target_wallet] = current_balances.get(target_wallet, 0) + amount
            else:
                # 正常交易：仍然检查余额
                # 确保不超过余额
                max_amount = min(max_amount, current_balances.get(source_wallet, 0))
                
                if max_amount < amount_range["min"]:
                    # 如果无法满足最小金额要求，回滚之前的所有交易
                    for j in range(i):
                        prev_tx = transactions[j]
                        current_balances[source_wallet] += prev_tx["amount"]
                        current_balances[prev_tx["dst"]] -= prev_tx["amount"]
                    return [], tx_id, current_time
                    
                amount = rng.uniform(amount_range["min"], max_amount)
                
                # 检查主钱包余额是否足够
                if current_balances.get(source_wallet, 0) < amount:
                    # 如果余额不足，回滚之前的所有交易
                    for j in range(i):
                        prev_tx = transactions[j]
                        current_balances[source_wallet] += prev_tx["amount"]
                        current_balances[prev_tx["dst"]] -= prev_tx["amount"]
                    return [], tx_id, current_time
                
                # 执行交易
                current_balances[source_wallet] -= amount
                current_balances[target_wallet] = current_balances.get(target_wallet, 0) + amount
            
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": source_wallet,
                "dst": target_wallet,
                "amount": round(amount, 2),
                "risk_type": risk_type,
                "motif_id": motif_id,  # 所有fan_out交易使用相同的motif_id
                "transaction_mode": "fan_out"
            })
            
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 第五步：最终验证 - 根据是否为异常交易设置不同的最小交易数量
        is_risk = self.manager.is_risk_transaction(risk_type)
        min_transactions = 3 if is_risk else 2
        if len(transactions) < min_transactions:
            # 如果交易数量不足，回滚所有操作并返回空列表
            for tx in transactions:
                current_balances[source_wallet] += tx["amount"]
                current_balances[tx["dst"]] -= tx["amount"]
            return [], tx_id - len(transactions), current_time
        
        
        return transactions, tx_id, current_time

    def _generate_fan_in_mode(self, risk_type: str, g: nx.DiGraph, source_wallet: str, 
                            current_balances: Dict[str, float], current_time: datetime, tx_id: int, 
                            max_frac_per_tx: float, min_abs_amount: float, rng, time_preference=None) -> Tuple[List[Dict], int, datetime]:
        """生成扇入模式：多个钱包（大于等于2个）→一个钱包（主钱包）"""
        transactions = []
        if not self.manager.is_preferred_time(current_time, risk_type):
            return [], tx_id, current_time
        # 第一步：检查是否有上游钱包
        predecessors = list(g.predecessors(source_wallet))
        if len(predecessors) < 2:  # 需要至少2个上游钱包
            return transactions, tx_id, current_time
        # 获取配置
        config = self.manager.get_transaction_config(risk_type)
        amount_range = config.get("amount_range", {"min": 1, "max": 10000})
        time_interval = config.get("time_interval", {"min": 60, "max": 1800})
        # 第二步：选择源钱包数量
        # 如果是异常交易，生成更多的上游账户；如果是正常交易，生成较少的上游账户
        is_risk = self.manager.is_risk_transaction(risk_type)
        if is_risk:
            # 异常交易：至少3个上游账户，不设上限
            max_sources = len(predecessors)  # 不设上限，使用所有可用钱包
            min_sources = 3
        else:
            # 正常交易：生成2-4个上游账户
            max_sources = min(4, len(predecessors))
            min_sources = 2
        
        if len(predecessors) < min_sources:
            return [], tx_id, current_time
        
        # 应用跨行偏好（如果配置中有）
        available_sources = predecessors
        if self.wallet_to_attrs and self.attr_headers:
            config = self.manager.get_transaction_config(risk_type)
            if config.get("cross_bank_preference") is not None:
                preferred_sources_cross_bank = self.manager.apply_cross_bank_preference(
                    predecessors, source_wallet, self.wallet_to_attrs, self.attr_headers, risk_type, rng
                )
                if len(preferred_sources_cross_bank) >= min_sources:
                    available_sources = preferred_sources_cross_bank
        
        num_sources = rng.integers(min_sources, min(max_sources + 1, len(available_sources) + 1))
        source_wallets = rng.choice(available_sources, size=num_sources, replace=False)
        # 第三步：检查所有源钱包余额是否足够
       
        # 第四步：生成每笔交易
        motif_id = str(tx_id)  # 所有fan_in交易使用相同的motif_id
        
        for i, src_wallet in enumerate(source_wallets):
            # 生成交易金额，考虑钱包限额
            max_amount = amount_range["max"]
            wallet_level = None
            
            # 获取源钱包等级限制
            if self.wallet_to_attrs and self.attr_headers:
                wallet_level = self.manager.get_wallet_level(src_wallet, self.wallet_to_attrs, self.attr_headers)
                limits = self.manager.get_wallet_level_limits(wallet_level)
                max_amount = min(max_amount, limits.get("single_transaction_limit", max_amount))
            
            # 确保不超过余额
            max_amount = min(max_amount, current_balances[src_wallet])
            
            if max_amount < amount_range["min"]:
                # 如果无法满足最小金额要求，回滚之前的所有交易
                for j in range(i):
                    prev_tx = transactions[j]
                    current_balances[prev_tx["src"]] += prev_tx["amount"]
                    current_balances[source_wallet] -= prev_tx["amount"]
                return [], tx_id, current_time
            
            # 如果是class4_laundering或异常的四类钱包，让金额靠近单笔限额
            if (is_risk and risk_type == 'class4_laundering') or \
               (is_risk and wallet_level == 4 and self.wallet_to_attrs and self.attr_headers):
                limits = self.manager.get_wallet_level_limits(4)
                limit_max = limits.get("single_transaction_limit", max_amount)
                amount = self._generate_amount_near_limit(amount_range["min"], min(limit_max, max_amount), rng)
            else:
                amount = rng.uniform(amount_range["min"], max_amount)
            
            # 检查源钱包余额是否足够
            if current_balances[src_wallet] < amount:
                # 如果余额不足，回滚之前的所有交易
                for j in range(i):
                    prev_tx = transactions[j]
                    current_balances[prev_tx["src"]] += prev_tx["amount"]
                    current_balances[source_wallet] -= prev_tx["amount"]
                return [], tx_id, current_time
            
            # 执行交易
            current_balances[src_wallet] -= amount
            current_balances[source_wallet] += amount
            
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": src_wallet,
                "dst": source_wallet,
                "amount": round(amount, 2),
                "risk_type": risk_type,
                "motif_id": motif_id,  # 所有fan_in交易使用相同的motif_id
                "transaction_mode": "fan_in"
            })
            
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 第五步：最终验证 - 根据是否为异常交易设置不同的最小交易数量
        is_risk = self.manager.is_risk_transaction(risk_type)
        min_transactions = 3 if is_risk else 2
        if len(transactions) < min_transactions:
            # 如果交易数量不足，回滚所有操作并返回空列表
            for tx in transactions:
                current_balances[tx["src"]] += tx["amount"]
                current_balances[source_wallet] -= tx["amount"]
            return [], tx_id - len(transactions), current_time
      
            
        
        return transactions, tx_id, current_time

    def _generate_forward_mode(self, risk_type: str, g: nx.DiGraph, source_wallet: str, 
                            current_balances: Dict[str, float], current_time: datetime, tx_id: int, 
                            max_frac_per_tx: float, min_abs_amount: float, rng, time_preference=None) -> Tuple[List[Dict], int, datetime]:
        """生成转发模式：A → 主钱包 → C 三段式交易"""
        transactions = []
        if not self.manager.is_preferred_time(current_time, risk_type):
            return [], tx_id, current_time
        
        # 获取配置
        config = self.manager.get_transaction_config(risk_type)
        amount_range = config.get("amount_range", {"min": 1, "max": 10000})
        time_interval = config.get("time_interval", {"min": 60, "max": 1800})
        
        # 第一步：严格检查网络连接性
        predecessors = list(g.predecessors(source_wallet))
        successors = list(g.successors(source_wallet))
        if len(predecessors) < 1 or len(successors) < 1:
            return [], tx_id, current_time
        
        # 第二步：选择上游钱包和下游钱包
        upstream_wallet = rng.choice(predecessors)
        downstream_wallet = rng.choice(successors)
        
        # 第三步：确保上游和下游钱包不同
        if upstream_wallet == downstream_wallet:
            return [], tx_id, current_time
        
        # 第四步：计算可用的交易金额
        max_amount = min(current_balances[upstream_wallet], amount_range["max"])
        min_amount = min(amount_range["min"], max_amount)
        
        # 第五步：检查余额是否足够
        if current_balances[upstream_wallet] < min_amount:
            return [], tx_id, current_time
        
        # 第六步：生成交易金额
        amount = rng.uniform(min_amount, max_amount)
        
        # 第七步：最终余额检查
        if current_balances[upstream_wallet] < amount:
            return [], tx_id, current_time
        
        # 第八步：生成motif_id（两笔交易使用相同的motif_id）
        motif_id = str(tx_id)
        
        # 第九步：执行第一段交易：A → 主钱包
        current_balances[upstream_wallet] -= amount
        current_balances[source_wallet] += amount
        
        first_transaction = {
            "tx_id": tx_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "src": upstream_wallet,
            "dst": source_wallet,
            "amount": round(amount, 2),
            "risk_type": risk_type,
            "motif_id": motif_id,
            "transaction_mode": "forward"
        }
        
        tx_id += 1
        current_time += timedelta(seconds=int(rng.integers(30, 300)))  # 短时间间隔
        
        # 第十步：执行第二段交易：主钱包 → C
        current_balances[source_wallet] -= amount
        current_balances[downstream_wallet] += amount
        
        second_transaction = {
            "tx_id": tx_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "src": source_wallet,
            "dst": downstream_wallet,
            "amount": round(amount, 2),
            "risk_type": risk_type,
            "motif_id": motif_id,  # 使用相同的motif_id
            "transaction_mode": "forward"
        }
        
        # 第十一步：将两笔交易添加到结果中
        transactions.append(first_transaction)
        transactions.append(second_transaction)
        
        tx_id += 1
        current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 第十二步：严格验证 - 确保生成了正好两笔forward交易
        if len(transactions) != 2:
            # 如果交易数量不对，回滚所有操作
            current_balances[upstream_wallet] += amount
            current_balances[source_wallet] -= amount
            current_balances[downstream_wallet] -= amount
            return [], tx_id - 2, current_time
        
        # 第十三步：验证两笔交易都是forward模式
        forward_count = sum(1 for tx in transactions if tx.get("transaction_mode") == "forward")
        if forward_count != 2:
            # 如果不是两笔forward交易，回滚所有操作
            current_balances[upstream_wallet] += amount
            current_balances[source_wallet] -= amount
            current_balances[downstream_wallet] -= amount
            return [], tx_id - 2, current_time
        
        # 第十四步：验证交易结构
        if not self._validate_forward_completeness(transactions, source_wallet):
            # 如果验证失败，回滚所有操作
            current_balances[upstream_wallet] += amount
            current_balances[source_wallet] -= amount
            current_balances[downstream_wallet] -= amount
            return [], tx_id - 2, current_time
        
        # 第十五步：最终验证 - 确保同一个motif_id正好对应两笔交易
        motif_ids = [tx.get("motif_id") for tx in transactions]
        if len(set(motif_ids)) != 1 or len(motif_ids) != 2:
            # 如果motif_id不统一或数量不对，回滚所有操作
            current_balances[upstream_wallet] += amount
            current_balances[source_wallet] -= amount
            current_balances[downstream_wallet] -= amount
            return [], tx_id - 2, current_time
        
        return transactions, tx_id, current_time
            
    def _generate_one_to_many_mode(self, risk_type: str, g: nx.DiGraph, source_wallet: str, 
                                current_balances: Dict[str, float], current_time: datetime, tx_id: int, 
                                max_frac_per_tx: float, min_abs_amount: float, rng, 
                                wallet_to_attrs: Dict = None, attr_headers: List[str] = None, time_preference=None) -> Tuple[List[Dict], int, datetime]:
        """生成一对多模式交易：一个钱包 → 主钱包 → 多个钱包（交易链必须完整）"""
        transactions = []
        if not self.manager.is_preferred_time(current_time, risk_type):
            return [], tx_id, current_time
        motif_id = str(tx_id)
        successors = list(g.successors(source_wallet))
        if len(successors) < 2:
            return [], tx_id, current_time  # 直接返回空
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if len(available_wallets) < 3:
            return transactions, tx_id, current_time
        
        # 选择1个上游钱包（转入主钱包）
        # 根据交易类型和钱包偏好过滤上游钱包
        is_risk = self.manager.is_risk_transaction(risk_type)
        upstream_wallets_all = [w for w in available_wallets if w != source_wallet]
        preferred_upstream, non_preferred_upstream = self._filter_wallets_by_preference(upstream_wallets_all, is_risk)
        
        # 应用跨行偏好（如果配置中有）
        if wallet_to_attrs and attr_headers:
            config = self.manager.get_transaction_config(risk_type)
            if config.get("cross_bank_preference") is not None:
                if preferred_upstream:
                    preferred_upstream = self.manager.apply_cross_bank_preference(
                        preferred_upstream, source_wallet, wallet_to_attrs, attr_headers, risk_type, rng
                    )
                elif non_preferred_upstream:
                    non_preferred_upstream = self.manager.apply_cross_bank_preference(
                        non_preferred_upstream, source_wallet, wallet_to_attrs, attr_headers, risk_type, rng
                    )
        
        if preferred_upstream:
            selected_upstream = rng.choice(preferred_upstream)
        elif non_preferred_upstream:
            selected_upstream = rng.choice(non_preferred_upstream)
        else:
            return [], tx_id, current_time
        
        # 获取配置
        config = self.manager.get_transaction_config(risk_type)
        amount_range = config.get("amount_range", {"min": 100, "max": 1000})
        time_interval = config.get("time_interval", {"min": 60, "max": 600})
        
        # 生成转入交易，考虑钱包限额
        max_input_amount = amount_range["max"]
        
        # 获取上游钱包等级限制
        if wallet_to_attrs and attr_headers:
            wallet_level = self.manager.get_wallet_level(selected_upstream, wallet_to_attrs, attr_headers)
            limits = self.manager.get_wallet_level_limits(wallet_level)
            max_input_amount = min(max_input_amount, limits.get("single_transaction_limit", max_input_amount))
        
        # 判断是否为异常交易，异常交易放宽余额检查
        is_risk = self.manager.is_risk_transaction(risk_type)
        
        if is_risk:
            # 异常交易：直接使用配置的最大金额，不检查余额
            # 如果是class4_laundering或异常的四类钱包，让金额靠近单笔限额
            if risk_type == 'class4_laundering' or (wallet_to_attrs and attr_headers):
                if wallet_to_attrs and attr_headers:
                    upstream_wallet_level = self.manager.get_wallet_level(selected_upstream, wallet_to_attrs, attr_headers)
                    if risk_type == 'class4_laundering' or upstream_wallet_level == 4:
                        limits = self.manager.get_wallet_level_limits(4)
                        limit_max = limits.get("single_transaction_limit", amount_range["max"])
                        input_amount = self._generate_amount_near_limit(amount_range["min"], min(limit_max, amount_range["max"]), rng)
                    else:
                        input_amount = rng.uniform(amount_range["min"], amount_range["max"])
                else:
                    input_amount = rng.uniform(amount_range["min"], amount_range["max"])
            else:
                input_amount = rng.uniform(amount_range["min"], amount_range["max"])
            current_balances[selected_upstream] = current_balances.get(selected_upstream, 0) - input_amount
            current_balances[source_wallet] = current_balances.get(source_wallet, 0) + input_amount
        else:
            # 正常交易：检查余额
            # 确保不超过余额
            max_input_amount = min(max_input_amount, current_balances.get(selected_upstream, 0))
            
            if max_input_amount < amount_range["min"]:
                return transactions, tx_id, current_time
                
            input_amount = rng.uniform(amount_range["min"], max_input_amount)
            
            # 检查上游钱包余额
            if current_balances.get(selected_upstream, 0) < input_amount:
                return transactions, tx_id, current_time
            
            current_balances[selected_upstream] -= input_amount
            current_balances[source_wallet] = current_balances.get(source_wallet, 0) + input_amount
        
        transactions.append({
            "tx_id": tx_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "src": selected_upstream,
            "dst": source_wallet,
            "amount": round(input_amount, 2),
            "risk_type": risk_type,
            "motif_id": motif_id,
            "transaction_mode": "one_to_many"
        })
        tx_id += 1
        current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 生成转出交易（扣除佣金）
        # 获取配置中的佣金率
        if risk_type =="merchant_laundering":
            config = self.manager.get_transaction_config(risk_type)
            commission_config = config.get("commission_rate")
            commission_rate = rng.uniform(commission_config["min"], commission_config["max"])
        else:
            commission_rate = 0.0
        available_amount = input_amount * (1 - commission_rate)
        
        # 选择下游钱包（从主钱包转出）
        downstream_wallets_all = [w for w in available_wallets if w != selected_upstream]
        
        # 判断是否为异常交易，异常交易生成更多的下游账户
        is_risk = self.manager.is_risk_transaction(risk_type)
        
        # 根据交易类型和钱包偏好过滤下游钱包
        preferred_downstream, non_preferred_downstream = self._filter_wallets_by_preference(downstream_wallets_all, is_risk)
        
        # 尝试多次选择下游钱包，确保有足够的钱包进行转出
        max_attempts = 5
        for attempt in range(max_attempts):
            # 根据是否为异常交易决定下游钱包数量
            if is_risk:
                # 异常交易：至少3个下游账户，不设上限
                min_outputs = 3
                max_outputs = len(downstream_wallets)  # 不设上限，使用所有可用钱包
            else:
                # 正常交易：生成2-5个下游账户
                min_outputs = 2
                max_outputs = min(5, len(downstream_wallets))
            
            if len(downstream_wallets) < min_outputs:
                break  # 没有足够的下游钱包，退出循环
            
            num_outputs = rng.integers(min_outputs, max_outputs + 1)
            selected_downstreams = rng.choice(downstream_wallets, size=num_outputs, replace=False)
            
            # 确保每笔转出交易有合理的金额
            num_outputs = len(selected_downstreams)
            min_per_output = max(min_abs_amount, available_amount * 0.01)  # 至少是总金额的1%
            max_per_output = available_amount * 0.5  # 单笔最多50%
            
            # 使用改进的分配策略：先分配最小金额，然后随机分配剩余金额
            amounts = [min_per_output] * num_outputs
            remaining = available_amount - (min_per_output * num_outputs)
            
            if remaining > 0:
                # 使用beta分布分配剩余金额（但使用更均匀的参数）
                for i in range(num_outputs - 1):
                    weight = rng.beta(2, 2)  # 更均匀的分布
                    amount = remaining * weight
                    amount = min(amount, max_per_output - min_per_output)  # 限制单笔上限
                    amounts[i] += amount
                    remaining -= amount
                amounts[-1] += remaining  # 最后一笔包含所有剩余
            
            # 确保总金额等于available_amount（四舍五入误差调整）
            total_allocated = sum(amounts)
            if abs(total_allocated - available_amount) > 0.01:
                diff = available_amount - total_allocated
                amounts[0] += diff  # 将差值加到第一笔
            
            # 检查是否有足够的钱包进行转出
            valid_downstreams = []
            valid_amounts = []
            
            for i, (dst_wallet, amount) in enumerate(zip(selected_downstreams, amounts)):
                # 对于异常交易，放宽余额检查
                if is_risk:
                    # 异常交易：直接使用，不检查余额
                    valid_downstreams.append(dst_wallet)
                    valid_amounts.append(amount)
                else:
                    # 正常交易：检查余额
                    if amount >= min_abs_amount and current_balances.get(source_wallet, 0) >= amount:
                        valid_downstreams.append(dst_wallet)
                        valid_amounts.append(amount)
            
            # 如果找到足够的下游钱包，生成转出交易
            # 根据是否为异常交易设置不同的最小下游钱包数量
            min_outputs_required = 3 if is_risk else 2
            if len(valid_downstreams) >= min_outputs_required:
                for i, (dst_wallet, amount) in enumerate(zip(valid_downstreams, valid_amounts)):
                    # 对于异常交易，放宽余额检查
                    if is_risk:
                        # 异常交易：直接生成交易，不检查余额
                        current_balances[source_wallet] = current_balances.get(source_wallet, 0) - amount
                        current_balances[dst_wallet] = current_balances.get(dst_wallet, 0) + amount
                    else:
                        # 正常交易：检查余额
                        current_balances[source_wallet] -= amount
                        current_balances[dst_wallet] = current_balances.get(dst_wallet, 0) + amount
                    
                    transactions.append({
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet,
                        "dst": dst_wallet,
                        "amount": round(amount, 2),
                        "risk_type": risk_type,
                        "motif_id": motif_id,
                        "transaction_mode": "one_to_many"
                    })
                    tx_id += 1
                    current_time += timedelta(seconds=int(rng.integers(30, max(30 + 1, 300))))
                
                # 成功生成完整交易链，验证完整性
                # 根据是否为异常交易设置不同的最小要求
                incoming_count = sum(1 for tx in transactions if tx.get("dst") == source_wallet)
                outgoing_count = sum(1 for tx in transactions if tx.get("src") == source_wallet)
                
                min_incoming = 1  # one_to_many 模式只需要1笔转入
                min_outgoing = 3 if is_risk else 2  # 异常交易至少3笔转出，正常交易至少2笔转出
                
                if incoming_count >= min_incoming and outgoing_count >= min_outgoing:
                    # 验证通过，返回
                    return transactions, tx_id, current_time
                else:
                    # 验证失败，回滚交易
                    for tx in transactions:
                        if tx["src"] == source_wallet:  # 转出交易
                            current_balances[source_wallet] += tx["amount"]
                            current_balances[tx["dst"]] -= tx["amount"]
                        else:  # 转入交易
                            current_balances[tx["src"]] += tx["amount"]
                            current_balances[source_wallet] -= tx["amount"]
                    transactions = []  # 清空交易列表
                    continue
            
            # 如果没有找到足够的下游钱包，尝试重新选择
            if attempt < max_attempts - 1:
                # 重新选择下游钱包
                downstream_wallets = [w for w in available_wallets if w != selected_upstream and w not in valid_downstreams]
                if len(downstream_wallets) < 2:
                    break
        
        # 如果多次尝试后仍然没有找到合适的下游钱包，返回空交易
        return [], tx_id, current_time

    def _generate_many_to_one_mode(self, risk_type: str, g: nx.DiGraph, source_wallet: str, 
                                 current_balances: Dict[str, float], current_time: datetime, tx_id: int, 
                                 max_frac_per_tx: float, min_abs_amount: float, rng, 
                                 wallet_to_attrs: Dict = None, attr_headers: List[str] = None, time_preference=None) -> Tuple[List[Dict], int, datetime]:
        """生成多对一模式交易：多个钱包 → 主钱包 → 一个钱包"""
        transactions = []
        if not self.manager.is_preferred_time(current_time, risk_type):
            return [], tx_id, current_time
        motif_id = str(tx_id)
        predecessors = list(g.predecessors(source_wallet))
        successors = list(g.successors(source_wallet))
        if len(predecessors) < 2 or len(successors) < 1:
            return [], tx_id, current_time 
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if len(available_wallets) < 3:
            return transactions, tx_id, current_time
        
        # 判断是否为异常交易，异常交易生成更多的上游账户
        is_risk = self.manager.is_risk_transaction(risk_type)
        
        # 选择上游钱包（转入主钱包）
        upstream_wallets = [w for w in available_wallets if w != source_wallet]
        
        if is_risk:
            # 异常交易：至少3个上游账户，不设上限
            min_inputs = 3
            max_inputs = len(upstream_wallets)  # 不设上限，使用所有可用钱包
        else:
            # 正常交易：生成2-4个上游账户
            min_inputs = 2
            max_inputs = min(4, len(upstream_wallets))
        
        if len(upstream_wallets) < min_inputs:
            return [], tx_id, current_time
        
        # 根据交易类型和钱包偏好过滤上游钱包
        preferred_upstream, non_preferred_upstream = self._filter_wallets_by_preference(upstream_wallets, is_risk)
        if len(preferred_upstream) >= min_inputs:
            available_upstream = preferred_upstream
        else:
            available_upstream = preferred_upstream + non_preferred_upstream
        
        # 应用跨行偏好（如果配置中有）
        if wallet_to_attrs and attr_headers:
            config = self.manager.get_transaction_config(risk_type)
            if config.get("cross_bank_preference") is not None:
                preferred_upstream_cross_bank = self.manager.apply_cross_bank_preference(
                    available_upstream, source_wallet, wallet_to_attrs, attr_headers, risk_type, rng
                )
                if len(preferred_upstream_cross_bank) >= min_inputs:
                    available_upstream = preferred_upstream_cross_bank
        
        if len(available_upstream) < min_inputs:
            return [], tx_id, current_time
        
        num_inputs = rng.integers(min_inputs, min(max_inputs + 1, len(available_upstream) + 1))
        selected_upstreams = rng.choice(available_upstream, size=num_inputs, replace=False)
        
        # 选择1个下游钱包（从主钱包转出）
        downstream_wallets_all = [w for w in available_wallets if w not in selected_upstreams]
        preferred_downstream, non_preferred_downstream = self._filter_wallets_by_preference(downstream_wallets_all, is_risk)
        
        # 应用跨行偏好（如果配置中有）
        if wallet_to_attrs and attr_headers:
            config = self.manager.get_transaction_config(risk_type)
            if config.get("cross_bank_preference") is not None:
                if preferred_downstream:
                    preferred_downstream = self.manager.apply_cross_bank_preference(
                        preferred_downstream, source_wallet, wallet_to_attrs, attr_headers, risk_type, rng
                    )
                elif non_preferred_downstream:
                    non_preferred_downstream = self.manager.apply_cross_bank_preference(
                        non_preferred_downstream, source_wallet, wallet_to_attrs, attr_headers, risk_type, rng
                    )
        
        if preferred_downstream:
            selected_downstream = rng.choice(preferred_downstream)
        elif non_preferred_downstream:
            selected_downstream = rng.choice(non_preferred_downstream)
        else:
            return [], tx_id, current_time
        
        # 获取配置
        config = self.manager.get_transaction_config(risk_type)
        amount_range = config.get("amount_range", {"min": 50, "max": 500})
        time_interval = config.get("time_interval", {"min": 60, "max": 600})
        
        # 生成转入交易
        total_input_amount = 0
        successful_inputs = 0
        for upstream_wallet in selected_upstreams:
            # 如果是class4_laundering或异常的四类钱包，让金额靠近单笔限额
            if risk_type == 'class4_laundering' or (is_risk and wallet_to_attrs and attr_headers):
                if wallet_to_attrs and attr_headers:
                    upstream_wallet_level = self.manager.get_wallet_level(upstream_wallet, wallet_to_attrs, attr_headers)
                    if risk_type == 'class4_laundering' or upstream_wallet_level == 4:
                        limits = self.manager.get_wallet_level_limits(4)
                        limit_max = limits.get("single_transaction_limit", amount_range["max"])
                        input_amount = self._generate_amount_near_limit(amount_range["min"], min(limit_max, amount_range["max"]), rng)
                    else:
                        input_amount = rng.uniform(amount_range["min"], amount_range["max"])
                else:
                    input_amount = rng.uniform(amount_range["min"], amount_range["max"])
            else:
                input_amount = rng.uniform(amount_range["min"], amount_range["max"])
            
            # 对于异常交易，放宽余额检查
            if is_risk:
                # 异常交易：直接生成交易，不检查余额
                current_balances[upstream_wallet] = current_balances.get(upstream_wallet, 0) - input_amount
                current_balances[source_wallet] = current_balances.get(source_wallet, 0) + input_amount
                total_input_amount += input_amount
                successful_inputs += 1
            else:
                # 正常交易：检查余额
                if current_balances.get(upstream_wallet, 0) < input_amount:
                    continue
                
                current_balances[upstream_wallet] -= input_amount
                current_balances[source_wallet] = current_balances.get(source_wallet, 0) + input_amount
                total_input_amount += input_amount
                successful_inputs += 1
            
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": upstream_wallet,
                "dst": source_wallet,
                "amount": round(input_amount, 2),
                "risk_type": risk_type,
                "motif_id": motif_id,
                "transaction_mode": "many_to_one"
            })
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 检查转入交易是否足够（异常交易至少3笔，正常交易至少2笔）
        min_inputs = 3 if is_risk else 2
        if successful_inputs < min_inputs or total_input_amount <= 0:
            # 回滚所有转入交易
            for tx in transactions:
                if tx.get("dst") == source_wallet:
                    current_balances[tx["src"]] += tx["amount"]
                    current_balances[source_wallet] -= tx["amount"]
            return [], tx_id - len(transactions), current_time
        
        # 生成转出交易（扣除佣金）
        if total_input_amount > 0:
            if risk_type =="merchant_laundering":
                config = self.manager.get_transaction_config(risk_type)
                commission_config = config.get("commission_rate")
                commission_rate = rng.uniform(commission_config["min"], commission_config["max"])
            else:
                commission_rate = 0.0
            available_amount = total_input_amount * (1 - commission_rate)
            
            # 对于异常交易，放宽余额检查
            if is_risk:
                # 异常交易：直接生成交易，不检查余额
                current_balances[source_wallet] = current_balances.get(source_wallet, 0) - available_amount
                current_balances[selected_downstream] = current_balances.get(selected_downstream, 0) + available_amount
                
                transactions.append({
                    "tx_id": tx_id,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src": source_wallet,
                    "dst": selected_downstream,
                    "amount": round(available_amount, 2),
                    "risk_type": risk_type,
                    "motif_id": motif_id,
                    "transaction_mode": "many_to_one"
                })
                tx_id += 1
                current_time += timedelta(seconds=int(rng.integers(30, max(30 + 1, 300))))
            else:
                # 正常交易：检查余额
                if current_balances.get(source_wallet, 0) >= available_amount:
                    current_balances[source_wallet] -= available_amount
                    current_balances[selected_downstream] = current_balances.get(selected_downstream, 0) + available_amount
                    
                    transactions.append({
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet,
                        "dst": selected_downstream,
                        "amount": round(available_amount, 2),
                        "risk_type": risk_type,
                        "motif_id": motif_id,
                        "transaction_mode": "many_to_one"
                    })
                    tx_id += 1
                    current_time += timedelta(seconds=int(rng.integers(30, max(30 + 1, 300))))
        
        # 验证交易完整性：根据是否为异常交易设置不同的最小要求
        incoming_count = sum(1 for tx in transactions if tx.get("dst") == source_wallet)
        outgoing_count = sum(1 for tx in transactions if tx.get("src") == source_wallet)
        
        min_incoming = 3 if is_risk else 2
        min_outgoing = 1  # many_to_one 模式只需要1笔转出
        
        if incoming_count >= min_incoming and outgoing_count >= min_outgoing:
            return transactions, tx_id, current_time
        else:
            # 验证失败，回滚所有交易
            for tx in transactions:
                if tx["src"] == source_wallet:  # 转出交易
                   current_balances[source_wallet] += tx["amount"]
                   current_balances[tx["dst"]] -= tx["amount"]
                else:  # 转入交易
                   current_balances[tx["src"]] += tx["amount"]
                   current_balances[source_wallet] -= tx["amount"]
            
            return [], tx_id, current_time

    def _generate_simplified_many_to_many_mode(self, transaction_type: str, g: nx.DiGraph, source_wallet: str, 
                                               current_balances: Dict[str, float], current_time: datetime, tx_id: int, 
                                               max_frac_per_tx: float, min_abs_amount: float, rng, 
                                               motif_id: str) -> Tuple[List[Dict], int, datetime]:
        """生成简化版的多对多模式交易（多个钱包 → 主钱包 → 多个钱包）"""
        transactions = []
        
        # 获取配置
        config = self.manager.get_transaction_config(transaction_type)
        amount_range = config.get("amount_range", {"min": 50, "max": 500})
        time_interval = config.get("time_interval", {"min": 60, "max": 600})
        
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if len(available_wallets) < 4:
            return [], tx_id, current_time
        
        # 使用 source_wallet 作为主钱包（多个钱包 → source_wallet → 多个钱包）
        main_wallet = source_wallet
        
        # 根据是否为异常交易决定交易数量
        is_risk = self.manager.is_risk_transaction(transaction_type)
        if is_risk:
            # 异常交易：生成更多交易（至少3个上游，至少4个下游）
            num_inputs = rng.integers(3, min(6, len(available_wallets)))
            num_outputs = rng.integers(4, min(8, len(available_wallets)))
        else:
            # 正常交易：生成较少交易（至少2个上游，至少2个下游）
            num_inputs = rng.integers(2, min(4, len(available_wallets)))
            num_outputs = rng.integers(2, min(5, len(available_wallets)))
        
        # 确保有足够的钱包
        if len(available_wallets) < (num_inputs + num_outputs):
            return [], tx_id, current_time
        
        # 选择上游钱包（转入主钱包）
        selected_upstreams = rng.choice(available_wallets, size=num_inputs, replace=False)
        # 选择下游钱包（从主钱包转出，不能与上游重复）
        remaining_wallets = [w for w in available_wallets if w not in selected_upstreams]
        selected_downstreams = rng.choice(remaining_wallets, size=num_outputs, replace=False)
        
        # 生成转入交易（多个上游钱包 → 主钱包）
        for upstream_wallet in selected_upstreams:
            # 如果是class4_laundering或异常的四类钱包，让金额靠近单笔限额
            if transaction_type == 'class4_laundering' or (is_risk and self.wallet_to_attrs and self.attr_headers):
                if self.wallet_to_attrs and self.attr_headers:
                    upstream_wallet_level = self.manager.get_wallet_level(upstream_wallet, self.wallet_to_attrs, self.attr_headers)
                    if transaction_type == 'class4_laundering' or upstream_wallet_level == 4:
                        limits = self.manager.get_wallet_level_limits(4)
                        limit_max = limits.get("single_transaction_limit", amount_range["max"])
                        amount = self._generate_amount_near_limit(amount_range["min"], min(limit_max, amount_range["max"]), rng)
                    else:
                        amount = rng.uniform(amount_range["min"], amount_range["max"])
                else:
                    amount = rng.uniform(amount_range["min"], amount_range["max"])
            else:
                amount = rng.uniform(amount_range["min"], amount_range["max"])
            
            # 对于异常交易，放宽余额检查
            if is_risk:
                # 异常交易：直接生成交易，不检查余额
                current_balances[upstream_wallet] = current_balances.get(upstream_wallet, 0) - amount
                current_balances[main_wallet] = current_balances.get(main_wallet, 0) + amount
            else:
                # 正常交易：检查余额
                if current_balances.get(upstream_wallet, 0) >= amount:
                    current_balances[upstream_wallet] -= amount
                    current_balances[main_wallet] = current_balances.get(main_wallet, 0) + amount
                else:
                    continue  # 余额不足，跳过这个上游钱包
            
            transaction = {
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": upstream_wallet,
                "dst": main_wallet,
                "amount": round(amount, 2),
                "risk_type": transaction_type,
                "motif_id": motif_id,
                "transaction_mode": "many_to_many"
            }
            transactions.append(transaction)
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 计算可用于转出的金额（主钱包收到的总金额）
        total_received = sum(tx["amount"] for tx in transactions)
        
        # 生成转出交易（主钱包 → 多个下游钱包）
        # 分配金额给下游钱包
        amounts = []
        remaining = total_received
        
        for i in range(num_outputs):
            if i == num_outputs - 1:
                # 最后一笔包含所有剩余金额
                amounts.append(remaining)
            else:
                # 随机分配金额
                amount = rng.uniform(amount_range["min"], min(remaining / (num_outputs - i), amount_range["max"]))
                amounts.append(amount)
                remaining -= amount
        
        # 生成转出交易
        for downstream_wallet, amount in zip(selected_downstreams, amounts):
            amount = max(amount, min_abs_amount)  # 确保满足最小金额
            
            # 对于异常交易，放宽余额检查
            if is_risk:
                # 异常交易：直接生成交易，不检查余额
                current_balances[main_wallet] = current_balances.get(main_wallet, 0) - amount
                current_balances[downstream_wallet] = current_balances.get(downstream_wallet, 0) + amount
            else:
                # 正常交易：检查余额
                if current_balances.get(main_wallet, 0) >= amount:
                    current_balances[main_wallet] -= amount
                    current_balances[downstream_wallet] = current_balances.get(downstream_wallet, 0) + amount
                else:
                    continue  # 余额不足，跳过这个下游钱包
            
            transaction = {
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": main_wallet,
                "dst": downstream_wallet,
                "amount": round(amount, 2),
                "risk_type": transaction_type,
                "motif_id": motif_id,
                "transaction_mode": "many_to_many"
            }
            transactions.append(transaction)
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        return transactions, tx_id, current_time
    
    def _generate_many_to_many_mode(self, risk_type: str, g: nx.DiGraph, source_wallet: str, 
                                current_balances: Dict[str, float], current_time: datetime, tx_id: int, 
                                max_frac_per_tx: float, min_abs_amount: float, rng, 
                                wallet_to_attrs: Dict = None, attr_headers: List[str] = None, time_preference=None) -> Tuple[List[Dict], int, datetime]:
        """生成多对多模式交易：多个钱包 → 主钱包 → 多个钱包（确保完整性）"""
        transactions = []
        if not self.manager.is_preferred_time(current_time, risk_type):
            return [], tx_id, current_time
        motif_id = str(tx_id)
        
        # 检查图结构是否满足 many_to_many 的要求
        predecessors = list(g.predecessors(source_wallet))
        successors = list(g.successors(source_wallet))
        
        # many_to_many 需要：至少2个上游钱包 + 至少2个下游钱包
        if len(predecessors) < 2 or len(successors) < 2:
            return [], tx_id, current_time
        
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if len(available_wallets) < 4:
            return [], tx_id, current_time
        
        # 获取配置
        config = self.manager.get_transaction_config(risk_type)
        amount_range = config.get("amount_range", {"min": 50, "max": 500})
        time_interval = config.get("time_interval", {"min": 60, "max": 600})
        
        # 多次尝试生成完整交易
        max_attempts = 5  # 增加尝试次数，提高成功率
        for attempt in range(max_attempts):
            transactions = []
            temp_balances = current_balances.copy()
            
            # 判断是否为异常交易，异常交易生成更多的交易笔数
            is_risk = self.manager.is_risk_transaction(risk_type)
            
            # 选择上游钱包（转入主钱包）
            # 对于异常交易，放宽余额限制
            if is_risk:
                # 异常交易：不要求余额，允许所有可用钱包
                graph_upstream = [w for w in predecessors if w in available_wallets]
                other_upstream = [w for w in available_wallets 
                                 if w != source_wallet 
                                 and w not in graph_upstream]
                upstream_wallets = graph_upstream + other_upstream
            else:
                # 正常交易：要求余额 >= amount_range["min"]
                graph_upstream = [w for w in predecessors if w in available_wallets and current_balances.get(w, 0) >= amount_range["min"]]
                other_upstream = [w for w in available_wallets 
                                 if w != source_wallet 
                                 and w not in graph_upstream
                                 and current_balances.get(w, 0) >= amount_range["min"]]
                upstream_wallets = graph_upstream + other_upstream
            
            if len(upstream_wallets) < 2:
                continue  # 没有足够的上游钱包，跳过此次尝试
            
            # 根据是否为异常交易决定上游钱包数量
            if is_risk:
                # 异常交易：至少3个上游钱包，不设上限
                min_inputs = 3
                max_inputs = len(upstream_wallets)  # 不设上限，使用所有可用钱包
                num_inputs = rng.integers(min_inputs, max_inputs + 1)
            else:
                # 正常交易：2-4个上游钱包
                num_inputs = rng.integers(2, min(5, len(upstream_wallets) + 1))
            
            # 根据交易类型和钱包偏好过滤上游钱包
            preferred_upstream, non_preferred_upstream = self._filter_wallets_by_preference(upstream_wallets, is_risk)
            if len(preferred_upstream) >= min_inputs:
                available_upstream = preferred_upstream
            else:
                available_upstream = preferred_upstream + non_preferred_upstream
            
            if len(available_upstream) < min_inputs:
                continue  # 没有足够的上游钱包，跳过此次尝试
            
            num_inputs = rng.integers(min_inputs, min(max_inputs + 1, len(available_upstream) + 1))
            selected_upstreams = rng.choice(available_upstream, size=num_inputs, replace=False)
            
            # 选择下游钱包（从主钱包转出，不能与上游重复）
            # 优先从图结构中的 successors 中选择，确保边存在
            graph_downstream = [w for w in successors if w in available_wallets and w not in selected_upstreams]
            # 如果图结构中的下游钱包不够，再从其他可用钱包中选择
            other_downstream = [w for w in available_wallets 
                               if w not in selected_upstreams 
                               and w not in graph_downstream]
            downstream_wallets_all = graph_downstream + other_downstream
            
            # 根据交易类型和钱包偏好过滤下游钱包
            preferred_downstream, non_preferred_downstream = self._filter_wallets_by_preference(downstream_wallets_all, is_risk)
            
            if is_risk:
                # 异常交易：需要至少4个下游钱包，不设上限
                min_outputs = 4
                # 优先从preferred_downstream中选择，如果不够再从non_preferred_downstream补充
                if len(preferred_downstream) >= min_outputs:
                    available_downstream = preferred_downstream
                else:
                    available_downstream = preferred_downstream + non_preferred_downstream
                
                if len(available_downstream) < min_outputs:
                    continue  # 没有足够的下游钱包，跳过此次尝试
                max_outputs = len(available_downstream)  # 不设上限，使用所有可用钱包
                num_outputs = rng.integers(min_outputs, max_outputs + 1)
            else:
                # 正常交易：需要至少3个下游钱包，生成3-6个
                min_outputs = 3
                # 优先从preferred_downstream中选择，如果不够再从non_preferred_downstream补充
                if len(preferred_downstream) >= min_outputs:
                    available_downstream = preferred_downstream
                else:
                    available_downstream = preferred_downstream + non_preferred_downstream
                
                if len(available_downstream) < min_outputs:
                    continue  # 没有足够的下游钱包，跳过此次尝试
                num_outputs = rng.integers(min_outputs, min(7, len(available_downstream) + 1))
            
            selected_downstreams = rng.choice(available_downstream, size=num_outputs, replace=False)
            
            # 生成转入交易（多个上游钱包 → 主钱包）
            total_input_amount = 0
            successful_inputs = 0
            
            for upstream_wallet in selected_upstreams:
                # 对于异常交易，放宽余额检查
                if is_risk:
                    # 异常交易：允许余额不足，直接使用配置的最大金额
                    # 如果是class4_laundering或异常的四类钱包，让金额靠近单笔限额
                    if risk_type == 'class4_laundering' or (self.wallet_to_attrs and self.attr_headers):
                        if self.wallet_to_attrs and self.attr_headers:
                            upstream_wallet_level = self.manager.get_wallet_level(upstream_wallet, self.wallet_to_attrs, self.attr_headers)
                            if risk_type == 'class4_laundering' or upstream_wallet_level == 4:
                                limits = self.manager.get_wallet_level_limits(4)
                                limit_max = limits.get("single_transaction_limit", amount_range["max"])
                                input_amount = self._generate_amount_near_limit(amount_range["min"], min(limit_max, amount_range["max"]), rng)
                            else:
                                input_amount = rng.uniform(amount_range["min"], amount_range["max"])
                        else:
                            input_amount = rng.uniform(amount_range["min"], amount_range["max"])
                    else:
                        input_amount = rng.uniform(amount_range["min"], amount_range["max"])
                    # 直接执行交易，不检查余额（余额会在后续账户开立数据中修正）
                    temp_balances[upstream_wallet] = temp_balances.get(upstream_wallet, 0) - input_amount
                    temp_balances[source_wallet] = temp_balances.get(source_wallet, 0) + input_amount
                    total_input_amount += input_amount
                    successful_inputs += 1
                else:
                    # 正常交易：确保金额在范围内，且不超过余额
                    max_available = min(amount_range["max"], temp_balances.get(upstream_wallet, 0))
                    if max_available < amount_range["min"]:
                        continue  # 跳过余额不足的钱包
                    input_amount = rng.uniform(amount_range["min"], max_available)
                    
                    # 再次检查余额（双重检查）
                    if temp_balances.get(upstream_wallet, 0) < input_amount:
                        continue  # 跳过余额不足的钱包
                    
                    temp_balances[upstream_wallet] -= input_amount
                    temp_balances[source_wallet] = temp_balances.get(source_wallet, 0) + input_amount
                    total_input_amount += input_amount
                    successful_inputs += 1
                
                # 添加转入交易记录
                transactions.append({
                    "tx_id": tx_id,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src": upstream_wallet,  # 上游钱包
                    "dst": source_wallet,    # 主钱包
                    "amount": round(input_amount, 2),
                    "risk_type": risk_type,
                    "motif_id": motif_id,
                    "transaction_mode": "many_to_many"
                })
                tx_id += 1
                current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
            
            # 检查转入交易是否足够
            # 异常交易至少需要3笔，正常交易至少需要2笔
            min_inputs = 3 if is_risk else 2
            if successful_inputs < min_inputs or total_input_amount <= 0:
                continue  # 尝试下一次
            
            # 生成转出交易（主钱包 → 多个下游钱包）
            if risk_type =="merchant_laundering":
                config = self.manager.get_transaction_config(risk_type)
                commission_config = config.get("commission_rate")
                commission_rate = rng.uniform(commission_config["min"], commission_config["max"])
            else:
                commission_rate = 0.0
            available_amount = total_input_amount * (1 - commission_rate)
            
            # 确保每笔转出交易有合理的金额
            # 对于 class4_laundering，使用更均匀的分配，避免金额过小
            num_outputs = len(selected_downstreams)
            min_per_output = max(min_abs_amount, available_amount * 0.01)  # 至少是总金额的1%
            max_per_output = available_amount * 0.5  # 单笔最多50%
            
            # 使用改进的分配策略：先分配最小金额，然后随机分配剩余金额
            amounts = [min_per_output] * num_outputs
            remaining = available_amount - (min_per_output * num_outputs)
            
            if remaining > 0:
                # 使用beta分布分配剩余金额（但使用更均匀的参数）
                for i in range(num_outputs - 1):
                    weight = rng.beta(2, 2)  # 更均匀的分布
                    amount = remaining * weight
                    amount = min(amount, max_per_output - min_per_output)  # 限制单笔上限
                    amounts[i] += amount
                    remaining -= amount
                amounts[-1] += remaining  # 最后一笔包含所有剩余
            
            # 确保总金额等于available_amount（四舍五入误差调整）
            total_allocated = sum(amounts)
            if abs(total_allocated - available_amount) > 0.01:
                diff = available_amount - total_allocated
                amounts[0] += diff  # 将差值加到第一笔
            
            # 生成转出交易
            successful_outputs = 0
            for i, (dst_wallet, amount) in enumerate(zip(selected_downstreams, amounts)):
                # 确保金额满足最小要求
                amount = max(amount, min_abs_amount)
                
                # 对于异常交易，放宽余额检查
                if is_risk:
                    # 异常交易：允许余额不足，直接执行交易（余额会在后续账户开立数据中修正）
                    temp_balances[source_wallet] = temp_balances.get(source_wallet, 0) - amount
                    temp_balances[dst_wallet] = temp_balances.get(dst_wallet, 0) + amount
                    successful_outputs += 1
                else:
                    # 正常交易：检查余额
                    if amount <= temp_balances.get(source_wallet, 0):
                        temp_balances[source_wallet] -= amount
                        temp_balances[dst_wallet] = temp_balances.get(dst_wallet, 0) + amount
                        successful_outputs += 1
                    else:
                        # 余额不足，跳过这笔转出交易
                        continue
                
                transactions.append({
                    "tx_id": tx_id,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src": source_wallet,  # 主钱包
                    "dst": dst_wallet,     # 下游钱包
                    "amount": round(amount, 2),
                    "risk_type": risk_type,
                    "motif_id": motif_id,
                    "transaction_mode": "many_to_many"
                })
                tx_id += 1
                current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
            
            # 检查转出交易是否足够
            # 异常交易至少需要4笔，正常交易至少需要3笔
            min_outputs = 4 if is_risk else 3
            if successful_outputs < min_outputs:
                # 回滚余额
                for tx in transactions:
                    if tx["src"] in temp_balances:
                        temp_balances[tx["src"]] += tx["amount"]
                    if tx["dst"] in temp_balances:
                        temp_balances[tx["dst"]] -= tx["amount"]
                continue  # 尝试下一次
            
            # 验证交易完整性：根据是否为异常交易设置不同的最小要求
            incoming_count = sum(1 for tx in transactions if tx.get("dst") == source_wallet)
            outgoing_count = sum(1 for tx in transactions if tx.get("src") == source_wallet)
            
            min_incoming = 3 if is_risk else 2
            min_outgoing = 4 if is_risk else 3
            
            if incoming_count >= min_incoming and outgoing_count >= min_outgoing:
                # 验证通过，更新实际余额
                current_balances.update(temp_balances)
                return transactions, tx_id, current_time
            else:
                # 验证失败，回滚余额并继续尝试
                for tx in transactions:
                    if tx["src"] in temp_balances:
                        temp_balances[tx["src"]] += tx["amount"]
                    if tx["dst"] in temp_balances:
                        temp_balances[tx["dst"]] -= tx["amount"]
                continue
        
        # 所有尝试都失败，返回空
        return [], tx_id, current_time

    def _validate_many_to_many_completeness(self, transactions: List[Dict], main_wallet: str) -> bool:
        """验证many_to_many交易的完整性（只验证结构）"""
        if len(transactions) < 4:
            return False
        
        # 检查转入和转出交易
        incoming_txs = [tx for tx in transactions if tx.get("dst") == main_wallet]
        outgoing_txs = [tx for tx in transactions if tx.get("src") == main_wallet]
        
        # 必须至少有2笔转入和2笔转出
        if len(incoming_txs) < 2 or len(outgoing_txs) < 2:
            return False
        
        return True

    def _validate_many_to_one_completeness(self, transactions: List[Dict], main_wallet: str) -> bool:
        """验证many_to_one交易的完整性（只验证结构）"""
        if len(transactions) < 3:
            return False
        
        # 按主钱包分组（出现次数最多的dst）
        dst_counts = {}
        for tx in transactions:
            dst = tx.get("dst")
            dst_counts[dst] = dst_counts.get(dst, 0) + 1
        
        # 找出主钱包（出现次数最多的dst）
        main_wallet_by_dst = max(dst_counts, key=dst_counts.get)
        
        # 检查转入和转出交易
        incoming_txs = [tx for tx in transactions if tx.get("dst") == main_wallet_by_dst]
        outgoing_txs = [tx for tx in transactions if tx.get("src") == main_wallet_by_dst]
        
        # 必须至少有2笔转入和1笔转出
        if len(incoming_txs) < 2 or len(outgoing_txs) < 1:
            return False
        
        return True
    def _validate_one_to_many_completeness(self, transactions: List[Dict], main_wallet: str) -> bool:
        """验证one_to_many交易的完整性（只验证结构）"""
        if len(transactions) < 3:
            return False
        
        # 按主钱包分组（出现次数最多的src）
        src_counts = {}
        for tx in transactions:
            src = tx.get("src")
            src_counts[src] = src_counts.get(src, 0) + 1
        
        # 找出主钱包（出现次数最多的src）
        main_wallet_by_src = max(src_counts, key=src_counts.get)
        
        # 检查转入和转出交易
        incoming_txs = [tx for tx in transactions if tx.get("dst") == main_wallet_by_src]
        outgoing_txs = [tx for tx in transactions if tx.get("src") == main_wallet_by_src]
        
        # 必须至少有1笔转入和2笔转出
        if len(incoming_txs) < 1 or len(outgoing_txs) < 2:
            return False
        
        return True
    def _validate_forward_completeness(self, transactions: List[Dict], main_wallet: str) -> bool:
        """验证forward交易的完整性（只验证结构）"""
        if len(transactions) != 2:
            return False
        
        # 检查两笔交易是否都通过主钱包
        first_tx = transactions[0]
        second_tx = transactions[1]
        
        # 第一笔交易：其他钱包 → 主钱包
        if first_tx.get("dst") != main_wallet:
            return False
        
        # 第二笔交易：主钱包 → 其他钱包
        if second_tx.get("src") != main_wallet:
            return False
        
        # 确保两笔交易使用相同的motif_id
        if first_tx.get("motif_id") != second_tx.get("motif_id"):
            return False
        
        return True

    def _generate_one_to_many_mode_with_preferences(self, transaction_type: str, g: nx.DiGraph,
                                                  source_wallet: str, current_balances: Dict[str, float],
                                                  current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                                  min_abs_amount: float, rng, wallet_to_attrs: Dict,
                                                  attr_headers: List[str]) -> Tuple[List[Dict], int, datetime]:
        """生成一对多模式交易，应用商户跑分偏好"""
        transactions = []
        if not self.manager.is_preferred_time(current_time, transaction_type):
            return [], tx_id, current_time
        
        motif_id = str(tx_id)
        successors = list(g.successors(source_wallet))
        if len(successors) < 2:
            return [], tx_id, current_time
        
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if len(available_wallets) < 3:
            return transactions, tx_id, current_time
        
        # 选择1个上游钱包（转入主钱包），应用偏好
        upstream_candidates = [w for w in available_wallets if w != source_wallet and current_balances.get(w, 0) >= 100]
        if not upstream_candidates:
            return transactions, tx_id, current_time
            
        if transaction_type == "merchant_laundering":
            # 对于merchant_laundering，只对主钱包应用偏好，上游钱包可以是任何类型
            selected_upstreams = self.manager.apply_merchant_laundering_preferences(
                upstream_candidates, source_wallet, wallet_to_attrs, attr_headers, rng
            )
            if not selected_upstreams:
                return transactions, tx_id, current_time
            selected_upstream = selected_upstreams[0]
        else:
            selected_upstream = rng.choice(upstream_candidates)
        
        # 获取配置
        config = self.manager.get_transaction_config(transaction_type)
        amount_range = config.get("amount_range", {"min": 100, "max": 1000})
        time_interval = config.get("time_interval", {"min": 60, "max": 600})
        
        # 生成转入交易
        input_amount = rng.uniform(amount_range["min"], amount_range["max"])
        
        # 检查上游钱包余额
        if current_balances[selected_upstream] < input_amount:
            return transactions, tx_id, current_time
        
        current_balances[selected_upstream] -= input_amount
        current_balances[source_wallet] += input_amount
        
        transactions.append({
            "tx_id": tx_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "src": selected_upstream,
            "dst": source_wallet,
            "amount": round(input_amount, 2),
            "risk_type": transaction_type,
            "motif_id": motif_id,
            "transaction_mode": "one_to_many"
        })
        tx_id += 1
        current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 选择2-4个下游钱包，应用偏好
        downstream_candidates = [w for w in available_wallets if w != selected_upstream and w != source_wallet]
        downstream_count = rng.integers(2, min(5, len(downstream_candidates) + 1))
        
        if transaction_type == "merchant_laundering":
            selected_downstreams = self.manager.apply_merchant_laundering_preferences(
                downstream_candidates, source_wallet, wallet_to_attrs, attr_headers, rng
            )[:downstream_count]
        else:
            selected_downstreams = rng.choice(downstream_candidates, size=downstream_count, replace=False)
        
        # 生成转出交易（扣除佣金）
        if transaction_type == "merchant_laundering":
            commission_config = config.get("commission_rate", {"min": 0.05, "max": 0.10})
            commission_rate = rng.uniform(commission_config["min"], commission_config["max"])
        else:
            commission_rate = 0.0
        
        available_amount = input_amount * (1 - commission_rate)
        
        # 使用改进的金额分配策略，确保总金额等于 available_amount
        num_outputs = len(selected_downstreams)
        min_per_output = max(min_abs_amount, available_amount * 0.01)  # 至少是总金额的1%
        max_per_output = available_amount * 0.5  # 单笔最多50%
        
        # 使用改进的分配策略：先分配最小金额，然后随机分配剩余金额
        amounts = [min_per_output] * num_outputs
        remaining = available_amount - (min_per_output * num_outputs)
        
        if remaining > 0:
            # 使用beta分布分配剩余金额（但使用更均匀的参数）
            for i in range(num_outputs - 1):
                weight = rng.beta(2, 2)  # 更均匀的分布
                amount = remaining * weight
                amount = min(amount, max_per_output - min_per_output)  # 限制单笔上限
                amounts[i] += amount
                remaining -= amount
            amounts[-1] += remaining  # 最后一笔包含所有剩余
        
        # 确保总金额等于available_amount（四舍五入误差调整）
        total_allocated = sum(amounts)
        if abs(total_allocated - available_amount) > 0.01:
            diff = available_amount - total_allocated
            amounts[0] += diff  # 将差值加到第一笔
        
        # 分配金额给下游钱包
        for i, (downstream_wallet, amount) in enumerate(zip(selected_downstreams, amounts)):
            amount = max(amount, min_abs_amount)  # 确保满足最小金额
            
            if current_balances.get(source_wallet, 0) < amount:
                break
                
            current_balances[source_wallet] = current_balances.get(source_wallet, 0) - amount
            current_balances[downstream_wallet] = current_balances.get(downstream_wallet, 0) + amount
            
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": source_wallet,
                "dst": downstream_wallet,
                "amount": round(amount, 2),
                "risk_type": transaction_type,
                "motif_id": motif_id,
                "transaction_mode": "one_to_many"
            })
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        return transactions, tx_id, current_time

    def _generate_many_to_one_mode_with_preferences(self, transaction_type: str, g: nx.DiGraph,
                                                  source_wallet: str, current_balances: Dict[str, float],
                                                  current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                                  min_abs_amount: float, rng, wallet_to_attrs: Dict,
                                                  attr_headers: List[str]) -> Tuple[List[Dict], int, datetime]:
        """生成多对一模式交易，应用商户跑分偏好"""
        transactions = []
        if not self.manager.is_preferred_time(current_time, transaction_type):
            return [], tx_id, current_time
        
        motif_id = str(tx_id)
        predecessors = list(g.predecessors(source_wallet))
        if len(predecessors) < 2:
            return [], tx_id, current_time
        
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if len(available_wallets) < 3:
            return transactions, tx_id, current_time
        
        # 选择2-4个上游钱包，应用偏好
        upstream_candidates = [w for w in available_wallets if w != source_wallet and current_balances.get(w, 0) >= 100]
        upstream_count = rng.integers(2, min(5, len(upstream_candidates) + 1))
        
        if transaction_type == "merchant_laundering":
            selected_upstreams = self.manager.apply_merchant_laundering_preferences(
                upstream_candidates, source_wallet, wallet_to_attrs, attr_headers, rng
            )[:upstream_count]
        else:
            selected_upstreams = rng.choice(upstream_candidates, size=upstream_count, replace=False)
        
        # 获取配置
        config = self.manager.get_transaction_config(transaction_type)
        amount_range = config.get("amount_range", {"min": 100, "max": 1000})
        time_interval = config.get("time_interval", {"min": 60, "max": 600})
        
        # 判断是否为异常交易，异常交易放宽余额检查
        is_risk = self.manager.is_risk_transaction(transaction_type)
        
        # 生成转入交易
        total_input_amount = 0
        for upstream_wallet in selected_upstreams:
            input_amount = rng.uniform(amount_range["min"], amount_range["max"])
            
            # 对于异常交易，放宽余额检查
            if is_risk:
                # 异常交易：直接生成交易，不检查余额
                current_balances[upstream_wallet] = current_balances.get(upstream_wallet, 0) - input_amount
                current_balances[source_wallet] = current_balances.get(source_wallet, 0) + input_amount
                total_input_amount += input_amount
            else:
                # 正常交易：检查余额
                if current_balances.get(upstream_wallet, 0) < input_amount:
                    continue
                    
                current_balances[upstream_wallet] -= input_amount
                current_balances[source_wallet] = current_balances.get(source_wallet, 0) + input_amount
                total_input_amount += input_amount
            
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": upstream_wallet,
                "dst": source_wallet,
                "amount": round(input_amount, 2),
                "risk_type": transaction_type,
                "motif_id": motif_id,
                "transaction_mode": "many_to_one"
            })
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 选择1个下游钱包，应用偏好
        downstream_candidates = [w for w in available_wallets if w not in selected_upstreams and w != source_wallet]
        if not downstream_candidates:
            return transactions, tx_id, current_time
            
        if transaction_type == "merchant_laundering":
            selected_downstreams = self.manager.apply_merchant_laundering_preferences(
                downstream_candidates, source_wallet, wallet_to_attrs, attr_headers, rng
            )
            if not selected_downstreams:
                return transactions, tx_id, current_time
            selected_downstream = selected_downstreams[0]
        else:
            selected_downstream = rng.choice(downstream_candidates)
        
        # 生成转出交易（扣除佣金）
        if total_input_amount > 0:
            if transaction_type == "merchant_laundering":
                commission_config = config.get("commission_rate", {"min": 0.05, "max": 0.10})
                commission_rate = rng.uniform(commission_config["min"], commission_config["max"])
            else:
                commission_rate = 0.0
            
            available_amount = total_input_amount * (1 - commission_rate)
            
            # 判断是否为异常交易，异常交易放宽余额检查
            is_risk = self.manager.is_risk_transaction(transaction_type)
            
            if is_risk:
                # 异常交易：直接生成交易，不检查余额
                current_balances[source_wallet] = current_balances.get(source_wallet, 0) - available_amount
                current_balances[selected_downstream] = current_balances.get(selected_downstream, 0) + available_amount
                
                transactions.append({
                    "tx_id": tx_id,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src": source_wallet,
                    "dst": selected_downstream,
                    "amount": round(available_amount, 2),
                    "risk_type": transaction_type,
                    "motif_id": motif_id,
                    "transaction_mode": "many_to_one"
                })
                tx_id += 1
            else:
                # 正常交易：检查余额
                if current_balances.get(source_wallet, 0) >= available_amount:
                    current_balances[source_wallet] -= available_amount
                    current_balances[selected_downstream] = current_balances.get(selected_downstream, 0) + available_amount
                    
                    transactions.append({
                        "tx_id": tx_id,
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "src": source_wallet,
                        "dst": selected_downstream,
                        "amount": round(available_amount, 2),
                        "risk_type": transaction_type,
                        "motif_id": motif_id,
                        "transaction_mode": "many_to_one"
                    })
                    tx_id += 1
        
        return transactions, tx_id, current_time

    def _generate_many_to_many_mode_with_preferences(self, transaction_type: str, g: nx.DiGraph,

                                                   source_wallet: str, current_balances: Dict[str, float],
                                                   current_time: datetime, tx_id: int, max_frac_per_tx: float,
                                                   min_abs_amount: float, rng, wallet_to_attrs: Dict,
                                                   attr_headers: List[str]) -> Tuple[List[Dict], int, datetime]:
        """生成多对多模式交易，应用商户跑分偏好"""
        transactions = []
        if not self.manager.is_preferred_time(current_time, transaction_type):
            return [], tx_id, current_time
        
        motif_id = str(tx_id)
        
        # 获取所有可用钱包
        all_wallets = list(g.nodes())
        available_wallets = [w for w in all_wallets if w != source_wallet and w in current_balances]
        
        if len(available_wallets) < 4:  # 至少需要2个上游+2个下游
            return transactions, tx_id, current_time
        
        # 选择2-3个上游钱包，应用偏好
        upstream_candidates = [w for w in available_wallets if w != source_wallet and current_balances.get(w, 0) >= 100]
        upstream_count = rng.integers(2, min(4, len(upstream_candidates) + 1))
        
        if transaction_type == "merchant_laundering":
            selected_upstreams = self.manager.apply_merchant_laundering_preferences(
                upstream_candidates, source_wallet, wallet_to_attrs, attr_headers, rng
            )[:upstream_count]
        else:
            selected_upstreams = rng.choice(upstream_candidates, size=upstream_count, replace=False)
        
        # 选择2-3个下游钱包，应用偏好
        downstream_candidates = [w for w in available_wallets if w not in selected_upstreams and w != source_wallet]
        downstream_count = rng.integers(2, min(4, len(downstream_candidates) + 1))
        
        if transaction_type == "merchant_laundering":
            selected_downstreams = self.manager.apply_merchant_laundering_preferences(
                downstream_candidates, source_wallet, wallet_to_attrs, attr_headers, rng
            )[:downstream_count]
        else:
            selected_downstreams = rng.choice(downstream_candidates, size=downstream_count, replace=False)
        
        # 获取配置
        config = self.manager.get_transaction_config(transaction_type)
        amount_range = config.get("amount_range", {"min": 100, "max": 1000})
        time_interval = config.get("time_interval", {"min": 60, "max": 600})
        
        # 生成转入交易
        total_input_amount = 0
        successful_inputs = 0
        for upstream_wallet in selected_upstreams:
            # 确保金额在范围内，且不超过余额
            max_available = min(amount_range["max"], current_balances.get(upstream_wallet, 0))
            if max_available < amount_range["min"]:
                continue  # 跳过余额不足的钱包
            
            input_amount = rng.uniform(amount_range["min"], max_available)
            
            if current_balances[upstream_wallet] < input_amount:
                continue  # 跳过余额不足的钱包
                
            current_balances[upstream_wallet] -= input_amount
            current_balances[source_wallet] += input_amount
            total_input_amount += input_amount
            successful_inputs += 1
            
            transactions.append({
                "tx_id": tx_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": upstream_wallet,
                "dst": source_wallet,
                "amount": round(input_amount, 2),
                "risk_type": transaction_type,
                "motif_id": motif_id,
                "transaction_mode": "many_to_many"
            })
            tx_id += 1
            current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
        
        # 检查转入交易是否足够（至少2笔）
        if successful_inputs < 2 or total_input_amount <= 0:
            return [], tx_id, current_time
        
        # 生成转出交易（扣除佣金）
        if total_input_amount > 0:
            if transaction_type == "merchant_laundering":
                commission_config = config.get("commission_rate", {"min": 0.05, "max": 0.10})
                commission_rate = rng.uniform(commission_config["min"], commission_config["max"])
            else:
                commission_rate = 0.0
            
            available_amount = total_input_amount * (1 - commission_rate)
            
            # 确保转出总金额不超过可用金额（扣除佣金后）
            # 使用改进的金额分配策略，确保总金额等于 available_amount
            num_outputs = len(selected_downstreams)
            min_per_output = max(min_abs_amount, available_amount * 0.01)  # 至少是总金额的1%
            max_per_output = available_amount * 0.5  # 单笔最多50%
            
            # 使用改进的分配策略：先分配最小金额，然后随机分配剩余金额
            amounts = [min_per_output] * num_outputs
            remaining = available_amount - (min_per_output * num_outputs)
            
            if remaining > 0:
                # 使用beta分布分配剩余金额（但使用更均匀的参数）
                for i in range(num_outputs - 1):
                    weight = rng.beta(2, 2)  # 更均匀的分布
                    amount = remaining * weight
                    amount = min(amount, max_per_output - min_per_output)  # 限制单笔上限
                    amounts[i] += amount
                    remaining -= amount
                amounts[-1] += remaining  # 最后一笔包含所有剩余
            
            # 确保总金额等于available_amount（四舍五入误差调整）
            total_allocated = sum(amounts)
            if abs(total_allocated - available_amount) > 0.01:
                diff = available_amount - total_allocated
                amounts[0] += diff  # 将差值加到第一笔
            
            # 生成转出交易
            successful_outputs = 0
            for i, (downstream_wallet, amount) in enumerate(zip(selected_downstreams, amounts)):
                amount = max(amount, min_abs_amount)  # 确保满足最小金额
                
                # 检查余额（虽然对于异常交易我们已经放宽了，但为了完整性仍然检查）
                if current_balances.get(source_wallet, 0) < amount:
                    # 如果余额不足，从其他已分配金额中调整
                    continue
                
                current_balances[source_wallet] = current_balances.get(source_wallet, 0) - amount
                current_balances[downstream_wallet] = current_balances.get(downstream_wallet, 0) + amount
                successful_outputs += 1
                
                transactions.append({
                    "tx_id": tx_id,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src": source_wallet,
                    "dst": downstream_wallet,
                    "amount": round(amount, 2),
                    "risk_type": transaction_type,
                    "motif_id": motif_id,
                    "transaction_mode": "many_to_many"
                })
                tx_id += 1
                current_time += timedelta(seconds=int(rng.integers(time_interval["min"], max(time_interval["min"] + 1, time_interval["max"]))))
            
            # 检查转出交易是否足够（至少2笔）
            if successful_outputs < 2:
                # 回滚所有交易
                for tx in transactions:
                    if tx.get("dst") == source_wallet:  # 转入交易
                        current_balances[tx["src"]] += tx["amount"]
                        current_balances[source_wallet] -= tx["amount"]
                    else:  # 转出交易
                        current_balances[source_wallet] += tx["amount"]
                        current_balances[tx["dst"]] -= tx["amount"]
                return [], tx_id, current_time
        
        # 验证交易完整性
        incoming_count = sum(1 for tx in transactions if tx.get("dst") == source_wallet)
        outgoing_count = sum(1 for tx in transactions if tx.get("src") == source_wallet)
        
        if incoming_count >= 2 and outgoing_count >= 2:
            return transactions, tx_id, current_time
        else:
            # 回滚所有交易
            for tx in transactions:
                if tx.get("dst") == source_wallet:  # 转入交易
                    current_balances[tx["src"]] += tx["amount"]
                    current_balances[source_wallet] -= tx["amount"]
                else:  # 转出交易
                    current_balances[source_wallet] += tx["amount"]
                    current_balances[tx["dst"]] -= tx["amount"]
            return [], tx_id, current_time
