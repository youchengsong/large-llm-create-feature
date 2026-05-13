#!/usr/bin/env python3
"""
基于分布的钱包开立模拟系统
根据正常和异常钱包的属性分布直接生成数据，无需后续打分
参照open_wallet_account_v2.py的格式和规则
钱包类型：1-对公钱包，2-个人钱包，3-商户钱包
"""

import json
import csv
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import random
import ipaddress

# 时间格式常量
_ts_fmt = "%Y-%m-%d %H:%M:%S"
_date_fmt = "%Y-%m-%d"

class WalletDistributionGenerator:
    """基于分布的钱包生成器"""
    
    def __init__(self, config_path: str = "conf_wallet_distribution_based.json"):
        """初始化生成器"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 设置随机种子
        np.random.seed(self.config.get("seed", 1243))
        random.seed(self.config.get("seed", 1243))
        
        # 基本参数
        self.accounts_num = self.config.get("accounts_num", 1000)
        self.laundering_ratio = self.config.get("laundering_ratio", 0.15)
        
        # 计算正常和异常钱包数量
        self.abnormal_num = int(self.accounts_num * self.laundering_ratio)
        self.normal_num = self.accounts_num - self.abnormal_num
        
        # 初始化IP池
        self._init_ip_pools()
        
        print(f"计划生成 {self.accounts_num} 个钱包:")
        print(f"  - 正常钱包: {self.normal_num} 个")
        print(f"  - 异常钱包: {self.abnormal_num} 个")
        print(f"  - 钱包类型: 1-对公钱包, 2-个人钱包, 3-商户钱包")
    
    def _init_ip_pools(self):
        """初始化IP池"""
        ip_ranges = {
            "domestic": [
                "192.168.1.0/24",
                "192.168.0.0/24",
                "10.0.0.0/8",
                "172.16.0.0/12"
            ],
            "mobile_networks": [
                "117.136.0.0/16",
                "117.137.0.0/16",
                "117.138.0.0/16",
                "117.139.0.0/16",
                "117.140.0.0/16",
                "117.141.0.0/16",
                "117.142.0.0/16",
                "117.143.0.0/16"
            ],
            "broadband": [
                "58.16.0.0/16",
                "58.17.0.0/16",
                "58.18.0.0/16",
                "58.19.0.0/16",
                "58.20.0.0/16",
                "58.21.0.0/16",
                "58.22.0.0/16",
                "58.23.0.0/16"
            ]
        }
        
        self.ip_pools = {}
        for ip_type, ranges in ip_ranges.items():
            self.ip_pools[ip_type] = []
            for range_str in ranges:
                try:
                    network = ipaddress.ip_network(range_str, strict=False)
                    # 生成该网络中的一些IP地址
                    for i in range(min(50, network.num_addresses)):
                        ip = str(network.network_address + i)
                        self.ip_pools[ip_type].append(ip)
                except:
                    continue
    
    def _generate_random_mac(self) -> str:
        """生成随机MAC地址"""
        mac_parts = []
        for i in range(6):
            mac_parts.append(f"{random.randint(0, 255):02X}")
        return ":".join(mac_parts)
    
    def _generate_device(self, wallet_id: str) -> str:
        """生成设备信息（MAC地址）"""
        # 10%的概率为柜台开立（返回None）
        hash_val = hash(str(wallet_id)) % 100
        if hash_val < 10:
            return None
        else:
            return self._generate_random_mac()
    
    def _generate_ip(self, wallet_id: str) -> str:
        """生成IP地址"""
        # 10%的概率为柜台开立（返回None）
        hash_val = hash(str(wallet_id)) % 100
        if hash_val < 10:
            return None
        else:
            # 根据钱包类型选择IP - 模拟手机网络
            ip_type = random.choices(
                ['mobile_networks', 'broadband', 'domestic'],
                weights=[0.6, 0.3, 0.1]  # 手机网络占主要比例
            )[0]
            
            if ip_type in self.ip_pools and self.ip_pools[ip_type]:
                return random.choice(self.ip_pools[ip_type])
            else:
                # 生成随机IP
                return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 254)}"
    
    def generate_wallet_id(self) -> str:
        """生成钱包ID"""
        wallet_config = self.config["wallet_id"]
        length = wallet_config["length"]
        
        if wallet_config["numeric_only"]:
            if wallet_config["no_leading_zero"]:
                # 第一位不能是0
                first_digit = str(random.randint(1, 9))
                remaining = ''.join([str(random.randint(0, 9)) for _ in range(length - 1)])
                return first_digit + remaining
            else:
                return ''.join([str(random.randint(0, 9)) for _ in range(length)])
        else:
            return ''.join([random.choice('0123456789') for _ in range(length)])
    
    def generate_wallet_type_and_level(self, is_abnormal: bool) -> Tuple[int, int]:
        """生成钱包类型和等级"""
        wallet_type_config = self.config["wallet_type"]
        wallet_level_config = self.config["wallet_level"]
        
        # 1. 先生成钱包类型（1=对公, 2=个人, 3=商户）
        wallet_type = int(np.random.choice(
            wallet_type_config["options"], 
            p=wallet_type_config["probs"]
        ))
        
        # 2. 再根据钱包类型生成等级
        level_options = wallet_level_config["options"]
        level_probs = wallet_level_config["probs"]

        if wallet_type == 1:
            """
            对公钱包（wallet_type=1）：
            - 只允许 1、2 类钱包
            - 不存在 3、4 类对公钱包
            这里从全局等级分布中“截取”出等级 1 和 2 的概率并重新归一化，
            这样整体等级比例仍然尽量贴近原始配置。
            """
            allowed_levels = [1, 2]
            allowed_probs = [
                p for lvl, p in zip(level_options, level_probs) if lvl in allowed_levels
            ]
            prob_sum = sum(allowed_probs)
            # 防御性处理，避免配置错误导致除以 0
            if prob_sum <= 0:
                # 如果配置异常，则退化为均匀分布
                norm_probs = [0.5, 0.5]
            else:
                norm_probs = [p / prob_sum for p in allowed_probs]

            wallet_level = int(np.random.choice(allowed_levels, p=norm_probs))
        else:
            """
            个人和商户钱包（wallet_type=2 或 3）：
            - 允许 1 / 2 / 3 / 4 所有等级
            - 使用原始等级分布配置
            """
            wallet_level = int(np.random.choice(
                level_options,
                p=level_probs
            ))

        return wallet_type, wallet_level
    
    def generate_region_code(self, is_abnormal: bool) -> str:
        """生成地区代码"""
        if is_abnormal:
            dist_config = self.config["distribution_based_generation"]["abnormal_wallet_distributions"]["region_code"]
        else:
            dist_config = self.config["distribution_based_generation"]["normal_wallet_distributions"]["region_code"]
        
        high_risk_ratio = dist_config["high_risk_ratio"]
        
        if random.random() < high_risk_ratio:
            return random.choice(dist_config["high_risk_codes"])
        else:
            return random.choice(dist_config["normal_codes"])
    
    def generate_cert_and_tel_distributions(self, is_abnormal: bool) -> Tuple[Dict[str, int], Dict[str, int]]:
        """生成身份证和手机号的分布
        
        Args:
            is_abnormal: 是否为异常钱包，True使用abnormal_wallet_distributions配置，False使用normal_wallet_distributions配置
        """
        if is_abnormal:
            # 使用异常钱包的身份证和手机号分布配置
            cert_config = self.config["distribution_based_generation"]["abnormal_wallet_distributions"]["wallet_open_cert_distribution"]
            tel_config = self.config["distribution_based_generation"]["abnormal_wallet_distributions"]["wallet_open_tel_distribution"]
            account_num = self.abnormal_num  # 使用异常账户数量
            wallet_type = "异常"
        else:
            # 使用正常钱包的身份证和手机号分布配置
            cert_config = self.config["distribution_based_generation"]["normal_wallet_distributions"]["wallet_open_cert_distribution"]
            tel_config = self.config["distribution_based_generation"]["normal_wallet_distributions"]["wallet_open_tel_distribution"]
            account_num = self.normal_num  # 使用正常账户数量
            wallet_type = "正常"
        
        print(f"  {wallet_type}钱包配置 - 身份证唯一比例: {cert_config['unique_ratio']:.2%}, 共享比例: {cert_config['shared_ratio']:.2%}")
        print(f"  {wallet_type}钱包配置 - 手机号唯一比例: {tel_config['unique_ratio']:.2%}, 共享比例: {tel_config['shared_ratio']:.2%}")
        
        # 生成身份证分布（使用对应类型的配置）
        cert_groups = self._generate_shared_groups(
            cert_config["unique_ratio"], 
            cert_config["shared_ratio"], 
            cert_config["group_size_probs"],
            "cert",
            account_num  # 传入正确的账户数量
        )
        
        # 生成手机号分布（使用对应类型的配置）
        tel_groups = self._generate_shared_groups(
            tel_config["unique_ratio"], 
            tel_config["shared_ratio"], 
            tel_config["group_size_probs"],
            "tel",
            account_num  # 传入正确的账户数量
        )
        
        return cert_groups, tel_groups
    
    def _generate_shared_groups(self, unique_ratio: float, shared_ratio: float, 
                               group_size_probs: Dict[str, float], prefix: str, account_num: int) -> Dict[str, int]:
        """生成共享组
        
        Args:
            unique_ratio: 唯一值比例
            shared_ratio: 共享值比例
            group_size_probs: 共享组大小概率分布
            prefix: 组ID前缀
            account_num: 账户数量（正常或异常账户的数量）
        """
        groups = {}
        group_id = 1
        
        # 生成唯一值（使用传入的账户数量，而不是总账户数）
        unique_count = int(account_num * unique_ratio)
        for i in range(unique_count):
            groups[f"{prefix}_{group_id}"] = 1
            group_id += 1
        
        # 生成共享组（使用传入的账户数量，而不是总账户数）
        shared_count = int(account_num * shared_ratio)
        remaining_shared = shared_count
        
        # 归一化概率
        group_sizes = list(group_size_probs.keys())
        group_probs = list(group_size_probs.values())
        total_prob = sum(group_probs)
        if total_prob > 0:
            group_probs = [p / total_prob for p in group_probs]
        
        while remaining_shared > 0:
            # 选择组大小
            group_size = int(np.random.choice(group_sizes, p=group_probs))
            
            actual_size = min(group_size, remaining_shared)
            groups[f"{prefix}_{group_id}"] = actual_size
            remaining_shared -= actual_size
            group_id += 1
        
        return groups
    
    def generate_uniform_date_list(self) -> List[str]:
        """生成均匀分布的开立日期列表"""
        date_config = self.config["wallet_open_date"]
        start_date = datetime.strptime(date_config["start"], "%Y-%m-%d")
        end_date = datetime.strptime(date_config["end"], "%Y-%m-%d")
        
        days_diff = (end_date - start_date).days + 1  # 包含起始日期和结束日期
        total_accounts = self.accounts_num
        
        # 计算每天应该有多少账户
        accounts_per_day = total_accounts // days_diff
        remaining_accounts = total_accounts % days_diff
        
        # 生成日期列表
        date_list = []
        current_date = start_date
        
        for day in range(days_diff):
            # 基础每天分配的账户数
            day_count = accounts_per_day
            # 剩余的账户随机分配到某些天（确保总数为total_accounts）
            if day < remaining_accounts:
                day_count += 1
            
            # 为该天添加日期
            for _ in range(day_count):
                date_list.append(current_date.strftime(_date_fmt))
            
            current_date += timedelta(days=1)
        
        # 打乱日期顺序，但保持总体分布均匀
        random.shuffle(date_list)
        
        return date_list
    
    def generate_wallet_open_timestamp(self, open_date: str, is_abnormal: bool) -> str:
        """生成开立时间戳"""
        # open_date 作为参数传入，不再重新生成
        
        if is_abnormal:
            dist_config = self.config["distribution_based_generation"]["abnormal_wallet_distributions"]["wallet_open_timestamp_hour"]
        else:
            dist_config = self.config["distribution_based_generation"]["normal_wallet_distributions"]["wallet_open_timestamp_hour"]
        
        # 选择小时
        if random.random() < dist_config["night_hours_prob"]:
            hour = random.choice(dist_config["night_hours"])
        else:
            if dist_config["distribution"] == "normal":
                # 正态分布，偏向工作时间
                hour = int(np.random.normal(13, 4))  # 均值13点，标准差4
                hour = max(0, min(23, hour))  # 限制在0-23之间
            else:
                # 均匀分布
                hour = random.randint(0, 23)
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        timestamp = datetime.strptime(open_date, _date_fmt).replace(hour=hour, minute=minute, second=second)
        return timestamp.strftime(_ts_fmt)
    
    def generate_realname_auth_time(self, wallet_open_timestamp: str, is_abnormal: bool) -> str:
        """生成实名认证时间"""
        if is_abnormal:
            dist_config = self.config["distribution_based_generation"]["abnormal_wallet_distributions"]["realname_auth_delay"]
        else:
            dist_config = self.config["distribution_based_generation"]["normal_wallet_distributions"]["realname_auth_delay"]
        
        open_time = datetime.strptime(wallet_open_timestamp, _ts_fmt)
        
        rand = random.random()
        if rand < dist_config["same_day_prob"]:
            delay_days = 0
        elif rand < dist_config["same_day_prob"] + dist_config["within_7_days_prob"]:
            delay_days = random.randint(1, 7)
        else:
            delay_days = random.randint(8, dist_config["max_delay_days"])
        
        auth_time = open_time + timedelta(days=delay_days)
        return auth_time.strftime(_ts_fmt)
    
    def generate_cert_expire_date(self, wallet_open_date: str, is_abnormal: bool) -> str:
        """生成身份证到期日期"""
        if is_abnormal:
            dist_config = self.config["distribution_based_generation"]["abnormal_wallet_distributions"]["cert_expire_days"]
        else:
            dist_config = self.config["distribution_based_generation"]["normal_wallet_distributions"]["cert_expire_days"]
        
        open_date = datetime.strptime(wallet_open_date, _date_fmt)
        
        if dist_config["distribution"] == "normal":
            days_to_expire = int(np.random.normal(dist_config["mean"], dist_config["std"]))
        else:
            days_to_expire = random.randint(dist_config["min"], dist_config["max"])
        
        days_to_expire = max(dist_config["min"], min(dist_config["max"], days_to_expire))
        expire_date = open_date + timedelta(days=days_to_expire)
        
        return expire_date.strftime(_date_fmt)
    
    def generate_bank_code(self) -> str:
        """生成银行代码"""
        prefixes = self.config["bank_code_prefixes"]
        prefix = random.choice(prefixes)
        suffix = "".join([str(random.randint(0, 9)) for _ in range(12)])
        return prefix + suffix
    
    def generate_bank_account_info(self) -> Tuple[str, str]:
        """生成银行账户信息，返回(完整卡号, 前四位)"""
        full_account_number = self.generate_bank_code()
        account_prefix = full_account_number[:4]  # 取前四位
        return full_account_number, account_prefix
    
    def generate_init_balance(self, wallet_level: int, is_abnormal: bool) -> float:
        """生成初始余额，异常账户和正常账户使用不同的配置"""
        # 根据是否异常选择不同的余额配置
        if is_abnormal:
            balance_config = self.config.get("abnormal_init_balance_mixture", self.config.get("init_balance_mixture", []))
        else:
            balance_config = self.config.get("normal_init_balance_mixture", self.config.get("init_balance_mixture", []))
        
        level_caps = self.config["wallet_level_balance_caps"]
        
        # 生成余额 - 直接根据概率选择配置项
        probs = [item["prob"] for item in balance_config]
        balance_config_item = np.random.choice(balance_config, p=probs)
        
        if balance_config_item["type"] == "point":
            balance = balance_config_item["value"]
        elif balance_config_item["type"] == "uniform":
            balance = random.uniform(balance_config_item["min"], balance_config_item["max"])
        elif balance_config_item["type"] == "lognormal":
            balance = np.random.lognormal(balance_config_item["mu"], balance_config_item["sigma"])
            # 添加范围限制
            if "min" in balance_config_item:
                balance = max(balance, balance_config_item["min"])
            if "max" in balance_config_item:
                balance = min(balance, balance_config_item["max"])
        
        # 应用等级限制 - 修改为更合理的分布
        cap = level_caps.get(str(wallet_level))
        if cap is not None:
            # 不是简单截断，而是在限制范围内重新生成
            if balance > cap:
                # 在0到cap之间重新生成
                balance = random.uniform(0, cap)
        
        return round(balance, 2)
    
    def generate_real_cert_number(self) -> str:
        """生成真实格式的身份证号"""
        # 生成18位身份证号
        # 前6位：地区代码（随机选择）
        region_codes = ["110101", "310101", "440101", "500101", "120101", "320101", "130101", "140101", "150101", "210101"]
        region = random.choice(region_codes)
        
        # 第7-14位：出生日期（1950-2000年）
        year = random.randint(1950, 2000)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # 简化处理，避免日期问题
        birth_date = f"{year:04d}{month:02d}{day:02d}"
        
        # 第15-17位：顺序码
        sequence = random.randint(100, 999)
        
        # 第18位：校验码（简化处理，随机生成）
        check_code = random.randint(0, 9)
        
        return f"{region}{birth_date}{sequence}{check_code}"
    
    def generate_real_phone_number(self) -> str:
        """生成真实格式的手机号"""
        # 生成11位手机号
        # 第1位：1
        # 第2位：3,4,5,6,7,8,9
        # 第3-11位：随机数字
        second_digit = random.choice([3, 4, 5, 6, 7, 8, 9])
        remaining = ''.join([str(random.randint(0, 9)) for _ in range(9)])
        return f"1{second_digit}{remaining}"
    
    def generate_accounts(self) -> List[Dict[str, Any]]:
        """生成所有账户数据"""
        print("开始生成账户数据...")
        
        # 生成身份证和手机号分布（分别使用正常和异常钱包的配置）
        print("生成身份证和手机号分布...")
        # 正常钱包：使用normal_wallet_distributions中的wallet_open_cert_distribution配置
        normal_cert_groups, normal_tel_groups = self.generate_cert_and_tel_distributions(False)
        # 异常钱包：使用abnormal_wallet_distributions中的wallet_open_cert_distribution配置
        abnormal_cert_groups, abnormal_tel_groups = self.generate_cert_and_tel_distributions(True)
        
        # 分别为正常和异常账户创建证书和手机号列表（不合并！）
        normal_cert_list = []
        normal_tel_list = []
        
        for cert_group, count in normal_cert_groups.items():
            # 为每个组生成真实格式的身份证号
            base_cert = self.generate_real_cert_number()
            for i in range(count):
                if i == 0:
                    normal_cert_list.append(base_cert)
                else:
                    # 对于共享的身份证，使用相同号码
                    normal_cert_list.append(base_cert)
        
        for tel_group, count in normal_tel_groups.items():
            # 为每个组生成真实格式的手机号
            base_tel = self.generate_real_phone_number()
            for i in range(count):
                if i == 0:
                    normal_tel_list.append(base_tel)
                else:
                    # 对于共享的手机号，使用相同号码
                    normal_tel_list.append(base_tel)
        
        abnormal_cert_list = []
        abnormal_tel_list = []
        
        for cert_group, count in abnormal_cert_groups.items():
            # 为每个组生成真实格式的身份证号
            base_cert = self.generate_real_cert_number()
            for i in range(count):
                if i == 0:
                    abnormal_cert_list.append(base_cert)
                else:
                    # 对于共享的身份证，使用相同号码
                    abnormal_cert_list.append(base_cert)
        
        for tel_group, count in abnormal_tel_groups.items():
            # 为每个组生成真实格式的手机号
            base_tel = self.generate_real_phone_number()
            for i in range(count):
                if i == 0:
                    abnormal_tel_list.append(base_tel)
                else:
                    # 对于共享的手机号，使用相同号码
                    abnormal_tel_list.append(base_tel)
        
        # 分别打乱正常和异常账户的证书和手机号列表
        random.shuffle(normal_cert_list)
        random.shuffle(normal_tel_list)
        random.shuffle(abnormal_cert_list)
        random.shuffle(abnormal_tel_list)
        
        # 生成均匀分布的开立日期列表
        print("生成均匀分布的开立日期...")
        date_list = self.generate_uniform_date_list()
        random.shuffle(date_list)  # 打乱日期顺序
        
        # 生成账户数据
        accounts_data = []
        
        # 生成正常钱包（只使用正常账户的证书和手机号）
        print(f"生成 {self.normal_num} 个正常钱包...")
        for i in range(self.normal_num):
            account = self._generate_single_account(False, normal_cert_list[i], normal_tel_list[i], date_list[i])
            accounts_data.append(account)
        
        # 生成异常钱包（只使用异常账户的证书和手机号）
        print(f"生成 {self.abnormal_num} 个异常钱包...")
        for i in range(self.abnormal_num):
            account = self._generate_single_account(
                True, 
                abnormal_cert_list[i], 
                abnormal_tel_list[i],
                date_list[self.normal_num + i]
            )
            accounts_data.append(account)
        
        # 打乱所有账户
        random.shuffle(accounts_data)
        
        print(f"成功生成 {len(accounts_data)} 个账户")
        return accounts_data
    
    def _generate_single_account(self, is_abnormal: bool, cert_value: str, tel_value: str, wallet_open_date: str) -> Dict[str, Any]:
        """生成单个账户数据"""
        wallet_id = self.generate_wallet_id()
        wallet_type, wallet_level = self.generate_wallet_type_and_level(is_abnormal)
        region_code = self.generate_region_code(is_abnormal)
        
        wallet_open_timestamp = self.generate_wallet_open_timestamp(wallet_open_date, is_abnormal)
        accociated_bank_account, bank_account_number = self.generate_bank_account_info()
        init_balance = self.generate_init_balance(wallet_level, is_abnormal)  # 传入is_abnormal参数
        
        # 生成设备和IP
        open_device = self._generate_device(wallet_id)
        open_ip = self._generate_ip(wallet_id)
        
        # 根据钱包等级设置字段
        account = {
            'wallet_id': wallet_id,
            'wallet_type': wallet_type,
            'wallet_level': wallet_level,
            'wallet_open_cert': cert_value,
            'wallet_open_cert_expire': None,
            'wallet_open_date': wallet_open_date,
            'wallet_open_timestamp': wallet_open_timestamp,
            'wallet_open_tel': tel_value,
            'realname_auth_time': None,
            'accociated_bank_account': accociated_bank_account,
            'bank_account_number': bank_account_number,
            'init_balance': init_balance,
            'region_code': region_code,
            'is_abnormal': is_abnormal,
            'open_device': open_device,
            'open_ip': open_ip
        }
        
        # 根据钱包类型和等级应用规则
        if wallet_type == 1:
            """
            对公钱包（wallet_type=1）：
            - 只会出现 1、2 类（在 generate_wallet_type_and_level 中已经保证）
            - 必须有银行卡、身份证、到期日和实名认证时间
            - 不允许出现“无银行卡/无证件”的情况
            """
            account['wallet_open_cert'] = cert_value
            account['realname_auth_time'] = self.generate_realname_auth_time(wallet_open_timestamp, is_abnormal)
            account['wallet_open_cert_expire'] = self.generate_cert_expire_date(wallet_open_date, is_abnormal)
            # accociated_bank_account / bank_account_number 已经在前面生成，直接保留
        else:
            """
            个人和商户钱包（wallet_type=2 或 3）：
            - 按照原有等级规则：
              * level 1 / 2：需要银行卡与身份证、实名认证时间、证件到期日
              * level 3：不需要银行卡，但需要身份证与实名认证、证件到期日
              * level 4：不需要银行卡与身份证相关字段
            """
        if wallet_level == 3:
            # 三类钱包：不需要绑定银行卡
            account['accociated_bank_account'] = None
            account['bank_account_number'] = None
            # 需要身份证和认证时间
            account['wallet_open_cert'] = cert_value
            account['realname_auth_time'] = self.generate_realname_auth_time(wallet_open_timestamp, is_abnormal)
            account['wallet_open_cert_expire'] = self.generate_cert_expire_date(wallet_open_date, is_abnormal)
        elif wallet_level == 4:
            # 四类钱包：不需要银行卡、身份证、到期日与认证时间
            account['accociated_bank_account'] = None
            account['bank_account_number'] = None
            account['wallet_open_cert'] = None
            account['wallet_open_cert_expire'] = None
            account['realname_auth_time'] = None
        else:
            # 一类和二类钱包：需要所有字段
            account['wallet_open_cert'] = cert_value
            account['realname_auth_time'] = self.generate_realname_auth_time(wallet_open_timestamp, is_abnormal)
            account['wallet_open_cert_expire'] = self.generate_cert_expire_date(wallet_open_date, is_abnormal)
        
        return account
    
    def save_to_csv(self, accounts_data: List[Dict[str, Any]]):
        """保存到CSV文件"""
        output_file = self.config["output_csv"]
        
        # 按照open_wallet_account_v2.py的格式保存
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入标题行
            writer.writerow([
                'wallet_id', 'wallet_type', 'wallet_level', 'wallet_open_cert', 
                'wallet_open_cert_expire', 'wallet_open_date', 'wallet_open_timestamp', 
                'wallet_open_tel', 'realname_auth_time', 'accociated_bank_account', 
                'bank_account_number', 'init_balance', 'region_code', 'is_abnormal',
                'open_device', 'open_ip'
            ])
            
            # 写入数据行
            for acc in accounts_data:
                writer.writerow([
                    acc['wallet_id'],
                    acc['wallet_type'],
                    acc['wallet_level'],
                    str(acc['wallet_open_cert']) if acc['wallet_open_cert'] is not None else '',
                    acc['wallet_open_cert_expire'],
                    acc['wallet_open_date'],
                    acc['wallet_open_timestamp'],
                    acc['wallet_open_tel'],
                    acc['realname_auth_time'],
                    str(acc['accociated_bank_account']) if acc['accociated_bank_account'] is not None else '',
                    str(acc['bank_account_number']) if acc['bank_account_number'] is not None else '',
                    acc['init_balance'],
                    acc['region_code'],
                    acc['is_abnormal'],
                    acc['open_device'] if acc['open_device'] is not None else '',
                    acc['open_ip'] if acc['open_ip'] is not None else ''
                ])
        
        print(f"账户数据已保存到: {output_file}")
        
        # 输出统计信息
        print("\n=== 生成统计 ===")
        print(f"总账户数: {len(accounts_data)}")
        
        normal_count = sum(1 for acc in accounts_data if not acc['is_abnormal'])
        abnormal_count = sum(1 for acc in accounts_data if acc['is_abnormal'])
        print(f"正常账户: {normal_count}")
        print(f"异常账户: {abnormal_count}")
        
        # 按钱包类型统计
        print("\n=== 钱包类型分布 ===")
        type_counts = {}
        for acc in accounts_data:
            wallet_type = acc['wallet_type']
            type_counts[wallet_type] = type_counts.get(wallet_type, 0) + 1
        
        total_accounts = len(accounts_data)
        for wallet_type in sorted(type_counts.keys()):
            count = type_counts[wallet_type]
            percentage = (count / total_accounts * 100) if total_accounts > 0 else 0
            if wallet_type == 1:
                type_name = "对公钱包"
            elif wallet_type == 2:
                type_name = "个人钱包"
            elif wallet_type == 3:
                type_name = "商户钱包"
            else:
                type_name = f"类型{wallet_type}"
            print(f"类型{wallet_type}({type_name}): {count}个 ({percentage:.2f}%)")
        
        # 输出配置中的期望比例
        print("\n配置中的期望比例:")
        wallet_type_config = self.config.get("wallet_type", {})
        options = wallet_type_config.get("options", [])
        probs = wallet_type_config.get("probs", [])
        for wallet_type, prob in zip(options, probs):
            if wallet_type == 1:
                type_name = "对公钱包"
            elif wallet_type == 2:
                type_name = "个人钱包"
            elif wallet_type == 3:
                type_name = "商户钱包"
            else:
                type_name = f"类型{wallet_type}"
            print(f"类型{wallet_type}({type_name}): {prob*100:.2f}%")
        
        # 按钱包等级统计
        print("\n=== 钱包等级分布 ===")
        level_counts = {}
        for acc in accounts_data:
            level = acc['wallet_level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        for level in sorted(level_counts.keys()):
            count = level_counts[level]
            print(f"等级{level}: {count}个")
        
        # 钱包类型和等级的交叉分布统计
        print("\n=== 钱包类型×等级交叉分布 ===")
        type_level_counts = {}
        for acc in accounts_data:
            wallet_type = acc['wallet_type']
            wallet_level = acc['wallet_level']
            key = (wallet_type, wallet_level)
            type_level_counts[key] = type_level_counts.get(key, 0) + 1
        
        # 按类型和等级排序输出
        type_names = {1: "对公钱包", 2: "个人钱包", 3: "商户钱包"}
        for wallet_type in sorted([1, 2, 3]):
            type_name = type_names.get(wallet_type, f"类型{wallet_type}")
            print(f"\n{type_name} (类型{wallet_type}):")
            for level in sorted([1, 2, 3, 4]):
                count = type_level_counts.get((wallet_type, level), 0)
                percentage = (count / total_accounts * 100) if total_accounts > 0 else 0
                print(f"  - 等级{level}: {count}个 ({percentage:.2f}%)")
        
        # 地区分布统计
        print("\n=== 地区分布 ===")
        region_counts = {}
        for acc in accounts_data:
            region = acc['region_code']
            region_counts[region] = region_counts.get(region, 0) + 1
        
        high_risk_codes = self.config["region_code"]["high_risk_codes"]
        normal_regions = sum(count for region, count in region_counts.items() if region not in high_risk_codes)
        high_risk_regions = sum(count for region, count in region_counts.items() if region in high_risk_codes)
        
        print(f"正常地区账户: {normal_regions}")
        print(f"高风险地区账户: {high_risk_regions}")
        
        # 身份证复用统计
        print("\n=== 身份证复用统计 ===")
        cert_counts = {}
        for acc in accounts_data:
            cert = acc['wallet_open_cert']
            if cert is not None:
                cert_counts[cert] = cert_counts.get(cert, 0) + 1
        
        unique_certs = sum(1 for count in cert_counts.values() if count == 1)
        shared_certs = sum(1 for count in cert_counts.values() if count > 1)
        print(f"唯一身份证: {unique_certs}")
        print(f"复用身份证: {shared_certs}")
        
        # 手机号复用统计
        print("\n=== 手机号复用统计 ===")
        tel_counts = {}
        for acc in accounts_data:
            tel = acc['wallet_open_tel']
            if tel is not None:
                tel_counts[tel] = tel_counts.get(tel, 0) + 1
        
        unique_tels = sum(1 for count in tel_counts.values() if count == 1)
        shared_tels = sum(1 for count in tel_counts.values() if count > 1)
        print(f"唯一手机号: {unique_tels}")
        print(f"复用手机号: {shared_tels}")
        
        # 设备和IP统计
        print("\n=== 设备和IP统计 ===")
        counter_open_count = sum(1 for acc in accounts_data if acc['open_device'] is None or acc['open_ip'] is None)
        device_count = sum(1 for acc in accounts_data if acc['open_device'] is not None)
        ip_count = sum(1 for acc in accounts_data if acc['open_ip'] is not None)
        
        print(f"柜台开立数（设备/IP为None）: {counter_open_count}")
        print(f"柜台开立比例: {counter_open_count/len(accounts_data)*100:.2f}%")
        print(f"有设备信息的账户: {device_count}")
        print(f"有IP信息的账户: {ip_count}")
        
        # 手机网络IP统计
        mobile_ip_count = sum(1 for acc in accounts_data 
                            if acc['open_ip'] is not None and str(acc['open_ip']).startswith('117.'))
        if ip_count > 0:
            print(f"手机网络IP数量: {mobile_ip_count}")
            print(f"手机网络比例: {mobile_ip_count/ip_count*100:.2f}%")

def main():
    """主函数"""
    print("=== 基于分布的钱包开立模拟系统 ===")
    
    # 创建生成器
    generator = WalletDistributionGenerator()
    
    # 生成账户数据
    accounts_data = generator.generate_accounts()
    
    # 保存到CSV
    generator.save_to_csv(accounts_data)
    
    print("\n生成完成！")

if __name__ == "__main__":
    main()
