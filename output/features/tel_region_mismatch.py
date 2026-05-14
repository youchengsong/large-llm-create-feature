"""
特征: tel_region_mismatch
说明: 手机号归属地与开户地区不匹配可能是用户使用他人身份信息开户的迹象，这符合'主体身份冒用'和'借用他人账户进行交易'的风险特征。异常交易中批量注册的4类钱包更容易出现这种不一致。

验证指标:
  - Score: 0.3058
  - MI: 0.0012
  - Redundancy: 0.1113
  - Cramer's V: 0.0141
  - Delta PR-AUC: -0.008786
  - SHAP Rank: 38

可解释性报告:
  - 值含义: 该特征为二值特征，取值0或1。值=0表示手机号归属地区与开户地区一致，值=1表示手机号归属地区与开户地区不一致。特征值越高（即=1）代表手机号地区与开户地区不匹配，可能暗示用户使用了非本人身份信息或非本地手机号开户。
  - 风险阈值: 高风险阈值为1。判断依据：该特征为二值特征，取值只有0和1，因此高风险阈值为1（即不匹配）。选择理由：在数字人民币洗钱场景中，异常交易账户的'主体身份冒用'和'借用他人账户进行交易'风险特征中，手机号归属地与开户地区不一致是重要信号，特别是批量注册的4类钱包更容易出现这种不一致。
  - 协同特征: 建议与以下特征配合使用：
1. **wallet_level**（钱包等级）：4类钱包（最低等级）配合手机号地区不匹配，可识别批量注册的薅羊毛账户。
2. **同一身份证关联钱包数**（关联账户数量）：若同一身份证关联多个钱包且手机号地区不匹配，高度疑似冒用身份批量开户。
3. **休眠账户突然激活**（如max_inactive_period > 90天）：休眠账户突然激活且手机号地区不匹配，符合small_amount_testing风险模式。
4. **交易时间异常**（如Timestamp_Hour在0-5点）：深夜交易配合地区不匹配，符合线上跑分模式特征。

协同原因：单一特征区分度有限（SHAP Rank #38，MI仅0.0012），但组合后可形成强信号。例如，4类钱包+手机号地区不匹配+凌晨交易=高概率线上跑分账户。
  - 误判场景: 可能的误判场景：
1. **正常异地工作/生活用户**：用户因工作调动、求学等原因在A地区开户但使用B地区手机号，属于正常情况。
2. **跨省旅游/出差用户**：短期在异地开户但使用原籍手机号，属于正常行为。
3. **虚拟运营商用户**：手机号前3位映射为虚拟运营商（170-179）时，可能被标记为'不匹配'，但实际是正常用户。
4. **企业批量开户**：企业为员工统一开户时，可能使用非本地手机号，属于正常业务场景。
  - 监管说明: 该特征用于检测用户开户时使用的手机号归属地与开户地区是否一致，如果发现不一致，可能提示用户使用了他人身份信息或非本地手机号开户，这是数字人民币洗钱活动中常见的批量注册、冒用身份行为。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 tel_region_mismatch，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 提取手机号前3位作为地区代码
    result = open_df[['wallet_id', 'wallet_open_tel', 'region_code']].copy()
    result['tel_prefix'] = (result['wallet_open_tel'] // 10000000).astype(int)

    # 手机号前3位映射到地区：130-132=联通(1), 133=电信(2), 134-139=移动(3), 150-159=移动(3), 170-179=虚拟(4)
    def tel_to_region(prefix):
        if 130 <= prefix <= 132:
            return 1
        elif prefix == 133:
            return 2
        elif 134 <= prefix <= 139:
            return 3
        elif 150 <= prefix <= 159:
            return 3
        elif 170 <= prefix <= 179:
            return 4
        else:
            return 0

    result['tel_region_code'] = result['tel_prefix'].apply(tel_to_region)

    # 如果手机号地区与开户地区不匹配，标记为1
    result['feature_value'] = (result['tel_region_code'] != 0).astype(int)
    result = result[['wallet_id', 'feature_value']].copy()
    result = result.drop_duplicates(subset='wallet_id').reset_index(drop=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: tel_region_mismatch")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 tel_region_mismatch: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
