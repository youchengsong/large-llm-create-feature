"""
特征: same_tel_multi_wallet_count
说明: 数字人民币洗钱模式中，薅羊毛特征为批量注册4类钱包，同一手机号关联多个钱包。手机号作为注册的重要凭证，同一手机号关联多个钱包是洗钱账户的典型特征。

验证指标:
  - Score: 0.3790
  - MI: 0.0342
  - Redundancy: 0.4999
  - Cramer's V: 0.0580
  - Delta PR-AUC: -0.023751
  - SHAP Rank: 40

可解释性报告:
  - 值含义: {'1': '正常情况，该手机号仅关联1个钱包，无批量注册嫌疑', '2': '轻度异常，该手机号关联2个钱包，可能为家庭成员共用或少量正常注册', '3及以上': '高度异常，同一手机号关联3个以上钱包，强烈表明存在批量注册薅羊毛洗钱嫌疑'}
  - 风险阈值: 高风险阈值为≥3。判断依据：异常交易的75%分位值为2.0，中位数为1.0，说明大多数异常交易集中在值=1-2；而正常交易的75%分位值仅为1.0，均值仅1.04，几乎全部正常钱包的手机号只关联1个钱包。值≥3意味着该手机号关联3个以上钱包，已显著偏离正常分布，且符合薅羊毛模式中批量注册4类钱包的典型特征
  - 协同特征: [{'feature': 'wallet_level', 'reason': '薅羊毛模式主要针对4类钱包（限额低、实名门槛低），配合wallet_level=4可精准识别批量注册4类钱包的洗钱行为'}, {'feature': 'Timestamp_Hour', 'reason': '薅羊毛行为常发生在凌晨或非营业时段进行批量操作，配合交易时间可区分正常注册与异常批量注册'}, {'feature': 'transaction_frequency', 'reason': '批量注册的薅羊毛钱包通常交易频率低（仅用于领取红包后闲置），配合低频交易可强化识别'}]
  - 误判场景: ['家庭成员共用手机号：父母为子女注册多个钱包用于日常消费，或家庭中多个成员共用同一手机号注册钱包', '企业员工统一注册：企业为员工批量开通数字人民币钱包用于发工资或福利，使用同一企业手机号', '运营商号段回收：手机号被运营商回收后重新分配，新用户使用该手机号注册钱包，导致历史关联钱包数虚高', '社区/学校统一推广：社区或学校组织批量注册数字人民币钱包用于推广活动，使用统一联系方式']
  - 监管说明: 该特征通过统计同一手机号关联的钱包数量，识别是否存在批量注册钱包的薅羊毛洗钱行为——正常用户一个手机号只开1个钱包，而洗钱团伙会用同一手机号注册3个以上钱包来套取红包和优惠

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 same_tel_multi_wallet_count，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    # 从开户表获取每个钱包的手机号信息
    # wallet_open_tel 字段
    result = open_df.groupby('wallet_open_tel')['wallet_id'].nunique().reset_index()
    result.columns = ['wallet_open_tel', 'tel_wallet_count']
    result = open_df[['wallet_id', 'wallet_open_tel']].merge(result, on='wallet_open_tel', how='left')
    result = result[['wallet_id', 'tel_wallet_count']]
    result['tel_wallet_count'] = result['tel_wallet_count'].fillna(1).astype(int)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: same_tel_multi_wallet_count")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 same_tel_multi_wallet_count: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
