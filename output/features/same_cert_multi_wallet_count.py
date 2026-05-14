"""
特征: same_cert_multi_wallet_count
说明: 数字人民币洗钱模式中，薅羊毛特征为批量注册4类钱包，同一身份证/手机号关联多个钱包。通过统计每个钱包的身份证关联的其他钱包数量，可以识别出批量注册的异常行为。

验证指标:
  - Score: 0.3285
  - MI: 0.0289
  - Redundancy: 0.6838
  - Cramer's V: 0.0588
  - Delta PR-AUC: -0.017062
  - SHAP Rank: 46

可解释性报告:
  - 值含义: 值=1表示该钱包的身份证仅关联了1个钱包（正常情况）；值=2表示关联了2个钱包；值≥3表示同一身份证关联了3个及以上钱包，存在批量注册嫌疑。异常交易中该特征的均值（1.3827）和标准差（0.6994）均高于正常交易（均值1.0131，标准差0.1422），且25%分位已出现2个关联钱包，表明异常交易中批量注册现象更集中。
  - 风险阈值: 高风险阈值为≥3，即同一身份证关联3个及以上钱包。依据：异常交易中75%分位值为2.0000，而正常交易中75%分位值仍为1.0000，说明2个关联钱包在异常交易中已较常见，但3个及以上关联钱包在正常交易中几乎不存在（正常均值1.0131，标准差0.1422），因此≥3可作为明确的风险信号。
  - 协同特征: 建议与以下特征配合使用：1) wallet_level（钱包等级）：薅羊毛模式中批量注册的通常是4类钱包，结合wallet_level=4可进一步确认；2) transaction_motif=small_amount_testing：批量注册后常伴随小额测试交易；3) counts_1H_From（1小时内转出次数）：批量注册后可能短时间内发起多笔交易；4) avg_inactive_period（平均不活跃期）：休眠账户突然激活是洗钱模式5的典型特征，与批量注册结合更可疑。这些特征从不同维度（账户属性、交易行为、时间模式）共同锁定薅羊毛或测试性洗钱行为。
  - 误判场景: 可能的误判场景：1) 家庭成员共用身份证注册多个钱包用于日常小额转账，例如父母为子女注册多个钱包管理零花钱；2) 企业为员工批量注册钱包用于发放福利或补贴，且均为正常交易模式；3) 同一身份证下注册了多个钱包但长期无交易或交易模式正常（如single_transaction），此时需结合交易行为特征排除误判。
  - 监管说明: 该特征通过识别同一身份证绑定的钱包数量，帮助发现利用批量注册4类钱包进行薅羊毛或洗钱测试的异常行为——正常用户通常只有一个钱包，而同一身份证下关联超过2个钱包时需引起关注。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 same_cert_multi_wallet_count，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    # 从开户表获取每个钱包的身份证信息
    # wallet_open_cert 字段可能为空，需要处理
    cert_df = open_df[['wallet_id', 'wallet_open_cert']].copy()
    # 过滤掉空值
    cert_df = cert_df[cert_df['wallet_open_cert'].notna()]
    # 统计每个身份证关联的钱包数量
    cert_counts = cert_df.groupby('wallet_open_cert')['wallet_id'].nunique().reset_index()
    cert_counts.columns = ['wallet_open_cert', 'cert_wallet_count']
    # 合并回每个钱包
    result = cert_df.merge(cert_counts, on='wallet_open_cert', how='left')
    result = result[['wallet_id', 'cert_wallet_count']]
    # 对于没有身份证信息的钱包，填充为1（只有自己）
    all_wallets = open_df[['wallet_id']].copy()
    result = all_wallets.merge(result, on='wallet_id', how='left')
    result['cert_wallet_count'] = result['cert_wallet_count'].fillna(1).astype(int)
    result = result[['wallet_id', 'cert_wallet_count']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: same_cert_multi_wallet_count")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 same_cert_multi_wallet_count: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
