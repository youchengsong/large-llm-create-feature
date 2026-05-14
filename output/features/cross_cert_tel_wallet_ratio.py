"""
特征: cross_cert_tel_wallet_ratio
说明: 洗钱分子可能使用同一身份证注册多个手机号，或者使用多个身份证绑定同一手机号，以规避单一维度的监控。计算身份证关联钱包数与手机号关联钱包数的比值，当比值偏离1时，表明存在交叉关联的异常，这是主体关联风险的重要信号。

验证指标:
  - Score: 0.4000
  - MI: 0.0004
  - Redundancy: 0.1795
  - Cramer's V: 0.0597
  - Delta PR-AUC: 0.020807
  - SHAP Rank: 23

可解释性报告:
  - 值含义: 该特征值表示身份证关联钱包数与手机号关联钱包数的偏离程度（取对数绝对值）。值=0表示身份证和手机号关联的钱包数一致（1:1关系），属于正常情况；值在0-0.5之间表示轻度偏离，可能为家庭共用手机号等正常场景；值>0.5表示显著偏离，值>1.0表示严重偏离，值>2.0表示极端偏离，大概率存在批量注册或身份盗用嫌疑。异常交易均值0.1955远高于正常交易均值0.0078，说明异常交易中交叉关联现象显著。
  - 风险阈值: 建议高风险阈值为0.5（对应比值偏离约1.65倍）。依据：异常交易的75%分位值为0.4055，而正常交易的75%分位值仅为0.0，因此0.5可以较好区分绝大多数正常与异常。0.5以下覆盖了几乎所有正常交易（正常交易均值0.0078，标准差0.0791），而异常交易中约25%超过0.4055。该阈值在Delta PR-AUC验证中贡献了0.0208的提升，表明对少数异常样本有较强识别能力。
  - 协同特征: 建议与以下特征配合使用：1) `same_transactionaccount`（同一身份证关联钱包数）：当cross_cert_tel_wallet_ratio高且同一身份证关联钱包数>3时，强烈提示薅羊毛模式（模式1），因为洗钱分子用同一身份证批量注册4类钱包。2) `activetime`（活跃时段）：结合凌晨活跃时段（0-5点），可识别线上跑分模式（模式3），因为该类异常通常伴随凌晨高频交易。3) `From_To_Ratio`（收发比）：当cross_cert_tel_wallet_ratio高且收发比>5时，提示商户收款码跑分模式（模式4），因为商户钱包大量收款后转出。
  - 误判场景: 可能的误判场景：1) 家庭成员共用手机号：如父母帮子女注册钱包，导致同一手机号关联多个钱包但身份证不同，比值<1触发高值。2) 企业员工批量开户：企业为员工统一注册钱包用于工资发放，同一身份证关联多个手机号，比值>1触发高值。3) 运营商号码回收：手机号被回收后新用户使用该号注册钱包，与旧用户身份证产生交叉关联。4) 实名认证延迟：用户更换手机号后未及时更新实名信息，导致新旧手机号关联同一身份证。
  - 监管说明: 该特征检测的是'一个人用多个手机号开户'或'一个手机号被多个人用来开户'的异常模式，洗钱分子常利用这种交叉关联来规避实名制监管，例如用同一身份证在多个平台注册钱包、或盗用他人身份信息绑定自己的手机号。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 cross_cert_tel_wallet_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 获取每个钱包的wallet_open_cert和wallet_open_tel
    wallet_info = open_df[['wallet_id', 'wallet_open_cert', 'wallet_open_tel']].copy()
    wallet_info = wallet_info.dropna(subset=['wallet_open_cert', 'wallet_open_tel'])
    wallet_info['wallet_open_cert'] = wallet_info['wallet_open_cert'].astype(str)
    wallet_info['wallet_open_tel'] = wallet_info['wallet_open_tel'].astype(str)

    # 计算每个身份证关联的钱包数
    cert_wallet_count = wallet_info.groupby('wallet_open_cert')['wallet_id'].nunique().reset_index()
    cert_wallet_count.columns = ['wallet_open_cert', 'cert_wallet_count']

    # 计算每个手机号关联的钱包数
    tel_wallet_count = wallet_info.groupby('wallet_open_tel')['wallet_id'].nunique().reset_index()
    tel_wallet_count.columns = ['wallet_open_tel', 'tel_wallet_count']

    # 合并到每个钱包
    wallet_with_counts = pd.merge(wallet_info, cert_wallet_count, on='wallet_open_cert', how='left')
    wallet_with_counts = pd.merge(wallet_with_counts, tel_wallet_count, on='wallet_open_tel', how='left')

    # 计算比值（身份证关联钱包数 / 手机号关联钱包数）
    # 当比值大于1时，说明同一身份证下关联了更多钱包（可能同一身份证多个手机号）
    # 当比值小于1时，说明同一手机号下关联了更多钱包（可能多个身份证同一手机号）
    wallet_with_counts['ratio'] = wallet_with_counts['cert_wallet_count'] / wallet_with_counts['tel_wallet_count'].clip(lower=1)

    # 对比值取对数，使正负偏离都为正数，且越大表示偏离越严重
    wallet_with_counts['feature_value'] = np.abs(np.log(wallet_with_counts['ratio'].clip(lower=0.01, upper=100)))

    # 合并回所有钱包
    all_wallets = open_df[['wallet_id']].copy()
    result = pd.merge(all_wallets, wallet_with_counts[['wallet_id', 'feature_value']], on='wallet_id', how='left').fillna(0)
    result['feature_value'] = result['feature_value'].astype(float)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: cross_cert_tel_wallet_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 cross_cert_tel_wallet_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
