"""
特征: cert_tel_mismatch_ratio
说明: 洗钱模式1（薅羊毛）中，批量注册4类钱包时，犯罪分子可能使用多个手机号对应同一个身份证，或者多个身份证对应同一个手机号，这种'证号不匹配'是批量注册的重要特征

验证指标:
  - Score: 0.1255
  - MI: 0.0006
  - Redundancy: 0.7321
  - Cramer's V: 0.0877
  - Delta PR-AUC: -0.011737
  - SHAP Rank: 22

可解释性报告:
  - 值含义: {'0.0': '正常：该钱包的身份证号和手机号均为唯一映射，不存在证号不匹配现象', '(0, 0.5)': '轻度嫌疑：存在一定程度的证号不匹配，如同一身份证关联了2个手机号，或同一手机号关联了2个身份证', '[0.5, 1.0]': '中度嫌疑：存在明显的证号不匹配，如同一身份证关联3个以上手机号，或同一手机号关联3个以上身份证', '>1.0': '高度嫌疑：证号严重不匹配，同时存在身份证关联多个手机号且手机号关联多个身份证，典型批量注册特征'}
  - 风险阈值: 0.5
  - 协同特征: [{'feature': 'wallet_level', 'reason': '洗钱模式1（薅羊毛）主要使用4类钱包批量注册。当cert_tel_mismatch_ratio高且wallet_level=4时，批量注册嫌疑极大提升。'}, {'feature': 'same_transactionaccount', 'reason': "证号不匹配的批量注册钱包往往使用同一银行账户进行充值提现。结合此特征可区分'真批量注册'与'偶然的证号共享（如家庭成员共用手机号）'。"}, {'feature': 'large_transaction_ratio', 'reason': '批量注册的薅羊毛钱包通常交易金额极小（0.01-0.1元测试交易）。高cert_tel_mismatch_ratio + 低large_transaction_ratio是薅羊毛的典型画像。'}]
  - 误判场景: ['家庭共享场景：一家三口使用同一手机号办理家庭套餐，但各自身份证分别注册钱包，此时tel_cert_unique_count=3，导致mismatch_score偏高', '企业员工场景：公司为员工批量开户时使用同一联系电话，但各员工身份证不同，也会导致同一手机号关联多个身份证', '运营商回收号码：某人新办理的手机号前主人曾用该号码注册过多个钱包，导致该手机号历史关联多个身份证', '代办开户：老年人或残障人士委托他人代办开户，代办人联系方式被用于多个钱包注册']
  - 监管说明: 该特征检测的是'同一身份证绑定多个手机号'或'同一手机号绑定多个身份证'的异常关联模式，这通常是不法分子为规避实名监管而批量注册钱包时留下的痕迹，正常用户几乎不会出现这种情况。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 cert_tel_mismatch_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包的身份证和手机号
    wallet_cert = open_df[['wallet_id', 'wallet_open_cert', 'wallet_open_tel']].dropna(subset=['wallet_open_cert'])

    # 统计同一身份证下使用的不同手机号数量
    cert_tel_count = wallet_cert.groupby('wallet_open_cert')['wallet_open_tel'].nunique().reset_index()
    cert_tel_count.columns = ['wallet_open_cert', 'cert_tel_unique_count']

    # 统计同一手机号下使用的不同身份证数量
    tel_cert_count = wallet_cert.groupby('wallet_open_tel')['wallet_open_cert'].nunique().reset_index()
    tel_cert_count.columns = ['wallet_open_tel', 'tel_cert_unique_count']

    # 合并到每个钱包
    wallet_feat = wallet_cert.merge(cert_tel_count, on='wallet_open_cert', how='left')
    wallet_feat = wallet_feat.merge(tel_cert_count, on='wallet_open_tel', how='left')

    # 计算不匹配比例：如果同一身份证下手机号数量>1，则说明存在证号不匹配
    # 使用 min(cert_tel_unique_count, tel_cert_unique_count) / 2 作为归一化指标
    # 值越大说明不匹配越严重
    wallet_feat['mismatch_score'] = np.where(
        (wallet_feat['cert_tel_unique_count'] > 1) | (wallet_feat['tel_cert_unique_count'] > 1),
        (wallet_feat['cert_tel_unique_count'] - 1) / wallet_feat['cert_tel_unique_count'].clip(lower=1) + 
        (wallet_feat['tel_cert_unique_count'] - 1) / wallet_feat['tel_cert_unique_count'].clip(lower=1),
        0
    )

    result = wallet_feat[['wallet_id', 'mismatch_score']].copy()
    result.columns = ['wallet_id', 'cert_tel_mismatch_ratio']
    result = result.fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: cert_tel_mismatch_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 cert_tel_mismatch_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
