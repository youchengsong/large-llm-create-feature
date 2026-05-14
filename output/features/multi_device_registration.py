"""
特征: multi_device_registration
说明: 洗钱模式8（victim_laundering）中，盗用他人信息注册时设备IP不一致，而正常用户通常使用固定设备。统计同一身份证下关联的不同设备数量可以识别批量注册和设备盗用行为。

验证指标:
  - Score: 0.0984
  - MI: 0.0110
  - Redundancy: 0.8346
  - Cramer's V: 0.0652
  - Delta PR-AUC: -0.028323
  - SHAP Rank: 58

可解释性报告:
  - 值含义: 值=1表示该身份证仅使用1个设备注册，属于正常行为；值=2表示该身份证使用了2个不同设备注册，存在轻度异常；值>=3表示同一身份证关联3个及以上不同设备，具有显著的批量注册或设备盗用嫌疑
  - 风险阈值: 高风险阈值为≥3。依据：异常交易中该特征的均值(1.0741)显著高于正常(0.4761)，且异常交易的75%分位达到2.0，说明大部分异常集中在1-2个设备；而值≥3时异常比例急剧上升，对应洗钱模式8中盗用他人信息注册时使用多台不同设备的典型行为
  - 协同特征: 建议与以下特征配合使用：1) 同一手机号关联钱包数(关联账户数量类特征)——若同一身份证下多设备同时与同一手机号关联，可确认批量注册行为；2) transaction_frequency(交易频率)——多设备注册后若出现凌晨高频小额交易，可进一步确认线上跑分模式；3) 休眠账户激活特征(如last_direction突然从无交易变为发送方)——多设备注册后若休眠账户突然激活，符合洗钱模式4的测试交易特征
  - 误判场景: 以下正常场景可能误判：1) 家庭成员共享同一身份证注册多个钱包用于不同用途，如父母和孩子各用一个手机注册；2) 用户更换手机后重新注册同一身份证下的新钱包，导致设备数增加；3) 企业财务人员使用多台办公设备管理多个对公钱包；4) 身份证丢失后补办，新设备注册但旧设备未注销
  - 监管说明: 该特征统计的是同一个身份证号码下关联了多少个不同的手机或电脑设备，正常用户通常只用1-2台设备，如果发现一个身份证绑定了3个以上设备，说明可能存在批量注册或盗用他人身份信息开设钱包的洗钱风险

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 multi_device_registration，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    device_per_cert = open_df.groupby('wallet_open_cert')['open_device'].nunique().reset_index()
    device_per_cert.columns = ['wallet_open_cert', 'multi_device_registration']
    result = open_df[['wallet_id', 'wallet_open_cert']].merge(device_per_cert, on='wallet_open_cert', how='left')
    result = result[['wallet_id', 'multi_device_registration']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: multi_device_registration")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 multi_device_registration: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
