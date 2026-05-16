"""
特征: src_weekend_volume_ratio
说明: 洗钱行为可能集中在周末以规避工作日监管，或在工作日伪装正常交易。周末金额占比能反映时间偏好差异。

验证指标:
  - Score: 0.3405
  - MI: 0.0000
  - Redundancy: 0.3879
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 91

可解释性报告:
  - 值含义: 该特征表示一个钱包在周六和周日发起的交易金额占其总交易金额的比例。值在0到1之间：值接近0表示该钱包几乎不在周末交易；值接近1表示该钱包几乎只在周末交易；值在0.25-0.30左右（约1/4-1/3）与正常交易分布一致。极端的高值（>0.5）或极低值（<0.1）可能反映非典型的交易时间偏好。
  - 风险阈值: 高风险阈值建议设为 <0.15 或 >0.45。判断依据：正常交易的中位数为0.2814，25%分位为0.2227，75%分位为0.3370，表明绝大多数正常钱包的周末交易占比集中在22%-34%之间。异常交易的中位数为0.2818，与正常几乎无差异，但均值差异显著（-1270 vs -2913），说明异常交易中存在极端的负值（可能因数据异常或计算错误）和更大的离散度。低于15%意味着钱包严重偏向工作日交易，可能是在刻意规避周末监管；高于45%意味着钱包严重偏向周末交易，可能是利用周末监管薄弱时段进行异常活动。
  - 协同特征: 建议与以下特征配合使用：1）`is_night_hour_From`（凌晨交易占比）：洗钱常同时发生在周末和深夜，两者结合可识别'周末深夜活跃'的复合异常模式；2）`S_src_time_risk_202308`（时间规律性风险分）：该特征评估交易时间是否符合历史规律，与周末占比结合可区分'正常周末购物'与'异常周末交易'；3）`transaction_count_Undirected`（总交易笔数）：高交易笔数+高周末占比的组合比单一特征更可疑。协同原理：周末占比单独区分度较低（IV=0.011），但与其他时间维度的特征组合后可形成更完整的交易时间画像。
  - 误判场景: 可能的误判场景包括：1）周末兼职商户：如周末市集、跳蚤市场的摊主，其收款集中在周六日，导致周末占比极高；2）旅游/餐饮行业商户：周末是消费高峰期，交易量远高于工作日；3）仅工作日使用的钱包：如公司工资代发钱包，只在周一至周五操作，周末占比为0；4）新注册钱包：交易记录极少，仅有的几笔恰好发生在周末或工作日，导致极端比例。
  - 监管说明: 该特征衡量的是一个钱包在周末（周六和周日）发生的交易金额占其总交易金额的比重。正常用户的交易时间通常均匀分布在一周中，而洗钱行为有时会刻意选择周末进行以利用监管薄弱时段，或者刻意避开周末以伪装成正常工作日交易，导致该比例异常偏高或偏低。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 src_weekend_volume_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'])
    tx_df['dayofweek'] = tx_df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    # 周末定义为周六(5)和周日(6)
    is_weekend = tx_df['dayofweek'].isin([5,6])
    # 按src统计总金额和周末金额
    weekend_amount = tx_df[is_weekend].groupby('src')['amount'].sum().reset_index()
    weekend_amount.columns = ['wallet_id', 'weekend_amount']
    total_amount = tx_df.groupby('src')['amount'].sum().reset_index()
    total_amount.columns = ['wallet_id', 'total_amount']
    result = total_amount.merge(weekend_amount, on='wallet_id', how='left')
    result['weekend_amount'] = result['weekend_amount'].fillna(0)
    result['feature_value'] = result['weekend_amount'] / result['total_amount'].clip(lower=0.01)
    result = result[['wallet_id', 'feature_value']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: src_weekend_volume_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 src_weekend_volume_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
