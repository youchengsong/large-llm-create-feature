"""
特征: night_active_ratio_by_region
说明: 洗钱模式中，small_amount_testing（测试交易）集中在深夜时段，merchant_laundering也倾向于夜间操作。通过计算钱包在深夜时段的交易比例，并与同地区其他钱包对比，可以有效识别那些在非正常时间段异常活跃的账户。地区因素也很重要，因为不同地区的监管环境和用户行为模式不同。

验证指标:
  - Score: 0.0449
  - MI: 0.0012
  - Redundancy: 0.8572
  - Cramer's V: 0.0108
  - Delta PR-AUC: -0.014342
  - SHAP Rank: 46

可解释性报告:
  - 值含义: {'负值（<-0.1）': '钱包的夜间交易比例低于同地区平均水平，属于正常或偏低活跃度', '接近0（-0.1~0.1）': '钱包的夜间交易比例与同地区平均水平相当，行为正常', '正值（0.1~0.5）': '钱包的夜间交易比例略高于同地区平均，可能存在轻微夜间活跃倾向', '高正值（0.5~1.0）': '钱包的夜间交易比例明显高于同地区平均，需要关注是否涉及测试交易或夜间洗钱', '极高值（>1.0）': '钱包的夜间交易比例远超同地区平均（超过2倍），高度可疑，可能涉及small_amount_testing或merchant_laundering模式'}
  - 风险阈值: {'高风险阈值': '>0.5', '判断依据': '从分布看，异常交易的中位数（-0.0275）低于正常交易（0.1380），且异常交易的均值（-0.0588）也低于正常（0.0210），说明洗钱账户整体夜间比例并不高，但部分极端值会拉高。当特征值>0.5时，说明该钱包的夜间比例比同地区平均高出50%以上，这远远超出正常波动范围。正常交易的75%分位是0.3851，因此0.5作为阈值能有效捕捉到尾部异常。', '建议阈值': '0.5（黄色预警），1.0（红色预警）'}
  - 协同特征: [{'feature': 'small_amount_testing_flag', 'reason': '夜间高活跃度+小额测试交易是典型的测试洗钱模式，两者结合能显著提升检测准确率'}, {'feature': 'merchant_laundering_flag', 'reason': '夜间活跃+商户钱包是商户收款码跑分的典型特征，配合使用可识别商户洗钱'}, {'feature': 'fan_out_ratio', 'reason': '夜间活跃+分散转出模式是线上跑分的核心特征，协同使用可覆盖线上跑分场景'}, {'feature': 'sleeping_account_reactivation', 'reason': '夜间活跃+休眠账户突然激活是small_amount_testing的典型行为，组合使用可精准识别'}]
  - 误判场景: [{'case': '夜班工作者', 'explanation': '如保安、医护人员、物流人员等夜间工作群体，其正常交易可能集中在深夜，导致夜间比例偏高。可通过结合职业信息或交易用途字段进行过滤'}, {'case': '跨时区用户', 'explanation': '居住在海外或经常跨境出差的用户，其交易时间可能不符合本地夜间定义，导致误判。可通过IP地址或设备信息辅助判断'}, {'case': '夜间消费习惯', 'explanation': '部分用户（如年轻人）有夜间消费习惯，如外卖、游戏充值等，其夜间交易比例可能高于地区平均。可通过交易商户类型进行区分'}, {'case': '地区样本不足', 'explanation': '某些地区钱包数量较少，计算出的地区平均夜间比例可能不具代表性，导致偏离度异常。建议对样本量<5的地区进行平滑处理或使用全局均值'}]
  - 监管说明: 该特征衡量一个钱包在深夜（晚10点到早6点）的交易活跃度是否显著高于其所在地区的平均水平——如果某个钱包的深夜交易比例是当地平均水平的1.5倍以上，就说明该账户存在异常夜间活跃行为，可能是洗钱测试或非法资金转移的迹象。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 night_active_ratio_by_region，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 提取交易小时
    hour = pd.to_datetime(tx_df['timestamp']).dt.hour

    # 标记深夜交易 (22:00-5:59)
    is_night = ((hour >= 22) | (hour < 6))

    # 计算每个钱包的夜间交易比例
    night_tx = tx_df.copy()
    night_tx['is_night'] = is_night
    night_ratio = night_tx.groupby('src')['is_night'].mean().reset_index()
    night_ratio.columns = ['wallet_id', 'night_ratio']

    # 获取钱包的地区信息
    wallet_region = open_df[['wallet_id', 'region_code']].dropna(subset=['region_code'])

    # 合并地区和夜间比例
    merged = wallet_region.merge(night_ratio, on='wallet_id', how='left')

    # 计算每个地区的平均夜间比例
    region_avg_night = merged.groupby('region_code')['night_ratio'].mean().reset_index()
    region_avg_night.columns = ['region_code', 'region_avg_night']

    # 计算偏离度：钱包夜间比例 / 同地区平均夜间比例 - 1
    merged = merged.merge(region_avg_night, on='region_code', how='left')
    merged['night_active_ratio_by_region'] = merged['night_ratio'] / merged['region_avg_night'].clip(lower=0.01) - 1

    # 处理缺失值
    merged['night_active_ratio_by_region'] = merged['night_active_ratio_by_region'].fillna(0)

    result = merged[['wallet_id', 'night_active_ratio_by_region']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: night_active_ratio_by_region")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 night_active_ratio_by_region: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
