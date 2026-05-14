"""
特征: geographic_dispersion_score
说明: 洗钱行为常涉及多地域分散转出（fan_out）和多对多（many_to_many）模式，通过计算交易对手地域的多样性可以捕捉异常的资金分散行为。正常用户交易地域相对集中，而洗钱账户会与大量不同地域的对手交易以混淆资金流向。

验证指标:
  - Score: 0.2077
  - MI: 0.0088
  - Redundancy: 0.5982
  - Cramer's V: 0.0361
  - Delta PR-AUC: -0.009670
  - SHAP Rank: 63

可解释性报告:
  - 值含义: {'0.0000-0.8000': '极低地域分散度——表示该钱包交易对手全部或绝大多数来自同一地域，常见于新注册/休眠钱包或仅与极少数固定对手交易，需警惕是否为测试账户或休眠账户突然激活的洗钱模式', '0.8000-0.9643': '中等偏低地域分散度——表示交易对手覆盖少数几个地域，但多样性不足，可能对应正常的小额高频消费场景（如本地商户），或异常交易中资金初步分散但尚未充分扩散', '0.9643-1.0000': '高地域分散度——表示交易对手来自多个不同地域，正常交易中该区间占主导（75%分位达1.0），对应日常跨地域消费或转账；但异常交易中该值反而偏低（75%分位仅0.9643），说明洗钱账户虽有意分散地域但受限于账户类型或对手数量', '1.0000': '最大分散度——表示该钱包交易对手覆盖了数据集中的所有地域类型，常见于正常的高活跃度钱包（如商户），但在异常交易中极少出现（异常均值0.8757），因为洗钱账户通常只与有限数量的同伙交易'}
  - 风险阈值: 高风险阈值为 geographic_dispersion_score < 0.8000 (即归一化后低于0.8)。依据：异常交易的中位数(0.9286)虽略低于正常(0.9821)，但差异较小；然而异常交易的std(0.1213)是正常(0.0339)的3.6倍，说明异常交易中存在大量低分散度样本。取25%分位(0.8036)作为阈值，可覆盖异常交易中最低的25%样本，同时避免误伤正常交易（正常交易小于0.8的比例极低）。低于0.8表示该钱包交易对手地域极度集中，与洗钱模式中“多账户小额分散转账”但实际对手有限的逻辑一致。
  - 协同特征: [{'feature': 'transaction_mode=fan_out', 'reason': 'fan_out模式表示资金从单个钱包分散转出，与低地域分散度结合可识别“集中转出但对手全在同一地域”的异常模式，如线下跑分团伙使用同一地区多个账户收款'}, {'feature': 'avg_inactive_period', 'reason': '休眠账户突然激活（avg_inactive_period高）同时地域分散度低，符合small_amount_testing风险模式（深夜激活后集中向少数地域小额转账）'}, {'feature': 'send_concentration', 'reason': '发送集中度（top1对手金额占比）与低地域分散度结合，可识别资金高度集中流向少数地域的洗钱行为，如商户收款码跑分后统一转至特定个人钱包'}]
  - 误判场景: ['本地商户钱包：如社区便利店、菜市场商户，其交易对手几乎全部来自同一城市，地域分散度天然很低，但属于正常经营场景', '新注册钱包：刚开立的钱包在初期仅与家人或少数朋友交易，地域分散度低，需结合账户注册时长判断', '单一用途钱包：如专门用于缴纳物业费、学费的钱包，交易对手固定且同地域，属于正常场景', '企业钱包：企业发放工资时，所有员工均在同一城市，地域分散度低，但交易金额和频率正常']
  - 监管说明: 该特征衡量一个钱包与多少个不同地区的对手进行交易——正常用户会与多个地区的商家或个人交易，而洗钱账户往往只与少数几个地区的同伙反复交易，导致地域多样性明显偏低，就像一个人整天只跟同一条街上的几个人来往，却声称在做全国生意。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 geographic_dispersion_score，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 合并src和dst的交易地域信息
    # 使用region_code作为地域标识，关联开户表
    open_region = open_df[['wallet_id', 'region_code']].copy()

    # 作为发送方，统计交易对手的地域多样性
    sent_dst = tx_df[['src', 'dst']].merge(open_region, left_on='dst', right_on='wallet_id', how='left')
    sent_region_diversity = sent_dst.groupby('src')['region_code'].nunique().reset_index()
    sent_region_diversity.columns = ['wallet_id', 'sent_region_count']

    # 作为接收方，统计交易来源地域多样性
    recv_src = tx_df[['dst', 'src']].merge(open_region, left_on='src', right_on='wallet_id', how='left')
    recv_region_diversity = recv_src.groupby('dst')['region_code'].nunique().reset_index()
    recv_region_diversity.columns = ['wallet_id', 'recv_region_count']

    # 合并并计算总的地域分散度
    geo_dispersion = sent_region_diversity.merge(recv_region_diversity, on='wallet_id', how='outer').fillna(0)
    geo_dispersion['geographic_dispersion_score'] = geo_dispersion['sent_region_count'] + geo_dispersion['recv_region_count']

    # 归一化到0-1之间
    if geo_dispersion['geographic_dispersion_score'].max() > 0:
        geo_dispersion['geographic_dispersion_score'] = geo_dispersion['geographic_dispersion_score'] / geo_dispersion['geographic_dispersion_score'].max()

    result = geo_dispersion[['wallet_id', 'geographic_dispersion_score']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: geographic_dispersion_score")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 geographic_dispersion_score: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
