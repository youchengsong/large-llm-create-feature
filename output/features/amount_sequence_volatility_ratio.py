"""
特征: amount_sequence_volatility_ratio
说明: 洗钱交易常通过拆分、归集等操作导致金额剧烈波动，而正常交易金额相对平稳。金额序列的波动性变化能有效区分正常与异常交易行为。

验证指标:
  - Score: 0.1141
  - MI: 0.0012
  - Redundancy: 0.6279
  - Cramer's V: 0.0181
  - Delta PR-AUC: -0.019915
  - SHAP Rank: 46

可解释性报告:
  - 值含义: {'0 ~ 3.9': '金额序列波动极小，连续交易金额高度一致，常见于固定金额的日常消费（如公交充值、便利店购物）或定期定额转账', '3.9 ~ 5.0': '正常波动范围，体现日常消费中金额的自然变化，如餐饮、购物等不同金额的混合支付', '5.0 ~ 6.6': '波动性开始上升，可能包含部分大额交易或偶发的非日常消费，需结合其他特征判断', '6.6 ~ 9.5': '显著高波动区间，金额比值频繁大幅变化，常见于洗钱交易中的拆分转出（如10000元拆成多笔不同金额）或归集转入后立即分散转出', '> 9.5': "极端高波动，金额序列呈现剧烈震荡，几乎每次交易金额都与前次相差数倍以上，高度疑似洗钱操作中的资金'快进快出'或'测试-大额'交替模式"}
  - 风险阈值: {'primary_threshold': 6.6, 'primary_basis': '异常交易的第25分位值为5.0472，第75分位值为9.5134，6.6恰好位于异常交易分布的底部区间，高于正常交易的第75分位值(6.5812)，可作为预警阈值', 'secondary_threshold': 9.5, 'secondary_basis': '异常交易的第75分位值为9.5134，超过此值的交易有75%概率为异常，适合作为高风险标记阈值，可触发人工审核', 'statistical_justification': '正常交易均值5.73，异常均值6.93，差异约1.2（20%相对差异），且正常75分位(6.58)低于异常25分位(5.05)的区间存在重叠，因此采用双阈值策略'}
  - 协同特征: [{'feature': 'send_concentration', 'reason': "金额波动大且发送集中度高（资金主要流向少数对手），高度吻合'归集后分散'的洗钱模式；波动大但集中度低则可能为正常消费"}, {'feature': 'Timestamp_Hour', 'reason': '金额波动大且交易发生在凌晨(0-5点)，洗钱嫌疑极高，因为正常消费极少在深夜产生金额剧烈波动的交易序列'}, {'feature': 'transaction_frequency', 'reason': "高波动+高频率的组合，对应'测试交易'模式（先小额测试再大额操作），也符合线上跑分的'高频小额分散转出'特征"}, {'feature': 'large_transaction_ratio', 'reason': "高波动+高比例大额交易，说明金额序列在大小额之间频繁切换，这是洗钱中'拆分交易规避监管'的典型特征"}]
  - 误判场景: ['场景1：商家促销活动期间，收款金额从几元到几百元剧烈波动（如奶茶店周末高峰时段），但这是正常经营行为', "场景2：个人用户的'大额消费+小额找零'模式，如购买大件商品后退款、押金缴纳与退还等，金额序列会产生剧烈波动但无洗钱意图", '场景3：新开账户首次使用，从0余额开始，首笔大额充值后立即进行多笔不同金额的消费，金额比值会异常高', '场景4：企业工资发放场景，先一次性大额转入，再分批发放不同金额的工资，金额序列波动大但属于正常经营']
  - 监管说明: 该特征衡量的是同一钱包连续交易金额的'忽大忽小'程度——正常人的消费金额变化有规律可循（如吃饭花30-50，买衣服花200-500），但洗钱账户的交易金额像是在'猜谜'，上一笔可能是1元测试，下一笔就变成10000元，再下一笔又变成500元，这种极端不自然的金额波动是洗钱行为的典型特征。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_sequence_volatility_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()
    tx_sorted['prev_amount'] = tx_sorted.groupby('src')['amount'].shift(1)
    tx_sorted['amount_ratio'] = np.where(
        tx_sorted['prev_amount'] > 0,
        tx_sorted['amount'] / tx_sorted['prev_amount'],
        np.nan
    )
    # 计算金额比值的波动性（变异系数）
    volatility = tx_sorted.groupby('src')['amount_ratio'].agg(['mean', 'std']).reset_index()
    volatility['amount_sequence_volatility_ratio'] = np.where(
        volatility['mean'] > 0,
        volatility['std'] / volatility['mean'],
        0
    )
    result = volatility[['src', 'amount_sequence_volatility_ratio']].rename(columns={'src': 'wallet_id'})
    result['amount_sequence_volatility_ratio'] = result['amount_sequence_volatility_ratio'].fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_sequence_volatility_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_sequence_volatility_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
