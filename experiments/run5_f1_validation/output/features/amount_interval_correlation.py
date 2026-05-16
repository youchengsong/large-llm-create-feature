"""
特征: amount_interval_correlation
说明: 洗钱行为（特别是'线上跑分'和'线下跑分'）通常表现为在短时间内进行多笔固定金额或相近金额的交易，导致交易金额与时间间隔之间存在特定的关联模式。通过计算每个钱包的金额与间隔的相关系数，可以识别出这种异常模式。例如，如果钱包的交易金额非常稳定（标准差小），但时间间隔却变化很大，或者反之，都可能意味着异常。这与'交易行为异常类风险'中的'交易频率'和'交易金额波动幅度'相关。

验证指标:
  - Score: 0.2000
  - MI: 0.0000
  - Redundancy: 0.0392
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 68

可解释性报告:
  - 值含义: {'接近0（-0.1 ~ 0.1）': '金额与时间间隔之间无明显线性关系，是大多数正常交易钱包的典型表现，金额和间隔之间相互独立', '负值（< -0.3）': "金额越大间隔越短，或金额越小间隔越长。可能对应'线下跑分'模式：大额资金到账后立即转出（大金额+短间隔），或小额测试交易后长时间等待（小金额+长间隔）", '正值（> 0.3）': "金额越大间隔越长，或金额越小间隔越短。可能对应'线上跑分'模式：固定金额（如10-1000元）的定时批量交易，金额和间隔呈正相关", '极端值（< -0.7 或 > 0.7）': '金额与间隔几乎完全线性相关，高度可疑。常见于程序化交易脚本，交易金额和时间间隔被严格预设，如每5分钟转出固定金额'}
  - 风险阈值: {'高风险阈值': '绝对值大于0.7', '判断依据': '0.5对应中度线性相关，可视为预警阈值，用于捕获可能存在结构化交易但尚未达到极端程度的钱包', '中等风险阈值': '绝对值大于0.5'}
  - 协同特征: [{'feature': 'transaction_frequency（交易频率）', 'reason': '高相关系数+高交易频率=高度可疑的程序化交易行为。线上跑分模式通常同时具备高频交易和金额-间隔强相关的特征，两者联合可显著提升检出率'}, {'feature': 'Std_Sent（发送金额标准差）', 'reason': '高相关系数+低标准差=固定金额的定时交易，典型线上跑分特征。高相关系数+高标准差=金额与间隔存在规律但金额波动大，可能对应线下跑分中的大额快进快出'}, {'feature': 'avg_inactive_period（平均不活跃间隔）', 'reason': '高相关系数+短不活跃间隔=持续性结构化交易，排除正常用户偶尔的规律性转账（如工资发放），聚焦于持续性异常'}]
  - 误判场景: ['定期定额理财用户：每月固定日期固定金额转入理财产品，金额和间隔完全固定（相关系数=1），但属于正常理财行为', '工资发放场景：企业钱包每月固定日期向员工发放固定工资，金额和间隔高度相关，但属正常经营行为', '定时缴费用户：每月固定日期缴纳水电费、房贷等固定金额，金额和间隔强相关，属正常生活缴费', '低频规律交易：钱包只有2-3笔交易且恰好金额和间隔线性相关（如第1笔100元间隔1天，第2笔200元间隔2天），样本量过小导致的伪相关']
  - 监管说明: 该特征衡量的是用户每笔转账金额与两笔交易之间的时间间隔是否存在固定规律——如果金额越大等得越久、金额越小转得越快，或者反过来金额越大转得越快，都说明交易行为可能被某种程序脚本控制，而非真实的人为操作，这在洗钱跑分活动中非常常见。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_interval_correlation，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包的金额与时间间隔的相关系数
    def calc_corr(group):
        # 需要至少3笔交易才能计算有意义的相关系数
        if len(group) < 3:
            return np.nan
        # 计算金额和间隔的相关系数
        corr = group['amount'].corr(group['interval'])
        # 处理可能的NaN（例如所有金额或所有间隔都相同）
        return corr if not np.isnan(corr) else 0.0

    # 按src分组，计算金额和间隔的相关系数
    correlation = tx_df.groupby('src').apply(calc_corr).reset_index()
    correlation.columns = ['wallet_id', 'amount_interval_correlation']
    # 填充NaN（交易笔数少于3的钱包）为0，表示无相关关系
    correlation['amount_interval_correlation'] = correlation['amount_interval_correlation'].fillna(0)

    result = correlation
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_interval_correlation")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_interval_correlation: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
