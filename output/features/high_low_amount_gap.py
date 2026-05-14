"""
特征: high_low_amount_gap
说明: 风险分类指标提到'多账户小额分散转账'和'大额转账记录'同时出现是典型的洗钱特征，利用金额极差比可以量化这种两极分化程度。

验证指标:
  - Score: 0.4294
  - MI: 0.0012
  - Redundancy: 0.5055
  - Cramer's V: 0.0266
  - Delta PR-AUC: -0.015704
  - SHAP Rank: 20

可解释性报告:
  - 值含义: {'low (0-6)': '交易金额波动小，所有交易金额接近，属于单一用途正常交易模式', 'medium (6-9)': '存在一定金额波动，可能有小额日常消费和大额转账混合，属于正常用户行为', 'high (9-12)': "金额波动显著，同时存在极小金额和较大金额交易，符合'多账户小额分散转账+大额转账记录'的洗钱特征", 'very high (12+)': '金额极度分化，极小金额测试交易与超大金额并存，强烈暗示测试+大额洗钱模式'}
  - 风险阈值: {'value': 9.5, 'rationale': '异常交易25%分位为9.2645，75%分位为12.4358，而正常交易75%分位为10.6642。取9.5作为阈值，可覆盖约75%的异常交易，同时误报率约30%（正常交易中25%超过10.66）。该阈值在召回率和精确率之间取得平衡。'}
  - 协同特征: [{'feature': 'small_amount_testing_ratio', 'reason': 'high_low_amount_gap高值说明金额两极分化，但不确定是否包含极小金额测试交易。配合small_amount_testing_ratio（极小金额交易占比），可确认是否存在测试行为，两者同时高值则洗钱嫌疑极高。'}, {'feature': 'activetime', 'reason': '洗钱交易常发生在凌晨（0-5点），而正常交易集中在白天。当high_low_amount_gap高且activetime异常（凌晨活跃），可有效区分正常用户（如白天大额转账+小额消费）和洗钱用户（深夜测试+转移）。'}, {'feature': 'fan_out_ratio', 'reason': '洗钱模式中fan_out（分散转出）比例高达54.3%，而正常仅35.5%。high_low_amount_gap高+fan_out_ratio高，说明钱包既做小额分散转出又做单笔大额转出，是典型的跑分特征。'}]
  - 误判场景: [{'scenario': '正常商户日常经营', 'explanation': '商户钱包既有大量小额收款（如零售交易），又有大额货款结算，金额波动自然大，high_low_amount_gap可达9-11。但商户交易时间正常（9-18点）、交易对手多样化，可与洗钱区分。'}, {'scenario': '个人用户混合用途', 'explanation': '个人用户既有日常小额消费（几十元）又有工资/大额转账（数千元），金额极差比可达8-10。但这类用户交易频率低、时间规律，且无测试交易特征。'}, {'scenario': '红包/优惠活动参与者', 'explanation': '参与薅羊毛活动的用户可能有0.01元测试交易和后续大额红包领取，金额波动大。但这类用户通常有多钱包关联，需结合same_id_card_wallet_count等特征排除。'}]
  - 监管说明: 该特征衡量一个钱包内交易金额的极端分化程度——如果同一个钱包既频繁进行几分钱的小额测试交易，又进行数万元的大额转账，极可能是洗钱分子在先用小额测试账户状态，再集中转移资金。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 high_low_amount_gap，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    amount_range = tx_df.groupby('src').agg({'amount': ['max', 'min']}).reset_index()
    amount_range.columns = ['wallet_id', 'max_amount', 'min_amount']
    amount_range['min_amount'] = amount_range['min_amount'].clip(lower=0.01)
    amount_range['high_low_amount_gap'] = np.log1p(amount_range['max_amount'] / amount_range['min_amount'])
    amount_range['high_low_amount_gap'] = amount_range['high_low_amount_gap'].fillna(0)
    result = amount_range[['wallet_id', 'high_low_amount_gap']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: high_low_amount_gap")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 high_low_amount_gap: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
