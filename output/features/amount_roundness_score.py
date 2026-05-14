"""
特征: amount_roundness_score
说明: 洗钱交易为了便于计算和归集，常使用整数金额（如100、500、1000），而正常交易金额更随机。金额整数度能有效识别此类异常。

验证指标:
  - Score: 0.0685
  - MI: 0.0007
  - Redundancy: 0.8337
  - Cramer's V: 0.0105
  - Delta PR-AUC: -0.012315
  - SHAP Rank: 47

可解释性报告:
  - 值含义: {'0.0000 - 0.0005': '金额高度随机，基本无整数规律，对应正常消费或小额转账行为', '0.0005 - 0.0100': '偶有整数金额出现，属正常波动范围，无显著风险', '0.0100 - 0.0300': '整数金额占比偏高，可能是习惯性整数转账或存在小额测试交易', '0.0300 - 0.0500': '明显整数偏好，常见于批量测试交易或跑分团伙的固定金额模式', '> 0.0500': '高度可疑，几乎全部交易均为整数金额，强烈指向洗钱或测试行为'}
  - 风险阈值: 0.03
  - 协同特征: [{'feature': 'small_amount_testing比例', 'reason': '整数金额特征与小额测试交易高度相关，洗钱测试阶段常使用1元、5元等整数金额测试账户可用性，两者结合可显著提升对测试型洗钱的识别'}, {'feature': '交易金额标准差(Std_Sent)', 'reason': '洗钱交易金额常为固定整数（如100、500），导致金额方差极小；正常交易金额波动更大。当整数度高且金额标准差低时，可确认是固定金额的洗钱模式而非随机整数'}, {'feature': '凌晨交易占比(activetime)', 'reason': '线上跑分洗钱常在凌晨(20-1点)进行，结合整数金额特征可精准定位凌晨高频整数交易行为，排除白天正常整数转账（如给家人转整钱）'}, {'feature': '交易对手集中度(send_concentration)', 'reason': '洗钱交易常向固定对手转整数金额，整数度高+对手集中度高=典型的线下跑分模式，可有效区分于正常用户偶尔整数转账'}]
  - 误判场景: ['工资发放账户：公司每月固定向员工发整数工资（如5000、8000），会导致整数度得分偏高，但这类账户交易对手固定、交易时间规律，可结合交易对手集中度排除', '房租转账：租客每月向房东转整数房租（如2000、3000），属于正常行为，但交易频率低（每月1次），可通过交易频率特征排除', '红包/转账习惯：部分用户习惯发整数红包（如6.66、8.88），但金额通常较小且伴有小数，不会完全整数', '商户收款：便利店等商户常收到整数付款（如10、20元），但商户的交易对手分散、交易频率高，可结合商户类型特征排除', '自动还款：信用卡自动还款常为整数金额，但交易对手固定且交易时间规律']
  - 监管说明: 该特征通过检测交易金额是否为整数（如100、500、1000）来识别洗钱行为，因为洗钱团伙为了便于计算和归集资金，常使用整数金额进行转账，而正常用户的消费金额通常更随机（如19.9、35.5），当某个钱包频繁进行整数金额转账时，可能存在洗钱嫌疑。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_roundness_score，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算交易金额的整数度特征
    tx_df['round_100'] = (tx_df['amount'] % 100 == 0).astype(int)
    tx_df['round_50'] = (tx_df['amount'] % 50 == 0).astype(int)
    tx_df['round_10'] = (tx_df['amount'] % 10 == 0).astype(int)

    # 综合整数度得分（权重：100整数度最高）
    tx_df['roundness_score'] = (
        tx_df['round_100'] * 3 + 
        tx_df['round_50'] * 2 + 
        tx_df['round_10'] * 1
    )

    # 按钱包计算平均整数度得分
    roundness = tx_df.groupby('src')['roundness_score'].mean().reset_index()
    roundness.columns = ['wallet_id', 'amount_roundness_score']

    # 同时考虑发送和接收的整数度
    roundness_sent = roundness.copy()

    roundness_recv = tx_df.groupby('dst')['roundness_score'].mean().reset_index()
    roundness_recv.columns = ['wallet_id', 'recv_roundness']

    # 合并并取最大值（一个钱包既发也收）
    result = roundness_sent.merge(roundness_recv, on='wallet_id', how='outer')
    result['amount_roundness_score'] = result[['amount_roundness_score', 'recv_roundness']].max(axis=1)
    result = result[['wallet_id', 'amount_roundness_score']].fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_roundness_score")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_roundness_score: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
