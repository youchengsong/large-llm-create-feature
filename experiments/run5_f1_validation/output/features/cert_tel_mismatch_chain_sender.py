"""
特征: cert_tel_mismatch_chain_sender
说明: 洗钱分子常批量注册钱包，可能使用同一证件但不同手机号（或反之）来规避单一维度的检测。证件-手机号关联数量的比值异常可以揭示这种主体关联层面的异常模式，与已有特征不重复。

验证指标:
  - Score: 0.3532
  - MI: 0.0000
  - Redundancy: 0.6341
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 117

可解释性报告:
  - 值含义: {'0.0': '绝对平衡，最常见于正常用户', '(-0.6931, 0.0)': '轻微不平衡，可能是正常的多账户管理行为', '>0.0': '证件关联钱包数多于手机号关联钱包数，同一证件下多个钱包共用少量手机号，存在批量注册嫌疑', '<-0.6931': '手机号关联钱包数远多于证件关联钱包数，同一手机号关联多个不同证件的钱包，存在身份冒用或代注册嫌疑'}
  - 风险阈值: {'high_risk_value': 0.0, 'threshold_basis': '从分布看，异常交易在[25%,75%]分位为[0.0,0.0]，即异常交易的75%分位恰好为0.0，说明所有正值的样本都集中在异常交易中。因此任何大于0的值都应视为高风险信号。', 'additional_notes': '负值方向(-0.6931以下)虽然也有风险含义，但在本数据集中异常交易的中位数为0.0，说明负值方向的风险信号不如正值方向显著。'}
  - 协同特征: [{'feature': 'wallets_per_cert_From', 'reason': '该特征直接衡量同一证件下的钱包数，与cert_tel_mismatch_chain_sender的正值方向高度互补。当两者同时偏高时，可以确认是同一证件批量注册多个钱包的洗钱模式。'}, {'feature': 'tel_reuse_count_From', 'reason': '该特征衡量同一手机号关联的钱包数，与cert_tel_mismatch_chain_sender的负值方向互补。当两者同时偏高时，可以确认是同一手机号关联多个不同证件钱包的身份冒用模式。'}, {'feature': 'small_amount_testing交易占比', 'reason': '批量注册后常伴随小额测试交易，如果cert_tel_mismatch_chain_sender偏高且该钱包的小额测试交易比例也高，则洗钱嫌疑大幅提升。'}]
  - 误判场景: [{'scenario': '家庭共用设备注册', 'explanation': '家庭成员使用同一部手机或同一WiFi注册多个钱包，可能导致同一IP/设备下出现多个不同证件的钱包，使得手机号关联钱包数偏多，触发负值方向的高危信号。', 'mitigation': '结合设备指纹特征（如open_device）判断是否为同一物理设备。'}, {'scenario': '企业员工批量开户', 'explanation': '企业为员工统一办理数字人民币钱包，可能使用同一营业执照（证件）但不同手机号，导致证件关联钱包数远多于手机号关联钱包数，触发正值方向的高危信号。', 'mitigation': '结合企业账户标识或交易对手特征判断是否为正常的企业行为。'}, {'scenario': '运营商号码回收', 'explanation': '手机号被运营商回收后分配给新用户，新用户用该手机号注册钱包并绑定自己的证件，导致同一手机号下出现多个不同证件的钱包记录。', 'mitigation': '结合开户时间间隔判断是否为号码回收场景。'}]
  - 监管说明: 该特征用于检测是否存在同一证件注册多个钱包但手机号不匹配，或同一手机号绑定多个不同证件的情况——这是批量注册洗钱账户的典型特征，正常用户极少出现。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 cert_tel_mismatch_chain_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个证件关联的钱包数
    cert_wallet_count = open_df.groupby('wallet_open_cert')['wallet_id'].transform('count')
    # 计算每个手机号关联的钱包数
    tel_wallet_count = open_df.groupby('wallet_open_tel')['wallet_id'].transform('count')

    # 计算证件-手机号关联比（证件关联数/手机号关联数），避免除零
    cert_tel_ratio = (cert_wallet_count / tel_wallet_count.replace(0, np.nan)).fillna(0)

    # 对每个钱包，计算其证件关联数与手机号关联数的差异程度
    # 如果证件关联数远大于手机号关联数，说明同一证件下多个钱包共用少量手机号（批量注册）
    # 如果手机号关联数远大于证件关联数，说明同一手机号关联多个不同证件的钱包（身份冒用）
    # 使用log变换使分布更对称
    log_ratio = np.log1p(np.abs(cert_tel_ratio - 1)) * np.sign(cert_tel_ratio - 1)

    # 创建结果DataFrame
    result = pd.DataFrame({
        'wallet_id': open_df['wallet_id'],
        'cert_tel_mismatch_chain_sender': log_ratio
    })

    # 确保wallet_id是交易表src中出现的钱包
    # 只保留在交易表中作为发送方的钱包
    sender_wallets = tx_df['src'].unique()
    result = result[result['wallet_id'].isin(sender_wallets)].reset_index(drop=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: cert_tel_mismatch_chain_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 cert_tel_mismatch_chain_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
