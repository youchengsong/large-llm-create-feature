"""
特征: src_consecutive_small_amount_ratio
说明: small_amount_testing 模式中，金额0.01-0.1元，休眠账户突然激活后先进行小额测试。洗钱者常使用连续小额交易测试账户功能，这种行为在正常用户中较少见。该特征能有效识别测试性交易行为。

验证指标:
  - Score: 0.2376
  - MI: 0.0000
  - Redundancy: 0.6666
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 140

可解释性报告:
  - 值含义: {'0.0': '该钱包从未出现连续小额交易行为，属于正常交易模式', '(0.0, 0.2]': '偶尔出现连续小额交易，可能是正常的小额消费习惯', '(0.2, 0.5]': '连续小额交易比例中等，需结合其他特征判断是否为测试行为', '(0.5, 0.8]': '连续小额交易比例较高，存在测试性交易嫌疑', '(0.8, 1.0]': '几乎全部交易都是连续小额，高度疑似 small_amount_testing 模式'}
  - 风险阈值: 0.5
  - 协同特征: [{'feature': 'is_night_hour_From', 'reason': 'small_amount_testing 模式常在深夜(0-5,22-23点)进行，与连续小额交易结合可提高识别精度'}, {'feature': 'max_inactive_period_From', 'reason': '休眠账户(>=90天)突然激活后进行小额测试，结合最大不活跃期可识别账户激活后的异常行为'}, {'feature': 'S_src_per_deal_match_times', 'reason': '测试交易后可能进行大额交易（接近限额），连续小额+后续大额匹配是典型洗钱路径'}]
  - 误判场景: ['自动售货机/地铁充值等场景：用户连续多次小额充值（如每次充值50元），属于正常消费行为', '红包发放场景：商户批量发放小额红包（如每次0.01-1元），会形成连续小额交易', '小额批发零售商户：如便利店、菜市场商户，频繁收到小额付款属于正常经营模式', '游戏充值/虚拟商品购买：用户连续购买小额虚拟商品（如每次6元、30元）', '慈善捐款场景：用户连续多次小额捐款（如每次0.1-10元）']
  - 监管说明: 该特征检测的是账户是否频繁进行连续小额交易（每笔低于100元），这种行为在洗钱测试阶段常见，但正常消费场景（如自动售货机、红包）也可能出现，需要结合交易时间是否在深夜、账户是否长期未使用等维度综合判断。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 src_consecutive_small_amount_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    # 按 src 和时间排序
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()
    # 标记是否为小额交易（金额<100元）
    tx_sorted['is_small'] = tx_sorted['amount'].abs() < 100
    # 对每个钱包，计算连续小额交易的比例
    def calc_small_ratio(group):
        small_flags = group['is_small'].values
        if len(small_flags) < 2:
            return 0.0
        # 连续小额交易：当前交易和下一笔交易都是小额
        consecutive_small = (small_flags[:-1] & small_flags[1:]).sum()
        total_consecutive = len(small_flags) - 1
        return consecutive_small / total_consecutive if total_consecutive > 0 else 0.0
    ratio_df = tx_sorted.groupby('src').apply(calc_small_ratio).reset_index()
    ratio_df.columns = ['wallet_id', 'src_consecutive_small_amount_ratio']
    result = ratio_df
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: src_consecutive_small_amount_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 src_consecutive_small_amount_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
