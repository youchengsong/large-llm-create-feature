"""Schema + 分布差异自动生成，供 prompt 注入。"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from knowfeat.data.loader import Stage2Data

logger = logging.getLogger(__name__)

_ID_LIKE = {
    "tx_id", "wallet_id", "src", "dst", "zs_id",
    "wallet_open_cert", "wallet_open_tel",
    "src_bank_account_number", "dst_bank_account_number",
    "bank_account_number", "accociated_bank_account",
    "src_device", "dst_device", "src_ip", "dst_ip",
    "open_device", "open_ip",
}

_DATETIME_LIKE = {
    "timestamp", "wallet_open_timestamp", "wallet_open_date",
    "wallet_open_cert_expire", "realname_auth_time", "zs_timestamp",
}


def _dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "str"


def _value_range(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        mn, mx = series.min(), series.max()
        return f"{mn} ~ {mx}"
    if pd.api.types.is_numeric_dtype(series):
        return f"[{series.min()}, {series.max()}], mean={series.mean():.4g}"
    nunique = series.nunique()
    if nunique <= 10:
        top = series.value_counts().head(10).index.tolist()
        return f"{nunique} values: {top}"
    return f"{nunique} unique values"


def generate_schema(df: pd.DataFrame, table_name: str) -> str:
    """Generate schema text for a single DataFrame."""
    lines = [f"### {table_name} ({df.shape[0]:,} rows × {df.shape[1]} cols)\n"]
    lines.append("| 字段 | 类型 | 取值范围/示例 | 缺失率 |")
    lines.append("|------|------|--------------|--------|")
    for col in df.columns:
        dtype = _dtype_label(df[col])
        vrange = _value_range(df[col])
        missing = df[col].isna().mean()
        lines.append(f"| {col} | {dtype} | {vrange} | {missing:.1%} |")
    return "\n".join(lines)


def generate_distribution_diff(df: pd.DataFrame, label_col: str,
                               table_name: str) -> str:
    """Generate normal vs abnormal distribution difference text."""
    if label_col not in df.columns:
        return f"### {table_name}: 无标签列 {label_col}\n"

    normal = df[df[label_col] == 0]
    abnormal = df[df[label_col] == 1]

    lines = [f"### {table_name} 正常 vs 异常分布差异"]
    lines.append(f"正常样本: {len(normal):,}, 异常样本: {len(abnormal):,}\n")

    skip = _ID_LIKE | _DATETIME_LIKE | {label_col}

    numeric_diffs = []
    categorical_diffs = []

    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            n_mean = normal[col].mean()
            a_mean = abnormal[col].mean()
            n_median = normal[col].median()
            a_median = abnormal[col].median()
            if n_mean != 0:
                ratio = a_mean / n_mean
            elif a_mean != 0:
                ratio = float("inf")
            else:
                ratio = 1.0
            numeric_diffs.append({
                "col": col,
                "n_mean": n_mean, "a_mean": a_mean,
                "n_median": n_median, "a_median": a_median,
                "ratio": ratio,
                "abs_diff": abs(ratio - 1) if ratio != float("inf") else float("inf"),
            })
        elif pd.api.types.is_object_dtype(df[col]) or hasattr(df[col], "cat"):
            n_props = normal[col].value_counts(normalize=True)
            a_props = abnormal[col].value_counts(normalize=True)
            all_vals = set(n_props.index) | set(a_props.index)
            max_diff = 0.0
            for v in all_vals:
                diff = abs(a_props.get(v, 0) - n_props.get(v, 0))
                max_diff = max(max_diff, diff)
            if max_diff > 0.05:
                categorical_diffs.append({"col": col, "max_prop_diff": max_diff})

    numeric_diffs.sort(key=lambda x: x["abs_diff"], reverse=True)

    if numeric_diffs:
        lines.append("| 字段 | 正常均值 | 异常均值 | 差异倍数 | 正常中位数 | 异常中位数 |")
        lines.append("|------|---------|---------|---------|-----------|-----------|")
        for d in numeric_diffs:
            r = f"{d['ratio']:.2f}x" if d["ratio"] != float("inf") else "∞"
            lines.append(
                f"| {d['col']} | {d['n_mean']:.4g} | {d['a_mean']:.4g} "
                f"| {r} | {d['n_median']:.4g} | {d['a_median']:.4g} |"
            )

    if categorical_diffs:
        lines.append("\n**分类字段差异**:")
        for d in categorical_diffs:
            lines.append(f"- {d['col']}: 最大比例差 {d['max_prop_diff']:.1%}")

    return "\n".join(lines)


def generate_join_stats(data: Stage2Data) -> str:
    """Generate join coverage statistics between the three tables."""
    lines = ["### 三表关联统计\n"]

    tx_src = set(data.tx_df["src"].unique())
    tx_dst = set(data.tx_df["dst"].unique())
    open_w = set(data.open_df["wallet_id"].unique())
    close_w = set(data.close_df["wallet_id"].unique())

    src_hit = len(tx_src & open_w)
    dst_hit = len(tx_dst & open_w)
    close_hit = len(close_w & open_w)

    lines.append(f"- tx_df.src → open_df.wallet_id: {src_hit}/{len(tx_src)} ({src_hit/len(tx_src):.1%})")
    lines.append(f"- tx_df.dst → open_df.wallet_id: {dst_hit}/{len(tx_dst)} ({dst_hit/len(tx_dst):.1%})")
    lines.append(f"- close_df → open_df.wallet_id: {close_hit}/{len(close_w)} ({close_hit/len(close_w):.1%})")
    lines.append(f"\n- open_df 钱包数: {len(open_w):,}")
    lines.append(f"- tx_df 涉及钱包数: {len(tx_src | tx_dst):,}")
    lines.append(f"- close_df 钱包数: {len(close_w):,}")

    return "\n".join(lines)


def generate_baseline_feature_list(data: Stage2Data) -> str:
    """Generate baseline feature name list text."""
    if data.baseline_features is None:
        return "### 基线特征: 未加载\n"
    exclude = {"is_laundering", "is_risk", "tx_id", "src", "dst"}
    cols = [c for c in data.baseline_features.columns if c not in exclude]
    lines = [f"### 基线特征 ({len(cols)} 个)\n"]
    lines.append(", ".join(cols))
    return "\n".join(lines)


def generate_all_schema_text(data: Stage2Data, label_col: str = "is_risk") -> str:
    """Generate complete schema + distribution diff text for prompt injection.

    Parameters
    ----------
    data : Stage2Data
        Stage 2 三表数据容器。
    label_col : str
        标签列名，默认 "is_risk"。

    Returns
    -------
    str
        格式化文本，可直接注入 System Prompt。
    """
    sections = [
        "## 数据 Schema\n",
        generate_schema(data.open_df, "accounts (open_df)"),
        generate_schema(data.tx_df, "transactions (tx_df)"),
        generate_schema(data.close_df, "close (close_df)"),
        "\n## 正常 vs 异常分布差异\n",
        generate_distribution_diff(data.tx_df, label_col, "transactions"),
    ]

    if "is_abnormal" in data.open_df.columns:
        sections.append(generate_distribution_diff(
            data.open_df, "is_abnormal", "accounts"))

    if "is_zs_laundering" in data.close_df.columns:
        sections.append(generate_distribution_diff(
            data.close_df, "is_zs_laundering", "close"))

    sections.append("\n## 三表关联\n")
    sections.append(generate_join_stats(data))
    sections.append("\n## 基线特征\n")
    sections.append(generate_baseline_feature_list(data))

    text = "\n\n".join(sections)
    logger.info("Generated schema text: %d chars", len(text))
    return text
