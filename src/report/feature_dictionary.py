"""特征字典生成模块：为每个通过验证的特征生成完整字典条目并导出。"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import shap

from src.config import ISLANDS, DEFAULT_CONFIG


def _infer_dimension(feature: dict) -> str:
    """根据特征代码中引用的字段推断所属维度岛屿。"""
    code = feature.get("code", "")
    best_match = ("未分类", "unknown", 0)
    for island in ISLANDS:
        hits = sum(1 for f in island.fields if f in code)
        if hits > best_match[2]:
            best_match = (island.name, island.dimension, hits)
    if feature.get("island"):
        return feature["island"]
    if feature.get("dimension"):
        return feature["dimension"]
    return best_match[0] if best_match[2] > 0 else "未分类"


def _compute_shap_rankings(X_final: pd.DataFrame, y: pd.Series) -> dict:
    """训练 XGBoost 并返回每个特征的 SHAP 均值排名。"""
    config = DEFAULT_CONFIG.xgboost
    model = XGBClassifier(
        scale_pos_weight=config.scale_pos_weight,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        eval_metric=config.eval_metric,
        verbosity=0,
    )
    model.fit(X_final, y)
    explainer = shap.TreeExplainer(model)
    shap_sample_size = min(5000, len(X_final))
    X_shap = X_final.sample(n=shap_sample_size, random_state=42) if shap_sample_size < len(X_final) else X_final
    shap_values = explainer.shap_values(X_shap)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_names = list(X_final.columns)
    ranked_indices = np.argsort(-mean_abs_shap)
    rankings = {}
    for rank, idx in enumerate(ranked_indices, start=1):
        rankings[feature_names[idx]] = {
            "rank": rank,
            "mean_abs_shap": float(mean_abs_shap[idx]),
        }
    return rankings


def generate_feature_dictionary(
    features_data: list[dict],
    X_final: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """为每个通过验证的特征生成完整字典条目。

    Args:
        features_data: output/features.json 中的特征列表
        X_final: 包含所有特征列的最终特征矩阵
        y: 标签列

    Returns:
        dict 包含 metadata 和 features 列表
    """
    shap_rankings = _compute_shap_rankings(X_final, y)

    entries = []
    for feat in features_data:
        name = feat["feature_name"]
        l2 = feat.get("l2_metrics", {})
        l3 = feat.get("l3_metrics", {})
        shap_info = shap_rankings.get(name, {"rank": None, "mean_abs_shap": None})

        entry = {
            "feature_name": name,
            "dimension": _infer_dimension(feat),
            "code": feat.get("code", ""),
            "explanation": feat.get("explanation", ""),
            "composite_score": feat.get("score", 0.0),
            "statistics": {
                "mutual_information": l2.get("mi"),
                "cramers_v": l2.get("cramers_v"),
                "redundancy": l2.get("redundancy"),
                "bootstrap_p": l2.get("bootstrap_p"),
            },
            "model_metrics": {
                "delta_prauc": l3.get("delta_prauc"),
                "shap_rank": shap_info["rank"],
                "mean_abs_shap": shap_info["mean_abs_shap"],
            },
        }
        entries.append(entry)

    entries.sort(key=lambda e: e["composite_score"], reverse=True)

    dictionary = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_features": len(entries),
            "dimensions": _summarize_dimensions(entries),
        },
        "features": entries,
    }
    return dictionary


def _summarize_dimensions(entries: list[dict]) -> dict:
    """统计各维度的特征数量。"""
    dim_counts: dict[str, int] = {}
    for e in entries:
        dim = e["dimension"]
        dim_counts[dim] = dim_counts.get(dim, 0) + 1
    return dim_counts


def _format_value(val, fmt=".6f") -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return str(val)


def _feature_to_markdown(entry: dict, index: int) -> str:
    """将单个特征字典条目转为 Markdown 片段。"""
    stats = entry["statistics"]
    model = entry["model_metrics"]

    lines = [
        f"### {index}. {entry['feature_name']}",
        "",
        f"- **维度**: {entry['dimension']}",
        f"- **综合评分**: {_format_value(entry['composite_score'], '.4f')}",
        "",
        "#### 可解释性说明",
        "",
        entry["explanation"],
        "",
        "#### 统计指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 互信息 (MI) | {_format_value(stats['mutual_information'])} |",
        f"| Cramér's V | {_format_value(stats['cramers_v'])} |",
        f"| 冗余度 | {_format_value(stats['redundancy'])} |",
        f"| Bootstrap p 值 | {_format_value(stats['bootstrap_p'], '.2e')} |",
        "",
        "#### 模型指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| ΔPR-AUC | {_format_value(model['delta_prauc'])} |",
        f"| SHAP 排名 | {_format_value(model['shap_rank'])} |",
        f"| 平均 |SHAP| | {_format_value(model['mean_abs_shap'])} |",
        "",
        "#### Python 代码",
        "",
        "```python",
        entry["code"],
        "```",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def export_feature_dictionary(dictionary: dict, output_path: str) -> None:
    """导出特征字典为 JSON + Markdown 格式。

    Args:
        dictionary: generate_feature_dictionary 的返回值
        output_path: 输出路径（不含扩展名），同时生成 .json 和 .md
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    json_path = out.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    md_path = out.with_suffix(".md")
    meta = dictionary["metadata"]
    features = dictionary["features"]

    md_lines = [
        "# 特征字典",
        "",
        f"生成时间: {meta['generated_at']}",
        "",
        f"通过验证的特征总数: **{meta['total_features']}**",
        "",
        "## 维度分布",
        "",
        "| 维度 | 特征数量 |",
        "|------|---------|",
    ]
    for dim, count in meta["dimensions"].items():
        md_lines.append(f"| {dim} | {count} |")

    md_lines.extend(["", "---", "", "## 特征详情", ""])

    for i, entry in enumerate(features, start=1):
        md_lines.append(_feature_to_markdown(entry, i))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"特征字典已导出: {json_path}, {md_path}")
