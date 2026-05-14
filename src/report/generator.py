"""验证报告生成模块：汇总搜索过程、特征字典、实验结果、稳定性验证，输出 Markdown 报告。"""

from datetime import datetime
from pathlib import Path


def _format_float(value, fmt=".4f"):
    if value is None:
        return "N/A"
    return f"{value:{fmt}}"


def _format_percent(value, fmt=".2f"):
    if value is None:
        return "N/A"
    return f"{value * 100:{fmt}}%"


def _section_search_summary(search_log: list[dict]) -> str:
    lines = [
        "## 1. 搜索过程摘要\n",
    ]

    total_rounds = len(search_log)
    total_accepted = sum(r.get("new_features", 0) for r in search_log)

    island_stats: dict[str, int] = {}
    for round_log in search_log:
        for island_log in round_log.get("islands", []):
            name = island_log.get("island", "unknown")
            island_stats[name] = island_stats.get(name, 0) + island_log.get("accepted", 0)

    converged = total_rounds > 1 and search_log[-1].get("new_features", 0) == 0
    convergence_text = "是（末轮无新特征）" if converged else "否"

    lines.append(f"- **总轮数**: {total_rounds}")
    lines.append(f"- **通过验证的特征总数**: {total_accepted}")
    lines.append(f"- **收敛**: {convergence_text}")
    lines.append("")
    lines.append("### 各岛屿贡献\n")
    lines.append("| 岛屿（维度） | 通过特征数 |")
    lines.append("|---|---|")
    for island_name, count in sorted(island_stats.items(), key=lambda x: -x[1]):
        lines.append(f"| {island_name} | {count} |")

    lines.append("")
    lines.append("### 各轮详情\n")
    lines.append("| 轮次 | 时间 | 新增特征 | 各岛屿候选/通过 |")
    lines.append("|---|---|---|---|")
    for r in search_log:
        round_num = r.get("round", "?")
        ts = r.get("timestamp", "")
        new_feat = r.get("new_features", 0)
        island_details = []
        for il in r.get("islands", []):
            island_details.append(f"{il.get('island', '?')}: {il.get('candidates', 0)}/{il.get('accepted', 0)}")
        lines.append(f"| {round_num} | {ts} | {new_feat} | {'; '.join(island_details)} |")

    lines.append("")
    return "\n".join(lines)


def _section_feature_dictionary(feature_dictionary: dict) -> str:
    lines = [
        "## 2. 特征字典\n",
    ]

    metadata = feature_dictionary.get("metadata", {})
    lines.append(f"- **特征总数**: {metadata.get('total_features', 0)}")
    dims = metadata.get("dimensions", {})
    if dims:
        dim_text = ", ".join(f"{k}: {v}" for k, v in dims.items())
        lines.append(f"- **维度分布**: {dim_text}")
    lines.append("")

    features = feature_dictionary.get("features", [])
    if not features:
        lines.append("*无通过验证的特征*\n")
        return "\n".join(lines)

    lines.append("| 排名 | 特征名 | 维度 | 综合得分 | MI | Cramér's V | ΔPR-AUC | SHAP排名 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, f in enumerate(features, 1):
        stats = f.get("statistics", {})
        model = f.get("model_metrics", {})
        lines.append(
            f"| {i} "
            f"| {f.get('feature_name', 'N/A')} "
            f"| {f.get('dimension', 'N/A')} "
            f"| {_format_float(f.get('composite_score'))} "
            f"| {_format_float(stats.get('mutual_information'))} "
            f"| {_format_float(stats.get('cramers_v'))} "
            f"| {_format_float(model.get('delta_prauc'))} "
            f"| {model.get('shap_rank', 'N/A')} |"
        )

    lines.append("")
    lines.append("### 特征详情\n")
    for i, f in enumerate(features, 1):
        lines.append(f"#### {i}. {f.get('feature_name', 'N/A')}\n")
        lines.append(f"- **维度**: {f.get('dimension', 'N/A')}")
        lines.append(f"- **综合得分**: {_format_float(f.get('composite_score'))}")
        explanation = f.get("explanation", "")
        if explanation:
            lines.append(f"- **说明**: {explanation}")
        code = f.get("code", "")
        if code:
            lines.append(f"\n```python\n{code}\n```\n")

    return "\n".join(lines)


def _section_experiments(experiment_results: dict) -> str:
    lines = [
        "## 3. 增量贡献实验结果\n",
    ]

    exp_names = {
        "exp1_baseline": "实验1: 仅人工基线特征",
        "exp2_augmented": "实验2: 人工 + LLM 特征",
        "exp3_llm_only": "实验3: 仅 LLM 特征",
        "exp4_random_control": "实验4: 人工 + 随机对照",
    }

    lines.append("### PR-AUC 对比\n")
    lines.append("| 实验 | PR-AUC (均值 ± 标准差) |")
    lines.append("|---|---|")
    for key, label in exp_names.items():
        exp = experiment_results.get(key, {})
        mean = exp.get("prauc_mean")
        std = exp.get("prauc_std")
        if mean is not None and std is not None:
            lines.append(f"| {label} | {mean:.4f} ± {std:.4f} |")
        else:
            lines.append(f"| {label} | N/A |")

    lines.append("")

    summary = experiment_results.get("summary", {})
    if summary:
        lines.append("### 关键指标\n")
        lines.append(f"- **增量 ΔPR-AUC（实验2 - 实验1）**: {_format_float(summary.get('delta_prauc'))}")
        lines.append(f"- **LLM 独立 PR-AUC**: {_format_float(summary.get('llm_independent_prauc'))}")
        lines.append(f"- **随机对照 ΔPR-AUC（实验4 - 实验1）**: {_format_float(summary.get('random_control_delta'))}")
        critical = summary.get("most_critical_feature")
        if critical:
            lines.append(f"- **最关键特征（消融）**: {critical}")
        lines.append("")

    ablation = experiment_results.get("exp5_ablation", {})
    if ablation:
        lines.append("### 消融实验（逐个移除 LLM 特征）\n")
        lines.append("| 特征 | 移除后 PR-AUC | 贡献（ΔPR-AUC） |")
        lines.append("|---|---|---|")
        sorted_ablation = sorted(ablation.items(), key=lambda x: -abs(x[1].get("contribution", 0)))
        for feat_name, info in sorted_ablation:
            lines.append(
                f"| {feat_name} "
                f"| {_format_float(info.get('prauc_mean'))} "
                f"| {_format_float(info.get('contribution'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def _section_stability(stability_results: dict) -> str:
    lines = [
        "## 4. L4 稳定性验证\n",
    ]

    si = stability_results.get("stability_index")
    passed = stability_results.get("pass")
    rankings = stability_results.get("per_round_rankings", [])

    lines.append(f"- **Nogueira 稳定性指数**: {_format_float(si)}")
    lines.append(f"- **Bootstrap 轮数**: {len(rankings)}")
    pass_text = "通过 ✓" if passed else "未通过 ✗"
    lines.append(f"- **阈值 > 0.8**: {pass_text}")
    lines.append("")

    if rankings:
        from collections import Counter
        freq: Counter = Counter()
        for ranking in rankings:
            for feat in ranking:
                freq[feat] += 1
        lines.append("### 特征出现频率（Top-20 排名中）\n")
        lines.append("| 特征 | 出现次数 | 出现率 |")
        lines.append("|---|---|---|")
        for feat, count in freq.most_common():
            rate = count / len(rankings)
            lines.append(f"| {feat} | {count}/{len(rankings)} | {_format_percent(rate)} |")
        lines.append("")

    return "\n".join(lines)


def _section_visualizations(plot_paths: dict[str, str] | None) -> str:
    lines = [
        "## 5. 可视化图表\n",
    ]

    if not plot_paths:
        lines.append("*未生成图表*\n")
        return "\n".join(lines)

    plot_labels = {
        "shap_summary": "SHAP 特征重要性",
        "experiment_comparison": "实验 PR-AUC 对比",
        "ablation_heatmap": "消融实验热力图",
        "stability_distribution": "稳定性 Bootstrap 分布",
        "island_contribution": "岛屿贡献分布",
    }

    for key, label in plot_labels.items():
        path = plot_paths.get(key)
        if path:
            lines.append(f"### {label}\n")
            lines.append(f"![{label}]({path})\n")

    return "\n".join(lines)


def generate_report(
    stability_results: dict,
    experiment_results: dict,
    feature_dictionary: dict,
    search_log: list[dict],
    plot_paths: dict[str, str] | None = None,
) -> str:
    """生成完整验证报告（Markdown）。

    Parameters
    ----------
    stability_results : L4 稳定性验证结果
    experiment_results : 5 组增量贡献实验结果
    feature_dictionary : 特征字典（generate_feature_dictionary 输出）
    search_log : 搜索日志（SearchResult.search_log）
    plot_paths : 可视化图表路径字典（generate_all_plots 输出），可选

    Returns
    -------
    str : Markdown 格式报告文本
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        f"# LLM 特征提取验证报告\n",
        f"> 生成时间: {now}\n",
        "---\n",
        _section_search_summary(search_log),
        "---\n",
        _section_feature_dictionary(feature_dictionary),
        "---\n",
        _section_experiments(experiment_results),
        "---\n",
        _section_stability(stability_results),
        "---\n",
        _section_visualizations(plot_paths),
    ]

    return "\n".join(sections)


def export_report(
    stability_results: dict,
    experiment_results: dict,
    feature_dictionary: dict,
    search_log: list[dict],
    output_path: str = "output/report.md",
    plot_paths: dict[str, str] | None = None,
) -> str:
    """生成并导出验证报告到文件。

    Returns
    -------
    str : 输出文件路径
    """
    report = generate_report(
        stability_results=stability_results,
        experiment_results=experiment_results,
        feature_dictionary=feature_dictionary,
        search_log=search_log,
        plot_paths=plot_paths,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return str(out)
