"""结果可视化模块：生成 SHAP、实验对比、消融热力图、稳定性分布、岛屿贡献图表。"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from xgboost import XGBClassifier

from src.config import DEFAULT_CONFIG, ISLANDS


def _ensure_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_shap_summary(
    X_final: pd.DataFrame,
    y: pd.Series,
    output_dir: str = "output/plots",
    top_k: int = 20,
) -> str:
    """SHAP summary plot（特征重要性）。"""
    import shap

    out = _ensure_dir(output_dir)
    cfg = DEFAULT_CONFIG.xgboost
    model = XGBClassifier(
        scale_pos_weight=cfg.scale_pos_weight,
        eval_metric=cfg.eval_metric,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        tree_method=cfg.tree_method,
        device=cfg.device,
    )
    model.fit(X_final, y, verbose=False)
    explainer = shap.TreeExplainer(model)
    shap_sample_size = min(5000, len(X_final))
    X_shap = X_final.sample(n=shap_sample_size, random_state=42) if shap_sample_size < len(X_final) else X_final
    shap_values = explainer.shap_values(X_shap)

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(-mean_abs)[:top_k]

    names = [X_final.columns[i] for i in top_indices]
    values = mean_abs[top_indices]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values[::-1], color="#1f77b4", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (Top {})".format(min(top_k, len(names))))
    ax.invert_yaxis()
    fig.tight_layout()

    path = out / "shap_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_experiment_comparison(
    experiment_results: dict,
    output_dir: str = "output/plots",
) -> str:
    """5 组实验 PR-AUC 对比柱状图（含误差线）。"""
    out = _ensure_dir(output_dir)

    labels = []
    means = []
    stds = []

    exp_map = [
        ("exp1_baseline", "Baseline\n(人工特征)"),
        ("exp2_augmented", "Augmented\n(人工+LLM)"),
        ("exp3_llm_only", "LLM Only\n(仅LLM)"),
        ("exp4_random_control", "Random\n(随机对照)"),
    ]
    for key, label in exp_map:
        if key in experiment_results:
            labels.append(label)
            means.append(experiment_results[key]["prauc_mean"])
            stds.append(experiment_results[key]["prauc_std"])

    if not labels:
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors[: len(labels)],
        edgecolor="white",
        width=0.6,
    )

    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("PR-AUC")
    ax.set_title("5-Fold CV PR-AUC Comparison")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = out / "experiment_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_ablation_heatmap(
    experiment_results: dict,
    output_dir: str = "output/plots",
) -> str:
    """消融实验热力图。"""
    out = _ensure_dir(output_dir)
    ablation = experiment_results.get("exp5_ablation", {})
    if not ablation:
        return ""

    names = list(ablation.keys())
    contributions = [ablation[n]["contribution"] for n in names]
    prauc_after = [ablation[n]["prauc_mean"] for n in names]

    sorted_idx = np.argsort(contributions)[::-1]
    names = [names[i] for i in sorted_idx]
    contributions = [contributions[i] for i in sorted_idx]
    prauc_after = [prauc_after[i] for i in sorted_idx]

    data = np.array([contributions, prauc_after])

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Contribution\n(ΔPR-AUC)", "PR-AUC\n(after removal)"], fontsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Ablation Study: Feature Contribution")
    fig.tight_layout()

    path = out / "ablation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_stability_distribution(
    stability_results: dict,
    output_dir: str = "output/plots",
) -> str:
    """特征稳定性 bootstrap 分布图。"""
    out = _ensure_dir(output_dir)
    rankings = stability_results.get("per_round_rankings", [])
    if not rankings:
        return ""

    all_features = sorted({f for r in rankings for f in r})
    freq = {f: 0 for f in all_features}
    for r in rankings:
        for f in r:
            freq[f] += 1

    n_rounds = len(rankings)
    sorted_features = sorted(freq.keys(), key=lambda f: freq[f], reverse=True)
    counts = [freq[f] for f in sorted_features]

    fig, ax = plt.subplots(figsize=(10, max(5, len(sorted_features) * 0.3)))
    y_pos = np.arange(len(sorted_features))
    colors = ["#2ca02c" if c == n_rounds else "#1f77b4" if c >= n_rounds * 0.8 else "#ff7f0e" if c >= n_rounds * 0.5 else "#d62728" for c in counts]

    ax.barh(y_pos, counts, color=colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=8)
    ax.set_xlabel(f"Selection Count (out of {n_rounds} bootstraps)")
    ax.set_title(
        "Feature Stability: Bootstrap Selection Frequency\n"
        f"Nogueira Index = {stability_results.get('stability_index', 0):.4f}"
    )
    ax.axvline(x=n_rounds * 0.8, color="gray", linestyle="--", linewidth=0.8, label="80% threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = out / "stability_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_island_contribution(
    features_data: list[dict],
    output_dir: str = "output/plots",
) -> str:
    """各维度岛屿贡献饼图。"""
    out = _ensure_dir(output_dir)

    island_names = {isl.name for isl in ISLANDS}
    dim_counts: dict[str, int] = {}

    for feat in features_data:
        code = feat.get("code", "")
        island = feat.get("island", "")
        dim = feat.get("dimension", "")

        matched = island if island in island_names else dim if dim in island_names else None
        if not matched:
            best = ("未分类", 0)
            for isl in ISLANDS:
                hits = sum(1 for f in isl.fields if f in code)
                if hits > best[1]:
                    best = (isl.name, hits)
            matched = best[0] if best[1] > 0 else "未分类"

        dim_counts[matched] = dim_counts.get(matched, 0) + 1

    if not dim_counts:
        return ""

    labels = list(dim_counts.keys())
    sizes = list(dim_counts.values())
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * sum(sizes)))})",
        colors=colors[: len(labels)],
        startangle=90,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Island Contribution: Validated Features per Dimension")
    fig.tight_layout()

    path = out / "island_contribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def generate_all_plots(
    X_final: pd.DataFrame,
    y: pd.Series,
    experiment_results: dict,
    stability_results: dict,
    features_data: list[dict],
    output_dir: str = "output/plots",
) -> dict[str, str]:
    """生成所有图表，返回 {plot_name: file_path}。"""
    paths = {}

    paths["shap_summary"] = plot_shap_summary(X_final, y, output_dir)
    paths["experiment_comparison"] = plot_experiment_comparison(experiment_results, output_dir)
    paths["ablation_heatmap"] = plot_ablation_heatmap(experiment_results, output_dir)
    paths["stability_distribution"] = plot_stability_distribution(stability_results, output_dir)
    paths["island_contribution"] = plot_island_contribution(features_data, output_dir)

    paths = {k: v for k, v in paths.items() if v}
    print(f"已生成 {len(paths)} 张图表到 {output_dir}/")
    return paths
