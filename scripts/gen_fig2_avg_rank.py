"""Generate the paper's average-rank figure from frozen-holdout JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analyze_public_results import DEFAULT_RESULTS_DIR, PROJECT_ROOT, analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "paper" / "figures" / "avg_rank",
    )
    args = parser.parse_args()

    ranks = analyze(args.results_dir)["average_ranks"]
    ordered = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
    methods = [method for method, _ in ordered]
    values = [rank for _, rank in ordered]
    colors = ["#1f5a94" if method == "KnowFeat" else "#b8bec6" for method in methods]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    bars = ax.barh(methods, values, color=colors, height=0.62)
    ax.set_xlim(0, 7)
    ax.set_xlabel("Average Rank (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", length=2.5, width=0.7)
    for bar, value in zip(bars, values):
        ax.text(value + 0.10, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    fig.tight_layout(pad=0.35)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
