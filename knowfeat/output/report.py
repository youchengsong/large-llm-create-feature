"""实验报告生成：Recall / CR@K / wallet_type 分场景结果表 + 消融对比。

生成内容:
  - 核心指标表（Recall, Precision, F1, CR@1%, CR@3%, CR@5%）
  - wallet_type 分场景结果表
  - 消融实验对比表
  - 导出 CSV + HTML
"""

from __future__ import annotations

import csv
import html
import json
import logging
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

WALLET_TYPE_LABELS = {
    1: "Corporate",
    2: "Personal",
    3: "Merchant",
}

CR_AT_K_PERCENTS = [1, 3, 5]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricsRow:
    """单行指标结果。"""
    name: str
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    cr_at_k: dict[int, float] = field(default_factory=dict)
    n_samples: int = 0
    n_positives: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
        }
        for k, v in self.cr_at_k.items():
            d[f"CR@{k}%"] = round(v, 4)
        return d


@dataclass
class ExperimentReport:
    """完整实验报告。"""
    stage: str
    overall: Optional[MetricsRow] = None
    wallet_type_rows: list[MetricsRow] = field(default_factory=list)
    ablation_rows: list[MetricsRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


def compute_capture_rate(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_percent: int,
) -> float:
    """CR@K%: Top K% 预测分数中捕获的正样本占比。"""
    n = len(y_true)
    if n == 0:
        return 0.0
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return 0.0
    top_n = max(1, int(n * k_percent / 100))
    top_indices = np.argsort(y_scores)[::-1][:top_n]
    captured = int(y_true[top_indices].sum())
    return captured / total_pos


def evaluate_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[float, float, float, dict[int, float]]:
    """XGBoost 5-fold CV 评估，返回 (recall, precision, f1, {k: CR@K%})。"""
    y_arr = y.values.astype(int)
    X_clean = X.copy()
    for col in X_clean.columns:
        X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")
    X_clean = X_clean.fillna(0)

    if X_clean.shape[1] == 0 or len(y_arr) == 0:
        return 0.0, 0.0, 0.0, {k: 0.0 for k in CR_AT_K_PERCENTS}

    pos_count = y_arr.sum()
    neg_count = len(y_arr) - pos_count
    scale = neg_count / pos_count if pos_count > 0 else 1.0

    actual_folds = min(n_folds, int(pos_count), int(neg_count))
    if actual_folds < 2:
        return 0.0, 0.0, 0.0, {k: 0.0 for k in CR_AT_K_PERCENTS}

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)

    recalls, precisions, f1s = [], [], []
    all_cr: dict[int, list[float]] = {k: [] for k in CR_AT_K_PERCENTS}

    for train_idx, val_idx in skf.split(X_clean, y_arr):
        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_scores = clf.predict_proba(X_val)[:, 1]

        r, p, f = _compute_binary_metrics(y_val, y_pred)
        recalls.append(r)
        precisions.append(p)
        f1s.append(f)

        for k in CR_AT_K_PERCENTS:
            cr = compute_capture_rate(y_val, y_scores, k)
            all_cr[k].append(cr)

    cr_means = {k: float(np.mean(vs)) for k, vs in all_cr.items()}
    return float(np.mean(recalls)), float(np.mean(precisions)), float(np.mean(f1s)), cr_means


# ---------------------------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------------------------

def evaluate_overall(
    X: pd.DataFrame,
    y: pd.Series,
    name: str = "KnowFeat",
    n_folds: int = 5,
    seed: int = 42,
) -> MetricsRow:
    """总体评估。"""
    recall, precision, f1, cr = evaluate_with_cv(X, y, n_folds=n_folds, seed=seed)
    return MetricsRow(
        name=name,
        recall=recall,
        precision=precision,
        f1=f1,
        cr_at_k=cr,
        n_samples=len(y),
        n_positives=int(y.sum()),
    )


# ---------------------------------------------------------------------------
# Wallet-type scenario evaluation
# ---------------------------------------------------------------------------

def evaluate_by_wallet_type(
    X: pd.DataFrame,
    y: pd.Series,
    wallet_type_series: pd.Series,
    n_folds: int = 5,
    seed: int = 42,
) -> list[MetricsRow]:
    """按 wallet_type 分场景评估。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵。
    y : pd.Series
        二分类标签。
    wallet_type_series : pd.Series
        每个样本对应的 wallet_type（1/2/3）。
    n_folds : int
    seed : int

    Returns
    -------
    list[MetricsRow]
        每种 wallet_type 对应一行。
    """
    rows: list[MetricsRow] = []
    for wt in sorted(wallet_type_series.unique()):
        mask = wallet_type_series == wt
        X_sub = X.loc[mask].copy()
        y_sub = y.loc[mask]

        if len(y_sub) == 0 or y_sub.sum() == 0:
            label = WALLET_TYPE_LABELS.get(int(wt), f"type_{wt}")
            rows.append(MetricsRow(
                name=label,
                n_samples=len(y_sub),
                n_positives=int(y_sub.sum()),
            ))
            continue

        label = WALLET_TYPE_LABELS.get(int(wt), f"type_{wt}")
        recall, precision, f1, cr = evaluate_with_cv(
            X_sub, y_sub, n_folds=n_folds, seed=seed,
        )
        rows.append(MetricsRow(
            name=label,
            recall=recall,
            precision=precision,
            f1=f1,
            cr_at_k=cr,
            n_samples=len(y_sub),
            n_positives=int(y_sub.sum()),
        ))

    return rows


# ---------------------------------------------------------------------------
# Ablation evaluation
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """单个消融实验配置。"""
    name: str
    feature_names: list[str]


def evaluate_ablation(
    X_full: pd.DataFrame,
    y: pd.Series,
    ablation_configs: list[AblationConfig],
    n_folds: int = 5,
    seed: int = 42,
) -> list[MetricsRow]:
    """消融实验：针对不同特征子集评估指标。

    Parameters
    ----------
    X_full : pd.DataFrame
        包含所有特征的完整矩阵。
    y : pd.Series
        标签。
    ablation_configs : list[AblationConfig]
        每个配置指定一个名称和要使用的特征子集。
    n_folds : int
    seed : int

    Returns
    -------
    list[MetricsRow]
    """
    rows: list[MetricsRow] = []
    for ac in ablation_configs:
        valid_cols = [c for c in ac.feature_names if c in X_full.columns]
        if not valid_cols:
            rows.append(MetricsRow(
                name=ac.name,
                n_samples=len(y),
                n_positives=int(y.sum()),
            ))
            continue

        X_sub = X_full[valid_cols]
        recall, precision, f1, cr = evaluate_with_cv(
            X_sub, y, n_folds=n_folds, seed=seed,
        )
        rows.append(MetricsRow(
            name=ac.name,
            recall=recall,
            precision=precision,
            f1=f1,
            cr_at_k=cr,
            n_samples=len(y),
            n_positives=int(y.sum()),
        ))

    return rows


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

def generate_report(
    X: pd.DataFrame,
    y: pd.Series,
    stage: str = "stage2",
    wallet_type_series: Optional[pd.Series] = None,
    ablation_configs: Optional[list[AblationConfig]] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> ExperimentReport:
    """生成完整实验报告。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵。
    y : pd.Series
        二分类标签。
    stage : str
        阶段标识。
    wallet_type_series : pd.Series, optional
        wallet_type 列（用于分场景评估）。
    ablation_configs : list[AblationConfig], optional
        消融实验配置列表。
    n_folds : int
    seed : int

    Returns
    -------
    ExperimentReport
    """
    report = ExperimentReport(stage=stage)

    logger.info("生成总体评估 ...")
    report.overall = evaluate_overall(X, y, name=f"{stage}_overall", n_folds=n_folds, seed=seed)

    if wallet_type_series is not None and len(wallet_type_series) > 0:
        logger.info("生成 wallet_type 分场景评估 ...")
        report.wallet_type_rows = evaluate_by_wallet_type(
            X, y, wallet_type_series, n_folds=n_folds, seed=seed,
        )

    if ablation_configs:
        logger.info("生成消融实验对比 (%d 配置) ...", len(ablation_configs))
        report.ablation_rows = evaluate_ablation(
            X, y, ablation_configs, n_folds=n_folds, seed=seed,
        )

    return report


# ---------------------------------------------------------------------------
# Formatting: Markdown
# ---------------------------------------------------------------------------

def _metrics_rows_to_markdown(
    title: str,
    rows: list[MetricsRow],
) -> str:
    if not rows:
        return ""
    lines = [f"## {title}\n"]
    k_cols = sorted({k for row in rows for k in row.cr_at_k})
    header_parts = ["| 配置 | Recall | Precision | F1 |"]
    sep_parts = ["|------|--------|-----------|-----|"]
    for k in k_cols:
        header_parts[0] += f" CR@{k}% |"
        sep_parts[0] += "--------|"
    header_parts[0] += " 样本数 | 正样本 |"
    sep_parts[0] += "--------|--------|"

    lines.append(header_parts[0])
    lines.append(sep_parts[0])

    for row in rows:
        parts = (
            f"| {row.name} | {row.recall:.4f} | {row.precision:.4f} | {row.f1:.4f} |"
        )
        for k in k_cols:
            parts += f" {row.cr_at_k.get(k, 0.0):.4f} |"
        parts += f" {row.n_samples} | {row.n_positives} |"
        lines.append(parts)

    lines.append("")
    return "\n".join(lines)


def format_report_markdown(report: ExperimentReport) -> str:
    """格式化实验报告为 Markdown。"""
    sections = [f"# 实验报告 ({report.stage})\n"]

    if report.overall:
        sections.append(_metrics_rows_to_markdown("总体结果", [report.overall]))

    if report.wallet_type_rows:
        sections.append(_metrics_rows_to_markdown(
            "Wallet Type 分场景结果", report.wallet_type_rows,
        ))

    if report.ablation_rows:
        sections.append(_metrics_rows_to_markdown("消融实验对比", report.ablation_rows))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Formatting: CSV
# ---------------------------------------------------------------------------

def _metrics_rows_to_csv_lines(rows: list[MetricsRow]) -> list[list[str]]:
    if not rows:
        return []
    k_cols = sorted({k for row in rows for k in row.cr_at_k})
    header = ["name", "recall", "precision", "f1"]
    for k in k_cols:
        header.append(f"CR@{k}%")
    header.extend(["n_samples", "n_positives"])

    lines = [header]
    for row in rows:
        vals = [row.name, f"{row.recall:.4f}", f"{row.precision:.4f}", f"{row.f1:.4f}"]
        for k in k_cols:
            vals.append(f"{row.cr_at_k.get(k, 0.0):.4f}")
        vals.extend([str(row.n_samples), str(row.n_positives)])
        lines.append(vals)
    return lines


# ---------------------------------------------------------------------------
# Formatting: HTML
# ---------------------------------------------------------------------------

def _table_html(title: str, rows: list[MetricsRow]) -> str:
    if not rows:
        return ""
    k_cols = sorted({k for row in rows for k in row.cr_at_k})
    headers = ["配置", "Recall", "Precision", "F1"]
    for k in k_cols:
        headers.append(f"CR@{k}%")
    headers.extend(["样本数", "正样本"])

    parts = [f"<h2>{html.escape(title)}</h2>", "<table>", "<thead><tr>"]
    for h in headers:
        parts.append(f"<th>{html.escape(h)}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")

    for row in rows:
        parts.append("<tr>")
        parts.append(f"<td>{html.escape(row.name)}</td>")
        parts.append(f"<td>{row.recall:.4f}</td>")
        parts.append(f"<td>{row.precision:.4f}</td>")
        parts.append(f"<td>{row.f1:.4f}</td>")
        for k in k_cols:
            parts.append(f"<td>{row.cr_at_k.get(k, 0.0):.4f}</td>")
        parts.append(f"<td>{row.n_samples}</td>")
        parts.append(f"<td>{row.n_positives}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "\n".join(parts)


def format_report_html(report: ExperimentReport) -> str:
    """格式化实验报告为 HTML。"""
    sections = [
        "<!DOCTYPE html>",
        "<html><head>",
        '<meta charset="utf-8">',
        f"<title>实验报告 ({html.escape(report.stage)})</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 2em; }",
        "table { border-collapse: collapse; margin: 1em 0; }",
        "th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: right; }",
        "th { background: #f5f5f5; }",
        "td:first-child, th:first-child { text-align: left; }",
        "</style>",
        "</head><body>",
        f"<h1>实验报告 ({html.escape(report.stage)})</h1>",
    ]

    if report.overall:
        sections.append(_table_html("总体结果", [report.overall]))

    if report.wallet_type_rows:
        sections.append(_table_html("Wallet Type 分场景结果", report.wallet_type_rows))

    if report.ablation_rows:
        sections.append(_table_html("消融实验对比", report.ablation_rows))

    sections.extend(["</body></html>"])
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_report(
    report: ExperimentReport,
    output_dir: Path,
) -> dict[str, Path]:
    """保存实验报告（CSV + HTML + JSON）。

    Returns
    -------
    dict[str, Path]
        {"csv": ..., "html": ..., "json": ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = report.stage

    # --- CSV ---
    csv_path = output_dir / f"{prefix}_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "recall", "precision", "f1",
                         "CR@1%", "CR@3%", "CR@5%", "n_samples", "n_positives"])
        if report.overall:
            _write_section_csv(writer, "overall", [report.overall])
        if report.wallet_type_rows:
            _write_section_csv(writer, "wallet_type", report.wallet_type_rows)
        if report.ablation_rows:
            _write_section_csv(writer, "ablation", report.ablation_rows)
    logger.info("报告 (CSV) → %s", csv_path)

    # --- HTML ---
    html_path = output_dir / f"{prefix}_report.html"
    html_content = format_report_html(report)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("报告 (HTML) → %s", html_path)

    # --- JSON ---
    json_path = output_dir / f"{prefix}_report.json"
    report_data = _report_to_dict(report)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    logger.info("报告 (JSON) → %s", json_path)

    return {"csv": csv_path, "html": html_path, "json": json_path}


def _write_section_csv(
    writer: Any,
    section: str,
    rows: list[MetricsRow],
) -> None:
    for row in rows:
        writer.writerow([
            section, row.name,
            f"{row.recall:.4f}", f"{row.precision:.4f}", f"{row.f1:.4f}",
            f"{row.cr_at_k.get(1, 0.0):.4f}",
            f"{row.cr_at_k.get(3, 0.0):.4f}",
            f"{row.cr_at_k.get(5, 0.0):.4f}",
            row.n_samples, row.n_positives,
        ])


def _report_to_dict(report: ExperimentReport) -> dict[str, Any]:
    d: dict[str, Any] = {"stage": report.stage}
    if report.overall:
        d["overall"] = report.overall.to_dict()
    if report.wallet_type_rows:
        d["wallet_type"] = [r.to_dict() for r in report.wallet_type_rows]
    if report.ablation_rows:
        d["ablation"] = [r.to_dict() for r in report.ablation_rows]
    return d


def load_report_json(path: Path) -> ExperimentReport:
    """从 JSON 文件反序列化实验报告。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = ExperimentReport(stage=data.get("stage", ""))

    if "overall" in data:
        report.overall = _dict_to_metrics_row(data["overall"])

    if "wallet_type" in data:
        report.wallet_type_rows = [_dict_to_metrics_row(r) for r in data["wallet_type"]]

    if "ablation" in data:
        report.ablation_rows = [_dict_to_metrics_row(r) for r in data["ablation"]]

    return report


def _dict_to_metrics_row(d: dict[str, Any]) -> MetricsRow:
    cr = {}
    for k in CR_AT_K_PERCENTS:
        key = f"CR@{k}%"
        if key in d:
            cr[k] = float(d[key])
    return MetricsRow(
        name=d.get("name", ""),
        recall=float(d.get("recall", 0.0)),
        precision=float(d.get("precision", 0.0)),
        f1=float(d.get("f1", 0.0)),
        cr_at_k=cr,
        n_samples=int(d.get("n_samples", 0)),
        n_positives=int(d.get("n_positives", 0)),
    )
