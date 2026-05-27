"""自动模型选择：全量运行 ModelSet 11 个模型，推荐最优模型。

流程:
  1. 调用 ModelSet/runner.py 全量跑 11 模型（train+test）
  2. 解析 results CSV，汇总各模型指标
  3. 分析数据特征（样本量、不平衡比、特征维度、特征类型分布）
  4. 按 Recall（Precision ≥ 90% 约束）自动推荐最优模型
  5. 输出模型对比表 + 推荐理由
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from knowfeat.config import KnowFeatConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

ALL_MODELS = [
    "catboost", "ga2m", "scorecard", "acxgb", "wav_xgb",
    "ensemble", "lgbm", "resample_boost", "gbdt_fm", "dnn_res", "mlp",
]

MODELSET_DIR = PROJECT_ROOT / "ModelSet"
RUNNER_SCRIPT = MODELSET_DIR / "runner.py"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """单个模型的评估结果。"""
    model_name: str
    precision: float
    recall: float
    f1_score: float
    auc: float
    threshold: float
    start_time: str = ""
    end_time: str = ""
    meets_precision: bool = False


@dataclass
class DataProfile:
    """数据集特征概览。"""
    n_samples: int
    n_features: int
    imbalance_ratio: float
    pos_count: int
    neg_count: int
    numeric_count: int
    categorical_count: int
    missing_rate: float
    feature_names: list[str] = field(default_factory=list)


@dataclass
class ModelSelectionResult:
    """模型选择综合结果。"""
    recommended_model: str
    recommendation_reason: str
    data_profile: DataProfile
    all_results: list[ModelResult] = field(default_factory=list)
    qualified_results: list[ModelResult] = field(default_factory=list)
    precision_min: float = 0.90


# ---------------------------------------------------------------------------
# Results parsing
# ---------------------------------------------------------------------------

def parse_results_csv(
    csv_path: Path,
    precision_min: float = 0.90,
) -> list[ModelResult]:
    """解析 ModelSet 输出的 global_results.csv。"""
    results: list[ModelResult] = []
    path = Path(csv_path)
    if not path.exists():
        logger.warning("结果文件不存在: %s", path)
        return results

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                precision = float(row.get("precision", 0))
                recall = float(row.get("recall", 0))
                f1 = float(row.get("f1_score", 0))
                auc = float(row.get("auc", 0))
                threshold = float(row.get("threshold", 0))
            except (ValueError, TypeError):
                logger.warning("跳过无效行: %s", row)
                continue

            mr = ModelResult(
                model_name=row.get("model_name", "unknown"),
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc=auc,
                threshold=threshold,
                start_time=row.get("start_time", ""),
                end_time=row.get("end_time", ""),
                meets_precision=precision >= precision_min,
            )
            results.append(mr)

    logger.info("解析 %d 条模型结果 from %s", len(results), path)
    return results


# ---------------------------------------------------------------------------
# Data profiling
# ---------------------------------------------------------------------------

def analyze_data_profile(
    X: pd.DataFrame,
    y: pd.Series,
) -> DataProfile:
    """分析数据特征概览。"""
    n_samples = len(y)
    n_features = X.shape[1]
    pos_count = int(y.sum())
    neg_count = n_samples - pos_count
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else float("inf")

    numeric_count = 0
    categorical_count = 0
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_count += 1
        else:
            categorical_count += 1

    total_cells = n_samples * n_features
    missing_rate = float(X.isnull().sum().sum() / total_cells) if total_cells > 0 else 0.0

    return DataProfile(
        n_samples=n_samples,
        n_features=n_features,
        imbalance_ratio=round(imbalance_ratio, 2),
        pos_count=pos_count,
        neg_count=neg_count,
        numeric_count=numeric_count,
        categorical_count=categorical_count,
        missing_rate=round(missing_rate, 4),
        feature_names=list(X.columns),
    )


# ---------------------------------------------------------------------------
# Model selection logic
# ---------------------------------------------------------------------------

def _build_recommendation_reason(
    best: ModelResult,
    data_profile: DataProfile,
    qualified: list[ModelResult],
    all_results: list[ModelResult],
) -> str:
    """构建推荐理由文本（数据特征 → 模型适配分析）。"""
    parts = []

    parts.append(
        f"数据概况: {data_profile.n_samples} 样本, "
        f"{data_profile.n_features} 特征 "
        f"({data_profile.numeric_count} 数值 + {data_profile.categorical_count} 类别), "
        f"不平衡比 1:{data_profile.imbalance_ratio:.0f} "
        f"(正样本 {data_profile.pos_count}, 负样本 {data_profile.neg_count})"
    )

    n_qualified = len(qualified)
    n_total = len(all_results)
    parts.append(
        f"满足 Precision 约束的模型: {n_qualified}/{n_total}"
    )

    parts.append(
        f"推荐模型: {best.model_name} "
        f"(Recall={best.recall:.4f}, Precision={best.precision:.4f}, "
        f"F1={best.f1_score:.4f}, AUC={best.auc:.4f})"
    )

    if data_profile.imbalance_ratio > 20:
        parts.append(
            "该数据集严重不平衡, 树类集成模型（CatBoost/LightGBM/XGBoost变体）"
            "通常表现更好, 因为它们内置不平衡处理机制"
        )
    elif data_profile.imbalance_ratio > 5:
        parts.append(
            "该数据集中等不平衡, 带重采样的集成模型通常有优势"
        )

    if data_profile.n_features > 50:
        parts.append(
            "特征维度较高, 正则化能力强的模型（如树模型、GA2M）更适合"
        )

    if data_profile.categorical_count > data_profile.numeric_count:
        parts.append(
            "类别特征占比高, CatBoost 原生支持类别特征, 具有天然优势"
        )

    return "; ".join(parts)


def select_best_model(
    results: list[ModelResult],
    precision_min: float = 0.90,
    data_profile: Optional[DataProfile] = None,
) -> ModelSelectionResult:
    """从模型结果中选择最优模型。

    选择标准: Recall 最高（Precision ≥ precision_min 约束）。
    若无模型满足 Precision 约束, 则从全部结果中选 Recall 最高的并附说明。
    """
    if data_profile is None:
        data_profile = DataProfile(
            n_samples=0, n_features=0, imbalance_ratio=0.0,
            pos_count=0, neg_count=0, numeric_count=0,
            categorical_count=0, missing_rate=0.0,
        )

    if not results:
        return ModelSelectionResult(
            recommended_model="",
            recommendation_reason="无模型结果可供选择",
            data_profile=data_profile,
            precision_min=precision_min,
        )

    for r in results:
        r.meets_precision = r.precision >= precision_min

    qualified = [r for r in results if r.meets_precision]
    qualified.sort(key=lambda r: r.recall, reverse=True)

    if qualified:
        best = qualified[0]
        reason = _build_recommendation_reason(best, data_profile, qualified, results)
    else:
        all_sorted = sorted(results, key=lambda r: r.recall, reverse=True)
        best = all_sorted[0]
        reason = (
            f"无模型满足 Precision≥{precision_min} 约束, "
            f"退而选择 Recall 最高的 {best.model_name} "
            f"(Recall={best.recall:.4f}, Precision={best.precision:.4f}); "
            + _build_recommendation_reason(best, data_profile, [], results)
        )

    return ModelSelectionResult(
        recommended_model=best.model_name,
        recommendation_reason=reason,
        data_profile=data_profile,
        all_results=sorted(results, key=lambda r: r.recall, reverse=True),
        qualified_results=qualified,
        precision_min=precision_min,
    )


# ---------------------------------------------------------------------------
# Runner invocation
# ---------------------------------------------------------------------------

def run_modelset(
    train_csv: Path,
    test_csv: Path,
    output_csv: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    models: Optional[list[str]] = None,
    timeout_sec: int = 3600,
) -> Path:
    """调用 ModelSet/runner.py 训练和测试所有模型。

    Returns
    -------
    Path
        结果 CSV 文件路径。
    """
    if output_csv is None:
        output_csv = MODELSET_DIR / "results" / "global_results.csv"
    if model_dir is None:
        model_dir = MODELSET_DIR / "saved_models"
    if models is None:
        models = ALL_MODELS

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(RUNNER_SCRIPT),
        "--model", *models,
        "--mode", "both",
        "--train_data", str(train_csv),
        "--test_data", str(test_csv),
        "--model_dir", str(model_dir),
        "--output", str(output_csv),
    ]

    logger.info("启动 ModelSet: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=str(MODELSET_DIR),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )

    if proc.returncode != 0:
        logger.error("ModelSet 执行失败:\nstdout: %s\nstderr: %s",
                      proc.stdout[-2000:] if proc.stdout else "",
                      proc.stderr[-2000:] if proc.stderr else "")
        raise RuntimeError(
            f"ModelSet runner 退出码 {proc.returncode}: "
            f"{proc.stderr[-500:] if proc.stderr else 'unknown error'}"
        )

    logger.info("ModelSet 完成, 输出: %s", output_csv)
    return output_csv


# ---------------------------------------------------------------------------
# Formatting & export
# ---------------------------------------------------------------------------

def format_comparison_table(result: ModelSelectionResult) -> str:
    """格式化模型对比表（Markdown 格式）。"""
    lines = ["## 模型对比表\n"]
    lines.append(
        f"Precision 约束: ≥ {result.precision_min:.0%} | "
        f"推荐模型: **{result.recommended_model}**\n"
    )

    header = "| 排名 | 模型 | Recall | Precision | F1 | AUC | 阈值 | 达标 |"
    sep = "|------|------|--------|-----------|-----|-----|------|------|"
    lines.append(header)
    lines.append(sep)

    for i, mr in enumerate(result.all_results, 1):
        mark = "**" if mr.model_name == result.recommended_model else ""
        check = "✓" if mr.meets_precision else "✗"
        lines.append(
            f"| {i} | {mark}{mr.model_name}{mark} | "
            f"{mr.recall:.4f} | {mr.precision:.4f} | "
            f"{mr.f1_score:.4f} | {mr.auc:.4f} | "
            f"{mr.threshold:.4f} | {check} |"
        )

    lines.append("")
    lines.append("## 推荐理由\n")
    lines.append(result.recommendation_reason)

    return "\n".join(lines)


def save_selection_report(
    result: ModelSelectionResult,
    output_dir: Path,
) -> dict[str, Path]:
    """保存模型选择报告（JSON + Markdown）。

    Returns
    -------
    dict[str, Path]
        {"json": ..., "markdown": ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "recommended_model": result.recommended_model,
        "recommendation_reason": result.recommendation_reason,
        "precision_min": result.precision_min,
        "data_profile": {
            "n_samples": result.data_profile.n_samples,
            "n_features": result.data_profile.n_features,
            "imbalance_ratio": result.data_profile.imbalance_ratio,
            "pos_count": result.data_profile.pos_count,
            "neg_count": result.data_profile.neg_count,
            "numeric_count": result.data_profile.numeric_count,
            "categorical_count": result.data_profile.categorical_count,
            "missing_rate": result.data_profile.missing_rate,
        },
        "all_results": [
            {
                "model_name": mr.model_name,
                "recall": mr.recall,
                "precision": mr.precision,
                "f1_score": mr.f1_score,
                "auc": mr.auc,
                "threshold": mr.threshold,
                "meets_precision": mr.meets_precision,
            }
            for mr in result.all_results
        ],
        "qualified_count": len(result.qualified_results),
    }

    json_path = output_dir / "model_selection.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    logger.info("模型选择报告 (JSON) → %s", json_path)

    md_path = output_dir / "model_selection.md"
    md_content = format_comparison_table(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("模型选择报告 (Markdown) → %s", md_path)

    csv_path = output_dir / "model_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "model_name", "recall", "precision",
            "f1_score", "auc", "threshold", "meets_precision",
        ])
        for i, mr in enumerate(result.all_results, 1):
            writer.writerow([
                i, mr.model_name, f"{mr.recall:.4f}", f"{mr.precision:.4f}",
                f"{mr.f1_score:.4f}", f"{mr.auc:.4f}",
                f"{mr.threshold:.4f}", mr.meets_precision,
            ])
    logger.info("模型对比表 (CSV) → %s", csv_path)

    return {"json": json_path, "markdown": md_path, "csv": csv_path}


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def run_model_selection(
    train_csv: Path,
    test_csv: Path,
    output_dir: Optional[Path] = None,
    precision_min: float = 0.90,
    data_X: Optional[pd.DataFrame] = None,
    data_y: Optional[pd.Series] = None,
    models: Optional[list[str]] = None,
    timeout_sec: int = 3600,
) -> ModelSelectionResult:
    """端到端模型选择：训练 → 测试 → 分析 → 推荐。

    Parameters
    ----------
    train_csv : Path
        训练数据 CSV。
    test_csv : Path
        测试数据 CSV。
    output_dir : Path, optional
        报告输出目录。默认为 outputs/。
    precision_min : float
        Precision 约束下限。
    data_X : pd.DataFrame, optional
        特征矩阵（用于数据分析）。如为 None，从 train_csv 加载。
    data_y : pd.Series, optional
        标签列。如为 None，从 train_csv 加载。
    models : list[str], optional
        要运行的模型列表。默认全部 11 个。
    timeout_sec : int
        ModelSet 运行超时秒数。

    Returns
    -------
    ModelSelectionResult
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs"

    results_csv = run_modelset(
        train_csv=train_csv,
        test_csv=test_csv,
        models=models,
        timeout_sec=timeout_sec,
    )

    results = parse_results_csv(results_csv, precision_min=precision_min)

    if data_X is not None and data_y is not None:
        data_profile = analyze_data_profile(data_X, data_y)
    else:
        try:
            df = pd.read_csv(train_csv)
            label_col = "is_risk" if "is_risk" in df.columns else df.columns[-1]
            data_y_loaded = df[label_col]
            data_X_loaded = df.drop(columns=[label_col])
            data_profile = analyze_data_profile(data_X_loaded, data_y_loaded)
        except Exception as e:
            logger.warning("无法从训练数据分析数据概况: %s", e)
            data_profile = DataProfile(
                n_samples=0, n_features=0, imbalance_ratio=0.0,
                pos_count=0, neg_count=0, numeric_count=0,
                categorical_count=0, missing_rate=0.0,
            )

    selection = select_best_model(results, precision_min, data_profile)
    save_selection_report(selection, output_dir)

    logger.info(
        "模型选择完成: 推荐 %s (Recall=%.4f, Precision=%.4f)",
        selection.recommended_model,
        next((r.recall for r in results if r.model_name == selection.recommended_model), 0),
        next((r.precision for r in results if r.model_name == selection.recommended_model), 0),
    )
    return selection


def select_from_existing_results(
    results_csv: Path,
    precision_min: float = 0.90,
    data_X: Optional[pd.DataFrame] = None,
    data_y: Optional[pd.Series] = None,
    output_dir: Optional[Path] = None,
) -> ModelSelectionResult:
    """从已有的 results CSV 选择最优模型（不重新训练）。"""
    results = parse_results_csv(results_csv, precision_min=precision_min)

    data_profile = None
    if data_X is not None and data_y is not None:
        data_profile = analyze_data_profile(data_X, data_y)

    selection = select_best_model(results, precision_min, data_profile)

    if output_dir is not None:
        save_selection_report(selection, output_dir)

    return selection
