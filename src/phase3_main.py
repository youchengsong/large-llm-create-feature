"""Phase 3 主入口：加载 Phase 2 产出 → 重建特征矩阵 → L4 稳定性 → 5 组实验 → 特征字典 → 可视化 → 报告。"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.data.loader import load_all_tables, extract_label
from src.data.baseline import run_baseline, get_baseline_feature_names
from src.data.merge import merge_feature_to_transactions
from src.evaluation.sandbox import execute_feature_code
from src.evaluation.stability import run_l4_stability
from src.evaluation.experiments import run_ablation_experiments
from src.report.feature_dictionary import generate_feature_dictionary, export_feature_dictionary
from src.report.visualization import generate_all_plots
from src.report.generator import export_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: 最终验证与产出")
    parser.add_argument(
        "--features-json",
        default=None,
        help="Phase 2 产出的 features.json 路径（默认: <output-dir>/features.json）",
    )
    parser.add_argument("--data-dir", default=None, help="数据目录")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    return parser.parse_args()


def load_features_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"features.json 应为列表，实际为 {type(data).__name__}")
    return data


def load_search_log(output_dir: str) -> list[dict]:
    log_path = Path(output_dir) / "search_log.json"
    if not log_path.exists():
        logger.warning("search_log.json 未找到，使用空日志")
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rounds", []) if isinstance(data, dict) else data


def rebuild_feature_matrix(
    features_data: list[dict],
    open_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    close_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """重新执行每个特征的代码，重建 LLM 特征矩阵。"""
    llm_features = {}
    successful_names = []

    for feat in features_data:
        name = feat.get("feature_name", "unknown")
        code = feat.get("code", "")
        if not code:
            logger.warning("特征 %s 无代码，跳过", name)
            continue

        logger.info("执行特征代码: %s", name)
        feature_df, error = execute_feature_code(code, open_df, tx_df, close_df, timeout=60)
        if error:
            logger.warning("特征 %s 执行失败: %s", name, error)
            continue

        merged = merge_feature_to_transactions(feature_df, tx_df)
        id_col = "tx_id" if "tx_id" in feature_df.columns else "wallet_id"
        feat_cols = [c for c in feature_df.columns if c != id_col]

        if "wallet_id" in feature_df.columns and "tx_id" not in feature_df.columns:
            feat_cols = [f"{c}_src" for c in feat_cols] + [f"{c}_dst" for c in feat_cols]

        for col in feat_cols:
            if col in merged.columns:
                llm_features[col] = merged[col].fillna(0).values
                successful_names.append(col)

    if not llm_features:
        return pd.DataFrame(index=range(len(tx_df))), []

    X_llm = pd.DataFrame(llm_features, index=range(len(tx_df)))
    return X_llm, successful_names


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir or DEFAULT_CONFIG.paths.data_dir
    output_dir = args.output_dir or DEFAULT_CONFIG.paths.output_dir
    features_json_path = args.features_json or str(Path(output_dir) / "features.json")

    logger.info("=== Phase 3: 最终验证与产出 ===")

    logger.info("加载 features.json: %s", features_json_path)
    features_data = load_features_json(features_json_path)
    logger.info("共 %d 个特征", len(features_data))

    logger.info("加载数据...")
    open_df, tx_df, close_df = load_all_tables(data_dir)
    label = extract_label(tx_df)
    logger.info("数据: open=%d, tx=%d, close=%d", len(open_df), len(tx_df), len(close_df))

    logger.info("=== 运行基线特征 ===")
    baseline_feature_names = get_baseline_feature_names()
    try:
        baseline_df = run_baseline(open_df, tx_df, close_df)
        available_baseline = [c for c in baseline_feature_names if c in baseline_df.columns]
        X_baseline = baseline_df[available_baseline].fillna(0)
        logger.info("基线特征: %d 个", len(available_baseline))
    except Exception as e:
        logger.warning("基线特征提取失败: %s，使用空基线", e)
        X_baseline = pd.DataFrame(index=range(len(label)))
        available_baseline = []

    logger.info("=== 重建 LLM 特征矩阵 ===")
    X_llm, llm_feature_names = rebuild_feature_matrix(features_data, open_df, tx_df, close_df)
    logger.info("LLM 特征: %d 个", len(llm_feature_names))

    if not llm_feature_names:
        logger.error("无可用 LLM 特征，无法继续")
        sys.exit(1)

    X_final = pd.concat(
        [X_baseline.reset_index(drop=True), X_llm.reset_index(drop=True)],
        axis=1,
    )
    X_final = X_final.fillna(0)
    logger.info("最终特征矩阵: %d 行 x %d 列", X_final.shape[0], X_final.shape[1])

    logger.info("=== L4 稳定性验证 ===")
    stability_results = run_l4_stability(X_final, label)
    logger.info(
        "稳定性指数: %.4f, 通过: %s",
        stability_results["stability_index"],
        stability_results["pass"],
    )

    logger.info("=== 5 组增量贡献实验 ===")
    experiment_results = run_ablation_experiments(
        X_baseline=X_baseline,
        X_llm=X_llm,
        y=label,
        baseline_feature_names=available_baseline,
        llm_feature_names=llm_feature_names,
    )
    summary = experiment_results.get("summary", {})
    logger.info(
        "增量 PR-AUC: %.6f, LLM 独立: %.4f",
        summary.get("delta_prauc", 0),
        summary.get("llm_independent_prauc", 0),
    )

    logger.info("=== 生成特征字典 ===")
    feature_dictionary = generate_feature_dictionary(features_data, X_final, label)
    dict_output_path = str(Path(output_dir) / "feature_dictionary")
    export_feature_dictionary(feature_dictionary, dict_output_path)
    logger.info("特征字典已导出: %s", dict_output_path)

    logger.info("=== 生成可视化 ===")
    plots_dir = str(Path(output_dir) / "plots")
    plot_paths = generate_all_plots(
        X_final=X_final,
        y=label,
        experiment_results=experiment_results,
        stability_results=stability_results,
        features_data=features_data,
        output_dir=plots_dir,
    )
    logger.info("生成 %d 张图表", len(plot_paths))

    logger.info("=== 生成验证报告 ===")
    search_log = load_search_log(output_dir)
    report_path = export_report(
        stability_results=stability_results,
        experiment_results=experiment_results,
        feature_dictionary=feature_dictionary,
        search_log=search_log,
        output_path=str(Path(output_dir) / "report.md"),
        plot_paths=plot_paths,
    )
    logger.info("验证报告已导出: %s", report_path)

    logger.info("=== Phase 3 完成 ===")
    logger.info("输出目录: %s", output_dir)
    logger.info("  - 特征字典: %s.json / %s.md", dict_output_path, dict_output_path)
    logger.info("  - 图表目录: %s/", plots_dir)
    logger.info("  - 验证报告: %s", report_path)


if __name__ == "__main__":
    main()
