"""主入口：加载数据 → 运行基线 → 组装 Prompt → 初始化搜索引擎 → 输出结果。"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from src.config import Config, DEFAULT_CONFIG, LLMConfig
from src.data.loader import load_all_tables, extract_schema, extract_label, extract_distribution_diff, extract_join_statistics
from src.data.baseline import run_baseline, get_baseline_feature_names, compute_baseline_prauc, get_baseline_feature_code_samples
from src.prompt.domain_knowledge import (
    load_transaction_patterns,
    load_expert_interview,
    load_risk_indicators,
    load_detection_rules,
    rules_to_python_examples,
)
from src.prompt.builder import build_system_prompt
from src.llm.interface import LLMInterface
from src.search.engine import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 特征搜索主程序")
    parser.add_argument("--provider", default=None, help="LLM provider (openai/gemini/deepseek/local/mock)")
    parser.add_argument("--model", default=None, help="模型名称")
    parser.add_argument("--max-rounds", type=int, default=None, help="最大搜索轮数")
    parser.add_argument("--data-dir", default=None, help="数据目录")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.provider:
        cfg.llm.provider = args.provider
    if args.model:
        cfg.llm.model_name = args.model
    if args.max_rounds:
        cfg.search.T = args.max_rounds
    if args.data_dir:
        cfg.paths.data_dir = args.data_dir
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
    cfg.llm.__post_init__()
    return cfg


def load_domain_knowledge(cfg: Config) -> tuple[str, str, str, str]:
    transaction_patterns = load_transaction_patterns()
    expert_knowledge = load_expert_interview()

    risk_indicators_path = os.path.join(cfg.paths.domain_knowledge_dir, "风险分类表_38指标.xlsx")
    try:
        risk_indicators = load_risk_indicators(risk_indicators_path)
    except FileNotFoundError:
        logger.warning("风险分类表未找到: %s，使用空文本", risk_indicators_path)
        risk_indicators = "（风险分类表未提供）"

    rules_path = os.path.join(cfg.paths.domain_knowledge_dir, "检测规则_15条.xlsx")
    try:
        rules = load_detection_rules(rules_path)
        seed_examples = rules_to_python_examples(rules, n=2)
    except FileNotFoundError:
        logger.warning("检测规则未找到: %s，使用空文本", rules_path)
        seed_examples = "（检测规则示例未提供）"

    return transaction_patterns, expert_knowledge, risk_indicators, seed_examples


def _strip_redundant_imports(code: str) -> str:
    """去除代码中重复的 import pandas/numpy（函数头已包含）。"""
    lines = code.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("import pandas as pd", "import numpy as np",
                        "import pandas", "import numpy"):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def _generate_feature_file(feature_name: str, code: str, explanation: str,
                           l2_metrics: dict, l3_metrics: dict, score: float) -> str:
    """将 code 片段包装为独立可执行的 .py 文件内容。"""
    clean_code = _strip_redundant_imports(code)
    indented_code = "\n".join("    " + line if line.strip() else ""
                              for line in clean_code.split("\n"))

    mi = l2_metrics.get("mi", 0)
    redundancy = l2_metrics.get("redundancy", 0)
    cramers_v = l2_metrics.get("cramers_v", 0)
    delta_prauc = l3_metrics.get("delta_prauc", 0)
    shap_rank = l3_metrics.get("shap_rank", 0)

    content = f'''"""
特征: {feature_name}
说明: {explanation}

验证指标:
  - Score: {score:.4f}
  - MI: {mi:.4f}
  - Redundancy: {redundancy:.4f}
  - Cramer's V: {cramers_v:.4f}
  - Delta PR-AUC: {delta_prauc:.6f}
  - SHAP Rank: {shap_rank}

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 {feature_name}，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
{indented_code}
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: {feature_name}")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{{args.data_dir}}/accounts.csv")
    tx_df = pd.read_csv(f"{{args.data_dir}}/transactions.csv")
    close_df = pd.read_csv(f"{{args.data_dir}}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 {feature_name}: {{len(result)}} 行, 列={{list(result.columns)}}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {{args.output}}")
'''
    return content


def export_feature_files(features_data: list[dict], output_dir: str) -> list[str]:
    """将通过验证的特征导出为独立 .py 文件，返回文件路径列表。"""
    features_dir = Path(output_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for f in features_data:
        name = f.get("feature_name", "unknown")
        code = f.get("code", "")
        if not code:
            continue

        content = _generate_feature_file(
            feature_name=name,
            code=code,
            explanation=f.get("explanation", ""),
            l2_metrics=f.get("l2_metrics", {}),
            l3_metrics=f.get("l3_metrics", {}),
            score=f.get("score", 0),
        )

        file_path = features_dir / f"{name}.py"
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write(content)
        exported.append(str(file_path))

    return exported


def save_results(result, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    features_path = out / "features.json"
    features_data = []
    for f in result.accepted_features:
        record = {
            "feature_name": f.get("feature_name", f.get("name", "")),
            "score": f.get("score", 0),
            "code": f.get("code", ""),
            "explanation": f.get("explanation", ""),
            "l2_metrics": f.get("l2_metrics", {}),
            "l3_metrics": f.get("l3_metrics", {}),
        }
        features_data.append(record)
    with open(features_path, "w", encoding="utf-8") as fp:
        json.dump(features_data, fp, ensure_ascii=False, indent=2)
    logger.info("特征结果已保存: %s (%d 个特征)", features_path, len(features_data))

    exported = export_feature_files(features_data, output_dir)
    for path in exported:
        logger.info("特征文件已导出: %s", path)

    log_path = out / "search_log.json"
    log_data = {
        "total_rounds": result.total_rounds,
        "converged": result.converged,
        "total_features": len(result.accepted_features),
        "rounds": result.search_log,
    }
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(log_data, fp, ensure_ascii=False, indent=2)
    logger.info("搜索日志已保存: %s", log_path)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    logger.info("=== 加载数据 ===")
    open_df, tx_df, close_df = load_all_tables(cfg.paths.data_dir)
    label = extract_label(tx_df)
    logger.info("数据加载完成: open=%d, tx=%d, close=%d", len(open_df), len(tx_df), len(close_df))

    logger.info("=== 提取 Schema ===")
    schema_parts = [
        extract_schema(open_df, "开户表(accounts)"),
        extract_schema(tx_df, "交易表(transactions)"),
        extract_schema(close_df, "销户表(close)"),
    ]
    schema_text = "\n\n".join(schema_parts)

    logger.info("=== 运行基线特征 ===")
    baseline_feature_names = get_baseline_feature_names()
    try:
        baseline_df = run_baseline(open_df, tx_df, close_df)
        available_cols = [c for c in baseline_feature_names if c in baseline_df.columns]
        X_baseline = baseline_df[available_cols].fillna(0)
        baseline_prauc = compute_baseline_prauc(X_baseline, label)
        logger.info("基线 PR-AUC: %.4f (%d 个特征)", baseline_prauc, len(available_cols))
    except Exception as e:
        logger.warning("基线特征提取失败: %s，使用空基线", e)
        X_baseline = pd.DataFrame(index=range(len(label)))
        baseline_prauc = 0.0

    logger.info("=== 加载领域知识 ===")
    transaction_patterns, expert_knowledge, risk_indicators, seed_examples = load_domain_knowledge(cfg)

    logger.info("=== 提取分布差异与关联统计 ===")
    distribution_diff = extract_distribution_diff(open_df, tx_df, close_df, label)
    join_statistics = extract_join_statistics(open_df, tx_df, close_df)
    baseline_code_samples = get_baseline_feature_code_samples()
    logger.info("分布差异: %d chars, 关联统计: %d chars", len(distribution_diff), len(join_statistics))

    logger.info("=== 组装 System Prompt ===")
    system_prompt = build_system_prompt(
        schema_text=schema_text,
        transaction_patterns=transaction_patterns,
        expert_knowledge=expert_knowledge,
        risk_indicators=risk_indicators,
        baseline_features=baseline_feature_names,
        distribution_diff=distribution_diff,
        join_statistics=join_statistics,
        baseline_code_samples=baseline_code_samples,
    )

    logger.info("=== 初始化 LLM ===")
    llm = LLMInterface.from_config(cfg.llm)
    logger.info("LLM provider: %s, model: %s", cfg.llm.provider, cfg.llm.model_name)

    logger.info("=== 初始化搜索引擎 ===")
    engine = SearchEngine(
        llm=llm,
        system_prompt=system_prompt,
        seed_examples=seed_examples,
        open_df=open_df,
        tx_df=tx_df,
        close_df=close_df,
        label=label,
        baseline_features=X_baseline,
        baseline_prauc=baseline_prauc,
        config=cfg,
    )

    logger.info("=== 开始搜索 (最多 %d 轮) ===", cfg.search.T)
    result = engine.run(max_rounds=cfg.search.T)
    logger.info(
        "搜索完成: %d 轮, %d 个特征通过验证, 收敛=%s",
        result.total_rounds, len(result.accepted_features), result.converged,
    )

    logger.info("=== 保存结果 ===")
    save_results(result, cfg.paths.output_dir)

    logger.info("=== 完成 ===")


if __name__ == "__main__":
    main()
