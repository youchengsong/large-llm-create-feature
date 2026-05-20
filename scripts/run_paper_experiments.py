#!/usr/bin/env python3
"""论文实验脚本：运行所有对比实验和消融实验。

实验矩阵：
  1. 多模型验证（XGBoost / LightGBM / RF / LR）
  2. 方法对比（Full Agent vs CAAFE-style vs 随机特征）
  3. 消融实验（去领域知识 / 去SHAP反馈 / 去候选缓冲池+前向选择）
  4. 多次运行稳定性（同配置跑 N 次，报告均值±标准差）

用法:
  # 只跑离线评估（不需要 LLM API，用已有的 Agent 特征）
  python scripts/run_paper_experiments.py --mode offline --agent-result outputs/single_agent_run/result_xxx.json

  # 跑消融实验（需要 LLM API）
  python scripts/run_paper_experiments.py --mode ablation --n-runs 3

  # 跑全部
  python scripts/run_paper_experiments.py --mode all --n-runs 3
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_CONFIG, Config, SearchParams, PendingPoolConfig
from src.data.baseline import run_baseline, compute_baseline_prauc
from src.data.loader import load_data, extract_distribution_diff, extract_schema, extract_join_statistics
from src.evaluation.calibration import calibrate_l2_thresholds
from src.evaluation.experiments import run_ablation_experiments
from src.evaluation.metrics import compute_cv_metrics
from src.evaluation.multi_model import evaluate_multi_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 数据加载（所有实验共用）
# ============================================================

def load_experiment_data(data_dir: str = "data"):
    """加载数据、基线特征、标签。"""
    logger.info("Loading data from %s ...", data_dir)
    open_df, tx_df, close_df = load_data(data_dir)

    logger.info("Computing baseline features ...")
    baseline_df = run_baseline(open_df, tx_df, close_df)

    label_col = "is_risk"
    if label_col not in tx_df.columns:
        for alt in ["is_laundering", "is_abnormal", "is_zs_laundering"]:
            if alt in tx_df.columns:
                label_col = alt
                break

    y = tx_df[label_col].astype(int)

    exclude = {"tx_id", "wallet_id", "src", "dst", "timestamp",
               "is_risk", "is_laundering", "is_abnormal",
               "is_zs_laundering", "is_src_victim"}
    feature_cols = [c for c in baseline_df.columns if c not in exclude]
    X_baseline = baseline_df[feature_cols].fillna(0)

    return open_df, tx_df, close_df, X_baseline, y, feature_cols


def load_agent_features(result_path: str, tx_df: pd.DataFrame) -> pd.DataFrame:
    """从 Agent 运行结果 JSON 加载生成的特征。"""
    with open(result_path) as f:
        result = json.load(f)

    accepted = result.get("accepted_features", [])
    if not accepted:
        raise ValueError(f"No accepted features in {result_path}")

    feature_dfs = []
    for feat in accepted:
        code = feat["code"]
        name = feat["name"]
        logger.info("  Executing feature: %s", name)

        from src.evaluation.sandbox import execute_feature_code
        open_df_g = globals().get("_open_df")
        tx_df_g = globals().get("_tx_df")
        close_df_g = globals().get("_close_df")
        feat_df, err = execute_feature_code(code, open_df_g, tx_df_g, close_df_g)
        if err:
            logger.warning("  Failed to execute %s: %s", name, err)
            continue

        if "tx_id" in feat_df.columns:
            feat_df = feat_df.set_index("tx_id")
        elif "wallet_id" in feat_df.columns:
            feat_df = feat_df.set_index("wallet_id")

        feature_dfs.append(feat_df)

    if not feature_dfs:
        raise ValueError("No features could be executed")

    return pd.concat(feature_dfs, axis=1).fillna(0)


# ============================================================
# 实验 1：多模型验证
# ============================================================

def experiment_multi_model(X_baseline, X_agent, y, output_dir):
    """在 4 个模型上评估基线 vs 增强。"""
    logger.info("=" * 60)
    logger.info("Experiment 1: Multi-Model Validation")
    logger.info("=" * 60)

    results = evaluate_multi_model(X_baseline, X_agent, y)

    out_path = os.path.join(output_dir, "exp1_multi_model.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)

    print("\n=== Multi-Model Results ===")
    print(f"{'Model':<20} {'Base Recall':>12} {'Aug Recall':>12} {'ΔRecall':>10} {'ΔF1':>10}")
    print("-" * 64)
    for model_name, m in results.items():
        if "error" in m:
            print(f"{model_name:<20} ERROR: {m['error']}")
            continue
        b = m["baseline"]
        a = m["augmented"]
        d = m["delta"]
        print(f"{model_name:<20} {b['recall_mean']:>12.4f} {a['recall_mean']:>12.4f} "
              f"{d['delta_recall']:>+10.4f} {d['delta_f1']:>+10.4f}")

    return results


# ============================================================
# 实验 2：方法对比（CAAFE-style baseline）
# ============================================================

def _run_caafe_style_baseline(
    open_df, tx_df, close_df, label, baseline_features, X_baseline, y,
    api_key: str, base_url: str, model: str,
) -> pd.DataFrame:
    """CAAFE 风格的单轮生成基线：无多轮反馈、无验证循环、无领域知识。

    模拟 CAAFE 的方式：给 LLM 数据 schema + 少量示例，一次性生成特征，
    不做 L2/L3 验证反馈，不做前向选择。
    """
    from openai import OpenAI

    schema_text = extract_schema(open_df, tx_df, close_df)

    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = f"""You are a feature engineer. Given the following data schema, generate 10 features for fraud detection.
Each feature should be a Python function with signature: def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame

{schema_text}

The target is binary classification (is_risk column in tx_df).
Return ONLY Python code, one function per feature, each ending with:
result = feature_xxx(open_df, tx_df, close_df)

Generate diverse features covering: amount patterns, time patterns, behavior sequences, entity associations.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8000,
    )

    code_text = response.choices[0].message.content

    import re
    functions = re.findall(
        r'(def feature_\w+\(open_df.*?\n(?:.*?\n)*?result\s*=\s*feature_\w+\(.*?\))',
        code_text, re.MULTILINE
    )

    from src.evaluation.sandbox import execute_feature_code

    feature_dfs = []
    for func_code in functions[:10]:
        try:
            feat_df, err = execute_feature_code(func_code, open_df, tx_df, close_df, timeout=60)
            if err:
                continue
            feature_dfs.append(feat_df)
        except Exception:
            continue

    if not feature_dfs:
        return pd.DataFrame()

    return pd.concat(feature_dfs, axis=1).fillna(0)


def experiment_method_comparison(
    open_df, tx_df, close_df, label,
    X_baseline, X_agent, y,
    feature_cols, agent_feature_names,
    api_key, base_url, model,
    output_dir,
):
    """对比 Full Agent vs CAAFE-style vs 随机特征。"""
    logger.info("=" * 60)
    logger.info("Experiment 2: Method Comparison")
    logger.info("=" * 60)

    results = {}

    # 2a. 基线（仅人工特征）
    base_metrics = compute_cv_metrics(X_baseline, y)
    results["baseline_only"] = base_metrics
    logger.info("Baseline: F1=%.4f, Recall=%.4f", base_metrics["f1"], base_metrics["recall"])

    # 2b. Full Agent（已有结果）
    X_aug = pd.concat([X_baseline, X_agent], axis=1).fillna(0)
    aug_metrics = compute_cv_metrics(X_aug, y)
    results["full_agent"] = aug_metrics
    logger.info("Full Agent: F1=%.4f, Recall=%.4f", aug_metrics["f1"], aug_metrics["recall"])

    # 2c. CAAFE-style（单轮生成，无反馈）
    logger.info("Running CAAFE-style baseline ...")
    try:
        X_caafe = _run_caafe_style_baseline(
            open_df, tx_df, close_df, label,
            X_baseline, X_baseline, y,
            api_key, base_url, model,
        )
        if not X_caafe.empty:
            X_caafe_aug = pd.concat([X_baseline, X_caafe], axis=1).fillna(0)
            caafe_metrics = compute_cv_metrics(X_caafe_aug, y)
            results["caafe_style"] = caafe_metrics
            results["caafe_style"]["n_features"] = X_caafe.shape[1]
            logger.info("CAAFE-style: F1=%.4f, Recall=%.4f (%d features)",
                        caafe_metrics["f1"], caafe_metrics["recall"], X_caafe.shape[1])
        else:
            results["caafe_style"] = {"error": "No valid features generated"}
    except Exception as e:
        results["caafe_style"] = {"error": str(e)}
        logger.error("CAAFE-style failed: %s", e)

    # 2d. 随机特征对照
    rng = np.random.RandomState(42)
    n_agent = X_agent.shape[1]
    X_random = pd.DataFrame(
        rng.randn(len(y), n_agent),
        columns=[f"random_{i}" for i in range(n_agent)],
        index=X_baseline.index,
    )
    X_random_aug = pd.concat([X_baseline, X_random], axis=1)
    random_metrics = compute_cv_metrics(X_random_aug, y)
    results["random_control"] = random_metrics
    logger.info("Random: F1=%.4f, Recall=%.4f", random_metrics["f1"], random_metrics["recall"])

    out_path = os.path.join(output_dir, "exp2_method_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)

    return results


# ============================================================
# 实验 3：消融实验（需要 LLM API，重新跑 Agent）
# ============================================================

def _run_agent_with_config(
    open_df, tx_df, close_df, label, baseline_features, y,
    api_key, base_url, model_name,
    ablation_name: str,
    inject_domain_knowledge: bool = True,
    use_shap_feedback: bool = True,
    use_pending_pool: bool = True,
    max_rounds: int = 10,
) -> dict:
    """用指定配置运行一次 Agent，返回结果。"""
    from src.agent.single_agent import FeatureEngineerAgent
    from src.evaluation.calibration import calibrate_l2_thresholds

    schema_text = extract_schema(open_df, tx_df, close_df)
    distribution_diff = extract_distribution_diff(open_df, tx_df, close_df, label)
    join_statistics = extract_join_statistics(open_df, tx_df, close_df)

    thresholds = calibrate_l2_thresholds(baseline_features, label)
    baseline_metrics = compute_cv_metrics(baseline_features, y)

    agent = FeatureEngineerAgent(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        open_df=open_df,
        tx_df=tx_df,
        close_df=close_df,
        label=label,
        baseline_features=baseline_features,
        max_steps=400,
        max_features=30,
        temperature=0.7,
        validation_thresholds=thresholds,
        baseline_metrics=baseline_metrics,
        schema_text=schema_text,
        distribution_diff=distribution_diff,
        join_statistics=join_statistics,
    )

    # 消融：去掉领域知识
    if not inject_domain_knowledge:
        original_build = agent._build_system_prompt

        def _build_no_knowledge():
            prompt = original_build()
            for section_start in [
                "## 数据集交易模式说明",
                "## 数字人民币洗钱模式",
                "## 风险分类指标",
                "## 检测规则 Python 实现示例",
            ]:
                idx = prompt.find(section_start)
                if idx >= 0:
                    next_section = prompt.find("\n## ", idx + len(section_start))
                    if next_section >= 0:
                        prompt = prompt[:idx] + prompt[next_section:]
                    else:
                        prompt = prompt[:idx]
            return prompt

        agent._build_system_prompt = _build_no_knowledge

    # 消融：去掉 SHAP 反馈（均匀预算分配）
    if not use_shap_feedback:
        def _uniform_budgets():
            n_islands = len(agent.island_manager.islands)
            b_per = DEFAULT_CONFIG.search.B_total // n_islands
            return {isl.dimension: b_per for isl in agent.island_manager.islands}

        agent._get_island_budgets = _uniform_budgets
        agent._get_shap_feedback = lambda: None

    # 消融：去掉候选缓冲池+前向选择
    if not use_pending_pool:
        agent.pending_pool = type(agent.pending_pool)(max_size=0, max_age=0, max_rejects=0)
        agent._run_pending_pool_evaluation = lambda: []

    logger.info("Running ablation '%s' (knowledge=%s, shap=%s, pool=%s) ...",
                ablation_name, inject_domain_knowledge, use_shap_feedback, use_pending_pool)

    start = time.time()
    result = agent.run()
    elapsed = time.time() - start

    result["ablation_name"] = ablation_name
    result["elapsed_seconds"] = elapsed
    result["config"] = {
        "inject_domain_knowledge": inject_domain_knowledge,
        "use_shap_feedback": use_shap_feedback,
        "use_pending_pool": use_pending_pool,
    }

    return result


def experiment_ablation(
    open_df, tx_df, close_df, label,
    X_baseline, y, feature_cols,
    api_key, base_url, model_name,
    n_runs: int,
    output_dir: str,
):
    """运行消融实验矩阵。"""
    logger.info("=" * 60)
    logger.info("Experiment 3: Ablation Studies (%d runs each)", n_runs)
    logger.info("=" * 60)

    ablation_configs = [
        {"name": "full_agent",       "knowledge": True,  "shap": True,  "pool": True},
        {"name": "no_knowledge",     "knowledge": False, "shap": True,  "pool": True},
        {"name": "no_shap",          "knowledge": True,  "shap": False, "pool": True},
        {"name": "no_pool",          "knowledge": True,  "shap": True,  "pool": False},
        {"name": "no_knowledge_shap","knowledge": False, "shap": False, "pool": True},
    ]

    all_results = {}

    for config in ablation_configs:
        run_results = []
        for run_i in range(n_runs):
            logger.info("--- %s, run %d/%d ---", config["name"], run_i + 1, n_runs)
            try:
                result = _run_agent_with_config(
                    open_df, tx_df, close_df, label, X_baseline, y,
                    api_key, base_url, model_name,
                    ablation_name=config["name"],
                    inject_domain_knowledge=config["knowledge"],
                    use_shap_feedback=config["shap"],
                    use_pending_pool=config["pool"],
                )

                n_accepted = len(result.get("accepted_features", []))
                n_rounds = result.get("rounds_completed", 0)
                elapsed = result.get("elapsed_seconds", 0)

                # 评估这次运行生成的特征
                agent_features = _extract_features_from_result(
                    result, open_df, tx_df, close_df
                )

                if agent_features is not None and not agent_features.empty:
                    X_aug = pd.concat([X_baseline, agent_features], axis=1).fillna(0)
                    metrics = compute_cv_metrics(X_aug, y)
                    base_metrics = compute_cv_metrics(X_baseline, y)
                    delta_recall = metrics["recall"] - base_metrics["recall"]
                    delta_f1 = metrics["f1"] - base_metrics["f1"]
                else:
                    metrics = compute_cv_metrics(X_baseline, y)
                    delta_recall = 0.0
                    delta_f1 = 0.0

                run_results.append({
                    "run": run_i,
                    "n_accepted": n_accepted,
                    "n_rounds": n_rounds,
                    "elapsed_seconds": elapsed,
                    "delta_recall": delta_recall,
                    "delta_f1": delta_f1,
                    "metrics": metrics,
                })

                logger.info("  Result: %d features, ΔRecall=%.4f, ΔF1=%.4f, %.1f min",
                            n_accepted, delta_recall, delta_f1, elapsed / 60)

            except Exception as e:
                logger.error("  Run %d failed: %s", run_i, e)
                run_results.append({"run": run_i, "error": str(e)})

        # 汇总统计
        valid_runs = [r for r in run_results if "error" not in r]
        if valid_runs:
            summary = {
                "n_accepted_mean": np.mean([r["n_accepted"] for r in valid_runs]),
                "n_accepted_std": np.std([r["n_accepted"] for r in valid_runs]),
                "delta_recall_mean": np.mean([r["delta_recall"] for r in valid_runs]),
                "delta_recall_std": np.std([r["delta_recall"] for r in valid_runs]),
                "delta_f1_mean": np.mean([r["delta_f1"] for r in valid_runs]),
                "delta_f1_std": np.std([r["delta_f1"] for r in valid_runs]),
                "n_valid_runs": len(valid_runs),
            }
        else:
            summary = {"error": "All runs failed"}

        all_results[config["name"]] = {
            "runs": run_results,
            "summary": summary,
        }

    out_path = os.path.join(output_dir, "exp3_ablation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", out_path)

    # 打印汇总表
    print("\n=== Ablation Results ===")
    print(f"{'Config':<25} {'#Features':>10} {'ΔRecall':>15} {'ΔF1':>15}")
    print("-" * 65)
    for name, data in all_results.items():
        s = data.get("summary", {})
        if "error" in s:
            print(f"{name:<25} {'FAILED':>10}")
        else:
            print(f"{name:<25} {s['n_accepted_mean']:>10.1f} "
                  f"{s['delta_recall_mean']:>+10.4f}±{s['delta_recall_std']:.4f} "
                  f"{s['delta_f1_mean']:>+10.4f}±{s['delta_f1_std']:.4f}")

    return all_results


def _extract_features_from_result(result, open_df, tx_df, close_df):
    """从 Agent 运行结果中提取特征 DataFrame。"""
    from src.evaluation.sandbox import execute_feature_code

    accepted = result.get("accepted_features", [])
    if not accepted:
        return None

    dfs = []
    for feat in accepted:
        code = feat.get("code", "")
        if not code:
            continue
        feat_df, err = execute_feature_code(code, open_df, tx_df, close_df)
        if err:
            continue
        id_col = "tx_id" if "tx_id" in feat_df.columns else "wallet_id"
        feat_cols = [c for c in feat_df.columns if c != id_col]
        dfs.append(feat_df[feat_cols])

    if not dfs:
        return None
    return pd.concat(dfs, axis=1).fillna(0)


# ============================================================
# 实验 4：多次运行稳定性
# ============================================================

def experiment_stability(
    open_df, tx_df, close_df, label,
    X_baseline, y, feature_cols,
    api_key, base_url, model_name,
    n_runs: int,
    output_dir: str,
):
    """同一配置跑 N 次 Full Agent，统计稳定性。"""
    logger.info("=" * 60)
    logger.info("Experiment 4: Stability (%d runs)", n_runs)
    logger.info("=" * 60)

    results = []
    for i in range(n_runs):
        logger.info("--- Stability run %d/%d ---", i + 1, n_runs)
        try:
            result = _run_agent_with_config(
                open_df, tx_df, close_df, label, X_baseline, y,
                api_key, base_url, model_name,
                ablation_name=f"stability_run_{i}",
            )

            agent_features = _extract_features_from_result(result, open_df, tx_df, close_df)
            if agent_features is not None and not agent_features.empty:
                X_aug = pd.concat([X_baseline, agent_features], axis=1).fillna(0)
                metrics = compute_cv_metrics(X_aug, y)
                base_metrics = compute_cv_metrics(X_baseline, y)
            else:
                metrics = compute_cv_metrics(X_baseline, y)
                base_metrics = metrics

            results.append({
                "run": i,
                "n_accepted": len(result.get("accepted_features", [])),
                "n_rounds": result.get("rounds_completed", 0),
                "total_steps": result.get("total_steps", 0),
                "elapsed_seconds": result.get("elapsed_seconds", 0),
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "delta_recall": metrics["recall"] - base_metrics["recall"],
                "feature_names": [f["name"] for f in result.get("accepted_features", [])],
            })
        except Exception as e:
            logger.error("  Run %d failed: %s", i, e)
            results.append({"run": i, "error": str(e)})

    out_path = os.path.join(output_dir, "exp4_stability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    valid = [r for r in results if "error" not in r]
    if valid:
        print(f"\n=== Stability ({len(valid)}/{n_runs} runs) ===")
        print(f"  #Features: {np.mean([r['n_accepted'] for r in valid]):.1f} ± {np.std([r['n_accepted'] for r in valid]):.1f}")
        print(f"  ΔRecall:   {np.mean([r['delta_recall'] for r in valid]):+.4f} ± {np.std([r['delta_recall'] for r in valid]):.4f}")
        print(f"  Steps:     {np.mean([r['total_steps'] for r in valid]):.0f} ± {np.std([r['total_steps'] for r in valid]):.0f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run paper experiments")
    parser.add_argument("--mode", choices=["offline", "ablation", "stability", "all"],
                        default="offline",
                        help="offline=multi-model+method(no LLM), ablation/stability/all=need LLM")
    parser.add_argument("--agent-result", type=str, default=None,
                        help="Path to existing agent result JSON (for offline mode)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default=f"outputs/paper_experiments_{TIMESTAMP}")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs for ablation/stability")
    parser.add_argument("--api-key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    open_df, tx_df, close_df, X_baseline, y, feature_cols = load_experiment_data(args.data_dir)

    # 全局变量供 load_agent_features 使用
    globals()["_open_df"] = open_df
    globals()["_tx_df"] = tx_df
    globals()["_close_df"] = close_df

    label_col = "is_risk"
    for alt in ["is_risk", "is_laundering", "is_abnormal", "is_zs_laundering"]:
        if alt in tx_df.columns:
            label_col = alt
            break
    label = tx_df[label_col].astype(int)

    logger.info("Data loaded: %d transactions, %d baseline features, %.4f%% positive",
                len(tx_df), X_baseline.shape[1], y.mean() * 100)

    # 尝试加载已有的 Agent 特征
    X_agent = None
    agent_feature_names = []
    if args.agent_result and os.path.exists(args.agent_result):
        logger.info("Loading agent features from %s", args.agent_result)
        X_agent = load_agent_features(args.agent_result, tx_df)
        agent_feature_names = list(X_agent.columns)
        logger.info("Loaded %d agent features: %s", len(agent_feature_names), agent_feature_names)

    # --- 实验 1: 多模型验证 ---
    if args.mode in ("offline", "all") and X_agent is not None:
        experiment_multi_model(X_baseline, X_agent, y, args.output_dir)

    # --- 实验 2: 方法对比 ---
    if args.mode in ("offline", "all") and X_agent is not None:
        experiment_method_comparison(
            open_df, tx_df, close_df, label,
            X_baseline, X_agent, y,
            feature_cols, agent_feature_names,
            args.api_key, args.base_url, args.model,
            args.output_dir,
        )

    # --- 实验 3: 消融实验 ---
    if args.mode in ("ablation", "all"):
        if not args.api_key:
            logger.error("Ablation requires --api-key or DEEPSEEK_API_KEY env var")
        else:
            experiment_ablation(
                open_df, tx_df, close_df, label,
                X_baseline, y, feature_cols,
                args.api_key, args.base_url, args.model,
                args.n_runs, args.output_dir,
            )

    # --- 实验 4: 稳定性 ---
    if args.mode in ("stability", "all"):
        if not args.api_key:
            logger.error("Stability requires --api-key or DEEPSEEK_API_KEY env var")
        else:
            experiment_stability(
                open_df, tx_df, close_df, label,
                X_baseline, y, feature_cols,
                args.api_key, args.base_url, args.model,
                args.n_runs, args.output_dir,
            )

    logger.info("All experiments complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
