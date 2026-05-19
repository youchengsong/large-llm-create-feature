#!/usr/bin/env python3
"""
单Agent特征工程系统 - 完整运行脚本（含评估+可视化报告）

用法（服务器上后台运行）:
    nohup python run_single_agent.py > /dev/null 2>&1 &
    # 查看日志: tail -f outputs/single_agent_run/run_*.log
"""

import sys
import os
import json
import logging
import time
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-59bd25d3db384cf09f63814989d7200c"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
MAX_FEATURES = 30
MAX_STEPS = 250
DATA_DIR = str(PROJECT_ROOT / "data")

# ============================================================
# 日志配置
# ============================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "single_agent_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = OUTPUT_DIR / f"run_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

logger.info("=" * 60)
logger.info("单Agent特征工程系统 - 启动")
logger.info("=" * 60)
logger.info(f"时间: {datetime.now().isoformat()}")
logger.info(f"模型: {MODEL}, 最大步数: {MAX_STEPS}, 目标特征数: {MAX_FEATURES}")
logger.info(f"日志文件: {log_file}")

# ============================================================
# Step 1: 加载数据
# ============================================================
logger.info("\n[Step 1] 加载数据...")

from src.data.loader import load_all_tables

open_df, tx_df, close_df = load_all_tables(DATA_DIR)
logger.info(f"  open_df: {open_df.shape}")
logger.info(f"  tx_df: {tx_df.shape}")
logger.info(f"  close_df: {close_df.shape}")

# ============================================================
# Step 2: 运行基线特征工程
# ============================================================
logger.info("\n[Step 2] 运行基线特征工程...")

import pandas as pd

BASELINE_CACHE = PROJECT_ROOT / "outputs" / "baseline_cache.parquet"

if BASELINE_CACHE.exists():
    logger.info("  从缓存加载基线特征...")
    baseline_df = pd.read_parquet(BASELINE_CACHE)
    logger.info(f"  基线特征矩阵(缓存): {baseline_df.shape}")
else:
    from src.data.baseline import run_baseline
    baseline_df = run_baseline(open_df, tx_df, close_df)
    baseline_df.to_parquet(BASELINE_CACHE)
    logger.info(f"  基线特征矩阵: {baseline_df.shape} (已缓存)")

label_col = "is_laundering"
if label_col not in baseline_df.columns:
    if "is_risk" in tx_df.columns:
        label_col = "is_risk"
        label = tx_df[label_col].astype(int)
    else:
        raise ValueError("找不到标签列")
else:
    label = baseline_df[label_col].astype(int)

# 去掉10个指定特征，以95个基线特征为基准
DROP_FEATURES = [
    "Min_Received", "Min_Sent", "wallets_per_cert_From", "Max_Sent",
    "S_src_counterparty_index", "Timestamp_Hour", "S_dst_counterparty_index",
    "1H_frequency_From", "1H_frequency_To", "min_inactive_period_From",
]
feature_cols = [c for c in baseline_df.columns if c != label_col and c not in DROP_FEATURES]
baseline_features = baseline_df[feature_cols].fillna(0)

logger.info(f"  特征列数: {len(feature_cols)} (去掉{len(DROP_FEATURES)}个指定特征)")
logger.info(f"  标签分布: pos={int(label.sum())}, neg={int((label==0).sum())}, rate={label.mean()*100:.4f}%")

# ============================================================
# Step 2.5: 校准L2阈值（与原系统一致）
# ============================================================
logger.info("\n[Step 2.5] 校准L2验证阈值...")

from src.evaluation.calibration import calibrate_l2_thresholds

calibrated_thresholds = calibrate_l2_thresholds(baseline_features, label)
logger.info(f"  校准结果: IV_min={calibrated_thresholds.iv_min:.6f}, "
            f"KS_min={calibrated_thresholds.ks_min:.6f}, corr_max={calibrated_thresholds.corr_max:.2f}")

# ============================================================
# Step 3: 运行单Agent
# ============================================================
logger.info("\n[Step 3] 计算基线CV指标（用于L3验证）...")

from src.evaluation.metrics import compute_cv_metrics

baseline_cv_metrics = compute_cv_metrics(baseline_features, label)
logger.info(f"  基线CV指标: F1={baseline_cv_metrics.get('f1', 0):.6f}, "
            f"Recall={baseline_cv_metrics.get('recall', 0):.6f}, "
            f"Precision={baseline_cv_metrics.get('precision', 0):.6f}, "
            f"PR-AUC={baseline_cv_metrics.get('prauc', 0):.6f}")

# ============================================================
# Step 3.5: 提取Schema、分布差异、关联统计（注入system prompt）
# ============================================================
logger.info("\n[Step 3.5] 提取数据Schema和分布差异...")

from src.data.loader import extract_schema, extract_distribution_diff, extract_join_statistics

schema_parts = [
    extract_schema(open_df, "开户表(accounts)"),
    extract_schema(tx_df, "交易表(transactions)"),
    extract_schema(close_df, "销户表(close)"),
]
schema_text = "\n\n".join(schema_parts)
distribution_diff = extract_distribution_diff(open_df, tx_df, close_df, label)
join_statistics = extract_join_statistics(open_df, tx_df, close_df)
logger.info(f"  Schema: {len(schema_text)} chars, 分布差异: {len(distribution_diff)} chars, 关联统计: {len(join_statistics)} chars")

# ============================================================
logger.info("\n[Step 4] 初始化并运行单Agent...")

from src.agent.single_agent import FeatureEngineerAgent

agent = FeatureEngineerAgent(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    open_df=open_df,
    tx_df=tx_df,
    close_df=close_df,
    label=label,
    baseline_features=baseline_features,
    validation_thresholds=calibrated_thresholds,
    baseline_metrics=baseline_cv_metrics,
    max_steps=MAX_STEPS,
    max_features=MAX_FEATURES,
    temperature=0.7,
    schema_text=schema_text,
    distribution_diff=distribution_diff,
    join_statistics=join_statistics,
)

start_time = time.time()
try:
    result = agent.run()
except Exception as e:
    logger.error(f"Agent运行异常: {e}", exc_info=True)
    result = {
        "accepted_features": agent.state.accepted_features,
        "failed_features": agent.state.failed_features,
        "total_steps": len(agent.step_log),
        "total_attempts": agent.state.total_attempts,
        "dimensions_explored": list(agent.state.dimensions_explored),
        "finish_reason": f"异常终止: {e}",
        "step_log": agent.step_log,
    }

elapsed = time.time() - start_time
logger.info(f"\n[Step 4] Agent完成，耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
logger.info(f"  接受: {len(result['accepted_features'])}, 失败: {len(result['failed_features'])}, 步数: {result['total_steps']}")

# ============================================================
# Step 4: 保存特征代码和结果JSON
# ============================================================
logger.info("\n[Step 5] 保存结果...")

result_file = OUTPUT_DIR / f"result_{timestamp}.json"
save_data = {
    "run_info": {
        "timestamp": timestamp,
        "elapsed_seconds": elapsed,
        "model": MODEL,
        "max_steps": MAX_STEPS,
        "max_features": MAX_FEATURES,
        "data_shape": {"tx_df": list(tx_df.shape), "open_df": list(open_df.shape)},
        "label_stats": {"total": int(len(label)), "positive": int(label.sum()), "rate": float(label.mean())},
    },
    "accepted_features": [
        {"name": f["name"], "code": f["code"], "explanation": f["explanation"],
         "reasoning": f.get("reasoning", ""), "dimension": f.get("dimension", ""),
         "l2_metrics": f.get("l2_metrics", {}), "l3_metrics": f.get("l3_metrics", {})}
        for f in result["accepted_features"]
    ],
    "failed_features": [{"name": f.get("name", ""), "error": f.get("error", "")}
                        for f in result["failed_features"][:50]],
    "step_log": result.get("step_log", []),
    "summary": {
        "accepted_count": len(result["accepted_features"]),
        "failed_count": len(result["failed_features"]),
        "total_steps": result["total_steps"],
        "dimensions_explored": result["dimensions_explored"],
        "finish_reason": result["finish_reason"],
    },
}
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
logger.info(f"  结果JSON: {result_file}")

# ============================================================
# Step 5: 评估增量效果 (5-fold CV) — 与原系统一致使用 compute_cv_metrics + forward_select
# ============================================================
logger.info("\n[Step 6] 评估增量效果...")

from src.evaluation.validator import run_l1_validation, forward_select_features
from src.data.merge import merge_feature_to_transactions
import pandas as pd

# 基线评估：直接复用Step 3已计算的baseline_cv_metrics（与原系统一致的compute_cv_metrics）
baseline_metrics = baseline_cv_metrics
logger.info(f"  基线: F1={baseline_metrics['f1']:.6f}, Recall={baseline_metrics['recall']:.6f}, "
            f"Precision={baseline_metrics['precision']:.6f}, PR-AUC={baseline_metrics['prauc']:.6f}")

# 重新执行接受的特征代码，构建候选特征
eval_results = {"baseline": baseline_metrics}
candidate_features = {}
execution_status = {}

if result["accepted_features"]:
    logger.info("  重新执行Agent生成的特征代码...")
    for feat in result["accepted_features"]:
        try:
            feature_df, error = run_l1_validation(feat["code"], open_df, tx_df, close_df)
            if feature_df is not None:
                merged = merge_feature_to_transactions(feature_df, tx_df)
                feat_cols_new = [c for c in merged.columns if c not in tx_df.columns]
                if feat_cols_new:
                    series = merged[feat_cols_new[0]].fillna(0).values
                    candidate_features[feat["name"]] = series
                    execution_status[feat["name"]] = "OK"
                    logger.info(f"    OK: {feat['name']}")
                else:
                    execution_status[feat["name"]] = "SKIP: 无新列"
                    logger.warning(f"    SKIP: {feat['name']} (无新列)")
            else:
                execution_status[feat["name"]] = f"FAIL: {error[:80]}"
                logger.warning(f"    FAIL: {feat['name']} ({error[:80]})")
        except Exception as e:
            execution_status[feat["name"]] = f"ERROR: {e}"
            logger.warning(f"    ERROR: {feat['name']} ({e})")

    # 保存候选特征数组为parquet，避免重新执行代码
    if candidate_features:
        candidate_df = pd.DataFrame(candidate_features)
        candidate_parquet = OUTPUT_DIR / f"candidate_features_{timestamp}.parquet"
        candidate_df.to_parquet(candidate_parquet)
        logger.info(f"  候选特征已保存: {candidate_parquet} ({len(candidate_features)} 个特征)")

        # 全部特征的CV指标
        logger.info(f"  评估全部 {len(candidate_features)} 个候选特征...")
        X_all = baseline_features.copy()
        for name, vals in candidate_features.items():
            X_all[name] = vals
        all_metrics = compute_cv_metrics(X_all, label)
        all_delta = {k: all_metrics[k] - baseline_metrics[k] for k in all_metrics}
        logger.info(f"  全部特征: F1={all_metrics['f1']:.6f} (dF1={all_delta['f1']:+.6f}), "
                    f"Recall={all_metrics['recall']:.6f} (dRecall={all_delta['recall']:+.6f}), "
                    f"Precision={all_metrics['precision']:.6f} (dPrecision={all_delta['precision']:+.6f}), "
                    f"PR-AUC={all_metrics['prauc']:.6f} (dPR-AUC={all_delta['prauc']:+.6f})")

        # Forward Selection
        logger.info(f"  开始Forward Selection...")
        selected_names, final_metrics = forward_select_features(
            baseline_features, candidate_features, label, baseline_metrics,
        )

        fs_delta = {k: final_metrics[k] - baseline_metrics[k] for k in final_metrics}

        if selected_names:
            logger.info(f"  Forward Selection选出: {len(selected_names)} 个特征: {selected_names}")
            logger.info(f"  FS特征: F1={final_metrics['f1']:.6f} (dF1={fs_delta['f1']:+.6f}), "
                        f"Recall={final_metrics['recall']:.6f} (dRecall={fs_delta['recall']:+.6f}), "
                        f"Precision={final_metrics['precision']:.6f} (dPrecision={fs_delta['precision']:+.6f}), "
                        f"PR-AUC={final_metrics['prauc']:.6f} (dPR-AUC={fs_delta['prauc']:+.6f})")
        else:
            logger.info("  Forward Selection未选出任何特征（无增量效果）")

        # 写入评估结果：包含全部中间结果
        eval_results["all_features"] = {
            "metrics": all_metrics,
            "delta": all_delta,
            "count": len(candidate_features),
            "names": list(candidate_features.keys()),
        }
        eval_results["forward_selection"] = {
            "metrics": final_metrics,
            "delta": fs_delta,
            "count": len(selected_names),
            "names": selected_names,
        }
        eval_results["execution_status"] = execution_status
        eval_results["candidate_parquet"] = str(candidate_parquet)

        # 保持向后兼容的字段
        eval_results["combined"] = final_metrics
        eval_results["delta"] = fs_delta
        eval_results["new_features_count"] = len(selected_names)
        eval_results["new_feature_names"] = selected_names

eval_file = OUTPUT_DIR / f"evaluation_{timestamp}.json"
with open(eval_file, "w", encoding="utf-8") as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)
logger.info(f"  评估结果: {eval_file}")

# ============================================================
# Step 6: 生成可视化HTML报告
# ============================================================
logger.info("\n[Step 7] 生成可视化报告...")

from src.report.agent_report import generate_agent_report

report_file = OUTPUT_DIR / f"report_{timestamp}.html"
generate_agent_report(
    result=result,
    eval_results=eval_results,
    elapsed=elapsed,
    model=MODEL,
    tx_count=len(tx_df),
    wallet_count=open_df.shape[0],
    baseline_feature_count=baseline_features.shape[1],
    positive_rate=label.mean(),
    output_path=str(report_file),
)
logger.info(f"  可视化报告: {report_file}")

# ============================================================
# 完成
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("全部流程完成！")
logger.info("=" * 60)
logger.info(f"输出目录: {OUTPUT_DIR}")
logger.info(f"  result_{timestamp}.json    - 完整结果")
logger.info(f"  evaluation_{timestamp}.json - 评估指标")
logger.info(f"  report_{timestamp}.html     - 可视化报告")
logger.info(f"  run_{timestamp}.log         - 运行日志")
logger.info("=" * 60)
