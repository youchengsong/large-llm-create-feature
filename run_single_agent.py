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
MAX_FEATURES = 25
MAX_STEPS = 80
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

from src.data.baseline import run_baseline

baseline_df = run_baseline(open_df, tx_df, close_df)
logger.info(f"  基线特征矩阵: {baseline_df.shape}")

label_col = "is_laundering"
if label_col not in baseline_df.columns:
    if "is_risk" in tx_df.columns:
        label_col = "is_risk"
        label = tx_df[label_col].astype(int)
    else:
        raise ValueError("找不到标签列")
else:
    label = baseline_df[label_col].astype(int)

feature_cols = [c for c in baseline_df.columns if c != label_col]
baseline_features = baseline_df[feature_cols].fillna(0)

logger.info(f"  特征列数: {len(feature_cols)}")
logger.info(f"  标签分布: pos={int(label.sum())}, neg={int((label==0).sum())}, rate={label.mean()*100:.4f}%")

# ============================================================
# Step 3: 运行单Agent
# ============================================================
logger.info("\n[Step 3] 初始化并运行单Agent...")

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
    max_steps=MAX_STEPS,
    max_features=MAX_FEATURES,
    temperature=0.7,
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
# Step 5: 评估增量效果 (5-fold CV)
# ============================================================
logger.info("\n[Step 6] 评估增量效果...")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
from src.evaluation.validator import run_l1_validation
from src.data.merge import merge_feature_to_transactions
import pandas as pd

y = label.values
X_baseline = baseline_features.values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
spw = (y == 0).sum() / max((y == 1).sum(), 1)


def eval_cv(X, y, cv, spw):
    """5-fold CV评估，返回平均指标。"""
    metrics_list = []
    for train_idx, val_idx in cv.split(X, y):
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="aucpr",
            use_label_encoder=False, verbosity=0, random_state=42,
        )
        model.fit(X[train_idx], y[train_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        prob_train = model.predict_proba(X[train_idx])[:, 1]
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y[train_idx], prob_train)
        f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-10)
        best_thr = thr_arr[np.argmax(f1_arr[:-1])]
        prob_val = model.predict_proba(X[val_idx])[:, 1]
        y_pred = (prob_val >= best_thr).astype(int)
        metrics_list.append({
            "f1": f1_score(y[val_idx], y_pred),
            "recall": recall_score(y[val_idx], y_pred),
            "precision": precision_score(y[val_idx], y_pred),
            "pr_auc": average_precision_score(y[val_idx], prob_val),
        })
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in ["f1", "recall", "precision", "pr_auc"]}


# 基线评估
logger.info("  计算基线指标...")
baseline_metrics = eval_cv(X_baseline, y, cv, spw)
logger.info(f"  基线: F1={baseline_metrics['f1']:.6f}, Recall={baseline_metrics['recall']:.6f}, "
            f"Precision={baseline_metrics['precision']:.6f}, PR-AUC={baseline_metrics['pr_auc']:.6f}")

# 重新执行接受的特征代码，构建组合特征
eval_results = {"baseline": baseline_metrics}
new_feature_list = []

if result["accepted_features"]:
    logger.info("  重新执行Agent生成的特征代码...")
    for feat in result["accepted_features"]:
        try:
            feature_df, error = run_l1_validation(feat["code"], open_df, tx_df, close_df)
            if feature_df is not None:
                merged = merge_feature_to_transactions(feature_df, tx_df)
                feat_cols_new = [c for c in merged.columns if c not in tx_df.columns]
                if feat_cols_new:
                    series = merged[feat_cols_new[0]].fillna(0)
                    series.name = feat["name"]
                    new_feature_list.append(series)
                    logger.info(f"    OK: {feat['name']}")
                else:
                    logger.warning(f"    SKIP: {feat['name']} (无新列)")
            else:
                logger.warning(f"    FAIL: {feat['name']} ({error[:80]})")
        except Exception as e:
            logger.warning(f"    ERROR: {feat['name']} ({e})")

    if new_feature_list:
        new_features_df = pd.concat(new_feature_list, axis=1)
        X_combined = np.hstack([X_baseline, new_features_df.values])
        logger.info(f"  组合特征矩阵: baseline({X_baseline.shape[1]}) + new({new_features_df.shape[1]}) = {X_combined.shape[1]}")

        logger.info("  计算组合指标...")
        combined_metrics = eval_cv(X_combined, y, cv, spw)
        delta_metrics = {k: combined_metrics[k] - baseline_metrics[k] for k in combined_metrics}

        eval_results["combined"] = combined_metrics
        eval_results["delta"] = delta_metrics
        eval_results["new_features_count"] = len(new_feature_list)
        eval_results["new_feature_names"] = [s.name for s in new_feature_list]

        logger.info(f"  组合: F1={combined_metrics['f1']:.6f}, Recall={combined_metrics['recall']:.6f}, "
                    f"Precision={combined_metrics['precision']:.6f}, PR-AUC={combined_metrics['pr_auc']:.6f}")
        logger.info(f"  增量: dF1={delta_metrics['f1']:+.6f}, dRecall={delta_metrics['recall']:+.6f}, "
                    f"dPrecision={delta_metrics['precision']:+.6f}, dPR-AUC={delta_metrics['pr_auc']:+.6f}")

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
