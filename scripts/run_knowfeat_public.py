"""在公开数据集上运行 KnowFeat 特征生成。

用法:
    python scripts/run_knowfeat_public.py --dataset credit_g
    python scripts/run_knowfeat_public.py --dataset heart --max-features 20
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowfeat.config import KnowFeatConfig, PublicDatasetConfig, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "credit_g": {
        "csv": PROJECT_ROOT / "data" / "public" / "credit_g.csv",
        "target": "class",
        "domain": "信贷风控",
        "task": "二分类（信贷违约）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "credit_g" / "domain_knowledge.json",
        "dimensions": ["客户属性", "财务状况", "信贷历史", "贷款特征"],
    },
    "diabetes": {
        "csv": PROJECT_ROOT / "data" / "public" / "diabetes.csv",
        "target": "class",
        "domain": "糖尿病诊断",
        "task": "二分类（糖尿病诊断）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "diabetes" / "domain_knowledge.json",
        "dimensions": ["代谢指标", "体质指标", "遗传因素", "临床检查"],
    },
    "blood_transfusion": {
        "csv": PROJECT_ROOT / "data" / "public" / "blood_transfusion.csv",
        "target": "Class",
        "domain": "献血行为预测",
        "task": "二分类（献血预测）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "blood_transfusion" / "domain_knowledge.json",
        "dimensions": ["RFM特征", "行为模式"],
    },
    "heart": {
        "csv": PROJECT_ROOT / "data" / "public" / "heart.csv",
        "target": "HeartDisease",
        "domain": "心脏病诊断",
        "task": "二分类（心脏病诊断）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "heart" / "domain_knowledge.json",
        "dimensions": ["风险因素", "运动试验", "ECG特征", "复合指标"],
    },
    "bank_marketing": {
        "csv": PROJECT_ROOT / "data" / "public" / "bank_marketing.csv",
        "target": "y",
        "domain": "银行营销",
        "task": "二分类（是否认购定期存款）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "bank_marketing" / "domain_knowledge.json",
        "dimensions": ["客户画像", "联系记录", "经济环境", "时机因素"],
    },
    "breast_w": {
        "csv": PROJECT_ROOT / "data" / "public" / "breast_w.csv",
        "target": "Class",
        "domain": "乳腺癌诊断",
        "task": "二分类（良性/恶性）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "breast_w" / "domain_knowledge.json",
        "dimensions": ["细胞特征", "核特征", "复合指标"],
    },
}


def build_config(dataset_name: str, max_features: int = 20) -> KnowFeatConfig:
    """为公开数据集构建 KnowFeat 配置。"""
    reg = DATASET_REGISTRY[dataset_name]

    pub_cfg = PublicDatasetConfig(
        dataset_name=dataset_name,
        csv_path=reg["csv"],
        target_col=reg["target"],
        id_col="row_id",
        domain=reg["domain"],
        task=reg["task"],
        domain_knowledge_path=reg["knowledge"],
    )

    cfg = load_config(stage="generic")
    cfg.public_dataset = pub_cfg
    cfg.verification.accept_metric = "auc"
    cfg.search.dimensions = reg["dimensions"]
    cfg.search.max_features = max_features
    cfg.search.initial_budget_per_dim = max(2, max_features // len(reg["dimensions"]))

    return cfg


def main():
    parser = argparse.ArgumentParser(description="在公开数据集上运行 KnowFeat")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knowledge-filter", default=None,
                        choices=["schema_only", "indicators", "rules", "expert"])
    args = parser.parse_args()

    cfg = build_config(args.dataset, args.max_features)
    cfg.search.max_rounds = args.max_rounds
    cfg.seed = args.seed

    logger.info("=== KnowFeat on %s ===", args.dataset)
    logger.info("Config: stage=%s, dataset=%s, target=%s, max_features=%d",
                cfg.stage, cfg.public_dataset.dataset_name,
                cfg.public_dataset.target_col, cfg.search.max_features)

    # 加载数据
    from knowfeat.data.loader import load_public_dataset
    data = load_public_dataset(cfg.public_dataset)
    label_series = pd.to_numeric(data.df[data.label_col], errors='coerce')
    pos_rate = label_series.mean() * 100 if not label_series.isna().all() else 0.0
    logger.info("Data loaded: %d rows, %d features, label=%s (pos_rate=%.2f%%)",
                len(data.df), len(data.feature_cols), data.label_col, pos_rate)

    # 加载领域知识
    from knowfeat.prompt.domain_knowledge import load_domain_knowledge
    dk = load_domain_knowledge(cfg)
    logger.info("Knowledge loaded: registry=%d entries", len(dk.knowledge_registry))

    # 运行 Agent 循环
    from knowfeat.main import run_stage, save_results
    result = run_stage(cfg, data, dk, knowledge_filter=args.knowledge_filter)
    suffix = f"_{args.knowledge_filter}" if args.knowledge_filter else ""
    save_results(result, cfg.paths.output_dir / (args.dataset + suffix))

    print(
        f"\n完成: {result.total_accepted} 特征, "
        f"{result.total_rounds} 轮, "
        f"{result.convergence_reason}"
    )


if __name__ == "__main__":
    main()
