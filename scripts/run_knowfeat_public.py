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
    "adult": {
        "csv": PROJECT_ROOT / "data" / "public" / "adult.csv",
        "target": "class",
        "domain": "收入预测",
        "task": "二分类（年收入是否超过50K美元）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "adult" / "domain_knowledge.json",
        "dimensions": ["人口统计", "教育就业", "财务资产", "社会经济"],
    },
    "elliptic": {
        "csv": PROJECT_ROOT / "data" / "elliptic" / "elliptic_labeled.csv",
        "target": "label",
        "domain": "Bitcoin AML detection",
        "task": "Binary classification (illicit vs licit transactions)",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "elliptic" / "domain_knowledge.json",
        "dimensions": ["Transaction Amount", "Network Topology", "Temporal Pattern", "Aggregated Features"],
    },
    "default_credit": {
        "csv": PROJECT_ROOT / "data" / "public" / "default_credit.csv",
        "target": "target",
        "domain": "信用卡违约预测",
        "task": "二分类（是否违约）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "default_credit" / "domain_knowledge.json",
        "dimensions": ["还款记录", "消费行为", "客户属性"],
    },
    "churn": {
        "csv": PROJECT_ROOT / "data" / "public" / "churn.csv",
        "target": "target",
        "domain": "电信客户流失预测",
        "task": "二分类（是否流失）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "churn" / "domain_knowledge.json",
        "dimensions": ["使用行为", "账户信息", "客户服务"],
    },
    "wine_quality": {
        "csv": PROJECT_ROOT / "data" / "public" / "wine_quality.csv",
        "target": "class",
        "domain": "红葡萄酒品质评估",
        "task": "二分类（品质>=6为好酒）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "wine_quality" / "domain_knowledge.json",
        "dimensions": ["酸度平衡", "氧化保护", "酒体结构", "矿质感"],
    },
    "spambase": {
        "csv": PROJECT_ROOT / "data" / "public" / "spambase.csv",
        "target": "class",
        "domain": "垃圾邮件检测",
        "task": "二分类（垃圾邮件识别）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "spambase" / "domain_knowledge.json",
        "dimensions": ["词频特征", "字符特征", "格式特征"],
    },
    "australian": {
        "csv": PROJECT_ROOT / "data" / "public" / "australian.csv",
        "target": "A15",
        "domain": "信贷审批",
        "task": "二分类（信贷审批）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "australian" / "domain_knowledge.json",
        "dimensions": ["个人信息", "财务状况", "信贷记录"],
    },
    "phoneme": {
        "csv": PROJECT_ROOT / "data" / "public" / "phoneme.csv",
        "target": "Class",
        "domain": "语音识别",
        "task": "二分类（元音/鼻音分类）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "phoneme" / "domain_knowledge.json",
        "dimensions": ["频谱特征", "时域特征"],
    },
    "wdbc": {
        "csv": PROJECT_ROOT / "data" / "public" / "wdbc.csv",
        "target": "Class",
        "domain": "乳腺癌诊断",
        "task": "二分类（良性/恶性）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "wdbc" / "domain_knowledge.json",
        "dimensions": ["均值特征", "标准差特征", "最大值特征"],
    },
    "climate_model": {
        "csv": PROJECT_ROOT / "data" / "public" / "climate_model.csv",
        "target": "outcome",
        "domain": "气候模型故障预测",
        "task": "二分类（模拟成功/失败）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "climate_model" / "domain_knowledge.json",
        "dimensions": ["对流参数", "混合参数", "效率参数"],
    },
    "phishing": {
        "csv": PROJECT_ROOT / "data" / "public" / "phishing.csv",
        "target": "Result",
        "domain": "钓鱼网站检测",
        "task": "二分类（正常/钓鱼）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "phishing" / "domain_knowledge.json",
        "dimensions": ["URL特征", "域名特征", "页面特征", "安全特征"],
    },
}


def build_config(dataset_name: str, max_features: int = 9999) -> KnowFeatConfig:
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
    cfg.verification.delta_auc_min = -0.002
    cfg.search.dimensions = reg["dimensions"]
    cfg.search.max_features = max_features
    cfg.search.initial_budget_per_dim = max(2, 20 // len(reg["dimensions"]))

    return cfg


def main():
    parser = argparse.ArgumentParser(description="在公开数据集上运行 KnowFeat")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--max-features", type=int, default=9999)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--result-dir", type=Path, default=None,
        help="Exact output directory for this run (keeps repeated runs isolated)",
    )
    parser.add_argument(
        "--enable-posthoc-selection", action="store_true",
        help="Run the optional second model-selection pass after forward selection",
    )
    parser.add_argument("--knowledge-filter", default=None,
                        choices=["schema_only", "indicators", "rules", "expert"])
    args = parser.parse_args()

    cfg = build_config(args.dataset, args.max_features)
    cfg.search.max_rounds = args.max_rounds
    cfg.search.enable_posthoc_selection = args.enable_posthoc_selection
    cfg.seed = args.seed
    if args.temperature is not None:
        cfg.llm.temperature = args.temperature

    logger.info("=== KnowFeat on %s ===", args.dataset)
    logger.info("Config: stage=%s, dataset=%s, target=%s, max_features=%d",
                cfg.stage, cfg.public_dataset.dataset_name,
                cfg.public_dataset.target_col, cfg.search.max_features)

    # 加载数据
    from knowfeat.data.loader import load_public_dataset, subset_stage1
    data_full = load_public_dataset(cfg.public_dataset)

    # 无泄漏协议：按冻结划分裁剪到 D_dev，特征生成/选择只见训练部分。
    # 划分缓存与 scripts/evaluate_holdout.py 共享，保证最终 test 集一致。
    from knowfeat.data.split import get_or_make_split
    full_labels = pd.to_numeric(data_full.df[data_full.label_col], errors="coerce").astype(int)
    split_cache = cfg.paths.output_dir / "splits" / f"{args.dataset}_split.json"
    split = get_or_make_split(len(data_full.df), labels=full_labels, cache_path=split_cache)
    data = subset_stage1(data_full, split.dev_idx)

    label_series = pd.to_numeric(data.df[data.label_col], errors='coerce')
    pos_rate = label_series.mean() * 100 if not label_series.isna().all() else 0.0
    logger.info("Data loaded: full=%d rows -> dev=%d rows (test=%d frozen), %d features, "
                "label=%s (dev pos_rate=%.2f%%)",
                len(data_full.df), len(data.df), len(split.test_idx),
                len(data.feature_cols), data.label_col, pos_rate)

    # Load domain knowledge (skip if no knowledge path configured, e.g. Elliptic schema-only)
    from knowfeat.prompt.domain_knowledge import load_domain_knowledge, DomainKnowledge
    if cfg.public_dataset.domain_knowledge_path and str(cfg.public_dataset.domain_knowledge_path) != "":
        dk = load_domain_knowledge(cfg)
        logger.info("Knowledge loaded: registry=%d entries", len(dk.knowledge_registry))
    else:
        dk = DomainKnowledge(knowledge_registry={}, formatted_text="")
        logger.info("No domain knowledge configured, using schema-only mode")

    # 运行 Agent 循环
    from knowfeat.main import run_stage, save_results
    result = run_stage(cfg, data, dk, knowledge_filter=args.knowledge_filter)
    suffix = f"_{args.knowledge_filter}" if args.knowledge_filter else ""
    result_dir = args.result_dir or cfg.paths.output_dir / (args.dataset + suffix)
    save_results(result, result_dir)

    print(
        f"\n完成: {result.total_accepted} 特征, "
        f"{result.total_rounds} 轮, "
        f"{result.convergence_reason}"
    )


if __name__ == "__main__":
    main()
