"""KnowFeat Web 演示服务：FastAPI + WebSocket 实时推送。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="KnowFeat Demo")

DATASET_REGISTRY = {
    "simecny": {
        "stage": "stage2",
        "domain": "数字人民币反洗钱",
        "task": "二分类（交易是否涉嫌洗钱）",
        "dimensions": ["金额频次", "时间地域", "行为序列", "主体关联"],
        "n_rows": 200000,
        "description": "SimECNY 合成数字人民币交易数据（3表：accounts 43K + transactions 200K + close 224）",
    },
    "adult": {
        "stage": "generic",
        "csv": PROJECT_ROOT / "data" / "public" / "adult.csv",
        "target": "class",
        "domain": "收入预测",
        "task": "二分类（年收入是否超过50K美元）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "adult" / "domain_knowledge.json",
        "dimensions": ["人口统计", "教育就业", "财务资产", "社会经济"],
    },
    "heart": {
        "stage": "generic",
        "csv": PROJECT_ROOT / "data" / "public" / "heart.csv",
        "target": "HeartDisease",
        "domain": "心脏病诊断",
        "task": "二分类（心脏病诊断）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "heart" / "domain_knowledge.json",
        "dimensions": ["风险因素", "运动试验", "ECG特征", "复合指标"],
    },
    "diabetes": {
        "stage": "generic",
        "csv": PROJECT_ROOT / "data" / "public" / "diabetes.csv",
        "target": "class",
        "domain": "糖尿病诊断",
        "task": "二分类（糖尿病诊断）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "diabetes" / "domain_knowledge.json",
        "dimensions": ["代谢指标", "体质指标", "遗传因素", "临床检查"],
    },
    "blood_transfusion": {
        "stage": "generic",
        "csv": PROJECT_ROOT / "data" / "public" / "blood_transfusion.csv",
        "target": "Class",
        "domain": "献血行为预测",
        "task": "二分类（献血预测）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "blood_transfusion" / "domain_knowledge.json",
        "dimensions": ["RFM特征", "行为模式"],
    },
    "credit_g": {
        "stage": "generic",
        "csv": PROJECT_ROOT / "data" / "public" / "credit_g.csv",
        "target": "class",
        "domain": "信贷风控",
        "task": "二分类（信贷违约）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "credit_g" / "domain_knowledge.json",
        "dimensions": ["客户属性", "财务状况", "信贷历史", "贷款特征"],
    },
    "breast_w": {
        "stage": "generic",
        "csv": PROJECT_ROOT / "data" / "public" / "breast_w.csv",
        "target": "Class",
        "domain": "乳腺癌诊断",
        "task": "二分类（良性/恶性）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "breast_w" / "domain_knowledge.json",
        "dimensions": ["细胞特征", "核特征", "复合指标"],
    },
    "bank_marketing": {
        "stage": "generic",
        "csv": PROJECT_ROOT / "data" / "public" / "bank_marketing.csv",
        "target": "y",
        "domain": "银行营销",
        "task": "二分类（是否认购定期存款）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "bank_marketing" / "domain_knowledge.json",
        "dimensions": ["客户画像", "联系记录", "经济环境", "时机因素"],
    },
}

event_queue: Queue = Queue()
pipeline_thread: Optional[threading.Thread] = None
pipeline_stop_flag = threading.Event()


def event_callback(event_type: str, data: dict):
    event = {"type": event_type, "time": time.time(), **data}
    event_queue.put(event)


def _sample_stage2_data(data, baseline_features, sample_size, seed=42):
    """Stratified sample of Stage2 data (transactions + aligned baseline)."""
    import pandas as pd
    import numpy as np

    n_total = len(data.tx_df)
    if sample_size >= n_total:
        return data, baseline_features

    rng = np.random.RandomState(seed)

    label = data.tx_df["is_risk"].values
    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]

    pos_rate = len(pos_idx) / n_total
    n_pos = max(1, int(sample_size * pos_rate))
    n_neg = sample_size - n_pos

    sampled_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    sampled_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    sampled_idx = np.sort(np.concatenate([sampled_pos, sampled_neg]))

    tx_sampled = data.tx_df.iloc[sampled_idx].reset_index(drop=True)

    wallet_ids = set(tx_sampled["src"].unique()) | set(tx_sampled["dst"].unique())
    open_sampled = data.open_df[data.open_df["wallet_id"].isin(wallet_ids)].reset_index(drop=True)
    close_sampled = data.close_df[data.close_df["wallet_id"].isin(wallet_ids)].reset_index(drop=True)

    bf_sampled = None
    if baseline_features is not None:
        bf_sampled = baseline_features.iloc[sampled_idx].reset_index(drop=True)

    from knowfeat.data.loader import Stage2Data
    sampled_data = Stage2Data(
        open_df=open_sampled,
        tx_df=tx_sampled,
        close_df=close_sampled,
        baseline_features=bf_sampled,
    )
    return sampled_data, bf_sampled


def run_pipeline_simecny(config: dict):
    """Run Stage2 (SimECNY) pipeline."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from knowfeat.config import load_config
        from knowfeat.data.loader import load_stage2
        from knowfeat.prompt.domain_knowledge import load_domain_knowledge
        from knowfeat.main import run_stage, save_results

        cfg = load_config(stage="stage2")
        cfg.search.max_rounds = config.get("max_rounds", 3)
        cfg.search.initial_budget_per_dim = config.get("budget_per_dim", 2)
        cfg.verification.noise_n_random_features = config.get("noise_samples", 20)
        cfg.search.convergence_rounds = config.get("convergence_rounds", 2)
        cfg.verification.cv_folds = config.get("cv_folds", 3)

        knowledge_filter = config.get("knowledge_filter") or None

        event_callback("data_loading", dataset="SimECNY", domain="数字人民币反洗钱")
        data = load_stage2(cfg=cfg)

        sample_size = config.get("sample_size", 0)
        if sample_size and sample_size < len(data.tx_df):
            event_callback("data_sampling",
                           original_size=len(data.tx_df),
                           sample_size=sample_size)
            data, _ = _sample_stage2_data(data, data.baseline_features, sample_size, cfg.seed)

        pos_count = int(data.tx_df["is_risk"].sum())
        event_callback("data_loaded",
                       n_rows=len(data.tx_df),
                       n_accounts=len(data.open_df),
                       n_close=len(data.close_df),
                       n_features=data.baseline_features.shape[1] if data.baseline_features is not None else 0,
                       pos_count=pos_count,
                       pos_rate=round(100 * pos_count / len(data.tx_df), 2))

        event_callback("knowledge_loading", dataset="SimECNY")
        dk = load_domain_knowledge(cfg)
        event_callback("knowledge_loaded",
                       n_indicators=len(dk.risk_indicators),
                       n_rules=len(dk.detection_rules),
                       n_patterns=len(dk.transaction_patterns),
                       n_expert=len(dk.expert_opinion) if dk.expert_opinion else 0,
                       n_court_factors=len(dk.court_doc_risk_factors),
                       registry_size=len(dk.knowledge_registry))

        result = run_stage(cfg, data, dk,
                           knowledge_filter=knowledge_filter,
                           event_callback=event_callback)

        output_dir = cfg.paths.output_dir / "simecny_web"
        save_results(result, output_dir)

        event_callback("pipeline_complete",
                       total_accepted=result.total_accepted,
                       total_rounds=result.total_rounds,
                       elapsed_sec=round(result.elapsed_sec, 1),
                       convergence_reason=result.convergence_reason,
                       accepted_features=result.accepted_feature_names,
                       feature_codes={k: v for k, v in result.feature_codes.items()},
                       feature_provenance={k: v for k, v in result.feature_provenance.items()})

    except Exception as e:
        logger.exception("Pipeline error")
        event_callback("pipeline_error", error=str(e))


def run_pipeline_generic(config: dict):
    """Run generic (public dataset) pipeline."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from knowfeat.config import PublicDatasetConfig, load_config
        from knowfeat.data.loader import load_public_dataset
        from knowfeat.prompt.domain_knowledge import load_domain_knowledge
        from knowfeat.main import run_stage, save_results
        import pandas as pd

        dataset_name = config["dataset"]
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
        cfg.search.max_rounds = config.get("max_rounds", 3)
        cfg.search.initial_budget_per_dim = config.get("budget_per_dim", 2)
        cfg.verification.noise_n_random_features = config.get("noise_samples", 20)
        cfg.search.convergence_rounds = config.get("convergence_rounds", 2)
        cfg.verification.cv_folds = config.get("cv_folds", 3)

        knowledge_filter = config.get("knowledge_filter") or None

        event_callback("data_loading", dataset=dataset_name, domain=reg["domain"])
        data = load_public_dataset(pub_cfg)
        label_series = pd.to_numeric(data.df[data.label_col], errors='coerce')
        pos_rate = label_series.mean() * 100 if not label_series.isna().all() else 0.0
        event_callback("data_loaded",
                       n_rows=len(data.df),
                       n_features=len(data.feature_cols),
                       feature_cols=data.feature_cols[:20],
                       pos_rate=round(pos_rate, 2))

        event_callback("knowledge_loading", dataset=dataset_name)
        dk = load_domain_knowledge(cfg)
        event_callback("knowledge_loaded",
                       n_indicators=len(dk.public_knowledge.indicators) if dk.public_knowledge else 0,
                       n_rules=len(dk.public_knowledge.rules) if dk.public_knowledge else 0,
                       n_expert=len(dk.public_knowledge.expert_consensus) if dk.public_knowledge else 0,
                       registry_size=len(dk.knowledge_registry))

        result = run_stage(cfg, data, dk,
                           knowledge_filter=knowledge_filter,
                           event_callback=event_callback)

        output_dir = cfg.paths.output_dir / dataset_name
        save_results(result, output_dir)

        event_callback("pipeline_complete",
                       total_accepted=result.total_accepted,
                       total_rounds=result.total_rounds,
                       elapsed_sec=round(result.elapsed_sec, 1),
                       convergence_reason=result.convergence_reason,
                       accepted_features=result.accepted_feature_names,
                       feature_codes={k: v for k, v in result.feature_codes.items()},
                       feature_provenance={k: v for k, v in result.feature_provenance.items()})

    except Exception as e:
        logger.exception("Pipeline error")
        event_callback("pipeline_error", error=str(e))


def run_pipeline_thread(config: dict):
    """Dispatch to the correct pipeline runner."""
    dataset_name = config.get("dataset", "")
    reg = DATASET_REGISTRY.get(dataset_name, {})

    if reg.get("stage") == "stage2":
        run_pipeline_simecny(config)
    else:
        run_pipeline_generic(config)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/datasets")
async def list_datasets():
    datasets = []
    for name, reg in DATASET_REGISTRY.items():
        if reg.get("stage") == "stage2":
            available = (PROJECT_ROOT / "data" / "transactions.csv").exists()
            datasets.append({
                "name": name,
                "stage": "stage2",
                "domain": reg["domain"],
                "task": reg["task"],
                "dimensions": reg["dimensions"],
                "description": reg.get("description", ""),
                "n_rows": reg.get("n_rows", 0),
                "available": available,
                "is_simecny": True,
            })
        else:
            csv_exists = reg["csv"].exists()
            knowledge_exists = reg["knowledge"].exists()
            datasets.append({
                "name": name,
                "stage": "generic",
                "domain": reg["domain"],
                "task": reg["task"],
                "dimensions": reg["dimensions"],
                "csv_exists": csv_exists,
                "knowledge_exists": knowledge_exists,
                "available": csv_exists,
                "is_simecny": False,
            })
    return JSONResponse(datasets)


@app.get("/api/knowledge/{dataset_name}")
async def get_knowledge(dataset_name: str):
    reg = DATASET_REGISTRY.get(dataset_name)
    if not reg:
        return JSONResponse({"error": "Dataset not found"}, status_code=404)

    if reg.get("stage") == "stage2":
        result = {
            "type": "stage2",
            "indicators": {"count": 38, "source": "风险分类表_38指标.xlsx"},
            "rules": {"count": 15, "source": "检测规则_15条.xlsx"},
            "patterns": {"count": 8, "types": "3 normal + 5 abnormal"},
            "expert": {"source": "邮储专家访谈_数币洗钱模式.docx"},
            "court_docs": {"count": 309, "source": "裁判文书docx/"},
        }
        return JSONResponse(result)

    knowledge_path = reg.get("knowledge")
    if not knowledge_path or not knowledge_path.exists():
        return JSONResponse({"indicators": [], "rules": [], "expert_consensus": []})

    with open(knowledge_path, encoding="utf-8") as f:
        data = json.load(f)

    return JSONResponse({
        "type": "generic",
        "indicators": data.get("indicators", []),
        "rules": data.get("rules", []),
        "expert_consensus": data.get("expert_consensus", []),
        "field_reference": data.get("field_reference", {}),
    })


@app.post("/api/start")
async def start_pipeline(config: dict):
    global pipeline_thread

    if pipeline_thread and pipeline_thread.is_alive():
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)

    while not event_queue.empty():
        try:
            event_queue.get_nowait()
        except Empty:
            break

    pipeline_stop_flag.clear()
    pipeline_thread = threading.Thread(
        target=run_pipeline_thread,
        args=(config,),
        daemon=True,
    )
    pipeline_thread.start()
    return JSONResponse({"status": "started"})


@app.get("/api/status")
async def get_status():
    running = pipeline_thread is not None and pipeline_thread.is_alive()
    return JSONResponse({"running": running})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            try:
                event = event_queue.get_nowait()
                for key, val in event.items():
                    if isinstance(val, Path):
                        event[key] = str(val)
                await ws.send_json(event)
            except Empty:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.warning("WebSocket error: %s", e)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")


if __name__ == "__main__":
    main()
