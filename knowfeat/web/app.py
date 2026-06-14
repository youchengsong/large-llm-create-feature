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
    "adult": {
        "csv": PROJECT_ROOT / "data" / "public" / "adult.csv",
        "target": "class",
        "domain": "收入预测",
        "task": "二分类（年收入是否超过50K美元）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "adult" / "domain_knowledge.json",
        "dimensions": ["人口统计", "教育就业", "财务资产", "社会经济"],
    },
    "heart": {
        "csv": PROJECT_ROOT / "data" / "public" / "heart.csv",
        "target": "HeartDisease",
        "domain": "心脏病诊断",
        "task": "二分类（心脏病诊断）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "heart" / "domain_knowledge.json",
        "dimensions": ["风险因素", "运动试验", "ECG特征", "复合指标"],
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
    "credit_g": {
        "csv": PROJECT_ROOT / "data" / "public" / "credit_g.csv",
        "target": "class",
        "domain": "信贷风控",
        "task": "二分类（信贷违约）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "credit_g" / "domain_knowledge.json",
        "dimensions": ["客户属性", "财务状况", "信贷历史", "贷款特征"],
    },
    "breast_w": {
        "csv": PROJECT_ROOT / "data" / "public" / "breast_w.csv",
        "target": "Class",
        "domain": "乳腺癌诊断",
        "task": "二分类（良性/恶性）",
        "knowledge": PROJECT_ROOT / "inputs" / "public_datasets" / "breast_w" / "domain_knowledge.json",
        "dimensions": ["细胞特征", "核特征", "复合指标"],
    },
    "bank_marketing": {
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


def run_pipeline_thread(config: dict):
    """Run the KnowFeat pipeline in a background thread."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from knowfeat.config import KnowFeatConfig, PublicDatasetConfig, load_config
        from knowfeat.data.loader import load_public_dataset
        from knowfeat.prompt.domain_knowledge import load_domain_knowledge
        from knowfeat.main import run_stage, save_results

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

        knowledge_filter = config.get("knowledge_filter")

        event_callback("data_loading", dataset=dataset_name, domain=reg["domain"])
        data = load_public_dataset(pub_cfg)
        import pandas as pd
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


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/datasets")
async def list_datasets():
    datasets = []
    for name, reg in DATASET_REGISTRY.items():
        csv_exists = reg["csv"].exists()
        knowledge_exists = reg["knowledge"].exists()
        datasets.append({
            "name": name,
            "domain": reg["domain"],
            "task": reg["task"],
            "dimensions": reg["dimensions"],
            "csv_exists": csv_exists,
            "knowledge_exists": knowledge_exists,
            "available": csv_exists,
        })
    return JSONResponse(datasets)


@app.get("/api/knowledge/{dataset_name}")
async def get_knowledge(dataset_name: str):
    reg = DATASET_REGISTRY.get(dataset_name)
    if not reg:
        return JSONResponse({"error": "Dataset not found"}, status_code=404)

    knowledge_path = reg["knowledge"]
    if not knowledge_path.exists():
        return JSONResponse({"indicators": [], "rules": [], "expert_consensus": []})

    with open(knowledge_path, encoding="utf-8") as f:
        data = json.load(f)

    return JSONResponse({
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
