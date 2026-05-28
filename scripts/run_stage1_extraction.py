"""Run Key-clue Extraction for Stage 1 knowledge sources."""
import sys
sys.path.insert(0, "/root/大模型提取特征")

from pathlib import Path
from knowfeat.config import KnowFeatConfig
from knowfeat.extractor.key_clue_extractor import batch_extract, save_results
from knowfeat.llm.client import LLMClient
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

cfg = KnowFeatConfig()
input_dir = cfg.paths.stage1_knowledge_dir
output_dir = cfg.paths.stage1_risk_factors_dir

skip = {"knowledge_sources.md"}
files = sorted([
    f for f in input_dir.glob("*.md")
    if f.name not in skip
])

print(f"Found {len(files)} files to process:")
for f in files:
    print(f"  - {f.name}")

client = LLMClient(cfg.llm)
results = batch_extract(client, files)
saved = save_results(results, output_dir)

total = sum(len(r.risk_factors) for r in results if r.success)
print(f"\nDone: {len(saved)} files saved, {total} total risk factors")
for r in results:
    status = f"{len(r.risk_factors)} factors" if r.success else f"FAILED: {r.error}"
    print(f"  {r.source}: {status}")
