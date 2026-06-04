"""Generate Figure 1: KnowFeat Framework Overview."""

import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/ubuntu/paperbanana/.env")

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings


async def main():
    source_context = """
    KnowFeat is a knowledge-guided feature engineering framework built on an LLM agent.
    The pipeline has the following components, flowing left to right:

    INPUT (left side):
    - Tabular Dataset D (with features x and labels y)
    - Domain Knowledge Base K, organized as a 5-level hierarchy:
      Level 0: Schema Only (column names, types, stats)
      Level 1: Risk Indicators (38 structured indicators)
      Level 2: Detection Rules (15 operational rules with logic)
      Level 3: Expert Opinions (tacit knowledge from interviews)
      Level 4: Court Document Evidence (extracted from 309 legal cases)
      These levels stack from general (top) to specific (bottom).

    CORE (center): ReAct LLM Agent
    The agent operates in a multi-round loop. It has 3 tools:
    - explore_data: inspect dataset distributions and patterns
    - generate_feature: propose a candidate feature with provenance metadata (knowledge_ids, reasoning chain, code)
    - execute_sandbox: trigger verification pipeline

    The agent receives knowledge context + failure memory + SHAP feedback each round.

    VERIFICATION (right of agent): Three-Stage Pipeline (L1 -> L2 -> L3)
    - L1 Execution: sandbox, 30s timeout, syntax/shape checks, up to 2 repair attempts
    - L2 Statistical Quality: IV, KS statistic, redundancy < 0.7, missing < 30%, thresholds calibrated from 100 random noise features at 95th percentile
    - L3 Model Effectiveness: XGBoost evaluation across 5 seeds, delta-AUC > 0 or delta-Recall > 0

    SEARCH STRATEGY (bottom or feedback loop):
    - 4 semantic dimensions: Amount & Frequency, Time & Geography, Behavior Sequence, Entity Association
    - SHAP-guided inverse-proportion budget reallocation after each round
    - Failure Memory prevents regenerating rejected features
    - Forward Selection discovers synergistic feature combinations
    - Cross-dimension propagation of successful patterns

    OUTPUT (right side):
    - Accepted feature set F* with executable code
    - Provenance cards for each feature (knowledge assets, reasoning chain, computation steps)

    The flow is: Dataset + Knowledge -> Agent proposes features -> L1->L2->L3 verification ->
    accepted features added to baseline, rejected features to failure memory ->
    SHAP analysis reallocates budget -> next round. Loop until convergence.
    """

    caption = (
        "Overview of the KnowFeat framework. The LLM agent generates features guided by "
        "hierarchical domain knowledge. Each candidate is verified through a three-stage "
        "pipeline (L1-L3). SHAP feedback dynamically reallocates the exploration budget "
        "across semantic dimensions."
    )

    settings = Settings(
        vlm_provider="gemini",
        vlm_model="gemini-2.5-pro",
        image_provider="google_imagen",
        image_model="gemini-3.1-flash-image",
        refinement_iterations=2,
    )

    pipeline = PaperBananaPipeline(settings=settings)

    result = await pipeline.generate(
        GenerationInput(
            source_context=source_context,
            communicative_intent=caption,
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    out = Path("/home/ubuntu/Projects/大模型提取特征/paper/figures/framework.png")
    shutil.copy(result.image_path, out)
    print(f"Saved to {out}")
    print(f"Iterations: {len(result.iterations)}")


if __name__ == "__main__":
    asyncio.run(main())
