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

    VISUAL STYLE: Use soft pastel colored background regions to group components.
    Light blue for Inputs, light yellow/cream for Core Agent, light green for Verification,
    light purple for Outputs, light gray for Search Strategy at the bottom.
    Use small icons (magnifying glass, gear, terminal) for the agent's tools.
    Rounded rectangle boxes. Professional academic diagram style with clear labeled arrows.
    Use Arial or Helvetica font. IEEE two-column figure width (~7 inches).

    INPUT (left side, light blue region):
    - Tabular Dataset D (with features x and labels y), show a small table/database icon
    - Domain Knowledge Base K, organized as a 5-type hierarchy stacked vertically:
      Schema Only (column names, types, stats)
      Risk Indicators (38 structured indicators)
      Detection Rules (15 operational rules with logic)
      Expert Opinions (tacit knowledge from interviews)
      Court Document Evidence (extracted from 309 legal cases)
      These types stack from general (top) to specific (bottom) with a gradient arrow.
      IMPORTANT: Do NOT use "Level 0/1/2/3/4" labels. Only use the type names above.

    CORE (center, light yellow region): ReAct LLM Agent
    Do NOT write any specific model name (no GPT-4o, no DeepSeek). Just "LLM Agent".
    The agent operates in a multi-round reasoning loop. It has 3 tools shown with icons:
    - explore_data: inspect dataset (magnifying glass icon)
    - generate_feature: propose candidate feature with provenance (gear/document icon)
    - execute_sandbox: trigger verification (terminal icon)

    VERIFICATION (right of agent, light green region): Three-Stage Pipeline (L1 -> L2 -> L3)
    - L1 Execution: syntax & shape checks, 30s timeout, up to 2 repair attempts
    - L2 Statistical Quality: IV, KS > threshold, redundancy < 0.7, missing < 50%
    - L3 Model Effectiveness: XGBoost evaluation, delta-AUC > 0 or delta-Recall > 0
    Show Pass arrow to Outputs and Fail arrow to Failure Context.

    OUTPUT (right side, light purple region):
    - Accepted Feature Set F* with executable code
    - Provenance Cards (knowledge source, reasoning chain, computation)

    SEARCH STRATEGY (bottom, light gray region spanning full width):
    - 4 semantic dimensions in a row: Amount & Frequency, Time & Geography, Behavior Sequence, Entity Association
    - SHAP-guided inverse-proportion budget reallocation
    - Failure Memory prevents regenerating rejected features
    - Forward Selection discovers synergistic feature combinations
    - Cross-dimension propagation of successful patterns
    Show SHAP Analysis feeding back reallocated budget to the agent (feedback loop).

    ARROWS: Use green arrows for main flow, orange/yellow for feedback loops, dashed for cross-dimension propagation.
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
        refinement_iterations=3,
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
