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
    The figure should span two columns in an IEEE paper (width ~7 inches, height ~3 inches).
    Use ONLY standard fonts: Times New Roman, Helvetica, or Arial. All text must be at least 8pt.
    Clean, professional academic style. No decorative elements. White background.

    The pipeline flows LEFT to RIGHT with these components:

    INPUT (left side, two stacked boxes):
    Box 1: "Tabular Dataset" (with icon or simple table representation)
    Box 2: "Hierarchical Domain Knowledge" with 5 types stacked vertically:
      - Schema Metadata (most general, at top)
      - Risk Indicators (38 items)
      - Detection Rules (15 items)
      - Expert Opinions (12 items)
      - Court Document Evidence (most specific, at bottom)
      Label the stack as going from "General" at top to "Specific" at bottom with a vertical arrow.
      Do NOT use "Level 0/1/2/3/4" labels. Just use the type names directly.

    CORE (center): "LLM Agent" box (do NOT mention any specific model name like GPT-4o or DeepSeek - just "LLM Agent")
    The agent has 3 tools shown as small sub-boxes or icons:
    - Data Exploration
    - Feature Registration (with provenance fields)
    - Verification Trigger
    Show a circular arrow or loop indicating multi-round generation.
    Also show "Failure Memory" feeding back into the agent.

    VERIFICATION (right of agent): Three-stage pipeline shown as 3 connected filter boxes:
    - L1: Code Execution Sandbox
    - L2: Statistical Quality (IV, KS, redundancy, missing rate)
    - L3: Model Effectiveness (XGBoost, 5 seeds)
    Show arrows: candidate -> L1 -> L2 -> L3 -> Accepted/Rejected
    Rejected features go back to Failure Memory (feedback arrow).

    OUTPUT (right side):
    - "Accepted Features F*" box
    - "Provenance Cards" box (showing traceability)

    FEEDBACK LOOP (bottom, COMPACT - keep this section small and tight):
    Show SHAP Feedback arrow from accepted features back to Budget Reallocation, then back to Agent.
    List 4 dimension labels in a SINGLE compact row or small grouped box: Amount & Frequency, Time & Geography, Behavior Sequence, Entity Association.
    Do NOT expand each dimension into a large separate box. Keep the SHAP/budget area visually small.

    Keep the diagram compact but readable. Use light colors for boxes (light blue, light green, light yellow).
    Arrows should be clear with labels where needed.
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
