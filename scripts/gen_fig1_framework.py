"""Generate Figure 1: KnowFeat Framework Overview — clean redesign."""

import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/ubuntu/paperbanana/.env")

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings


async def main():
    source_context = """
    Draw a clean, professional framework diagram for an IEEE conference paper (two-column, ~7x3.5 inches).

    This diagram illustrates KnowFeat, an LLM-based feature engineering system.
    The main flow goes LEFT → RIGHT, with a feedback loop returning from right to left at the bottom.

    There are four major blocks arranged horizontally:

    BLOCK 1 — INPUTS (left):
    Two stacked items:
    (a) "Tabular Dataset D" — a simple table icon with "features x, labels y".
    (b) "Domain Knowledge K" — five types stacked top-to-bottom, labeled with an arrow
        from "General" at top to "Specific" at bottom:
        • Schema Metadata
        • Risk Indicators (38)
        • Detection Rules (15)
        • Expert Opinions (12)
        • Case Evidence (309)
    Use a light blue background for this block.

    BLOCK 2 — LLM AGENT (center-left):
    A rounded box labeled "ReAct LLM Agent" (no specific model name).
    Inside or around it, show three tool icons:
        • explore_data (magnifying glass)
        • generate_feature (pencil/document)
        • execute_sandbox (terminal)
    A curved arrow around the box indicates "Multi-round Generation".
    Use a light orange/cream background.

    BLOCK 3 — VERIFICATION PIPELINE (center-right):
    Three stages shown as a horizontal pipeline with arrows:
        L1: Execution → L2: Quality Filter → L3: Model Test
    Below each stage, one line of key info:
        L1: "code sandbox, auto-repair"
        L2: "IV, KS, redundancy"
        L3: "XGBoost, 5 seeds"
    A "Pass" arrow exits right; a "Reject" arrow exits downward.
    Use a light green background.

    BLOCK 4 — OUTPUTS (right):
    Two stacked items:
        • "Accepted Features F*"
        • "Provenance Cards" (with sub-text: knowledge trace, reasoning chain)
    Use a light purple background.

    BOTTOM — FEEDBACK LOOP (spanning full width, light gray background):
    Shows the adaptive search strategy:
        • Rejected features → "Failure Memory" (prevents repeats)
        • Accepted features → "SHAP Analysis" → "Budget Reallocation"
          Budget feeds back to the LLM Agent for the next round.
        • Four semantic dimensions shown as small tags:
          "Amount & Frequency", "Time & Geography", "Behavior Sequence", "Entity Association"
        • "Forward Selection" box for synergistic combinations

    ARROWS:
        • Solid dark arrows for the main left-to-right flow
        • Dashed or colored arrows for feedback (SHAP → Agent, Reject → Failure Memory)

    STYLE:
        • White overall background, pastel colored blocks
        • Clean sans-serif font (Arial/Helvetica), minimum 8pt
        • Simple flat icons, no 3D effects
        • Enough whitespace between blocks for readability
    """

    caption = (
        "Overview of the KnowFeat framework. The LLM agent generates features "
        "guided by hierarchical domain knowledge. Each candidate is verified "
        "through a three-stage pipeline (L1–L3). SHAP feedback dynamically "
        "reallocates the exploration budget across semantic dimensions."
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
