"""Generate Figure 1: KnowFeat Framework Overview — based on reference images."""

import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/ubuntu/paperbanana/.env")

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings


async def main():
    source_context = """
    Create a detailed academic framework diagram for an IEEE two-column paper (~7 inches wide, ~3.5 inches tall).
    This is a methodology overview figure for a system called "KnowFeat".

    LAYOUT: Five labeled column headers across the top: "Inputs", "Core Agent", "Three-Stage Verification Pipeline", "Outputs".
    Below the main row, a full-width bottom band labeled "Search Strategy & Dynamic Budgeting".
    Use pastel colored background rectangles behind each column group.

    === COLUMN 1: "Inputs" (light blue background) ===
    Top item: A box with a small database/table icon, labeled "Tabular Dataset D", subtitle "Features x, Labels y".
    Below it: A larger box titled "Domain Knowledge Base K" with subtitle "(column names, types, stats)".
    Inside this box, 5 knowledge types stacked vertically, each as a small rounded pill/tag:
        • Schema Only
        • Risk Indicators (38 structured indicators)
        • Detection Rules (15 operational rules with logic)
        • Expert Opinions (tacit knowledge)
        • Court Document Evidence (309 legal cases)
    On the left edge, a vertical arrow labeled "General" at top → "Specific" at bottom.
    At the bottom of this column, repeat label "Domain Knowledge Base K" in italics.
    CRITICAL: Do NOT write "Level 0", "Level 1", "Level 2", "Level 3", "Level 4" anywhere.

    === COLUMN 2: "Core Agent" (light cream/yellow background) ===
    Top: Label "Multi-round Reasoning Loop" with a circular arrow icon.
    Center: A dark rounded box containing "ReAct LLM Agent" (bold text).
    Do NOT write any model name like GPT-4o or DeepSeek. Only "ReAct LLM Agent".
    Three tools arranged around the agent box, each with a small icon:
        • explore_data (magnifying glass icon, top-right)
        • generate_feature (document/gear icon, left) with sub-label "Candidate Feature & Provenance"
        • execute_sandbox (terminal/command icon, bottom)
    An arrow from Inputs → Agent labeled "Knowledge Context".
    An arrow from Agent → Verification labeled "Candidate Feature".
    At bottom: "Reallocated Budget" arrow coming up from the Search Strategy area.

    === COLUMN 3: "Three-Stage Verification Pipeline" (light green background) ===
    Three boxes arranged left to right, each with a small icon at top:
        L1: "Execution" — with details: "Syntax & Shape Checks", "30s Timeout", "Up to 2 Repair Attempts"
        L2: "Statistical Quality" — with details: "IV, KS > Threshold", "Redundancy < 0.7", "Missing < 50%"
        L3: "Model Effectiveness" — with details: "XGBoost Evaluation", "ΔAUC > 0 or", "ΔRecall > 0"
    Arrow from L3 → Outputs labeled "Pass".
    Arrow from L3 going down labeled "Fail", connecting to "Failure Context" which feeds to the bottom Search Strategy area.

    === COLUMN 4: "Outputs" (light purple background) ===
    Two stacked boxes:
        Top: "Accepted Feature Set F*" with sub-text "Executable Code"
        Bottom: "Provenance Cards" with sub-text "Knowledge Source, Reasoning Chain, Computation"

    === BOTTOM BAND: "Search Strategy & Dynamic Budgeting" (light gray background, full width) ===
    Left side: "SHAP-guided Inverse-Proportion Budget Reallocation" box.
    Center: "4 Semantic Dimensions" box containing a 2×2 grid of four items:
        Amount & Frequency | Time & Geography
        Behavior Sequence  | Entity Association
    Right side: Three items in a row:
        "Forward Selection" box → "Cross-dimension propagation" (dashed arrow label) → "Failure Memory" box with sub-text "Prevents Repeated Failures"
    Arrows: SHAP box → feeds "Reallocated Budget" back up to Core Agent.
    "Failure Context" from Verification feeds down into this band.

    === ARROW STYLES ===
    Main flow arrows (Inputs→Agent→Verification→Outputs): solid dark green, thick
    Feedback arrows (SHAP→Agent, Fail→Failure Memory): dashed or orange/yellow
    Cross-dimension propagation: dashed gray arrow

    === VISUAL STYLE ===
    White background, pastel colored region blocks.
    Sans-serif font (Arial or Helvetica), all text at least 7pt.
    Small flat icons for tools and verification stages.
    Clean, professional, information-dense but well-organized with clear visual hierarchy.
    No 3D effects, no gradients on text, no decorative elements.
    """

    caption = (
        "Overview of the KnowFeat framework. The LLM agent generates features guided by "
        "hierarchical domain knowledge. Each candidate is verified through a three-stage "
        "pipeline (L1–L3). SHAP feedback dynamically reallocates the exploration budget "
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
