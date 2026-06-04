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
    Reproduce this EXACT layout for an IEEE two-column paper framework diagram (width ~7 inches, height ~3.2 inches).
    Use ONLY IEEE-compliant fonts: Arial or Helvetica for labels, Times New Roman for body text. Minimum 7pt font size.
    White background, clean professional academic style with light pastel colors for regions.

    The figure has 5 major regions flowing LEFT to RIGHT, with a feedback loop at the BOTTOM:

    === REGION 1: "Inputs" (left side, light blue background region) ===
    Two stacked boxes:
    Top box: "Tabular Dataset D" with subtitle "Features x, Labels y" and a small database/table icon.
    Bottom box: "Domain Knowledge Base K" with subtitle "(column names, types, stats)" at the top,
    then 5 knowledge types stacked vertically with colored rounded boxes, each slightly indented:
      - "Schema Only" (lightest shade) — at top, most general
      - "Risk Indicators" with "(38 structured indicators)" in small text
      - "Detection Rules" with "(15 operational rules with logic)" in small text
      - "Expert Opinions" with "(tacit knowledge)" in small text
      - "Court Document Evidence" with "(309 legal cases)" in small text — at bottom, most specific
    IMPORTANT: Do NOT use "Level 0/1/2/3/4" labels. Only show the type names above.
    Show a vertical gradient arrow on the left side of this stack from "General" (top) to "Specific" (bottom).
    Label the whole region "Domain Knowledge Base K" at the bottom.

    === REGION 2: "Core Agent" (center, light yellow/cream background region) ===
    Title at top: "Core Agent"
    Show "Multi-round Reasoning Loop" as a circular arrow at the top.
    Three tool icons/boxes arranged around the agent:
      - "explore_data" (with magnifying glass icon)
      - "generate_feature" (with gear icon) — subtitle "Candidate Feature & Provenance"
      - "execute_sandbox" (with terminal icon)
    Center label: "ReAct LLM Agent" (bold, NO model name like GPT-4o or DeepSeek)
    At the bottom-left: "Reallocated Budget" arrow feeding into the agent from the feedback loop below.

    === REGION 3: "Three-Stage Verification Pipeline" (right-center, light green background region) ===
    Title at top: "Three-Stage Verification Pipeline"
    Three boxes flowing left to right:
      - "L1: Execution" with small text: "Syntax & Shape Checks, 30s Timeout, Up to 2 Repair Attempts"
      - "L2: Statistical Quality" with small text: "IV, KS > Threshold, Redundancy < 0.7, Missing < 50%"
      - "L3: Model Effectiveness" with small text: "XGBoost Evaluation, ΔAUC > 0 or ΔRecall > 0"
    Show "Pass" arrow from L3 going right to Outputs.
    Show "Fail" arrow going down to "Failure Context" which feeds back to the agent.

    === REGION 4: "Outputs" (far right, light purple background region) ===
    Title at top: "Outputs"
    Two stacked boxes:
      - "Accepted Feature Set F*" with subtitle "Executable Code"
      - "Provenance Cards" with subtitle "Knowledge Source, Reasoning Chain, Computation"

    === REGION 5: "Search Strategy & Dynamic Budgeting" (bottom, light gray background region) ===
    This spans the full width at the bottom.
    Left side: "SHAP-guided Inverse-Proportion Budget Reallocation" box
    Center: "4 Semantic Dimensions" box with 4 sub-boxes in a 2x2 grid:
      - "Amount & Frequency" | "Time & Geography"
      - "Behavior Sequence" | "Entity Association"
    Right side: "Forward Selection" box, then "Cross-dimension propagation" label, then "Failure Memory" box
    with label "Prevents Repeated Failures"
    Show arrows connecting: SHAP reallocation -> dimensions -> forward selection -> failure memory
    Show "Failure Context" arrow from verification (Region 3) feeding into this bottom region.

    === ARROWS (use colored arrows to show flow) ===
    - Green arrows: main data flow (inputs -> agent -> verification -> outputs)
    - Orange/yellow arrows: feedback loops (failure context, SHAP reallocation)
    - Dashed arrows: secondary connections (cross-dimension propagation)

    The overall feel should match a polished IEEE conference paper figure: structured, professional, information-dense but readable.
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
