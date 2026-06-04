"""Generate Figure 4: Provenance Card Example - compact for single column."""

import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/ubuntu/paperbanana/.env")

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings


async def main():
    source_context = """
    A COMPACT provenance card for an IEEE single-column figure (width ~3.5 inches, height ~3.5 inches MAX).
    Use standard fonts: Times New Roman or Helvetica, 7-8pt for body text, 9pt for headers.
    White background, clean academic style. Must be readable but space-efficient.

    This card shows the full traceability of ONE generated feature.
    Layout should be a SINGLE COLUMN card with sections stacked vertically:

    HEADER (top, bold, in a colored banner):
    Feature: fan_in_fan_out_ratio

    SECTION 1 - KNOWLEDGE ASSETS (light blue background, compact list):
    - RULE_006: Scatter-in/gather-out layering
    - RULE_013: Pyramid fund networks
    - PAT_A03: Money mule behavior
    - IND_15: Abnormal transaction modes

    SECTION 2 - REASONING CHAIN (white background, paragraph text, NOT numbered list):
    Display as a flowing causal narrative paragraph (because...therefore... style), NOT a numbered sequence:
    "Because RULE_006 and RULE_013 indicate that scatter-in/gather-out layering and pyramid fund networks both manifest as fan-in/fan-out asymmetry, we first compute the ratio of fan-in to fan-out counts per wallet. Since PAT_A03 shows that money mule wallets exhibit fan-in >> fan-out, this ratio captures the core topology signal. Furthermore, IND_15 identifies abnormal transaction modes (many-to-many, one-to-many), so we weight the ratio by the abnormal mode proportion to amplify suspicious patterns, yielding: ratio x (1 + abnormal_mode_ratio)."

    SECTION 3 - CODE (light gray background, monospace font, compact):
    fi = mode_counts['fan_in'] / n
    fo = mode_counts['fan_out'] / n
    ratio = fi / (fo + eps)
    abnormal = fi + m2m + o2m
    result = ratio * (1 + abnormal)

    SECTION 4 - VERIFICATION (light green background, single line):
    L1: PASS | L2: IV=0.34, KS=0.28 | L3: delta-Recall=+0.8pp

    Key constraints:
    - Keep it COMPACT - this must fit in a single column (~3.5 inch width)
    - Use thin borders, small padding, tight spacing
    - Each section should have a small section label/header
    - The card should look professional and structured, like a form or report card
    - Use color sparingly: light blue, light gray, light green for section backgrounds
    """

    caption = (
        "Provenance card for the fan_in_fan_out_ratio feature, showing "
        "knowledge assets, reasoning chain, generated code, and verification results."
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

    out = Path("/home/ubuntu/Projects/大模型提取特征/paper/figures/provenance_card.png")
    shutil.copy(result.image_path, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
