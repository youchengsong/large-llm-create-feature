"""Generate Figure 2: Average Rank Bar Chart - compact for single column."""

import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/ubuntu/paperbanana/.env")

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings


async def main():
    source_context = """
    Horizontal bar chart for an IEEE single-column figure (width ~3.5 inches, height ~1.8 inches).
    COMPACT layout - minimize vertical space. MUST use Times New Roman font for ALL text (axis labels, tick labels, bar annotations) to match the IEEE paper body font. Font size 8-9pt.
    White background, no grid lines, clean academic style.

    Data (sorted by rank, best at top, 7 methods, 12 public datasets):
    | Method    | Avg. Rank |
    |-----------|-----------|
    | KnowFeat  | 2.33      |
    | OpenFE    | 3.77      |
    | AutoFeat  | 3.78      |
    | LLM-FE    | 3.83      |
    | No-FE     | 4.00      |
    | CAAFE     | 4.45      |
    | FeatLLM   | 4.62      |

    - KnowFeat bar should be a distinct darker blue color, all others in light gray
    - Show the numeric rank value at the end of each bar
    - X-axis: "Average Rank (lower is better)" from 0 to 7
    - Y-axis: method names
    - Thin bars with small spacing to keep the figure compact
    - No title (caption will be in LaTeX)
    - Remove top and right spines
    """

    raw_data = {
        "methods": ["KnowFeat", "OpenFE", "AutoFeat", "LLM-FE", "No-FE", "CAAFE", "FeatLLM"],
        "avg_rank": [2.33, 3.77, 3.78, 3.83, 4.00, 4.45, 4.62],
    }

    caption = "Average rank across 12 public datasets (lower is better). KnowFeat achieves the best average rank (2.3) among 7 methods."

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
            diagram_type=DiagramType.STATISTICAL_PLOT,
            raw_data=raw_data,
        )
    )

    out = Path("/home/ubuntu/Projects/大模型提取特征/paper/figures/avg_rank.png")
    shutil.copy(result.image_path, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
