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
    COMPACT layout - minimize vertical space. Use standard fonts: Times New Roman or Helvetica, 8-9pt.
    White background, no grid lines, clean academic style.

    Data (sorted by rank, best at top, 8 methods):
    | Method    | Avg. Rank |
    |-----------|-----------|
    | KnowFeat  | 1.4       |
    | OpenFE    | 2.3       |
    | No-FE     | 4.7       |
    | FeatLLM   | 5.1       |
    | LLM-FE    | 5.1       |
    | AutoFeat  | 5.5       |
    | CAAFE     | 5.9       |
    | OCTree    | 5.9       |

    - KnowFeat bar should be a distinct darker blue color, all others in light gray
    - Show the numeric rank value at the end of each bar
    - X-axis: "Average Rank (lower is better)" from 0 to 7
    - Y-axis: method names
    - Thin bars with small spacing to keep the figure compact
    - No title (caption will be in LaTeX)
    - Remove top and right spines
    """

    raw_data = {
        "methods": ["KnowFeat", "OpenFE", "No-FE", "FeatLLM", "LLM-FE", "AutoFeat", "CAAFE", "OCTree"],
        "avg_rank": [1.4, 2.3, 4.7, 5.1, 5.1, 5.5, 5.9, 5.9],
    }

    caption = "Average rank across 7 public datasets (lower is better). KnowFeat achieves the best average rank (1.4) among 8 methods."

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
