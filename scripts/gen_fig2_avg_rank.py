"""Generate Figure 2: Average Rank Bar Chart."""

import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/ubuntu/paperbanana/.env")

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings


async def main():
    source_context = """
    Table: Average rank of 7 automated feature engineering methods across 7 public datasets.
    Lower rank is better (rank 1 = best on that dataset).

    | Method    | Avg. Rank |
    |-----------|-----------|
    | KnowFeat  | 1.4       |
    | OpenFE    | 2.1       |
    | No-FE     | 4.5       |
    | FeatLLM   | 4.6       |
    | LLM-FE    | 4.7       |
    | AutoFeat  | 5.1       |
    | CAAFE     | 5.5       |

    This is for an academic paper comparing automated feature engineering methods.
    KnowFeat (our method) achieves the best average rank (1.4), substantially
    outperforming the second-best OpenFE (2.1). The chart should clearly highlight
    KnowFeat as the best method. Use a horizontal bar chart sorted by rank (best at top).
    Use a distinct color (e.g., blue or teal) for KnowFeat and a neutral gray for others.
    The x-axis should show "Average Rank" from 0 to 6, and y-axis lists method names.
    Include the numeric rank value at the end of each bar.
    Academic style: clean, no grid clutter, white background, serif or sans-serif font.
    """

    raw_data = {
        "methods": ["KnowFeat", "OpenFE", "No-FE", "FeatLLM", "LLM-FE", "AutoFeat", "CAAFE"],
        "avg_rank": [1.4, 2.1, 4.5, 4.6, 4.7, 5.1, 5.5],
    }

    caption = "Average rank across 7 public datasets (lower is better). KnowFeat achieves the best average rank (1.4)."

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
