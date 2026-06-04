"""Generate Figure 3: Feature Count vs Average Rank Scatter Plot."""

import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/ubuntu/paperbanana/.env")

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings


async def main():
    source_context = """
    Scatter plot showing the relationship between the average number of features
    generated and the average rank across 7 public datasets.

    Data:
    | Method    | Avg Features Added | Avg Rank |
    |-----------|-------------------|----------|
    | KnowFeat  | 28                | 1.4      |
    | OpenFE    | 42                | 2.1      |
    | No-FE     | 0                 | 4.5      |
    | FeatLLM   | 151               | 4.6      |
    | LLM-FE    | 17                | 4.7      |
    | AutoFeat  | 41                | 5.1      |
    | CAAFE     | 6                 | 5.5      |

    X-axis: Average number of features generated (linear scale, 0 to 160).
    Y-axis: Average Rank (inverted, so 1 is at the top and 6 at the bottom).
    Label each data point with the method name next to the marker.
    Use a filled circle marker for all methods. Use a different color for
    KnowFeat (e.g. dark blue) vs others (e.g. gray).
    Clean academic style with white background.
    """

    raw_data = {
        "methods": ["KnowFeat", "OpenFE", "No-FE", "FeatLLM", "LLM-FE", "AutoFeat", "CAAFE"],
        "avg_features_added": [28, 42, 0, 151, 17, 41, 6],
        "avg_rank": [1.4, 2.1, 4.5, 4.6, 4.7, 5.1, 5.5],
    }

    caption = (
        "Trade-off between feature quantity and average rank. "
        "KnowFeat achieves the best rank with a moderate number of verified features."
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
            diagram_type=DiagramType.STATISTICAL_PLOT,
            raw_data=raw_data,
        )
    )

    out = Path("/home/ubuntu/Projects/大模型提取特征/paper/figures/feat_vs_rank.png")
    shutil.copy(result.image_path, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
