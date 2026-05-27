"""跨维度知识传播：高分特征信息传播到其他维度作为灵感来源。

每轮 SHAP 反馈后，将各维度高分特征的摘要（名称、代码、指标、SHAP 排名）
传播到其他维度的 prompt 中，可能激发跨维度交叉特征。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from knowfeat.config import KnowFeatConfig, SearchConfig
from knowfeat.prompt.builder import CrossDimExample
from knowfeat.search.shap_feedback import FeatureScore

logger = logging.getLogger(__name__)


@dataclass
class AcceptedFeature:
    """已接受特征的完整信息（用于跨维度传播）。"""
    name: str
    dimensions: list[str]
    code: str
    shap_rank: int = 0
    composite_score: float = 0.0
    iv: float = 0.0
    delta_recall: float = 0.0


class CrossDimPropagator:
    """跨维度知识传播管理器。

    维护已接受特征的注册表，每轮根据 SHAP 反馈更新排名，
    为指定目标维度生成排除自身维度的高分特征示例列表。

    Parameters
    ----------
    cfg : KnowFeatConfig, optional
        全局配置。
    search_cfg : SearchConfig, optional
        搜索配置，优先于 cfg.search。
    """

    def __init__(
        self,
        cfg: KnowFeatConfig | None = None,
        search_cfg: SearchConfig | None = None,
    ):
        if search_cfg is None:
            search_cfg = cfg.search if cfg is not None else SearchConfig()
        self._search_cfg = search_cfg
        self._features: dict[str, AcceptedFeature] = {}

    @property
    def feature_count(self) -> int:
        return len(self._features)

    def register_feature(
        self,
        name: str,
        dimensions: list[str],
        code: str,
        iv: float = 0.0,
        delta_recall: float = 0.0,
    ) -> None:
        """注册一个已接受的特征。

        Parameters
        ----------
        name : 特征名。
        dimensions : 所属维度列表。
        code : 特征生成代码。
        iv : IV 值。
        delta_recall : delta-recall 值。
        """
        self._features[name] = AcceptedFeature(
            name=name,
            dimensions=dimensions,
            code=code,
            iv=iv,
            delta_recall=delta_recall,
        )
        logger.debug("注册跨维度特征: %s (维度=%s)", name, dimensions)

    def update_from_shap(self, feature_scores: list[FeatureScore]) -> None:
        """根据 SHAP 反馈更新已注册特征的排名和分数。

        Parameters
        ----------
        feature_scores : SHAP 分析产出的特征评分列表。
        """
        score_map = {fs.name: fs for fs in feature_scores}
        for name, feat in self._features.items():
            if name in score_map:
                fs = score_map[name]
                feat.shap_rank = fs.shap_rank
                feat.composite_score = fs.composite_score
                feat.iv = fs.iv
                feat.delta_recall = fs.delta_recall

        logger.info(
            "跨维度传播: 更新 %d/%d 特征的 SHAP 排名",
            sum(1 for n in self._features if n in score_map),
            len(self._features),
        )

    def get_examples_for_dimension(
        self,
        target_dimension: str,
        top_k: int | None = None,
    ) -> list[CrossDimExample]:
        """为目标维度生成跨维度高分特征示例。

        排除属于目标维度的特征，按复合分数降序取 top_k 个。

        Parameters
        ----------
        target_dimension : 当前 Agent 正在探索的维度。
        top_k : 最多返回多少个示例。默认使用配置的 cross_dim_top_k。

        Returns
        -------
        list[CrossDimExample]
            供 prompt 注入的跨维度特征示例。
        """
        if top_k is None:
            top_k = self._search_cfg.cross_dim_top_k

        candidates = [
            f for f in self._features.values()
            if target_dimension not in f.dimensions
        ]

        candidates.sort(key=lambda f: f.composite_score, reverse=True)
        selected = candidates[:top_k]

        examples = []
        for feat in selected:
            source_dim = feat.dimensions[0] if feat.dimensions else "未知"
            examples.append(CrossDimExample(
                feature_name=feat.name,
                source_dimension=source_dim,
                code=feat.code,
                metrics={
                    "composite_score": feat.composite_score,
                    "iv": feat.iv,
                    "delta_recall": feat.delta_recall,
                },
                shap_rank=feat.shap_rank,
            ))

        return examples

    def remove_feature(self, name: str) -> bool:
        """移除一个已注册的特征。

        Parameters
        ----------
        name : 特征名。

        Returns
        -------
        bool
            是否成功移除。
        """
        if name in self._features:
            del self._features[name]
            return True
        return False

    def get_all_features(self) -> list[AcceptedFeature]:
        """返回所有已注册特征（按复合分数降序）。"""
        return sorted(
            self._features.values(),
            key=lambda f: f.composite_score,
            reverse=True,
        )

    def summary(self) -> str:
        """返回跨维度传播状态摘要文本。"""
        if not self._features:
            return "跨维度传播: 暂无已注册特征。"

        lines = ["跨维度传播状态:"]
        lines.append(f"  已注册特征: {len(self._features)}")

        dim_counts: dict[str, int] = {}
        for feat in self._features.values():
            for d in feat.dimensions:
                dim_counts[d] = dim_counts.get(d, 0) + 1

        for dim, count in sorted(dim_counts.items()):
            lines.append(f"  {dim}: {count} 个特征")

        top = self.get_all_features()[:3]
        if top:
            lines.append("  Top 3:")
            for f in top:
                lines.append(
                    f"    {f.name} (score={f.composite_score:.4f}, "
                    f"rank={f.shap_rank})"
                )

        return "\n".join(lines)


def migrate_best(
    feature_scores: list[FeatureScore],
    feature_codes: dict[str, str],
    feature_dimensions: dict[str, list[str]],
    target_dimension: str,
    top_k: int = 5,
) -> list[CrossDimExample]:
    """一次性从 SHAP 反馈中提取跨维度高分特征示例（无状态便捷函数）。

    Parameters
    ----------
    feature_scores : SHAP 分析产出的特征评分列表。
    feature_codes : {特征名: 代码}。
    feature_dimensions : {特征名: [维度列表]}。
    target_dimension : 当前 Agent 正在探索的维度。
    top_k : 最多返回多少个示例。

    Returns
    -------
    list[CrossDimExample]
        供 prompt 注入的跨维度特征示例。
    """
    candidates = [
        fs for fs in feature_scores
        if target_dimension not in feature_dimensions.get(fs.name, [])
        and fs.name in feature_codes
    ]

    candidates.sort(key=lambda fs: fs.composite_score, reverse=True)
    selected = candidates[:top_k]

    examples = []
    for fs in selected:
        dims = feature_dimensions.get(fs.name, [])
        source_dim = dims[0] if dims else "未知"
        examples.append(CrossDimExample(
            feature_name=fs.name,
            source_dimension=source_dim,
            code=feature_codes[fs.name],
            metrics={
                "composite_score": fs.composite_score,
                "iv": fs.iv,
                "delta_recall": fs.delta_recall,
            },
            shap_rank=fs.shap_rank,
        ))

    logger.info(
        "migrate_best: 为维度「%s」提取 %d 个跨维度示例",
        target_dimension, len(examples),
    )
    return examples
