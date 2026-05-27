"""维度管理：4 维度定义、预算分配与动态更新。

管理特征生成的四个维度（金额频次 / 时间地域 / 行为序列 / 主体关联），
跟踪每个维度的预算消耗，根据 SHAP 反馈按贡献反比调整下一轮预算。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from knowfeat.config import KnowFeatConfig, SearchConfig

logger = logging.getLogger(__name__)


@dataclass
class DimensionState:
    """单个维度的运行时状态。"""
    name: str
    budget: int = 0
    generated: int = 0
    accepted: int = 0
    candidates: int = 0
    rejected: int = 0
    shap_contribution: float = 0.0


class DimensionManager:
    """四维度预算管理器。

    Parameters
    ----------
    cfg : KnowFeatConfig, optional
        全局配置，使用其中的 SearchConfig。
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

        self._dims: dict[str, DimensionState] = {}
        for dim in search_cfg.dimensions:
            self._dims[dim] = DimensionState(
                name=dim,
                budget=search_cfg.initial_budget_per_dim,
            )

    @property
    def dimensions(self) -> list[str]:
        return list(self._dims.keys())

    @property
    def total_budget(self) -> int:
        return sum(d.budget for d in self._dims.values())

    @property
    def total_accepted(self) -> int:
        return sum(d.accepted for d in self._dims.values())

    def get_state(self, dimension: str) -> DimensionState:
        """返回指定维度的状态快照。"""
        if dimension not in self._dims:
            raise ValueError(f"未知维度: {dimension}")
        return self._dims[dimension]

    def all_states(self) -> list[DimensionState]:
        """返回所有维度状态（按名称顺序）。"""
        return [self._dims[d] for d in self._search_cfg.dimensions
                if d in self._dims]

    def get_budget(self, dimension: str) -> int:
        return self.get_state(dimension).budget

    def record_round(
        self,
        dimension: str,
        generated: int,
        accepted: int,
        candidates: int = 0,
        rejected: int = 0,
    ) -> None:
        """记录一轮 Agent 探索的结果（累加）。

        Parameters
        ----------
        dimension : 维度名称。
        generated : 本轮生成特征数。
        accepted : 本轮直接通过 L3 的特征数。
        candidates : 本轮进入候选池的特征数。
        rejected : 本轮被拒绝的特征数。
        """
        state = self.get_state(dimension)
        state.generated += generated
        state.accepted += accepted
        state.candidates += candidates
        state.rejected += rejected

    def record_feature(self, dimensions: list[str], *, accepted: bool) -> None:
        """记录单个特征的维度归属（支持多维度标签）。

        对于多维度标签的特征，每个相关维度都会计数。

        Parameters
        ----------
        dimensions : 特征声明的维度列表。
        accepted : True 表示通过 L3 / 前向选择被接受，False 表示进入候选池。
        """
        for dim in dimensions:
            if dim not in self._dims:
                logger.warning("特征声明了未知维度 '%s'，忽略", dim)
                continue
            if accepted:
                self._dims[dim].accepted += 1
            else:
                self._dims[dim].candidates += 1

    def update_budgets_from_shap(
        self,
        dimension_contributions: dict[str, float],
        total_budget: int | None = None,
        min_budget: int = 1,
    ) -> dict[str, int]:
        """根据 SHAP 维度贡献的反比分配下一轮预算。

        薄弱维度（SHAP 贡献低）获得更多预算，强维度获得较少。

        Parameters
        ----------
        dimension_contributions : {维度名: SHAP 贡献分数}。
        total_budget : 下一轮总预算。默认使用 initial_budget_per_dim * 维度数。
        min_budget : 每个维度的最低预算保障。

        Returns
        -------
        dict[str, int]
            更新后的预算分配 {维度名: 预算}。
        """
        n_dims = len(self._dims)
        if total_budget is None:
            total_budget = self._search_cfg.initial_budget_per_dim * n_dims

        for dim, contrib in dimension_contributions.items():
            if dim in self._dims:
                self._dims[dim].shap_contribution = contrib

        active_dims = [d for d in self._dims if d in dimension_contributions]
        if not active_dims:
            budget_each = max(min_budget, total_budget // n_dims)
            for d in self._dims.values():
                d.budget = budget_each
            return {d: budget_each for d in self._dims}

        contribs = {d: max(dimension_contributions.get(d, 0.0), 1e-9)
                    for d in active_dims}

        inv = {d: 1.0 / c for d, c in contribs.items()}
        inv_sum = sum(inv.values())

        # Inverse-proportion allocation with minimum guarantee
        allocatable = total_budget - min_budget * n_dims
        if allocatable < 0:
            allocatable = 0

        budgets: dict[str, int] = {}
        for dim in self._dims:
            if dim in inv:
                raw = min_budget + allocatable * (inv[dim] / inv_sum)
            else:
                raw = min_budget + allocatable / n_dims
            budgets[dim] = max(min_budget, int(round(raw)))

        # Correct rounding errors
        diff = total_budget - sum(budgets.values())
        if diff != 0:
            sorted_dims = sorted(budgets, key=lambda d: budgets[d])
            if diff > 0:
                for d in sorted_dims:
                    budgets[d] += 1
                    diff -= 1
                    if diff == 0:
                        break
            else:
                for d in reversed(sorted_dims):
                    if budgets[d] > min_budget:
                        reduce = min(budgets[d] - min_budget, -diff)
                        budgets[d] -= reduce
                        diff += reduce
                    if diff == 0:
                        break

        for dim, b in budgets.items():
            self._dims[dim].budget = b

        logger.info(
            "SHAP 反比预算分配: %s (总预算=%d)",
            {d: budgets[d] for d in self._search_cfg.dimensions if d in budgets},
            total_budget,
        )
        return budgets

    def reset_budgets(self, budget_per_dim: int | None = None) -> None:
        """重置所有维度预算为均匀分配。"""
        b = budget_per_dim or self._search_cfg.initial_budget_per_dim
        for d in self._dims.values():
            d.budget = b

    def summary(self) -> str:
        """返回维度状态摘要文本。"""
        lines = ["维度 | 预算 | 已生成 | 已接受 | 候选 | 拒绝 | SHAP贡献"]
        lines.append("--- | --- | --- | --- | --- | --- | ---")
        for dim in self._search_cfg.dimensions:
            s = self._dims[dim]
            lines.append(
                f"{s.name} | {s.budget} | {s.generated} | "
                f"{s.accepted} | {s.candidates} | {s.rejected} | "
                f"{s.shap_contribution:.4f}"
            )
        return "\n".join(lines)
