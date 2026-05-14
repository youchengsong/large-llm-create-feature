"""搜索引擎主程序：维度岛屿进化搜索，整合 LLM 生成、沙盒执行、三级验证。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from src.config import Config, DEFAULT_CONFIG
from src.data.merge import merge_feature_to_transactions
from src.evaluation.validator import (
    run_l1_validation,
    run_l2_validation,
    run_l3_validation,
    compute_composite_score,
)
from src.llm.interface import LLMInterface
from src.llm.parser import parse_llm_output
from src.prompt.builder import build_user_prompt, build_user_prompt_cold_start
from src.search.island import Island, IslandManager
from src.search.memory import FailureMemory
from src.search.shap_feedback import generate_shap_feedback, shap_feedback_to_prompt, compute_island_budgets

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    accepted_features: list[dict] = field(default_factory=list)
    search_log: list[dict] = field(default_factory=list)
    total_rounds: int = 0
    converged: bool = False


class SearchEngine:
    def __init__(
        self,
        llm: LLMInterface,
        system_prompt: str,
        seed_examples: str,
        open_df: pd.DataFrame,
        tx_df: pd.DataFrame,
        close_df: pd.DataFrame,
        label: pd.Series,
        baseline_features: pd.DataFrame,
        baseline_prauc: float,
        config: Config | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.seed_examples = seed_examples
        self.open_df = open_df
        self.tx_df = tx_df
        self.close_df = close_df
        self.label = label
        self.baseline_prauc = baseline_prauc

        self.cfg = config or DEFAULT_CONFIG
        self.island_manager = IslandManager(self.cfg.islands)
        self.failure_memory = FailureMemory()

        self.existing_features = baseline_features.copy()
        self.accepted: list[dict] = []
        self.all_scores: dict[str, list[float]] = {
            "shap_rank": [],
            "mi": [],
            "redundancy": [],
            "delta_prauc": [],
        }
        self._rounds_without_new = 0

    def run_single_island_step(
        self, island: Island, round_num: int, n_candidates: int = 3
    ) -> list[dict]:
        """单岛屿一轮：LLM 生成 → 解析 → L1 验证（失败重试最多 2 次）。"""
        if round_num == 1 and not island.memory:
            user_prompt = build_user_prompt_cold_start(
                island_name=island.name,
                island_fields=island.fields,
                seed_examples=self.seed_examples,
                n_candidates=n_candidates,
            )
        else:
            shap_fb = {}
            if self.existing_features.shape[1] > 0:
                try:
                    shap_fb = generate_shap_feedback(self.existing_features, self.label)
                except Exception as e:
                    logger.warning("SHAP feedback failed: %s", e)

            shap_text = shap_feedback_to_prompt(shap_fb, [island.dimension])

            top_features = island.boltzmann_sample(
                k=self.cfg.search.k, tau=self.cfg.search.tau
            )

            user_prompt = build_user_prompt(
                island_name=island.name,
                island_fields=island.fields,
                shap_feedback=shap_text,
                top_features=top_features,
                failed_features=self.failure_memory.get_recent(5),
                n_candidates=n_candidates,
            )

        raw_output = self.llm.generate(self.system_prompt, user_prompt)
        candidates = parse_llm_output(raw_output)

        validated: list[dict] = []
        for cand in candidates:
            code = cand.get("code", "")
            name = cand.get("feature_name", "unknown")

            feature_df, error = self._try_l1_with_retry(code, name, max_retries=2)
            if feature_df is not None:
                cand["_feature_df"] = feature_df
                validated.append(cand)
            else:
                self.failure_memory.add_failure(name, code, error)
                logger.info("L1 failed for %s: %s", name, error)

        return validated

    def _try_l1_with_retry(
        self, code: str, name: str, max_retries: int = 2
    ) -> tuple[pd.DataFrame | None, str]:
        """L1 验证，失败时让 LLM 修复代码并重试。"""
        feature_df, error = run_l1_validation(
            code, self.open_df, self.tx_df, self.close_df
        )
        if feature_df is not None:
            return feature_df, ""

        for attempt in range(max_retries):
            fix_prompt = (
                f"特征 '{name}' 的代码执行失败。\n"
                f"错误信息: {error}\n\n"
                f"原始代码:\n```python\n{code}\n```\n\n"
                f"请修复代码并重新输出完整的 JSON 数组（只包含这一个特征）。"
            )
            try:
                raw = self.llm.generate(self.system_prompt, fix_prompt)
                parsed = parse_llm_output(raw)
                if not parsed:
                    continue
                code = parsed[0].get("code", "")
                feature_df, error = run_l1_validation(
                    code, self.open_df, self.tx_df, self.close_df
                )
                if feature_df is not None:
                    return feature_df, ""
            except Exception as e:
                error = str(e)
                logger.warning("L1 retry %d failed for %s: %s", attempt + 1, name, e)

        return None, error

    def evaluate_candidates(
        self, candidates: list[dict], island: Island
    ) -> list[dict]:
        """L1 已通过的候选 → L2 统计验证 → L3 模型验证 → 打分。"""
        accepted_this_step: list[dict] = []

        for cand in candidates:
            feature_df = cand.pop("_feature_df", None)
            if feature_df is None:
                continue

            name = cand.get("feature_name", "unknown")

            try:
                merged = merge_feature_to_transactions(feature_df, self.tx_df)
            except Exception as e:
                self.failure_memory.add_failure(name, cand.get("code", ""), f"merge 失败: {e}")
                logger.info("Merge failed for %s: %s", name, e)
                continue

            if len(merged) != len(self.tx_df):
                self.failure_memory.add_failure(
                    name, cand.get("code", ""),
                    f"merge 行数不匹配: {len(merged)} vs {len(self.tx_df)}"
                )
                logger.info("Merge row mismatch for %s: %d vs %d", name, len(merged), len(self.tx_df))
                continue

            feat_cols = [c for c in merged.columns if c not in self.tx_df.columns]
            if not feat_cols:
                self.failure_memory.add_failure(name, cand.get("code", ""), "merge 后无新列")
                continue

            feature_series = merged[feat_cols[0]].fillna(0)
            feature_series.name = name

            l2_passed, l2_metrics = run_l2_validation(
                feature_series, self.label, self.existing_features
            )
            cand["l2_metrics"] = l2_metrics

            if not l2_passed:
                reason = (
                    f"L2 未通过: MI={l2_metrics['mi']:.4f}, "
                    f"redundancy={l2_metrics['redundancy']:.4f}, "
                    f"cramers_v={l2_metrics['cramers_v']:.4f}, "
                    f"bootstrap_p={l2_metrics['bootstrap_p']:.4e}"
                )
                self.failure_memory.add_failure(name, cand.get("code", ""), reason)
                logger.info("L2 failed for %s: %s", name, reason)
                continue

            l3_passed, l3_metrics = run_l3_validation(
                self.existing_features, feature_series, self.label, self.baseline_prauc
            )
            cand["l3_metrics"] = l3_metrics

            if not l3_passed:
                reason = f"L3 未通过: delta_prauc={l3_metrics['delta_prauc']:.6f}"
                self.failure_memory.add_failure(name, cand.get("code", ""), reason)
                logger.info("L3 failed for %s: %s", name, reason)
                continue

            self.all_scores["shap_rank"].append(l3_metrics["shap_rank"])
            self.all_scores["mi"].append(l2_metrics["mi"])
            self.all_scores["redundancy"].append(l2_metrics["redundancy"])
            self.all_scores["delta_prauc"].append(l3_metrics["delta_prauc"])

            score = compute_composite_score(
                shap_rank=l3_metrics["shap_rank"],
                mi=l2_metrics["mi"],
                redundancy=l2_metrics["redundancy"],
                delta_prauc=l3_metrics["delta_prauc"],
                all_scores=self.all_scores,
            )
            cand["score"] = score

            self.existing_features[name] = feature_series.values

            feature_record = {
                "name": name,
                "feature_name": name,
                "score": score,
                "code": cand.get("code", ""),
                "explanation": cand.get("explanation", cand.get("rationale", "")),
                "l2_metrics": l2_metrics,
                "l3_metrics": l3_metrics,
                "shap_rank": l3_metrics["shap_rank"],
                "mi": l2_metrics["mi"],
            }

            island.add_feature(feature_record)
            self.island_manager.migrate_best(feature_record)
            self.accepted.append(feature_record)
            accepted_this_step.append(feature_record)

            logger.info(
                "Accepted feature %s: score=%.4f, delta_prauc=%.6f",
                name, score, l3_metrics["delta_prauc"],
            )

        return accepted_this_step

    def _compute_budgets(self, round_num: int) -> dict[str, int]:
        """计算每个岛屿的候选预算。第 1 轮均匀分配，之后根据 SHAP 动态分配。"""
        island_dims = [isl.dimension for isl in self.island_manager.islands]

        if round_num == 1 or self.existing_features.shape[1] == 0:
            return {dim: self.cfg.search.b for dim in island_dims}

        try:
            shap_fb = generate_shap_feedback(self.existing_features, self.label)
            dim_importance = {}
            for ic in self.cfg.islands:
                dim_vals = [
                    shap_fb["feature_importance"].get(f, 0.0)
                    for f in self.existing_features.columns
                    if any(field.lower() in f.lower() for field in ic.fields)
                ]
                dim_importance[ic.dimension] = sum(dim_vals) if dim_vals else 0.0

            return compute_island_budgets(
                dim_importance, island_dims,
                B_total=self.cfg.search.B_total,
                b_min=self.cfg.search.b_min,
                b_max=self.cfg.search.b_max,
            )
        except Exception as e:
            logger.warning("Dynamic budget computation failed: %s, using uniform", e)
            return {dim: self.cfg.search.b for dim in island_dims}

    def run_round(self, round_num: int) -> dict:
        """一轮完整迭代：遍历所有岛屿，根据 SHAP 动态分配候选预算。"""
        budgets = self._compute_budgets(round_num)

        round_log = {
            "round": round_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "islands": [],
            "new_features": 0,
            "budgets": {dim: b for dim, b in budgets.items()},
        }

        new_count = 0
        for island in self.island_manager.islands:
            n_candidates = budgets.get(island.dimension, self.cfg.search.b)
            island_log = {
                "island": island.name,
                "budget": n_candidates,
                "candidates": 0,
                "accepted": 0,
                "details": [],
            }

            try:
                validated = self.run_single_island_step(island, round_num, n_candidates)
                island_log["candidates"] = len(validated)

                accepted = self.evaluate_candidates(validated, island)
                island_log["accepted"] = len(accepted)
                island_log["details"] = [
                    {"name": f["name"], "score": f["score"]} for f in accepted
                ]
                new_count += len(accepted)
            except Exception as e:
                logger.error("Error in island %s round %d: %s", island.name, round_num, e)
                island_log["error"] = str(e)

            round_log["islands"].append(island_log)

        round_log["new_features"] = new_count
        return round_log

    def check_convergence(self, round_log: dict) -> bool:
        """连续 2 轮无新特征则收敛。"""
        if round_log["new_features"] == 0:
            self._rounds_without_new += 1
        else:
            self._rounds_without_new = 0
        return self._rounds_without_new >= 2

    def run(self, max_rounds: int | None = None) -> SearchResult:
        """主循环。"""
        if max_rounds is None:
            max_rounds = self.cfg.search.T

        result = SearchResult()

        for round_num in range(1, max_rounds + 1):
            logger.info("=== Round %d/%d ===", round_num, max_rounds)

            round_log = self.run_round(round_num)
            result.search_log.append(round_log)
            result.total_rounds = round_num

            logger.info(
                "Round %d: %d new features, %d total accepted",
                round_num, round_log["new_features"], len(self.accepted),
            )

            if self.check_convergence(round_log):
                logger.info("Converged after %d rounds", round_num)
                result.converged = True
                break

        result.accepted_features = list(self.accepted)
        return result
