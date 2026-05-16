"""搜索引擎主程序：维度岛屿进化搜索，整合 LLM 生成、沙盒执行、三级验证。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from src.config import Config, DEFAULT_CONFIG, ValidationThresholds
from src.data.merge import merge_feature_to_transactions
from src.evaluation.validator import (
    run_l1_validation,
    run_l2_validation,
    run_l3_validation,
    backward_eliminate_pending,
    compute_composite_score,
)
from src.llm.interface import LLMInterface
from src.data.stats_query import execute_stats_query
from src.llm.parser import parse_llm_output, parse_llm_output_mixed, extract_json_from_text
from src.prompt.builder import build_user_prompt, build_user_prompt_cold_start
from src.search.island import Island, IslandManager
from src.search.memory import FailureMemory
from src.search.pending_pool import PendingPool, PendingFeature
from src.search.shap_feedback import generate_shap_feedback, shap_feedback_to_prompt, compute_island_budgets

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    accepted_features: list[dict] = field(default_factory=list)
    search_log: list[dict] = field(default_factory=list)
    total_rounds: int = 0
    converged: bool = False
    calibrated_thresholds: dict | None = None
    baseline_metrics: dict | None = None
    combined_metrics: dict | None = None


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
        validation_thresholds: ValidationThresholds | None = None,
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
        self.validation_thresholds = validation_thresholds
        self.island_manager = IslandManager(self.cfg.islands)
        self.failure_memory = FailureMemory()
        self.pending_pool = PendingPool(
            max_size=self.cfg.pending_pool.max_size,
            max_age=self.cfg.pending_pool.max_age,
            max_rejects=self.cfg.pending_pool.max_rejects,
        )

        self.existing_features = baseline_features.copy()
        self.accepted: list[dict] = []
        self.all_scores: dict[str, list[float]] = {
            "shap_rank": [],
            "iv": [],
            "redundancy": [],
            "delta_f1": [],
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

    def _generate_explainability(
        self,
        name: str,
        code: str,
        explanation: str,
        l2_metrics: dict,
        l3_metrics: dict,
        feature_series: pd.Series,
    ) -> dict:
        """通过 L3 后调用 LLM 生成特征可解释性报告。"""
        normal_vals = feature_series[self.label == 0]
        abnormal_vals = feature_series[self.label == 1]

        prompt = f"""特征 {name} 已通过三级验证，请为它生成可解释性报告。

## 特征信息
- 名称: {name}
- 说明: {explanation}
- 代码: ```python
{code}
```

## 验证指标
- SHAP Rank: #{l3_metrics.get('shap_rank', 'N/A')}（在模型所有特征中的重要性排名）
- IV: {l2_metrics.get('iv', 0):.4f}
- KS: {l2_metrics.get('ks', 0):.4f}
- Delta F1: {l3_metrics.get('delta_f1', 0):.6f}
- Delta Recall: {l3_metrics.get('delta_recall', 0):.6f}
- Delta Precision: {l3_metrics.get('delta_precision', 0):.6f}

## 该特征在正常 vs 异常交易上的分布
- 正常交易: 中位数={normal_vals.median():.4f}, 均值={normal_vals.mean():.4f}, std={normal_vals.std():.4f}
- 异常交易: 中位数={abnormal_vals.median():.4f}, 均值={abnormal_vals.mean():.4f}, std={abnormal_vals.std():.4f}
- 正常交易的 [25%, 75%] 分位: [{normal_vals.quantile(0.25):.4f}, {normal_vals.quantile(0.75):.4f}]
- 异常交易的 [25%, 75%] 分位: [{abnormal_vals.quantile(0.25):.4f}, {abnormal_vals.quantile(0.75):.4f}]

请输出一个 JSON 对象，包含以下字段：
- value_meaning: 不同取值区间的业务含义（如"值=1表示正常，值>3表示批量注册嫌疑"）
- risk_threshold: 高风险阈值及判断依据（具体数值+为什么选这个阈值）
- synergy: 建议与哪些已有特征配合使用、为什么能协同
- false_positive_cases: 可能的误判场景（什么情况下正常用户也会触发高值）
- regulatory_explanation: 面向监管审计人员的一句话解释（简洁、非技术语言）"""

        try:
            raw = self.llm.generate(self.system_prompt, prompt)
            json_str = extract_json_from_text(raw)
            if json_str:
                import json as _json
                report = _json.loads(json_str)
                if isinstance(report, dict):
                    logger.info("Explainability report generated for %s", name)
                    return report
            if isinstance(raw, str) and len(raw) > 20:
                return {"raw_explanation": raw}
        except Exception as e:
            logger.warning("Explainability generation failed for %s: %s", name, e)

        return {}

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

            if any(f["name"] == name for f in self.accepted):
                logger.info("Skipped duplicate feature %s (already accepted)", name)
                continue

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
                feature_series, self.label, self.existing_features,
                thresholds=self.validation_thresholds,
            )
            cand["l2_metrics"] = l2_metrics

            if not l2_passed:
                th = self.validation_thresholds or self.cfg.validation
                failed_checks = []
                if l2_metrics["iv"] < th.iv_min:
                    failed_checks.append(f"IV={l2_metrics['iv']:.6f}<{th.iv_min:.6f}")
                if l2_metrics["ks"] < th.ks_min:
                    failed_checks.append(f"KS={l2_metrics['ks']:.6f}<{th.ks_min:.6f}")
                if l2_metrics["redundancy"] >= th.corr_max:
                    failed_checks.append(f"redundancy={l2_metrics['redundancy']:.4f}>={th.corr_max}")
                if l2_metrics["missing_rate"] > th.missing_rate_max:
                    failed_checks.append(f"missing={l2_metrics['missing_rate']:.2%}>{th.missing_rate_max:.0%}")
                reason = f"L2 未通过: {'; '.join(failed_checks)}"
                self.failure_memory.add_failure(name, cand.get("code", ""), reason)
                logger.info("L2 failed for %s: %s", name, reason)
                continue

            l3_passed, l3_metrics = run_l3_validation(
                self.existing_features, feature_series, self.label,
                baseline_metrics=getattr(self, '_baseline_metrics', {}),
            )
            cand["l3_metrics"] = l3_metrics

            if not l3_passed:
                pending_feat = PendingFeature(
                    name=name,
                    code=cand.get("code", ""),
                    explanation=cand.get("explanation", cand.get("rationale", "")),
                    reasoning=cand.get("reasoning", ""),
                    feature_values=feature_series.values,
                    l2_metrics=l2_metrics,
                    l3_metrics=l3_metrics,
                    island_name=island.name,
                )
                self.pending_pool.add(pending_feat)
                logger.info("L3 failed for %s (delta_f1=%.6f, delta_recall=%.6f, delta_precision=%.6f), "
                            "added to pending pool (%d in pool)",
                            name, l3_metrics.get("delta_f1", 0), l3_metrics.get("delta_recall", 0),
                            l3_metrics.get("delta_precision", 0), len(self.pending_pool))
                continue

            self.all_scores["shap_rank"].append(l3_metrics["shap_rank"])
            self.all_scores["iv"].append(l2_metrics["iv"])
            self.all_scores["redundancy"].append(l2_metrics["redundancy"])
            self.all_scores["delta_f1"].append(l3_metrics.get("delta_f1", 0))

            score = compute_composite_score(
                shap_rank=l3_metrics["shap_rank"],
                iv=l2_metrics["iv"],
                redundancy=l2_metrics["redundancy"],
                delta_f1=l3_metrics.get("delta_f1", 0),
                all_scores=self.all_scores,
            )
            cand["score"] = score

            self.existing_features[name] = feature_series.values

            explanation_text = cand.get("explanation", cand.get("rationale", ""))
            explainability = self._generate_explainability(
                name=name,
                code=cand.get("code", ""),
                explanation=explanation_text,
                l2_metrics=l2_metrics,
                l3_metrics=l3_metrics,
                feature_series=feature_series,
            )

            feature_record = {
                "name": name,
                "feature_name": name,
                "score": score,
                "code": cand.get("code", ""),
                "explanation": explanation_text,
                "reasoning": cand.get("reasoning", ""),
                "l2_metrics": l2_metrics,
                "l3_metrics": l3_metrics,
                "shap_rank": l3_metrics["shap_rank"],
                "iv": l2_metrics["iv"],
                "explainability": explainability,
            }

            island.add_feature(feature_record)
            self.island_manager.migrate_best(feature_record)
            self.accepted.append(feature_record)
            accepted_this_step.append(feature_record)

            logger.info(
                "Accepted feature %s: score=%.4f, delta_f1=%.6f, delta_recall=%.6f",
                name, score, l3_metrics.get("delta_f1", 0), l3_metrics.get("delta_recall", 0),
            )

        return accepted_this_step

    def _accept_pending_feature(self, pf: PendingFeature) -> dict:
        """将缓刑池特征正式接受。"""
        feature_series = pd.Series(pf.feature_values, name=pf.name)

        self.all_scores["shap_rank"].append(pf.l3_metrics.get("shap_rank", 999))
        self.all_scores["iv"].append(pf.l2_metrics.get("iv", 0))
        self.all_scores["redundancy"].append(pf.l2_metrics.get("redundancy", 0))
        self.all_scores["delta_f1"].append(pf.l3_metrics.get("delta_f1", 0))

        score = compute_composite_score(
            shap_rank=pf.l3_metrics.get("shap_rank", 999),
            iv=pf.l2_metrics.get("iv", 0),
            redundancy=pf.l2_metrics.get("redundancy", 0),
            delta_f1=pf.l3_metrics.get("delta_f1", 0),
            all_scores=self.all_scores,
        )

        self.existing_features[pf.name] = pf.feature_values

        explainability = self._generate_explainability(
            name=pf.name,
            code=pf.code,
            explanation=pf.explanation,
            l2_metrics=pf.l2_metrics,
            l3_metrics=pf.l3_metrics,
            feature_series=feature_series,
        )

        feature_record = {
            "name": pf.name,
            "feature_name": pf.name,
            "score": score,
            "code": pf.code,
            "explanation": pf.explanation,
            "reasoning": pf.reasoning,
            "l2_metrics": pf.l2_metrics,
            "l3_metrics": pf.l3_metrics,
            "shap_rank": pf.l3_metrics.get("shap_rank", 999),
            "iv": pf.l2_metrics.get("iv", 0),
            "explainability": explainability,
            "accepted_via": "pending_pool",
        }

        island = self.island_manager.get_island(pf.island_name)
        if island:
            island.add_feature(feature_record)
        self.island_manager.migrate_best(feature_record)
        self.accepted.append(feature_record)
        return feature_record

    def run_pending_pool_evaluation(self) -> list[dict]:
        """每轮结束后对缓刑池做反向淘汰评估，返回新接受的特征记录。"""
        if self.pending_pool.is_empty():
            return []

        if self.existing_features.shape[1] == 0:
            logger.info("No accepted features yet, skipping pending pool evaluation")
            return []

        pending_features = {
            pf.name: pf.feature_values
            for pf in self.pending_pool.get_all()
        }

        accepted_names, rejected_names = backward_eliminate_pending(
            self.existing_features, pending_features, self.label,
            baseline_metrics=getattr(self, '_baseline_metrics', None),
        )

        logger.info("Pending pool evaluation: %d accepted, %d rejected out of %d",
                     len(accepted_names), len(rejected_names), len(pending_features))

        self.pending_pool.mark_rejected(rejected_names)

        accepted_pfs = self.pending_pool.remove_accepted(accepted_names)
        new_records = []
        for pf in accepted_pfs:
            record = self._accept_pending_feature(pf)
            new_records.append(record)
            logger.info("Accepted from pending pool: %s (score=%.4f)", pf.name, record["score"])

        self.pending_pool.tick_age()
        purged = self.pending_pool.purge_expired()
        if purged:
            logger.info("Purged %d expired features from pending pool: %s", len(purged), purged)

        return new_records

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

    def _execute_queries(self, queries: list[dict], max_queries: int = 3) -> list[dict]:
        """执行 LLM 的数据探索查询，返回结果列表。"""
        results = []
        for q in queries[:max_queries]:
            code = q.get("code", "")
            question = q.get("question", "")
            result_text, error = execute_stats_query(
                code, self.open_df, self.tx_df, self.close_df,
            )
            results.append({
                "question": question,
                "result": result_text if result_text else f"查询失败: {error}",
            })
            logger.info("Stats query '%s': %s", question, "OK" if result_text else error)
        return results

    def _build_turn_feedback_prompt(
        self,
        island: Island,
        turn_accepted: list[dict],
        turn_failed: list[dict],
        n_candidates: int,
        query_results: list[dict] | None = None,
    ) -> str:
        """构建岛屿内多轮对话的反馈 Prompt。"""
        lines = [f"## 上一轮生成结果反馈（{island.name}）\n"]

        if query_results:
            lines.append("### 数据查询结果")
            for qr in query_results:
                lines.append(f"**{qr['question']}**")
                lines.append(f"```\n{qr['result']}\n```")
            lines.append("")

        if turn_accepted:
            lines.append("通过验证的特征：")
            for f in turn_accepted:
                lines.append(
                    f"  - {f['name']}: score={f['score']:.4f}, "
                    f"delta_f1={f.get('l3_metrics', {}).get('delta_f1', 0):.6f}, "
                    f"delta_recall={f.get('l3_metrics', {}).get('delta_recall', 0):.6f}"
                )
            lines.append("")

        if turn_failed:
            lines.append("未通过的特征（请避免类似方向）：")
            for f in turn_failed:
                lines.append(f"  - {f.get('name', 'unknown')}: {f.get('reason', '')}")
            lines.append("")

        lines.append(
            f"请基于以上反馈，围绕 {island.name} 维度再生成 {n_candidates} 个"
            f"不同角度的新特征。尝试与上轮不同的思路。输出 JSON 数组格式。"
        )
        return "\n".join(lines)

    def run_round(self, round_num: int) -> dict:
        """一轮完整迭代：遍历所有岛屿，根据 SHAP 动态分配候选预算。
        每个岛屿内进行多轮对话（intra_island_turns），实时反馈验证结果。"""
        budgets = self._compute_budgets(round_num)
        max_turns = self.cfg.search.intra_island_turns

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
                "turns": [],
                "candidates": 0,
                "accepted": 0,
                "details": [],
            }

            try:
                pending_query_results: list[dict] | None = None
                for turn in range(1, max_turns + 1):
                    if turn == 1:
                        validated = self.run_single_island_step(island, round_num, n_candidates)
                        pending_query_results = None
                    else:
                        feedback_prompt = self._build_turn_feedback_prompt(
                            island, prev_accepted, prev_failed, n_candidates,
                            query_results=pending_query_results,
                        )
                        raw_output = self.llm.generate(self.system_prompt, feedback_prompt)
                        candidates, queries = parse_llm_output_mixed(raw_output)

                        if queries:
                            pending_query_results = self._execute_queries(queries)
                        else:
                            pending_query_results = None

                        validated = []
                        for cand in candidates:
                            code = cand.get("code", "")
                            name = cand.get("feature_name", "unknown")
                            feature_df, error = self._try_l1_with_retry(code, name, max_retries=2)
                            if feature_df is not None:
                                cand["_feature_df"] = feature_df
                                validated.append(cand)
                            else:
                                self.failure_memory.add_failure(name, code, error)

                    accepted = self.evaluate_candidates(validated, island)

                    prev_accepted = [
                        {"name": f["name"], "score": f["score"], "l3_metrics": f.get("l3_metrics", {})}
                        for f in accepted
                    ]
                    prev_failed = []
                    for cand in validated:
                        name = cand.get("feature_name", "unknown")
                        if not any(a["name"] == name for a in accepted):
                            recent = self.failure_memory.get_recent(1)
                            reason = recent[0].get("failure_reason", "") if recent else "验证未通过"
                            prev_failed.append({"name": name, "reason": reason})

                    has_queries = pending_query_results is not None
                    turn_log = {
                        "turn": turn,
                        "candidates": len(validated),
                        "accepted": len(accepted),
                        "queries": len(pending_query_results) if has_queries else 0,
                    }
                    island_log["turns"].append(turn_log)
                    island_log["candidates"] += len(validated)
                    island_log["accepted"] += len(accepted)
                    island_log["details"].extend(
                        [{"name": f["name"], "score": f["score"], "status": "accepted"} for f in accepted]
                    )
                    for pf in prev_failed:
                        detail = {"name": pf["name"], "status": "rejected", "reason": pf["reason"]}
                        cand_match = next((c for c in validated if c.get("feature_name") == pf["name"]), None)
                        if cand_match and "l2_metrics" in cand_match:
                            detail["l2_metrics"] = cand_match["l2_metrics"]
                        if cand_match and "l3_metrics" in cand_match:
                            detail["l3_metrics"] = cand_match["l3_metrics"]
                        island_log["details"].append(detail)
                    new_count += len(accepted)

                    if len(accepted) == 0 and not has_queries and turn >= 2:
                        logger.info("Island %s turn %d: no new features and no queries, stopping", island.name, turn)
                        break

            except Exception as e:
                logger.error("Error in island %s round %d: %s", island.name, round_num, e)
                island_log["error"] = str(e)

            round_log["islands"].append(island_log)

        pool_accepted = self.run_pending_pool_evaluation()
        if pool_accepted:
            round_log["pending_pool_accepted"] = [
                {"name": f["name"], "score": f["score"]} for f in pool_accepted
            ]
            new_count += len(pool_accepted)

        round_log["new_features"] = new_count
        round_log["pending_pool_size"] = len(self.pending_pool)
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
        if self.validation_thresholds:
            th = self.validation_thresholds
            result.calibrated_thresholds = {
                "iv_min": th.iv_min,
                "ks_min": th.ks_min,
                "corr_max": th.corr_max,
                "missing_rate_max": th.missing_rate_max,
            }

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
