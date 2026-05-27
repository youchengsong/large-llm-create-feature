"""KnowFeat 主编排：两阶段特征生成 + 搜索策略循环。

Stage 2 完整流程:
  1. 加载数据 + 基线特征
  2. 生成噪声校准阈值
  3. 每轮循环: 4 维度 Agent 探索 → 前向选择 → SHAP → 更新预算
  4. 收敛检测（连续 2 轮无新特征 / 达上限 / max_rounds 轮）
  5. 输出结果
Stage 1 同样流程（数据和知识不同）。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from knowfeat.agent.react_agent import ReactAgent, RoundResult
from knowfeat.agent.tools import ToolContext
from knowfeat.config import KnowFeatConfig, load_config
from knowfeat.data.loader import Stage2Data, get_label_series, load_stage2
from knowfeat.data.schema_stats import generate_all_schema_text
from knowfeat.llm.client import LLMClient
from knowfeat.prompt.builder import (
    CrossDimExample,
    FailureRecord as PromptFailureRecord,
    ShapFeedback as PromptShapFeedback,
)
from knowfeat.prompt.domain_knowledge import DomainKnowledge, load_domain_knowledge
from knowfeat.search.cross_dim import CrossDimPropagator
from knowfeat.search.dimension_manager import DimensionManager
from knowfeat.search.failure_memory import FailureMemory, FailureRecord
from knowfeat.search.forward_selection import forward_select
from knowfeat.search.shap_feedback import (
    ShapFeedback,
    analyze_shap_feedback,
    format_shap_summary,
)
from knowfeat.verification.statistics import NoiseThresholds, calibrate_noise_thresholds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Single stage (Stage 1 or Stage 2) run outcome."""

    stage: str
    total_rounds: int = 0
    total_accepted: int = 0
    total_candidates: int = 0
    total_rejected: int = 0
    accepted_feature_names: list[str] = field(default_factory=list)
    round_results: list[RoundResult] = field(default_factory=list)
    convergence_reason: str = ""
    elapsed_sec: float = 0.0
    feature_codes: dict[str, str] = field(default_factory=dict)
    feature_dimensions: dict[str, list[str]] = field(default_factory=dict)
    feature_provenance: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Type conversion helpers (search types → prompt types)
# ---------------------------------------------------------------------------

def _shap_to_prompt(shap_fb: ShapFeedback, round_num: int) -> PromptShapFeedback:
    top_features = []
    for fs in shap_fb.feature_scores[:10]:
        top_features.append({
            "name": fs.name,
            "importance": fs.shap_importance,
            "dimension": "/".join(fs.dimensions) if fs.dimensions else "",
        })
    return PromptShapFeedback(
        dimension_contributions=shap_fb.dimension_contributions,
        top_features=top_features,
        round_num=round_num,
    )


def _failures_to_prompt(records: list[FailureRecord]) -> list[PromptFailureRecord]:
    return [
        PromptFailureRecord(
            feature_name=r.feature_name,
            code=r.code,
            reason=r.reason,
            round_num=r.round_num,
            dimension=r.dimension,
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def run_stage(
    cfg: KnowFeatConfig,
    data: Stage2Data,
    domain_knowledge: DomainKnowledge,
    noise_thresholds: Optional[NoiseThresholds] = None,
) -> StageResult:
    """Run one stage of KnowFeat feature generation.

    Parameters
    ----------
    cfg : KnowFeatConfig
    data : Stage2Data
    domain_knowledge : DomainKnowledge
    noise_thresholds : NoiseThresholds, optional
        Pre-computed noise calibration. If *None* it will be computed here.

    Returns
    -------
    StageResult
    """
    t0 = time.time()
    result = StageResult(stage=cfg.stage)

    # -- initialise components ------------------------------------------------
    llm = LLMClient(cfg.llm)
    label = get_label_series(data)
    schema_text = generate_all_schema_text(data)

    dim_mgr = DimensionManager(cfg=cfg)
    failure_mem = FailureMemory()
    cross_prop = CrossDimPropagator(cfg=cfg)

    if noise_thresholds is None:
        logger.info("噪声校准开始 (n_random=%d) ...", cfg.verification.noise_n_random_features)
        noise_thresholds = calibrate_noise_thresholds(
            label=label,
            n_samples=len(label),
            n_random=cfg.verification.noise_n_random_features,
            percentile=cfg.verification.noise_percentile,
            iv_fallback=cfg.verification.iv_min_fallback,
            ks_fallback=cfg.verification.ks_min_fallback,
            seed=cfg.seed,
        )

    tool_ctx = ToolContext(
        data=data, cfg=cfg, noise_thresholds=noise_thresholds,
    )
    agent = ReactAgent(
        llm=llm, tool_ctx=tool_ctx, cfg=cfg,
        domain_knowledge=domain_knowledge, schema_text=schema_text,
    )

    prompt_shap: Optional[PromptShapFeedback] = None
    consecutive_no_new = 0

    # -- main loop ------------------------------------------------------------
    for round_num in range(1, cfg.search.max_rounds + 1):
        accepted_before = len(tool_ctx.accepted_features)
        rejected_before = len(tool_ctx.rejected_features)

        logger.info("=" * 60)
        logger.info(
            "Round %d/%d 开始 (已接受=%d, 候选=%d)",
            round_num, cfg.search.max_rounds,
            accepted_before, len(tool_ctx.candidate_pool),
        )

        # ---- per-dimension agent exploration --------------------------------
        for dim in dim_mgr.dimensions:
            budget = dim_mgr.get_budget(dim)
            if budget <= 0:
                continue

            failure_records = failure_mem.get_recent(
                cfg.search.failure_memory_inject_n,
            )
            prompt_failures = _failures_to_prompt(failure_records)
            cross_examples = cross_prop.get_examples_for_dimension(dim)

            rr = agent.run_round(
                round_num=round_num,
                dimension=dim,
                budget=budget,
                shap_feedback=prompt_shap,
                failure_records=prompt_failures,
                cross_dim_examples=cross_examples,
            )
            result.round_results.append(rr)

            dim_mgr.record_round(
                dimension=dim,
                generated=rr.n_generated,
                accepted=rr.n_accepted,
                candidates=rr.n_candidates,
                rejected=rr.n_rejected,
            )

            # Register newly accepted features in cross-dim propagator
            for fname in rr.features_accepted:
                af = tool_ctx.accepted_features.get(fname)
                if af is not None:
                    cross_prop.register_feature(
                        name=fname,
                        dimensions=af.dimension,
                        code=af.code,
                    )

        # Collect new L2-rejected features into failure memory
        for rej in tool_ctx.rejected_features[rejected_before:]:
            failure_mem.record_failure(
                feature_name=rej.get("feature_name", ""),
                code=rej.get("code", ""),
                reason=rej.get("reason", "unknown"),
                round_num=round_num,
                dimension=rej.get("dimension", ""),
            )

        # ---- forward selection on candidate pool ----------------------------
        if tool_ctx.candidate_pool:
            logger.info(
                "前向选择: %d 个候选特征", len(tool_ctx.candidate_pool),
            )
            fs_result = forward_select(
                baseline_X=tool_ctx.baseline_X,
                label=label,
                candidates=list(tool_ctx.candidate_pool),
                cfg=cfg,
                seed=cfg.seed,
            )

            for cand in fs_result.accepted:
                tool_ctx.accepted_features[cand.name] = cand
                tool_ctx.baseline_X[cand.name] = cand.series.values
                cross_prop.register_feature(
                    name=cand.name,
                    dimensions=cand.dimension,
                    code=cand.code,
                )

            accepted_set = {a.name for a in fs_result.accepted}
            tool_ctx.candidate_pool = [
                c for c in tool_ctx.candidate_pool if c.name not in accepted_set
            ]

            for rej_cand in fs_result.rejected:
                detail = fs_result.eval_details.get(rej_cand.name)
                reason = (
                    "; ".join(detail.reject_reasons) if detail else "未通过"
                )
                failure_mem.record_failure(
                    feature_name=rej_cand.name,
                    code=rej_cand.code,
                    reason=f"前向选择: {reason}",
                    round_num=round_num,
                    dimension=(
                        "/".join(rej_cand.dimension)
                        if rej_cand.dimension else ""
                    ),
                )

            # Remove forward-selection rejects from candidate pool
            rejected_set = {r.name for r in fs_result.rejected}
            tool_ctx.candidate_pool = [
                c for c in tool_ctx.candidate_pool if c.name not in rejected_set
            ]

        # ---- SHAP feedback --------------------------------------------------
        if tool_ctx.accepted_features and tool_ctx.baseline_X.shape[1] > 0:
            feature_dims: dict[str, list[str]] = {}
            for fname, cand in tool_ctx.accepted_features.items():
                feature_dims[fname] = cand.dimension

            shap_fb = analyze_shap_feedback(
                X=tool_ctx.baseline_X,
                label=label,
                feature_dimensions=feature_dims,
                cfg=cfg,
                seed=cfg.seed,
            )
            cross_prop.update_from_shap(shap_fb.feature_scores)
            dim_mgr.update_budgets_from_shap(shap_fb.dimension_contributions)
            prompt_shap = _shap_to_prompt(shap_fb, round_num)
            logger.info("SHAP 反馈:\n%s", format_shap_summary(shap_fb))

        # ---- convergence check ----------------------------------------------
        accepted_after = len(tool_ctx.accepted_features)
        new_this_round = accepted_after - accepted_before
        result.total_rounds = round_num

        logger.info(
            "Round %d 完成: +%d 特征, 总计 %d/%d",
            round_num, new_this_round,
            accepted_after, cfg.search.max_features,
        )
        logger.info("维度状态:\n%s", dim_mgr.summary())

        if new_this_round == 0:
            consecutive_no_new += 1
        else:
            consecutive_no_new = 0

        if accepted_after >= cfg.search.max_features:
            result.convergence_reason = f"达到特征上限 {cfg.search.max_features}"
            logger.info("收敛: %s", result.convergence_reason)
            break

        if consecutive_no_new >= cfg.search.convergence_rounds:
            result.convergence_reason = (
                f"连续 {consecutive_no_new} 轮无新特征"
            )
            logger.info("收敛: %s", result.convergence_reason)
            break
    else:
        result.convergence_reason = f"达到最大轮次 {cfg.search.max_rounds}"

    # -- collect final results ------------------------------------------------
    result.total_accepted = len(tool_ctx.accepted_features)
    result.accepted_feature_names = list(tool_ctx.accepted_features.keys())
    result.total_candidates = len(tool_ctx.candidate_pool)
    result.total_rejected = len(tool_ctx.rejected_features)

    for fname, cand in tool_ctx.accepted_features.items():
        result.feature_codes[fname] = cand.code
        result.feature_dimensions[fname] = cand.dimension
        result.feature_provenance[fname] = {
            "explanation": cand.explanation,
            "reasoning": cand.reasoning,
            "knowledge_source": cand.knowledge_source,
            "data_fields_used": cand.data_fields_used,
        }

    result.elapsed_sec = time.time() - t0
    logger.info(
        "Stage %s 完成: %d 特征, %d 轮, %.1fs, %s",
        cfg.stage, result.total_accepted, result.total_rounds,
        result.elapsed_sec, result.convergence_reason,
    )
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(result: StageResult, output_dir: Path) -> dict[str, Path]:
    """Save accepted features (JSON), run summary (JSON), and provenance cards.

    Returns
    -------
    dict[str, Path]
        {"features": ..., "summary": ..., "provenance_json": ..., "provenance_md": ...}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    features_info = []
    for fname in result.accepted_feature_names:
        features_info.append({
            "name": fname,
            "dimensions": result.feature_dimensions.get(fname, []),
            "code": result.feature_codes.get(fname, ""),
        })

    features_path = output_dir / f"{result.stage}_features.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features_info, f, ensure_ascii=False, indent=2)
    logger.info("特征信息 → %s", features_path)

    summary = {
        "stage": result.stage,
        "total_rounds": result.total_rounds,
        "total_accepted": result.total_accepted,
        "total_candidates": result.total_candidates,
        "total_rejected": result.total_rejected,
        "convergence_reason": result.convergence_reason,
        "elapsed_sec": round(result.elapsed_sec, 1),
        "accepted_features": result.accepted_feature_names,
    }
    summary_path = output_dir / f"{result.stage}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("运行摘要 → %s", summary_path)

    paths: dict[str, Path] = {
        "features": features_path,
        "summary": summary_path,
    }

    from knowfeat.output.provenance import ProvenanceCard, build_card, save_provenance
    cards = []
    for fname in result.accepted_feature_names:
        prov = result.feature_provenance.get(fname, {})
        cards.append(build_card(
            feature_name=fname,
            dimensions=result.feature_dimensions.get(fname, []),
            code=result.feature_codes.get(fname, ""),
            explanation=prov.get("explanation", ""),
            reasoning=prov.get("reasoning", ""),
            knowledge_source=prov.get("knowledge_source", ""),
            data_fields_used=prov.get("data_fields_used", []),
        ))
    if cards:
        prov_paths = save_provenance(cards, output_dir, stage=result.stage)
        paths["provenance_json"] = prov_paths["json"]
        paths["provenance_md"] = prov_paths["markdown"]

    return paths


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(stage: str = "stage2", **config_overrides) -> StageResult:
    """KnowFeat main entry point.

    Parameters
    ----------
    stage : "stage1" or "stage2"
    **config_overrides : forwarded to :func:`load_config`.

    Returns
    -------
    StageResult
    """
    cfg = load_config(stage=stage, **config_overrides)

    logging.basicConfig(
        level=logging.INFO if cfg.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        force=True,
    )
    logger.info("KnowFeat 启动 (stage=%s)", stage)

    data = load_stage2(cfg=cfg)
    domain_knowledge = load_domain_knowledge(cfg)
    result = run_stage(cfg, data, domain_knowledge)
    save_results(result, cfg.paths.output_dir)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="KnowFeat 特征生成系统")
    parser.add_argument(
        "--stage", default="stage2", choices=["stage1", "stage2"],
    )
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument(
        "--quiet", action="store_true", help="suppress verbose logging",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.max_rounds is not None:
        overrides["max_rounds"] = args.max_rounds
    if args.max_features is not None:
        overrides["max_features"] = args.max_features
    if args.quiet:
        overrides["verbose"] = False

    result = run(stage=args.stage, **overrides)
    print(
        f"\n完成: {result.total_accepted} 特征, "
        f"{result.total_rounds} 轮, "
        f"{result.convergence_reason}"
    )


if __name__ == "__main__":
    main()
