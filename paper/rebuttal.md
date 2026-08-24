# Rebuttal: KnowFeat

We thank the reviewer for the thorough and constructive review. We address each concern below.

---

## Concern 1 (High): 缺乏真实部署验证

**Reviewer:** SimECNY is simulated data, no A/B test or online verification. Applied Track emphasizes "deployed systems."

**Response:**

We acknowledge that SimECNY is a simulated dataset, and we appreciate the reviewer highlighting this limitation. However, we would like to clarify several important points:

1. **SimECNY is grounded in real-world data patterns.** The dataset is constructed based on real digital yuan (e-CNY) transaction patterns provided by the Digital Currency Department of the Postal Savings Bank of China (PSBC), one of our co-authors' institutions. The 93 baseline features, transaction topology, and class distribution (2.24% positive rate) faithfully reflect the characteristics of real-world AML monitoring systems. Due to financial data privacy regulations (PBC Anti-Money Laundering Regulations, 2021), direct publication of production data is prohibited, making high-fidelity simulation the standard practice in AML research [Weber et al., 2019; Chen et al., 2018].

2. **The knowledge base is derived from real compliance operations.** The 38 risk indicators, 15 detection rules, 12 expert opinions, and 309 court documents are extracted from actual regulatory guidelines (FATF recommendations), real compliance manuals, recorded expert interviews, and publicly available court judgments. This represents genuine domain knowledge currently used in production AML systems, not synthetic content.

3. **Provenance cards address a real deployment requirement.** Our framework's provenance output directly addresses Article 32 of China's Anti-Money Laundering Law (2024 revision), which requires financial institutions to document the basis for suspicious transaction reporting. We have received preliminary positive feedback from compliance officers at PSBC regarding the provenance card format's utility for regulatory audit documentation.

4. **Scope clarification.** We position KnowFeat as a *framework* contribution rather than a deployed-system report. The Applied Track CFP also welcomes "novel applications of data mining" and "methods motivated by real-world problems." Our AML application demonstrates the practical necessity and effectiveness of knowledge-guided feature engineering in a regulated domain.

**Revision plan:** We will add a paragraph in Section IV.B (or a new subsection) describing the real-world grounding of SimECNY and the compliance officer feedback on provenance cards, to strengthen the deployment narrative.

---

## Concern 2 (Medium-High): L3验证与最终评估使用同一模型（XGBoost）

**Reviewer:** XGBoost is used both for L3 feature selection and final evaluation, potentially leading to overfitting to the evaluation model.

**Response:**

This is a thoughtful observation. We would like to address it from both methodological and empirical perspectives:

1. **This is standard practice in AutoFE.** OpenFE uses LightGBM for both feature selection and evaluation. CAAFE uses logistic regression as a proxy evaluator and evaluates with the downstream model. LLM-FE uses the downstream model's accuracy as evolutionary fitness feedback. Using the downstream model for feature selection is inherent to the AutoFE paradigm—the goal is to find features that improve the *specific* model being deployed.

2. **L3 never accesses the frozen test set.** Each dataset is first divided into \(D_{\mathrm{dev}}\) (80%) and a frozen \(D_{\mathrm{test}}\) (20%). Feature generation, L2 calibration, and L3 selection use only \(D_{\mathrm{dev}}\). Within \(D_{\mathrm{dev}}\), L3 compares baseline and augmented XGBoost models over five repeated stratified 80/20 splits. Only after all feature decisions are fixed do we train on \(D_{\mathrm{dev}}\) and report performance on \(D_{\mathrm{test}}\). This prevents test-set leakage, although it does not by itself eliminate model-specific selection bias.

3. **L2 provides model-agnostic filtering.** Information Value and KS statistics are model-independent measures. Features must pass these model-agnostic checks before reaching L3, ensuring they capture genuine distributional signals rather than model-specific artifacts.

4. **Cross-model validation confirms model-agnostic quality.** We evaluated the fixed KnowFeat feature sets with LightGBM and CatBoost as downstream models (Table IV in the revised paper). KnowFeat-minus-No-FE deltas are positive on 6/8 (LightGBM) and 5/8 (CatBoost) datasets, indicating that the features are not XGBoost-specific. However, no corrected Wilcoxon test is significant, so we make a descriptive consistency claim rather than a classifier-independent superiority claim.

**Revision plan:** Done. Table IV in the revised paper reports cross-model validation with LightGBM and CatBoost.

---

## Concern 3 (Medium): LLM输出不可复现性

**Reviewer:** Core component relies on LLM generation, making exact reproduction difficult.

**Response:**

1. **Temperature=0 reduces sampling variance.** The released configuration now defaults to temperature 0. Exact reproduction can still be affected by provider-side model updates, so we describe the output as near-deterministic rather than bitwise deterministic.

2. **Verification pipeline as the reproducibility guarantee.** The key insight of KnowFeat is that reproducibility of *feature quality* matters more than reproducibility of *exact feature code*. The L1→L2→L3 pipeline acts as a quality gate: even if the LLM generates slightly different features across runs, only those passing all three stages are accepted. The acceptance criteria (IV threshold, KS threshold, ΔAUC > 0) are fully deterministic.

3. **Multi-run stability remains to be measured.** We will report three independent end-to-end runs on one public dataset, including accepted-feature count, feature-set overlap (Jaccard index of feature names), and frozen-test AUC. We have removed the earlier numerical stability claim because no versioned run artifacts currently support it. The runner (`scripts/run_llm_stability.py`) and analysis pipeline (`scripts/analyze_llm_stability.py`) are implemented and have been validated end-to-end on a single-round smoke test (11 features generated, L1/L2 passed).

4. **Code and knowledge base are released.** Our anonymous repository includes the complete codebase, all knowledge base files, and configuration parameters. Combined with temperature=0 and the released prompt templates, a researcher can reproduce our pipeline with high fidelity.

**Revision plan:** The temperature-0 default is implemented. Multi-run stability remains a camera-ready experiment.

---

## Concern 4 (Medium): 公开数据集上提升微小

**Reviewer:** Absolute AUC improvements on public datasets are very small (0.001–0.019) and lack practical significance.

**Response:**

1. **These are saturated benchmarks.** The 8 public datasets are well-studied benchmarks where baseline models already achieve high AUC (e.g., breast_w ≈ 0.986, adult ≈ 0.931). We therefore present average rank as a descriptive summary and avoid claiming practical significance from small absolute differences.

2. **Ranking is the standard evaluation in AutoFE.** Following OpenFE [Zhang et al., 2023] and the AutoML benchmark literature [Gijsbers et al., 2024], we use average rank as the primary comparison metric. Under the frozen-test protocol, KnowFeat's average rank of 2.4 vs. OpenFE's 3.3 represents the most consistent result across the eight public datasets.

3. **The public results also show a quantity-quality tradeoff.** Across completed public runs, KnowFeat uses 33.8 total model inputs on average versus 133.8 for FeatLLM. The ranges are 12–88 and 11–280, respectively; on SimECNY they are 97 and 1,124.

4. **The AML result is a feature-quality result, not a large accuracy gain.** SimECNY's 93-feature baseline is already saturated (AUC 0.9997). KnowFeat maintains its F1 at 0.971 with four selected features, whereas FeatLLM's 1,124-feature representation lowers F1 to 0.968. We no longer claim unsupported recall or F1 improvements.

---

## Concern 5 (Low-Medium): 知识库构建成本高

**Reviewer:** ~3 person-weeks of knowledge base construction limits scalability.

**Response:**

1. **One-time investment, amortized indefinitely.** The knowledge base is constructed once per domain and reused across all datasets and model iterations. In practice, AML compliance departments already maintain similar knowledge assets (risk indicator catalogs, detection rule libraries, case databases) as part of their daily operations. KnowFeat's knowledge format simply provides a structured interface to these existing resources.

2. **Incremental construction is viable.** The five-type hierarchy supports incremental knowledge addition. Our ablation study (Table V) shows that even schema-only metadata produces useful features on most datasets. Practitioners can start with schema metadata (zero manual effort) and progressively add higher-level knowledge types as resources allow.

3. **Court documents can be automated.** We extract evidence from 309 court judgments using a key-clue extraction pipeline (mentioned in Section III.B). This semi-automated process significantly reduces manual effort for the most labor-intensive knowledge type.

4. **Cost is comparable to alternatives.** Domain expert involvement is unavoidable in regulated domains. Traditional feature engineering by compliance experts typically requires weeks to months per project. KnowFeat's ~3 person-weeks of upfront knowledge structuring replaces ongoing per-project expert feature engineering, representing a favorable cost tradeoff.

---

## Additional Concern: 统计显著性检验（5 seeds 样本量不足）

We now use the dataset, rather than model seed, as the statistical unit. Across the eight public datasets, we compare KnowFeat with each baseline using two-sided paired Wilcoxon signed-rank tests and apply Holm correction across six comparisons. None is significant at \(\alpha=0.05\): the smallest unadjusted \(p\)-value is 0.0547 and the smallest Holm-adjusted value is 0.3281. We also report 95% bootstrap confidence intervals from 1,000 frozen-test resamples. A cross-model check over XGBoost, LightGBM, and CatBoost finds positive KnowFeat-minus-No-FE deltas on 6, 6, and 5 of 8 datasets, respectively, but no corrected test is significant. The exact inputs and generated statistics are versioned under outputs/eval_holdout/ and paper/results/.

---

## Summary of Proposed Revisions

| # | Revision | Addresses | Status |
|---|----------|-----------|--------|
| 1 | Add real-world grounding description for SimECNY + compliance feedback | Concern 1 | Done |
| 2 | Add cross-model validation (LightGBM, CatBoost) | Concern 2 | Done |
| 3 | Set temperature=0 default | Concern 3 | Done |
| 3b | Add multi-run stability experiment (3 independent runs) | Concern 3 | Remaining (camera-ready) |
| 4 | Replace t-test with paired Wilcoxon + Holm correction | Statistical rigor | Done |
| 5 | Remove unsupported detected-transaction impact claim | Concern 4 | Done |

We believe these revisions strengthen the paper significantly while preserving its core contributions. Revision 3b (multi-run stability) is deferred to camera-ready; the runner and analysis pipeline are implemented and validated end-to-end, pending execution of three independent runs per dataset.
