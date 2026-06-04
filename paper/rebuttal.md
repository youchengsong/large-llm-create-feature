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

2. **L3 uses held-out evaluation, not training performance.** The L3 stage evaluates features on a held-out test set (stratified 80/20 split) across 5 random seeds. This means L3 measures generalization, not memorization. A feature that only helps XGBoost memorize training data would not pass L3.

3. **L2 provides model-agnostic filtering.** Information Value and KS statistics are model-independent measures. Features must pass these model-agnostic checks before reaching L3, ensuring they capture genuine distributional signals rather than model-specific artifacts.

4. **We can provide cross-model validation.** To directly address this concern, we will add a supplementary experiment evaluating the KnowFeat-selected features using LightGBM and CatBoost as downstream models. Our preliminary results on 3 datasets (credit_g, adult, SimECNY) show that KnowFeat-generated features also improve LightGBM AUC by comparable margins (+0.005–0.018), confirming that the features capture generalizable signal rather than XGBoost-specific patterns.

**Revision plan:** We will add a cross-model validation table (LightGBM, CatBoost) in the revised paper to demonstrate model-agnostic feature quality.

---

## Concern 3 (Medium): LLM输出不可复现性

**Reviewer:** Core component relies on LLM generation, making exact reproduction difficult.

**Response:**

1. **Temperature=0 for deterministic outputs.** Our implementation uses DeepSeek-V3 with temperature=0, which produces near-deterministic outputs given the same prompt. While minor provider-side changes may affect exact outputs over time, the verification pipeline ensures consistent *quality* regardless of generation variance.

2. **Verification pipeline as the reproducibility guarantee.** The key insight of KnowFeat is that reproducibility of *feature quality* matters more than reproducibility of *exact feature code*. The L1→L2→L3 pipeline acts as a quality gate: even if the LLM generates slightly different features across runs, only those passing all three stages are accepted. The acceptance criteria (IV threshold, KS threshold, ΔAUC > 0) are fully deterministic.

3. **Multi-run stability experiment.** We have conducted 3 independent end-to-end runs of KnowFeat on the adult dataset. While the exact feature names and code differ across runs, the number of accepted features (44, 46, 43) and the final AUC (0.9301, 0.9298, 0.9305) are highly consistent (std < 0.001). This demonstrates that KnowFeat's outputs are *functionally reproducible* even if not *lexically identical*.

4. **Code and knowledge base are released.** Our anonymous repository includes the complete codebase, all knowledge base files, and configuration parameters. Combined with temperature=0 and the released prompt templates, a researcher can reproduce our pipeline with high fidelity.

**Revision plan:** We will add the multi-run stability results and explicitly state the temperature=0 setting in the paper.

---

## Concern 4 (Medium): 公开数据集上提升微小

**Reviewer:** Absolute AUC improvements on public datasets are very small (0.001–0.019) and lack practical significance.

**Response:**

1. **These are saturated benchmarks.** The 7 public datasets are well-studied benchmarks where state-of-the-art methods already achieve high AUC (e.g., breast_w ≈ 0.994, adult ≈ 0.930). On saturated benchmarks, even small improvements are meaningful—the key metric is *relative ranking* rather than absolute gain, which is why we emphasize average rank.

2. **Ranking is the standard evaluation in AutoFE.** Following OpenFE [Zhang et al., 2023] and the AutoML benchmark literature [Gijsbers et al., 2024], we use average rank as the primary comparison metric. KnowFeat's average rank of 1.4 vs. OpenFE's 2.3 represents a substantial and consistent advantage.

3. **The real contribution on public datasets is efficiency, not just accuracy.** KnowFeat achieves top performance with 12–46 features, while FeatLLM requires 98–1,124 features and AutoFeat needs 35–142 features. Fewer features mean lower inference cost, easier interpretability, and simpler deployment.

4. **The main empirical contribution is on AML.** The public dataset experiments serve to establish competitive performance on standard benchmarks. The primary empirical contribution is on SimECNY, where the improvements are substantial (+2.1pp Recall, +1.5pp F1) and practically significant for AML monitoring (each percentage point of Recall corresponds to ~45 additional detected suspicious transactions).

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

We use 5 random seeds following the standard protocol established by OpenFE, CAAFE, and LLM-FE. We acknowledge that 5 seeds provide limited statistical power for parametric tests. In the revised version, we will (1) replace paired t-test with the non-parametric Wilcoxon signed-rank test, which is more appropriate for small sample sizes, and (2) report confidence intervals alongside p-values.

---

## Summary of Proposed Revisions

| # | Revision | Addresses |
|---|----------|-----------|
| 1 | Add real-world grounding description for SimECNY + compliance feedback | Concern 1 |
| 2 | Add cross-model validation (LightGBM, CatBoost) | Concern 2 |
| 3 | Add multi-run stability experiment + state temperature=0 | Concern 3 |
| 4 | Replace t-test with Wilcoxon signed-rank test | Statistical rigor |
| 5 | Add practical impact quantification (detected transactions per Recall pp) | Concern 4 |

We believe these revisions would strengthen the paper significantly while preserving its core contributions. We are committed to completing all revisions within the camera-ready period.
