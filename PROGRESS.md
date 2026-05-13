# 进展记录

## 2026-05-13 文献调研完成

调研了 10+ 篇 LLM 特征生成/提取相关论文，形成 LITERATURE_REVIEW.md。

关键发现：
- Balek 2025 的方法最直接可用：62 个 LLM 特征接近 768 维嵌入性能，成本仅 $12
- CAAFE (NeurIPS 2023) 是该领域开创性工作，生成特征转换代码的思路值得借鉴
- RiskTagger 的多维度分析框架（模式/流向/关联/时序）适合金融场景

经验教训：
- OpenReview 链接 JhJPJtau8B 对应的是 LLM-FE（进化优化），与 Balek 2025 是不同论文
- nikhilsab/llmfe 对应 LLM-FE，Balek 的代码在 vojtech-balek/llm-features
