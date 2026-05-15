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

## 2026-05-14 Phase 0-3 代码实现完成

### 核心搜索引擎
- 实现了 4 维度岛屿搜索 + Boltzmann 采样 + SHAP 反馈闭环
- 三级验证流水线（L1 沙盒 → L2 统计 → L3 增量）
- 171 个测试全部通过
- Commits: `550343c`, `113ea8b`, `ff1d34a`, `78154b9`, `f72bc2a`

### Prompt 增强（commit `113ea8b`）
- 注入正常 vs 异常分布差异（extract_distribution_diff）
- 注入三表关联统计（extract_join_statistics）
- 注入 15 个基线特征代码示例
- 添加 CoT 推理步骤
- 多轮岛内对话（每岛最多 3 轮）

### 动态预算分配（commit `550343c`）
- SHAP 维度重要性 → 逆重要性加权 → 薄弱维度分配更多候选预算
- 第 1 轮均匀分配，第 2 轮起动态分配（b_min=1, b_max=5, B_total=12）

### 按需数据探索（commit `78154b9`）
- 方案 A（被动）：预计算分布差异和关联统计注入 System Prompt
- 方案 B（主动）：LLM 可输出 `{"action": "query", ...}` 在沙盒中执行统计查询
- 查询结果在多轮对话中反馈给 LLM

### 特征可解释性（commit `ff1d34a`）
- L3 通过后额外调用 LLM 生成结构化报告
- 5 个字段：value_meaning, risk_threshold, synergy, false_positive_cases, regulatory_explanation
- 报告写入 features.json 和导出的 .py 文件

### 缓刑池 + 反向淘汰（commit `f72bc2a`）
- 解决特征组合效应问题：单独无增量但组合有效的特征不再被遗漏
- L2 通过 + L3 未通过 → 进缓刑池（最多 20 个）
- 每轮末全量加入训练 → SHAP 排序 → 从弱到强逐个剔除 → 分数下降则保留
- 淘汰规则：连续 2 次被剔除 或 age > 3 轮 → 永久淘汰

经验教训：
- 论文调研发现 CAAFE/LLM-FE/OCTree 都没有用单特征增量作为唯一判定标准，而是看整体模型分数
- 反向淘汰比正向逐个评估更能捕捉交互效应，计算成本可控（缓刑池上限 20 个）

## 2026-05-14 L2 阈值自动校准（commit `a7ef094`）

### 问题
- L2 验证的 4 个阈值（MI>=0.001, Cramer's V>=0.05, redundancy<0.9, bootstrap p<0.05）全部硬编码
- 2 轮搜索共 39 个候选特征，0 个通过 L2，系统连续 2 轮无新特征直接收敛退出
- search_log 中 details 为空数组，无法诊断哪些指标导致被拒

### 解决方案
- 新增 `src/evaluation/calibration.py`：用基线特征的指标分布自动校准阈值
  - 对 ~30 个基线特征计算 MI 和 Cramer's V
  - 取 p10 分位数 × 0.5 作为新阈值（新特征只需不比最弱基线差太多）
  - 保持 redundancy<0.9 和 bootstrap p<0.05 不变（这两个已经足够宽松/标准）
- `run_l2_validation` 支持接收外部阈值参数
- search_log 记录被拒候选的指标值和具体失败原因，方便调试
- search_log 头部记录校准后的阈值值

经验教训：
- 验证阈值不应硬编码，应根据数据特性自动适配
- 必须记录被拒候选的指标值，否则无法诊断问题
- 168 个测试通过（新增 6 个校准测试）

## 2026-05-14 DeepSeek 全流程搜索完成

### 配置
- LLM: DeepSeek (deepseek-chat)
- 校准后阈值: MI>=0.000131 (原0.001, 降7.6x), Cramer's V>=0.006661 (原0.05, 降7.5x)
- 搜索轮数: 5 轮收敛

### 结果
- 基线 PR-AUC: 0.6305 → 最终 ~0.678 (+0.048 提升)
- 52 个特征通过验证，导出到 output/features/ 目录
- 4 个维度均有特征被接受：金额频次、时间地域、行为序列、主体关联

### 关键验证
- L2 自动校准有效解决了阈值过严问题（之前 0/39 通过 → 现在 52 个通过）
- L2 转变为粗筛角色，L3 增量验证 + 反向淘汰承担最终质量控制
- 搜索耗时约 5.5 小时（含 LLM 调用和 5-fold CV）

经验教训：
- 长时间运行的搜索任务需要用 nohup 后台执行，避免 Bash 超时
- 校准阈值的核心思想：新特征只需不比最弱基线差太多（p10 × 0.5）

## 2026-05-15 反向淘汰逻辑修复 + 特征工程升级 + GPU 支持

### 反向淘汰 bug 修复
- **问题**：旧版反向淘汰以 full_score（所有已接受+待定特征的组合分数）为比较基准，但 full_score 始终低于 baseline_prauc（0.868~0.880 vs 0.885）。特征在"已变差的集合"中有相对贡献就被接受，导致越多特征被接受、模型越差
- **实测数据**：R1→R6 的 full_score 持续下降：0.877→0.877→0.880→0.878→0.873→0.868，12个"已接受"特征实际拉低了 baseline 1.7 个百分点
- **修复方案（方案B 两阶段筛选）**：
  1. 阶段1（反向淘汰去噪）：照旧从弱到强逐个剔除，清除组合中无贡献的噪音
  2. 阶段2（增量验证）：对幸存特征逐个加入原始 baseline 计算 delta_prauc，只有 delta > 0 才接受
- **改动文件**：`src/evaluation/validator.py`（backward_eliminate_pending）、`src/search/engine.py`（传入 baseline_prauc）
- 177 个测试全部通过

### 特征工程升级
- 替换 `inputs/baseline/feature_engineering.py` 为新版本
- 新增 5 个 S 交易行为指标（ibank_index, ibank_amount_ratio, counterparty_index, per_deal_match_times, daily_deal_match_times）
- 新增 9 个账户开立衍生特征（open_hour, is_work_hour, auth_delay_days, wallets_per_cert 等）
- 修复 feature_df 未合并回 df_quantized 的 bug（账户级汇总特征 activetime, transaction_frequency 等现在正确通过 From/To 合并回交易表）
- `_BASELINE_FEATURE_NAMES` 从 62 个更新为 107 个

### GPU 加速
- XGBoostParams 新增 tree_method="hist" + device（通过 XGB_DEVICE 环境变量控制）
- 8 处 XGBClassifier 调用全部加上 GPU 参数
- 租用 GPU 服务器（RTX PRO 6000 48GB vGPU），环境已搭建并验证
- 50K×70 训练耗时从 CPU ~10s 降至 GPU 0.69s（~15x 加速）

经验教训：
- 反向淘汰的比较基准必须是原始 baseline，而不是包含待定特征后的 full_score
- full_score < baseline 时，整批待定特征是负向贡献，旧逻辑在此场景下完全失效
- 两阶段方案兼顾了组合效应检测（阶段1）和绝对质量保证（阶段2）
