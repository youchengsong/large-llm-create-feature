# ICDM 2026 投稿计划与 Novelty 增强策略

> 记录于 2026-05-20，待系统打磨完成后执行。

## 1. 会议信息

| 项目 | 详情 |
|------|------|
| 会议 | IEEE International Conference on Data Mining (ICDM 2026) |
| Abstract 截止 | 2026-05-30 (AoE) |
| Paper 截止 | 2026-06-06 (AoE) |
| 格式 | IEEE 2-column，10 页（含参考文献） |
| Applied Track | 单盲审稿，侧重实际应用 |
| Research Track | 三盲审稿，侧重理论创新 |
| 建议投稿 | **Applied Track**（数字人民币反洗钱场景契合度高） |

## 2. 当前系统概述

LLM Agent 驱动的自动特征工程系统，用于数字人民币反洗钱：

- **领域知识注入**：邮储银行专家访谈（4 种洗钱模式）+ 309 份裁判文书（9 个子维度）→ 结构化后注入 Agent system prompt
- **SHAP 反馈闭环**：每轮结束后计算 SHAP 重要性，反馈给 Agent 指导下一轮生成方向
- **候选缓冲池 + Forward Selection**：生成候选特征 → IV/KS 统计验证 → 前向选择去冗余
- **噪声基线校准**：100 个随机噪声特征的 IV/KS p95 分位数作为验证阈值
- **多轮迭代**：多 round × 多 island × 多 turn，带收敛检测

## 3. 中稿概率评估

**Applied Track: 25-35%，Research Track: 10-15%**

### 3.1 优势

- **真实场景 + 真实数据**：数字人民币反洗钱，监管刚需，应用价值明确
- **完整系统**：domain knowledge → LLM 生成 → 统计验证 → SHAP 反馈，闭环设计
- **实验设计合理**：多模型验证、消融实验、稳定性分析框架已就绪

### 3.2 短板（审稿人视角）

1. **Novelty 不够硬**：LLM 做 feature engineering 不新鲜（CAAFE, AutoFeat 等已有），核心区别是加了 domain knowledge 和 SHAP feedback loop，但这更像 engineering contribution 而非 algorithmic contribution
2. **缺乏理论支撑**：为什么这个闭环能收敛？为什么 domain knowledge 注入能提升？没有理论保证或形式化分析
3. **单数据集**：只有一个数字人民币数据集，泛化性存疑
4. **缺强 baseline**：目前只对比了 CAAFE-style 和 random，没有对比 OpenFE、DIFER 等自动特征工程 SOTA

## 4. Novelty 增强策略

### 4.1 改 Framing 为「知识蒸馏」（P0，1 天）

**当前叙事**：LLM 帮你做 feature engineering → 太工程化

**更好的叙事**：将领域专家知识通过 LLM 蒸馏为可计算的特征表示

- 输入：非结构化领域知识（专家访谈文本、裁判文书）
- 中间：LLM Agent 作为知识蒸馏器，在统计反馈引导下迭代转化
- 输出：结构化、可验证、模型无关的特征
- 贡献：一种将 expert knowledge 自动转化为 ML-ready features 的通用框架

这个 framing 更学术，novelty 更清晰——不是"用 LLM 写代码"，而是"解决领域知识到特征的鸿沟"。

### 4.2 ~~补公开数据集实验~~ → 已决定：单数据集 + 领域深绑定

**决策（2026-05-20）**：不补公开数据集，聚焦数字人民币反洗钱单一场景。

理由：
- 本文的核心卖点就是深度领域绑定（专家访谈 + 裁判文书），换数据集会稀释这个优势
- Applied Track 接受 domain-specific 研究，泛化性可放 future work
- 审稿人质疑泛化性时的回应："本文聚焦数字人民币反洗钱这一高价值监管场景，框架设计具有通用性（领域知识模块可替换），跨领域验证留作后续工作"

### 4.3 补强 Baseline 对比（P0，提升优先级）

单数据集策略下，实验必须足够扎实来弥补。Baseline 对比从 P1 提升为 P0。

必须加的自动特征工程方法：

| 方法 | 出处 | 核心思路 | 对比意义 |
|------|------|----------|----------|
| **OpenFE** | NeurIPS 2022 | 枚举 + 增量评估 | 数据驱动 SOTA |
| **DIFER** | ICLR 2024 | 强化学习搜索特征变换 | RL-based SOTA |
| **AutoFeat** | 经典 | 多项式展开 + 选择 | 经典 baseline |
| **CAAFE** | 已有 | LLM 单轮生成 | 最直接对比（无反馈、无领域知识） |
| **手工特征** | 领域专家 | 人工设计 | 证明 Agent 接近/超越人类专家 |

关键论点：在专业领域场景下，LLM + domain knowledge 的自动化特征工程能达到甚至超越纯数据驱动方法和人工特征工程，同时生成的特征具有领域可解释性。

如果某些 baseline 跑赢了我们：强调"我们的方法在保持领域可解释性的同时接近 SOTA 数据驱动方法的性能，且无需人工特征设计"。

### 4.4 形式化 Agent 搜索过程（P1，1-2 天）

把多轮生成过程建模为 **信息论引导的特征空间搜索**：

#### 核心想法

- 定义特征空间 $\mathcal{F}$，Agent 每轮从中采样候选特征子集
- SHAP 反馈 → 计算当前特征集与标签的互信息 $I(X_{selected}; Y)$
- Domain knowledge 约束搜索空间：$\mathcal{F}_{DK} \subset \mathcal{F}$
- 证明（或 empirically 展示）：在 $\mathcal{F}_{DK}$ 中搜索，有效搜索空间从指数级降到多项式级

#### 收敛性分析

- 每轮的 $\Delta \text{F1}$ 或 $\Delta \text{PR-AUC}$ 随轮次递减（empirical bound）
- 连续 2 轮无有效特征 → 收敛（系统已实现此检测）
- 可以画出 round vs cumulative metric improvement 的曲线，展示收敛行为

#### 不需要改代码

这部分纯论文写作，在 Section 4 或 Section 5 加一节理论分析即可。

### 4.5 引入 Domain Alignment Score（P2，1 天）

量化生成特征与领域知识的对齐程度：

- **定义**：生成特征的代码中引用了多少条 detection rules、覆盖了多少种 transaction motifs
- **计算方式**：关键词匹配 / LLM 评判 / 规则覆盖率
- **用途**：
  - 消融实验中展示：alignment score 越高的特征，IV/KS 越强
  - 证明不是随便注入知识，而是有度量地引导
  - 可作为一个副产物贡献

## 5. 优先级与时间规划

| 优先级 | 行动 | 预估耗时 | 对中稿概率影响 | 备注 |
|--------|------|----------|----------------|------|
| **P0** | 改 framing 为知识蒸馏 | 1 天 | +5-10% | 纯写作，不改代码 |
| **P0** | 补强 Baseline（OpenFE/DIFER/AutoFeat/手工特征） | 2 天 | +10% | 单数据集下实验必须过硬 |
| **P1** | 形式化搜索过程 | 1-2 天 | +5% | 纯写作 + 画图 |
| **P2** | Domain Alignment Score | 1 天 | +2-3% | 锦上添花 |
| ~~P0~~ | ~~补公开数据集~~ | — | — | 已决定不做，聚焦单一场景 |

**策略：单数据集 + 领域深绑定 + 多 baseline 对比。**

**Applied Track 中稿概率预估：35-45%。**

## 6. 论文结构草案

```
1. Introduction
   - 数字人民币反洗钱的挑战
   - 人工特征工程的瓶颈：领域知识难以转化为特征
   - 本文贡献：LLM Agent 作为知识蒸馏器

2. Related Work
   - 自动特征工程：AutoFeat, OpenFE, DIFER
   - LLM for Data Science：CAAFE, DS-Agent
   - 反洗钱/欺诈检测

3. Problem Formulation
   - 形式化定义特征空间搜索问题
   - Domain knowledge 约束下的搜索空间缩减

4. Methodology
   4.1 System Overview（架构图）
   4.2 Domain Knowledge Structuring
   4.3 LLM Agent with SHAP Feedback Loop
   4.4 Noise-Calibrated Statistical Validation
   4.5 Forward Selection with Candidate Pool

5. Theoretical Analysis
   5.1 搜索空间缩减的形式化
   5.2 收敛性的 empirical bound

6. Experiments
   6.1 Dataset & Setup（数字人民币反洗钱数据集）
   6.2 Multi-model Validation（XGBoost/LightGBM/RF/LR）
   6.3 Comparison with Baselines（OpenFE, DIFER, AutoFeat, CAAFE, 手工特征, Random）
   6.4 Ablation Studies
   6.5 Stability Analysis
   6.6 Case Study: Generated Features Interpretation

7. Conclusion
```

## 7. 相关文献

需要引用和对比的关键论文：

- Hollmann et al., "CAAFE: LLM-based Feature Engineering for Tabular Data" (2023)
- Zhang et al., "OpenFE: Automated Feature Generation with Expert-level Performance" (NeurIPS 2022)
- Zhu et al., "DIFER: Differentiable Automated Feature Engineering" (ICLR 2024)
- DS-Agent: "Automated Data Science by Empowering LLMs" (2024)
- Weber et al., "Anti-Money Laundering in Bitcoin" (关于 Elliptic 数据集)
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)

## 8. 备注

- 当前优先级：先打磨系统本身，实验代码已就绪（`scripts/run_paper_experiments.py`）
- GPU 服务器信息见 memory，需要确认服务器是否可用
- lightgbm 依赖可能需要在服务器上安装
- 已有实验代码支持：多模型验证、CAAFE-style 对比、5 种消融配置、稳定性测试
