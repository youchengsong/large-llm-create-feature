# LLM 特征工程验证方法深度分析

> 对五篇核心论文的特征验证方法进行系统性梳理，并总结通用最佳实践

---

## 一、五篇论文的验证方法详解

### 1. LLM-FE（Abhyankar et al., ICLR 2026 Workshop, arXiv:2503.14434）

#### 1.1 特征级验证

LLM-FE **不单独评估个体特征**，而是评估整个特征转换程序（transformation program）。其核心是**双层优化问题**：

```
max_𝒯  ℰ(f*(𝒯(𝒳_val)), 𝒴_val)
s.t.   f* = arg min_f L(f, 𝒯(𝒳_train), 𝒴_train)
```

其中 `𝒯` 是特征转换程序，`ℰ` 是评估指标，`f*` 是在转换后训练集上训练的最优模型。

**适应度函数（Fitness Function）**：直接使用下游模型在验证集上的性能得分。
- 分类任务：**Accuracy**
- 回归任务：**RMSE**

#### 1.2 集合级验证

- 将新生成的特征转换程序应用到数据集，生成增广数据集
- 在增广训练集上训练预测模型 `f*`
- 在增广验证集 `𝒯(𝒳_val)` 上计算性能得分
- 该得分即为适应度信号

#### 1.3 冗余处理

通过**多岛屿聚类机制**间接处理：
- 产生相同验证分数的特征转换程序被合并到同一聚类
- 选择时使用 **Boltzmann 采样**：

```
P_i = exp(s_i / τ_c) / Σ_j exp(s_j / τ_c)
```

其中 `s_i` 是聚类均分，`τ_c` 控制探索-利用权衡。但**没有显式的特征冗余/共线性检测**。

#### 1.4 交叉验证设置

- **数据划分**：80/20 训练-测试划分
- 训练数据中再划分出子训练集和验证集
- **5次独立实验**，每次使用不同随机种子和划分
- **无嵌套交叉验证**

#### 1.5 消融实验

三个关键消融（图 2）：

| 消融变体 | 性能 | 说明 |
|---------|------|------|
| 完整 LLM-FE | 0.848 | 基线 |
| w/o 领域知识（特征名匿名化）| 0.626 | **领域知识至关重要** |
| w/o 进化精炼（无迭代反馈）| 0.587 | **迭代优化最关键** |
| w/o 数据样本 | ~0.838 | 影响较小 |

#### 1.6 关键公式

**多岛屿选择**：m=3 个岛屿独立进化，每轮选一个岛屿，从中选 k 个程序作为 in-context 示例。

---

### 2. CAAFE（Hollmann et al., NeurIPS 2023, arXiv:2305.03403）

#### 2.1 特征级验证

CAAFE 采用**贪心迭代**策略，逐个评估新生成的特征：

- LLM 生成 Python 代码块（code block），每个代码块可能包含一个或多个新特征
- 代码通过**语法白名单检查**（只允许安全操作执行）
- 在**10次随机验证划分**上评估性能变化

**接受准则**：
```
特征被保留 ⟺ 10次验证划分的平均 ΔROC_AUC > 0
```

如果加入新特征后，10次验证的平均 ROC AUC 提升（哪怕很小），则保留该特征；否则丢弃。

#### 2.2 集合级验证

- 使用 **TabPFN**（Prior-Data Fitted Networks）作为默认下游模型
- 在 14 个 OpenML 数据集上评估
- 主指标：**ROC AUC**（二分类）/ **准确率**（多分类）
- 每个数据集执行 **10轮特征工程迭代**

#### 2.3 冗余处理

**无显式冗余检测**。依赖贪心机制——如果加入冗余特征不能提升验证得分，该特征自然被丢弃。

#### 2.4 交叉验证设置

- 10次随机训练/验证划分
- 使用 TabPFN（无需超参数调整，避免了 HPO 与 FE 的交互混淆）
- **数据泄露防护**：特征生成代码在训练集上拟合，在验证集上应用

#### 2.5 消融实验

CAAFE 论文未包含系统性消融实验，主要通过与以下基线对比：
- 原始数据 + TabPFN
- 原始数据 + 逻辑回归
- 原始数据 + 随机森林
- AutoML 基线（AutoGluon）

**关键结果**：14个数据集中11个获得提升，平均 ROC AUC 0.798 → 0.822。

#### 2.6 错误反馈机制

如果 LLM 生成的代码执行失败（语法错误、运行时错误等），错误信息会反馈给 LLM 进行修正，形成自我纠错循环。

---

### 3. MedFeat（arXiv:2603.02221）

#### 3.1 SHAP 评估的具体计算

MedFeat 的 SHAP 评估是**整个系统的导航核心**：

**Step 1：计算 SHAP 重要性**
```python
# 在每轮迭代 t 结束后
# 使用验证集 D_val 计算 SHAP 值
shap_values = shap.TreeExplainer(model_t).shap_values(D_val)
# 对时间序列特征，同一变量的重复测量作为"时间组"平均
s_t_i = mean(|SHAP(feature_group_i)|)  # 第i个特征组的归一化重要性
```

**Step 2：构建采样概率分布**
```
p_t(i) = s̃_t,i  # 归一化后的 SHAP 重要性
```
高重要性特征被更频繁地采样到"岛屿"中。

**Step 3：岛屿构建**
- 每个岛屿包含 m=3 个特征（基于重要性采样）
- 随机选择采样"静态特征"还是"聚合时间组"
- 岛屿特征子集被传入 LLM prompt，引导生成新特征

#### 3.2 特征接受/拒绝准则

```
k* = arg min_k L_{t,k}   # 选择损失最小的候选

接受条件：L_{t,k*} ≥ L_{t,baseline} + β   （β = 0.01 默认容忍度）
```

即新特征必须使验证损失至少降低 β（0.01）才被接受。

#### 3.3 模型感知机制

根据下游模型类型调整 LLM 生成策略：
- **逻辑回归**：引导生成非线性变换、交互项（弥补模型表达能力不足）
- **XGBoost**：引导生成复杂时序模式、全局统计量（利用模型强项）

#### 3.4 冗余处理

**记忆库机制**：
- 维护"接受特征"和"拒绝特征"的记忆
- 拒绝的特征被追踪，防止重复生成类似的低价值特征
- 接受的特征影响后续重要性估计和岛屿采样

#### 3.5 交叉验证设置

- **60/20/20 分层随机划分**（训练/验证/测试）
- 验证集**双重用途**：(1) 选择成功的特征转换 (2) 计算 SHAP 信号
- 测试集仅用于最终评估

#### 3.6 指标

- **主指标**：AUC（ROC 曲线下面积）
- **辅助指标**：F1 分数（使用验证集上 Youden's J 统计量选择的阈值）

```
Youden's J = max(Sensitivity + Specificity - 1)
```

#### 3.7 消融实验

| 消融变体 | AUC 下降 | F1 下降 |
|---------|---------|--------|
| w/o 模型感知 | 1-4% | 最高 35.9% |
| w/o 岛屿采样 | 1-2% | — |
| w/o SHAP 引导 | 2-3% | 9-26% |

**关键发现**：SHAP 引导对 F1 的影响远大于对 AUC 的影响，说明 SHAP 引导在决策边界处最有价值。

---

### 4. OCTree（Nam et al., NeurIPS 2024, arXiv:2406.08527）

#### 4.1 决策树验证机制

OCTree 的独特之处是用**决策树推理**作为特征质量的反馈信号：

**Step 1：生成特征规则**
```
LLM 根据 prompt 生成一条特征生成规则 r_t
```

**Step 2：执行规则生成新列**
```
新列 = apply(r_t, D_train)
```

**Step 3：训练模型并评估**
```
s_t = ℒ_f*(D_val ⊕ r_t)
```
其中 `⊕` 表示将新列拼接到数据集。

**Step 4：训练 CART 决策树（最大深度=4）**
```python
cart = DecisionTreeClassifier(max_depth=4)
cart.fit(D_train_augmented, y_train)
d_t = extract_rules_as_natural_language(cart)
```

**Step 5：构建优化轨迹**
```
Τ_t = {(s_i, d_i, r_i)}_{i=0}^{t}  # 按分数排序
```
轨迹中包含：验证分数 `s_i`、决策树描述 `d_i`、生成规则 `r_i`。

**关键设计**：轨迹按分数排序传入 LLM，因为 LLM 倾向于生成与列表末尾（高分）相似的建议。

#### 4.2 特征接受/拒绝准则

- **无显式拒绝机制**
- 运行固定迭代次数后，**选择验证分数最高的规则**
- 低分规则保留在轨迹中作为反面教材

#### 4.3 交叉验证设置

- **60/20/20 划分**（训练/验证/测试）
- 时间序列数据按**时间索引**划分（非随机），避免未来信息泄露

#### 4.4 指标

| 任务类型 | 指标 | 方向 |
|---------|------|------|
| 分类 | 测试错误率（%）| 越低越好 |
| 回归 | 平均绝对误差 MAE（×10⁻³）| 越低越好 |

报告方式：**相对错误率降低百分比**。

#### 4.5 冗余处理

通过 prompt 约束实现：要求 LLM "生成不在轨迹 Τ_t 中出现过的新规则"。无统计层面的冗余检测。

#### 4.6 消融实验

| 配置 | 改进幅度 |
|------|---------|
| 仅特征生成（无决策树反馈）| 1.4-17.2% |
| 特征生成 + 决策树推理 | 2.6-20.1% |

**特征可迁移性实验**：在 XGBoost 上优化的特征迁移到 MLP 和 HyperFast 后仍有 1.6-7.5% 提升。

---

### 5. CoFEH（Xu et al., ASPLOS 2026, arXiv:2602.09851）

#### 5.1 评估框架

CoFEH 使用**最完整的评估框架**，包含三层评估：

**层 1：节点级验证（MCTS 每个节点）**
```
v_new = evaluate(model, transform(D_val, t_new))
r = 𝕀(v_new > v_{s0}^max)  # 二值奖励：是否超过全局最优
```

**层 2：子树级传播（回溯更新）**
```
N_s ← N_s + 1            # 访问次数
R_s ← R_s + r             # 累积奖励
v_s^max ← max(v_s^max, v_new)  # 子树最优性能
```

**层 3：UCT 选择（平衡探索与利用）**
```
s* = arg max_{s'} [Q(s') + C₁ · √(ln N_s / N_{s'})]

其中 Q(s') = R_{s'}/N_{s'} + ṽ_{s'}^max
     C₁ = √2
```

#### 5.2 MCTS + 贝叶斯优化联合框架

**特征工程（FE）端**：MCTS 搜索最优特征转换管道
**超参数优化（HPO）端**：贝叶斯优化调整下游模型超参数

**动态调度器（MAB-PUCB）**：
```
a* = arg max_{a ∈ {FE, HPO}} [Q(a) + C₂ · ω_a(m) · √(Σ_{a'} N_{a'}) / (1 + N_a)]

ω_FE(m) = p₁ - (p₁ - 0.5) · (m/M)    # 从 0.9 衰减到 0.5
ω_HPO(m) = p₂ + (0.5 - p₂) · (m/M)    # 从 0.1 增长到 0.5
```

**设计意图**：早期优先搜索特征（FE），后期转向超参数优化（HPO），因为好特征的边际收益递减。

#### 5.3 交叉验证设置

- **60/20/20 划分**（训练/验证/测试）
- 验证集驱动优化决策
- 最终报告时：在训练+验证集上重新拟合最优配置，在测试集上评估
- **3次独立运行**，报告均值 ± 标准差

#### 5.4 指标

| 任务 | 指标 | 方向 |
|------|------|------|
| 分类 | 错误率（1-accuracy）| ↓ |
| 回归 | MSE | ↓ |
| 排名 | 平均排名 | ↓ |
| 改进 | 相对错误降低（%）| ↑ |

#### 5.5 冗余处理

通过**全局记忆架构**管理：
```
每个操作存储为元组 (ℛ, ℱ_t, v, Δv)
- ℛ = 推理链
- ℱ_t = 特征依赖
- v = 性能结果
- Δv = v - v_base（相对改进）
```

**两阶段检索**（利用阶段）：
1. **功能过滤**：匹配所需特征与当前数据集可用特征
2. **Pareto 筛选**：在性能 v 和改进增益 Δv 之间找非支配解

#### 5.6 消融实验

| 变体 | 平均测试排名 |
|------|------------|
| **完整 CoFEH** | **1.25** |
| w/o 条件化（去掉 HPO→FE 反馈）| 2.75 |
| w/o 调度器（固定交替）| 2.25 |
| 贪心（先100轮FE再100轮HPO）| 3.75 |

**关键结论**：贪心顺序方法最差，证明 FE 和 HPO 的交互优化至关重要。

#### 5.7 公平对比协议

- 所有方法统一使用**100轮迭代预算**（单独 FE）或**200轮**（联合 FE+HPO）
- 统一 LLM backbone（Gemini 2.0 Flash）
- 统一 HPO 模块（SMAC）
- 报告 API 费用和运行时间

---

## 二、五篇论文验证方法对比总结

| 维度 | LLM-FE | CAAFE | MedFeat | OCTree | CoFEH |
|------|--------|-------|---------|--------|-------|
| **特征级评估** | 无（评估整个程序）| 有（逐个代码块）| 有（SHAP 驱动）| 有（逐规则）| 有（逐操作）|
| **集合级评估** | 下游模型验证得分 | ROC AUC 变化 | 验证损失变化 | 验证错误率 | MCTS 节点性能 |
| **冗余检测** | 聚类合并（间接）| 无 | 记忆库避免重复 | Prompt 约束 | Pareto 筛选 |
| **统计检验** | 无 | 无 | 无 | 无 | 无 |
| **分类指标** | Accuracy | ROC AUC | AUC + F1 | 错误率 | 错误率(1-acc) |
| **回归指标** | RMSE | — | — | MAE | MSE |
| **CV 设置** | 80/20 × 5种子 | 10次随机划分 | 60/20/20 分层 | 60/20/20 | 60/20/20 × 3次 |
| **数据泄露防护** | 分离训练/验证 | 代码在训练集拟合 | 分离验证/测试 | 时序数据按时间划分 | 最终在 train+val 上重拟合 |
| **消融实验** | 3项消融 | 无系统消融 | 3项消融 | 2项消融 | 4项消融 |
| **SHAP 使用** | 否 | 否 | **核心驱动** | 否 | 否 |
| **可解释反馈** | 进化记忆 | 错误信息 | SHAP + 模型感知 | 决策树规则 | 推理链 + Pareto |

---

## 三、通用最佳实践：特征验证方法论

### 3.1 特征相关性统计检验

#### 卡方检验（Chi-Square Test）

**适用**：分类特征 vs 分类目标

```python
from scipy.stats import chi2_contingency

# 构建列联表
contingency = pd.crosstab(feature, target)
chi2, p_value, dof, expected = chi2_contingency(contingency)

# 判断标准
if p_value < 0.05:
    print("特征与目标显著相关")
```

**Balek 2025 的增强方法**：使用 2500 次 bootstrap 卡方检验，更稳健。

#### 互信息（Mutual Information）

**适用**：任意类型特征 vs 目标（能捕捉非线性关系）

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
# MI = 0 → 完全独立
# MI 越高 → 依赖性越强
```

**优势**：不假设线性关系，能检测任何形式的统计依赖。
**注意**：MI 需要估计，不同估计器会影响结果（k-NN 估计器较稳健）。

#### Cramer's V（效应大小）

**适用**：衡量分类特征与分类目标的关联强度，不受样本量影响

```
V = √(χ² / [N × (min(r,c) - 1)])
```

**解释标准**：
| V 值 | 关联强度 |
|------|---------|
| < 0.10 | 可忽略 |
| 0.10 - 0.30 | 弱 |
| 0.30 - 0.50 | 中等 |
| > 0.50 | 强 |

**Balek 2025 报告**：其 LLM 生成特征的 Cramer's V 为 0.191-0.865（中等到强关联）。

#### mRMR（最大相关最小冗余）

```
mRMR_score(f) = MI(f, target) - (1/|S|) × Σ_{s∈S} MI(f, s)
```

同时最大化与目标的相关性，最小化与已选特征集的冗余。

### 3.2 SHAP 特征重要性 vs 置换特征重要性

| 维度 | SHAP | 置换重要性 |
|------|------|-----------|
| **计算复杂度** | 高（TreeSHAP 较快）| 低 |
| **粒度** | 实例级 + 全局 | 仅全局 |
| **方向信息** | 有（正/负贡献）| 无 |
| **理论基础** | Shapley 值公理 | 无 |
| **共线性处理** | 差（假设独立）| 差（低估共线特征）|
| **适用场景** | 需要解释为什么 | 需要快速排名 |

**推荐策略**：
1. 先用置换重要性快速筛选（计算快，全局排名）
2. 再用 SHAP 深入分析 top-k 特征（理解方向和交互）
3. 两者不一致时，以 SHAP 为准（更精确的归因）

### 3.3 增量特征价值评估

#### 方法 1：边际贡献检验

```python
# 基线模型（已有特征）
baseline_auc = cross_val_score(model, X_existing, y, scoring='roc_auc', cv=5).mean()

# 增量模型（已有特征 + 新特征）
augmented_auc = cross_val_score(model, X_augmented, y, scoring='roc_auc', cv=5).mean()

# 增量贡献
delta = augmented_auc - baseline_auc
```

**MCI 公理（Marginal Contribution Feature Importance）**：
> 如果一个特征在所有其他特征都存在的情况下仍能带来性能提升，则其重要性至少等于该额外增益。

#### 方法 2：Shapley 值增量

```python
# 计算新特征在完整模型中的 SHAP 值
# 如果 mean(|SHAP(new_feature)|) 排名在 top-k，则有增量价值
```

#### 方法 3：嵌套对比实验

```
实验 1：仅人工特征 → AUC_baseline
实验 2：人工特征 + LLM 特征 → AUC_augmented
实验 3：仅 LLM 特征 → AUC_llm_only
实验 4：人工特征 + 随机特征 → AUC_random（对照组）

增量贡献 = AUC_augmented - AUC_baseline
独立贡献 = AUC_llm_only
随机对照 = AUC_random - AUC_baseline（应接近0）
```

### 3.4 特征稳定性分析

#### Nogueira 稳定性指数

评估特征选择在不同数据子集上的一致性：

```python
import numpy as np

def nogueira_stability(feature_sets, n_features):
    """
    feature_sets: list of binary vectors [1,0,1,...] 表示哪些特征被选中
    n_features: 总特征数
    """
    M = len(feature_sets)
    Z = np.array(feature_sets)
    p_f = Z.mean(axis=0)  # 每个特征被选中的概率
    k_bar = p_f.sum()      # 平均选中特征数
    
    numerator = (1/n_features) * np.sum(p_f * (1 - p_f))
    denominator = (k_bar / n_features) * (1 - k_bar / n_features)
    
    stability = 1 - numerator / denominator
    return stability  # 越接近1越稳定
```

#### Bootstrap 稳定性检验

```python
from sklearn.utils import resample

stability_results = []
for _ in range(100):
    X_boot, y_boot = resample(X, y, stratify=y)
    # 在 bootstrap 样本上运行特征选择
    selected = feature_selection(X_boot, y_boot)
    stability_results.append(selected)

# 计算 Jaccard 相似度矩阵
# 稳定的特征选择方法应该在不同 bootstrap 样本上选出相似的特征集
```

**实践建议**：
- 使用 90% 子采样（小扰动）和 63.2% 子采样（大扰动，等效 bootstrap）两种设置
- 报告 Kuncheva/Nogueira/Jaccard 指数
- 稳定性 > 0.8 认为可接受

### 3.5 不平衡分类场景的特征评估（AML 场景）

#### 为什么 Accuracy 和 ROC-AUC 不够？

AML 数据通常正样本比例 ~5%。此时：
- **Accuracy 无用**：全预测为负可得 95% accuracy
- **ROC-AUC 过度乐观**：大量真负例使 FPR 极低，ROC 曲线看起来很好，但实际正类检出能力差

#### 推荐指标体系

| 指标 | 优先级 | 理由 |
|------|--------|------|
| **PR-AUC（精确率-召回率曲线下面积）** | **最高** | 仅关注正类表现，对类别不平衡敏感 |
| **F1-score（使用 Youden's J 选阈值）** | 高 | 平衡精确率和召回率 |
| **ROC-AUC** | 中 | 全局排序能力，但可能过度乐观 |
| **Recall @ fixed Precision** | 高 | AML 实战：在可接受误报率下最大化检出率 |
| **Precision @ fixed Recall** | 高 | AML 实战：确保检出率的前提下减少误报 |

#### PR-AUC 为何更适合 AML

```
ROC-AUC 高 + PR-AUC 低 → 模型善于排除正常交易，但不擅长找出可疑交易
ROC-AUC 高 + PR-AUC 高 → 模型真正有效

不平衡越严重，ROC-AUC 与 PR-AUC 的差距越大
```

#### 特征评估中的不平衡处理

```python
# 1. 使用分层采样的 CV
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True)

# 2. 用 PR-AUC 作为特征选择的评估指标
from sklearn.metrics import average_precision_score
feature_score = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

# 3. 对每个新特征，计算其在少数类上的区分度
from sklearn.metrics import roc_auc_score
# 关注少数类的 recall 提升，而非整体 accuracy
```

#### SMOTE / 过采样的注意事项

- **必须在 CV 的 fold 内部**做过采样，不能在划分前做（否则数据泄露）
- 特征评估时建议**不做过采样**，在原始分布上评估更真实
- 过采样仅用于下游模型训练

---

## 四、对本项目的具体建议

### 4.1 推荐验证框架

结合五篇论文的最佳实践，为数字人民币反洗钱项目设计的验证框架：

```
                        ┌──────────────────────────────┐
                        │  LLM 生成候选特征（代码）       │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Level 1: 代码验证              │
                        │  - 语法检查（CAAFE 白名单）      │
                        │  - 执行超时检查（30s）           │
                        │  - 输出分布检查（非全0/全1）      │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Level 2: 统计验证              │
                        │  - 互信息 MI > 阈值             │
                        │  - 与已有特征相关系数 < 0.9      │
                        │  - Cramer's V（分类特征）        │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Level 3: 模型验证              │
                        │  - 加入 XGBoost，5折分层CV       │
                        │  - 计算 ΔPR-AUC（增量贡献）     │
                        │  - 计算 SHAP 排名               │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  Level 4: 稳定性验证            │
                        │  - 10次 bootstrap 重采样        │
                        │  - 检查特征重要性排名是否稳定    │
                        │  - Nogueira 稳定性指数 > 0.8    │
                        └──────────────────────────────┘
```

### 4.2 具体指标配置

```python
EVALUATION_CONFIG = {
    # Level 2: 统计阈值
    "min_mutual_information": 0.01,         # MI 最低阈值
    "max_correlation_with_existing": 0.9,    # 冗余度阈值
    "min_cramers_v": 0.05,                  # Cramer's V 最低阈值
    "chi2_p_value_threshold": 0.05,         # 卡方检验显著性水平
    
    # Level 3: 模型验证
    "cv_folds": 5,
    "cv_strategy": "StratifiedKFold",
    "primary_metric": "average_precision",   # PR-AUC
    "secondary_metrics": ["roc_auc", "f1", "recall"],
    "min_delta_pr_auc": 0.005,              # 最低增量贡献
    
    # Level 4: 稳定性
    "bootstrap_iterations": 10,
    "min_stability_index": 0.8,
    
    # MedFeat 式 SHAP 引导
    "shap_feedback": True,
    "shap_top_k_for_prompt": 10,
    "acceptance_tolerance_beta": 0.01,       # MedFeat 的 β 参数
}
```

### 4.3 综合评分公式

借鉴 LLM_FEATURE_PLAN.md 中的设计，扩展为：

```python
def compute_feature_score(feature, existing_features, model, X_train, y_train, X_val, y_val):
    """
    综合评分 = w1 * SHAP_rank + w2 * MI + w3 * (1 - redundancy) + w4 * stability
    """
    # 1. SHAP 排名得分（归一化到 0-1）
    shap_rank = get_shap_rank_normalized(feature, model, X_val)
    
    # 2. 互信息得分
    mi = mutual_info_classif(feature.values.reshape(-1, 1), y_train)[0]
    mi_normalized = mi / max_possible_mi
    
    # 3. 冗余度（与所有已有特征的最大相关系数）
    max_corr = max([abs(feature.corr(existing)) for existing in existing_features])
    redundancy = max_corr
    
    # 4. 稳定性（可选，计算成本较高）
    stability = bootstrap_stability(feature, X_train, y_train)
    
    # 综合得分
    score = (0.35 * shap_rank + 
             0.25 * mi_normalized + 
             0.25 * (1 - redundancy) + 
             0.15 * stability)
    
    return score
```

### 4.4 最终报告所需的增量贡献实验

```
实验 1（基线）：          40个人工特征 → XGBoost → PR-AUC_baseline
实验 2（增量）：          40个人工 + N个LLM特征 → XGBoost → PR-AUC_augmented  
实验 3（独立）：          仅N个LLM特征 → XGBoost → PR-AUC_llm_only
实验 4（随机对照）：      40个人工 + N个随机特征 → XGBoost → PR-AUC_random
实验 5（消融-逐特征）：   逐个移除LLM特征，观察PR-AUC变化

关键报告指标：
- 增量 PR-AUC = PR-AUC_augmented - PR-AUC_baseline
- 增量 F1 = F1_augmented - F1_baseline
- 增量 Recall@Precision=0.5 = Recall_aug - Recall_base
- 各LLM特征的 SHAP 排名（在完整模型中）
```

---

## 参考文献

1. Abhyankar et al. "LLM-FE: Automated Feature Engineering with LLMs as Evolutionary Optimizers." arXiv:2503.14434, 2025.
2. Hollmann et al. "Large Language Models for Automated Data Science: Introducing CAAFE." NeurIPS 2023.
3. MedFeat: "Model-Aware and Explainability-Driven Feature Engineering with LLMs." arXiv:2603.02221, 2026.
4. Nam et al. "OCTree: Optimized Feature Generation via LLMs with Decision Tree Reasoning." NeurIPS 2024.
5. Xu et al. "CoFEH: LLM-driven Feature Engineering Empowered by Collaborative Bayesian HPO." arXiv:2602.09851, 2026.
6. Balek et al. "LLM-based Feature Generation from Text for Interpretable ML." Machine Learning, 2025.
7. Catav et al. "Marginal Contribution Feature Importance." ICML 2021.
8. Nogueira et al. "On the Stability of Feature Selection Algorithms." JMLR 18(174):1-54, 2018.
