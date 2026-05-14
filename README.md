# 大模型提取特征

利用 LLM（大语言模型）自动为数字人民币反洗钱检测模型生成高质量、可解释的特征。

## 核心思路

让 LLM 充当特征工程师 — 给它领域知识和数据 Schema，让它生成 Python 特征提取代码，再通过严格的验证筛选出真正有效的特征。

## 系统架构

```
原始数据（三表）+ 领域知识
       │
   Phase 0: 数据准备与基线构建
       │
   Phase 1: LLM 迭代生成特征代码
       │   ├── 4 维度岛屿搜索（金额频次/时间地域/行为序列/主体关联）
       │   ├── SHAP 动态预算分配
       │   ├── 多轮岛内对话（最多 3 轮/岛）
       │   ├── 按需数据探索（LLM 主动查询统计）
       │   └── Boltzmann 采样 + 失败记忆
       │
   Phase 2: 特征验证与接受
       │   ├── 直通路径: L1 沙盒 → L2 统计 → L3 增量 → 直接接受
       │   ├── 缓刑路径: L2 通过 + L3 未通过 → 缓刑池 → 反向淘汰
       │   └── 可解释性报告生成
       │
   Phase 3: 最终验证与产出
       │   ├── L4 稳定性验证（Nogueira 指数）
       │   ├── 5 组增量贡献实验
       │   └── 特征字典 + 可视化 + 报告
       ▼
  经过验证的特征代码 + 可解释性说明
```

## 特征验证机制

### 直通接受（路径一）

LLM 生成的候选特征逐个通过三级验证：

- **L1 沙盒验证**：AST 安全检查 + 子进程隔离执行 + 输出格式验证
- **L2 统计验证**：MI ≥ 0.001、Redundancy < 0.9、Cramer's V ≥ 0.05、Bootstrap χ² p < 0.05
- **L3 增量验证**：Delta PR-AUC ≥ 0.005（XGBoost 5 折 CV）

三级都通过 → 直接接受，加入特征集。

### 缓刑池 + 反向淘汰（路径二）

L2 通过但 L3 单独增量不够的特征进入缓刑池。每轮结束后：

1. 已接受特征 + 缓刑池特征一起训练模型
2. 按 SHAP 从弱到强逐个尝试剔除缓刑池特征
3. 剔除后分数下降 → 说明在组合中有贡献 → 接受
4. 剔除后分数不降 → 确认无贡献 → 留池等下轮或淘汰

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# Phase 0-2: 搜索生成特征（使用 DeepSeek）
DEEPSEEK_API_KEY=your_key python src/main.py --provider deepseek --model deepseek-chat

# Phase 3: 最终验证与产出
python src/phase3_main.py
```

支持的 LLM 提供商：OpenAI、Gemini、DeepSeek、本地模型（OpenAI 兼容接口）、Mock（测试用）。

## 项目结构

```
src/
├── config.py              # 超参数配置
├── main.py                # Phase 0-2 主入口
├── phase3_main.py         # Phase 3 主入口
├── data/
│   ├── loader.py          # 数据加载 + schema + 分布差异 + 关联统计
│   ├── merge.py           # 特征 merge 到交易表
│   ├── baseline.py        # 人工基线特征
│   └── stats_query.py     # 按需统计查询沙盒
├── prompt/
│   ├── builder.py         # System/User Prompt 组装
│   └── domain_knowledge.py # 领域知识加载
├── llm/
│   ├── interface.py       # LLM 可插拔接口（5 种提供商）
│   └── parser.py          # LLM 输出解析
├── evaluation/
│   ├── sandbox.py         # 代码安全沙盒
│   ├── metrics.py         # MI/冗余度/Cramer's V/Delta PR-AUC/SHAP
│   ├── validator.py       # L1-L3 验证 + 反向淘汰
│   ├── stability.py       # L4 稳定性验证
│   └── experiments.py     # 5 组增量实验
├── search/
│   ├── engine.py          # 搜索引擎主程序
│   ├── island.py          # 维度岛屿 + Boltzmann 采样
│   ├── memory.py          # 失败记忆
│   ├── pending_pool.py    # 缓刑池管理
│   └── shap_feedback.py   # SHAP 反馈 + 动态预算
└── report/
    ├── feature_dictionary.py
    ├── visualization.py
    └── generator.py
```

## 核心创新

| 能力 | LLM-FE | CAAFE | 本系统 |
|------|--------|-------|--------|
| 多表支持 | x | x | 三表关联 |
| 领域知识注入 | 仅特征名 | 数据集描述 | 深度注入（专家访谈+规则+指标） |
| 搜索策略 | 随机岛屿 | 单向 | 维度岛屿 + SHAP 动态预算 |
| 统计验证 | x | x | MI + Cramer's V + Bootstrap χ² |
| 组合效应 | x | x | 缓刑池反向淘汰 |
| 数据探索 | x | x | 预计算注入 + 按需查询 |
| 可解释性 | x | x | 结构化报告（阈值/协同/误判/监管） |

## 文献参考

详见 [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)，涵盖 CAAFE (NeurIPS 2023)、OCTree (NeurIPS 2024)、LLM-FE (2025)、Balek 2025 等核心论文。

## 项目状态

- [x] 文献调研
- [x] 方案设计
- [x] 代码实现（3000+ 行，171 个测试）
- [ ] 实验验证（接入真实 LLM 运行）
