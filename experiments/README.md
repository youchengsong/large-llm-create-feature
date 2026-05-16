# 实验记录

## 版本说明

### run3_wrong_data (2025-05-15)
- **数据**: 错误数据（14,983 条交易，非完整数据集）
- **验证方法**: 旧版 PR-AUC delta 验证
- **结果**: 无参考价值，仅作为记录保留
- **日志**: search_run2.log (旧数据旧方法), search_run3.log (错误数据)

### run4_prauc_validation (2025-05-16)
- **数据**: 正确数据（1,384,512 条交易，100K 账户）
- **验证方法**: 旧版 PR-AUC delta 验证（L2: MI + Cramers V, L3: delta_prauc > 0）
- **结果**: 0 个特征通过（基线 PR-AUC=0.9966 太高，增量几乎不可能为正）
- **日志**: search_run4.log

### run5_f1_validation (2025-05-16) ⭐ 当前最佳
- **数据**: 正确数据（1,384,512 条交易，100K 账户）
- **验证方法**: 
  - L2: IV + KS + redundancy + missing_rate（噪声基线校准）
  - L3: F1/Recall/Precision delta 判定
- **已知问题**: 阈值在验证集上寻找（threshold leakage），指标略有膨胀。代码已修复但本次运行使用旧代码。
- **结果**:
  - 10 轮搜索，71 个特征通过验证，导出 70 个特征文件
  - 基线: F1=0.9769, Recall=0.9708, Precision=0.9832, PR-AUC=0.9966
  - 组合: F1=0.9846, Recall=0.9837, Precision=0.9855, PR-AUC=0.9984
  - **增量: F1+0.0077, Recall+0.0129, Precision+0.0024, PR-AUC+0.0018**

## 目录结构

```
experiments/
├── README.md
├── run3_wrong_data/
│   └── logs/
│       ├── search_run2.log
│       └── search_run3.log
├── run4_prauc_validation/
│   └── logs/
│       └── search_run4.log
└── run5_f1_validation/
    ├── logs/
    │   ├── search_run5.log
    │   ├── eval_combined.log
    │   ├── eval_simple.log
    │   ├── eval_v2.log
    │   └── eval_v3.log
    └── output/
        ├── features.json          # 71个特征的完整信息（代码+指标+解释）
        ├── search_log.json        # 搜索过程详细日志
        ├── feature_dictionary.json
        ├── feature_dictionary.md
        ├── report.md
        ├── pipeline_explanation.html
        ├── features/              # 70个特征的独立Python文件
        └── plots/                 # 可视化图表
```

## 下一步计划

1. 使用修复后的指标代码（threshold on train set）重新运行搜索
2. 修复 L3 接受条件（避免 delta_f1 和 delta_recall 同时为负时仍接受）
3. 考虑使用 DeepSeek-R1 (deepseek-reasoner) 模型提升特征质量
