# 反洗钱算法模型集 (AML ModelSet) 自动化框架手册

## 1. 环境配置 (Conda)

本项目集成了 11 种前沿的反洗钱识别模型，涵盖了集成学习、深度学习、规则融合及可解释 AI 等方向。所有模型均已完成“工程化重构”，遵循统一的接口规范，支持通过中央调度器进行一键化批处理运行和性能对标。
### 1.1 初始化目录结构
在开始训练之前，运行初始化脚本以自动生成模型存放、结果输出及图表保存所需的文件夹：
```bash
python init_env.py
```
#### 自动生成的目录说明：
* /data: 存放原始及处理后的 CSV 数据（请将你的训练/测试集放入此目录）。
* /saved_models: 统一存放训练生成的 .joblib 模型序列化文件。
* /results: 存放 global_results.csv 性能对标报表。
* /plots: 存放 SHAP 可解释性图表及规则逻辑图。

### 1.2 创建环境
创建名为 `aml_model_set` 的环境：
```bash
conda env create -f environment.yml
conda activate aml_model_set
```

## 2. 数据规范
所有模型输入的 CSV 文件需满足以下基本要求：
1. 标签列：必须包含名为 is_laundering 的列（0 为正常，1 为洗钱）。
2. 缺失值：系统在加载数据时会自动执行中位数填充（防崩溃机制），但建议在特征工程阶段预先处理严重缺失的异常值。
3. **特征排除：** 系统在训练前会自动剔除 Timestamp、From_Account、To_Account 等业务主键，你无需手动在 CSV 中删除它们。

## 中央调度器使用指南 (runner.py)
runner.py 是整个项目的唯一执行入口，通过命令行参数控制所有 11 个模型的行为。
### 核心参数详解
以下是 `runner.py` 支持的所有命令行参数。请根据具体的执行模式传递相应的路径。

| 参数名            | 是否必填 | 默认值 | 功能描述                                                                            |
|:---------------| :---: | :--- |:--------------------------------------------------------------------------------|
| `--model`      | **是** | - | 指定运行的模型 ID（如 `catboost`、 `acxgb`,具体见下方`核心模型矩阵库`）。支持空格分隔多个名称，或直接输入 `all` 运行全部模型。 |
| `--mode`       | **是** | - | 运行模式选择：<br>`train`: 仅训练并保存模型<br>`test`: 仅加载模型进行预测<br>`both`: 串行执行训练与测试          |
| `--train_data` | 否* | - | 训练集 CSV 路径。在 `train` 或 `both` 模式下为**必填项**。                                      |
| `--test_data`  | 否* | - | 测试集 CSV 路径。在 `test` 或 `both` 模式下为**必填项**。                                       |
| `--model_dir`  | 否 | `./saved_models` | 指定模型序列化文件 (`.joblib`) 的存放与读取根目录。                                                |
| `--output`     | 否 | `results/global_results.csv`| 指定性能评估指标追加保存的 CSV 文件路径。                                                         |

> **⚠️ 参数依赖提示**：
> 虽然 `--train_data` 和 `--test_data` 在全局定义中是非必填项，但如果你的 `--mode` 设置为 `both`，则**必须同时提供这两个数据路径**，否则系统会跳过对应模型的执行。

## 常用操作命令示例
### 场景 A：一键跑完所有模型的全流程对标实验
```bash
python runner.py --model all --mode both --train_data data/train.csv --test_data data/test.csv
```
### 场景 B：指定特定模型组合运行
```bash
python runner.py --model catboost lgbm dnn_res --mode both --train_data data/train.csv --test_data data/test.csv
```

### 场景 C：仅加载已有模型进行批量测试
```bash
python runner.py --model all --mode test --test_data data/new_test_set.csv --output results/global_results.csv
```

## 测试结果说明 (global_results.csv)
所有模型的测试表现都会按照统一格式追加到结果 CSV 中。使用 Excel 打开该文件，你将看到以下字段：
* model_name: 模型唯一 ID（如 GBDT_FM_Hybrid）。
* start_time: 任务调度启动时间。
* end_time: 模型完成推理与指标计算的结束时间。
* threshold: 模型判定为“洗钱”的概率分界线（阈值）。
* precision: 查准率（判定为洗钱的案例中，真实的洗钱比例）。
* recall: 查全率（实际洗钱案例中，被模型成功拦截的比例，反洗钱核心指标）。
* f1_score: 查准率和查全率的调和平均数。
* auc: ROC 曲线下面积，衡量模型区分正常与洗钱交易的排序能力。

## 核心模型矩阵库
本项目包含以下 11 种经过工程化重构的算法模型：
1. catboost: 结合动态阈值优化的 CatBoost 分类器。
2. ga2m: 高度可解释的广义加性模型 (EBM)。
3. scorecard: 基于 Toad 库的银行级标准评分卡流水线。
4. acxgb: 结合熵不确定性采样的主动学习 XGBoost。
5. wav_xgb: 融合小波变换特征空间扩展的 XGBoost。
6. ensemble: XGB + 决策树 + 逻辑回归的 Stacking 多层集成。
7. lgbm: 结合 ADASYN 过采样的 LightGBM。
8. resample_boost: 结合 SMOTE 与 AdaBoost 随机搜索调参的模型。
9. gbdt_fm: GBDT 特征离散化后接入 PyTorch 因子分解机 (FM)。
10. dnn_res: LightGBM 概率驱动的 PyTorch 残差修正网络。
11. mlp: 决策树规则挖掘并注入深度学习网络的融合模型。