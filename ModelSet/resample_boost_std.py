import pandas as pd
import numpy as np
import joblib
import time
import os
import argparse
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, \
    classification_report

warnings.filterwarnings("ignore")


# --- 1. 辅助功能函数 (保持原源码逻辑) ---

def load_and_preprocess(filepath, target_col='is_laundering'):
    """加载数据并剔除无关列"""
    df = pd.read_csv(filepath)
    # 按照原源码逻辑剔除特定字段
    drop_columns = [
        'id', 'account_is_laundering_From', 'account_is_laundering_To',
        'src_bank_account_number', 'dst_bank_account_number', 'Timestamp'
    ]
    actual_drop = [c for c in drop_columns if c in df.columns]
    X = df.drop(columns=actual_drop + [target_col], axis=1)
    y = df[target_col].astype(int)

    # 【新增】使用中位数填充缺失值（对异常值更稳健）
    df = df.fillna(df.median())
    # 或者用 0 填充：df = df.fillna(0)

    X = df.drop(columns=actual_drop + [target_col], axis=1)
    y = df[target_col].astype(int)
    return X, y


# --- 2. 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：执行 SMOTE、自动调参训练并打包保存"""
    print(f"--- [ResampleBoost 训练模式] ---")
    X, y = load_and_preprocess(train_csv)

    # 获取特征名称
    feature_names = X.columns.tolist()

    # 1. 模拟原源码中的 SMOTE 逻辑
    pNum = (y == 1).sum()
    nNum = (y == 0).sum()
    original_ratio = pNum / nNum

    # 设定 SMOTE 比例（默认增加 0.005 比例，最高 0.5）
    smote_ratio = min(0.5, original_ratio + 0.005)
    print(f"正在启动 SMOTE... 目标比例: {smote_ratio:.4f}")
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 2. 初始化超参数搜索 (RandomizedSearchCV)
    print("正在进行随机超参数搜索 (10次迭代)...")
    ada = AdaBoostClassifier(random_state=42)
    param_grid = {
        'n_estimators': randint(10, 800),
        'learning_rate': uniform(0.01, 0.99),
        'estimator': [DecisionTreeClassifier(max_depth=5), DecisionTreeClassifier(max_depth=20)],
    }

    random_search = RandomizedSearchCV(
        estimator=ada,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        verbose=1
    )

    start_time = time.time()
    random_search.fit(X_resampled, y_resampled)
    duration = time.time() - start_time

    best_model = random_search.best_estimator_
    print(f"训练完成，耗时: {duration:.2f}s. 最佳 F1 分数: {random_search.best_score_:.4f}")

    # 3. 【核心修改】全组件打包
    bundle = {
        'model': best_model,
        'feature_names': feature_names,
        'model_name': 'Resample_AdaBoost'
    }

    joblib.dump(bundle, model_save_path)
    print(f"最优模型已保存至: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载打包文件，执行测试并输出统一结果"""
    print(f"--- [ResampleBoost 测试模式] ---")
    if not os.path.exists(model_load_path):
        print("错误：找不到模型文件")
        return

    # 1. 加载组件包
    bundle = joblib.load(model_load_path)
    model = bundle['model']
    saved_features = bundle['feature_names']

    # 2. 读入测试集并进行特征对齐
    X_test, y_test = load_and_preprocess(test_csv)

    # 自动补齐或对齐列
    X_test_aligned = pd.DataFrame()
    for col in saved_features:
        X_test_aligned[col] = X_test[col] if col in X_test.columns else 0
    X_test_aligned = X_test_aligned[saved_features]

    # 3. 预测
    print("正在执行预测...")
    y_pred = model.predict(X_test_aligned)
    # 如果模型支持概率预测则计算 AUC
    y_prob = model.predict_proba(X_test_aligned)[:, 1] if hasattr(model, 'predict_proba') else None

    # 4. 计算指标
    auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else "N/A"
    report = classification_report(y_test, y_pred, output_dict=True)

    # 5. 写入统一结果模板
    result_entry = {
        "model_name": bundle['model_name'],
        "start_time": start_time or "N/A",
        "end_time": end_time or "N/A",
        "threshold": bundle['threshold'] if 'threshold' in bundle else 0.5,
        "precision": round(report['1']['precision'], 4),
        "recall": round(report['1']['recall'], 4),
        "f1_score": round(report['1']['f1-score'], 4),
        "auc": round(auc, 4)
    }

    # 定义所有模型共用的标准列顺序
    res_df = pd.DataFrame([result_entry])
    header = not os.path.exists(result_csv)
    res_df.to_csv(result_csv, mode='a', index=False, header=header)
    print(f"测试完成，结果已追加至: {result_csv}")


# --- 命令行交互入口 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--data", required=True, help="输入 CSV 路径")
    parser.add_argument("--model_file", required=True, help="模型 joblib 路径")
    parser.add_argument("--output", default="global_results.csv", help="结果保存路径")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_save(args.data, args.model_file)
    else:
        load_and_test(args.data, args.model_file, args.output)
