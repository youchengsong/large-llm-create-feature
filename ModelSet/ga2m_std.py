import os
import time
import joblib
import pandas as pd
import numpy as np
import argparse
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score


# --- 数据预处理函数 (保持原源码逻辑) ---

def load_and_preprocess(filepath, target_col='is_laundering'):
    """加载数据并执行 GA2M 特有的预处理"""
    print(f"正在处理数据: {filepath}")
    df = pd.read_csv(filepath)

    # 按照原源码逻辑：仅保留数值型特征并填充0
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col].astype(int)
    return X, y


# --- 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：训练并保存"""
    print(f"--- [GA2M 训练模式] ---")
    X_train, y_train = load_and_preprocess(train_csv)

    # 初始化 EBM (保持你源码中的超参数配置)
    ebm = ExplainableBoostingClassifier(
        interactions=100,
        learning_rate=0.05,
        max_leaves=100,
        min_samples_leaf=40,
        outer_bags=20,
        n_jobs=1,
        random_state=42
    )

    print("开始训练 GA2M (EBM) 模型...")
    start_time = time.time()
    ebm.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"训练完成，耗时: {duration:.2f}s")

    # 保存模型
    joblib.dump(ebm, model_save_path)
    print(f"模型已保存至: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载并测试 (输出至统一模板)"""
    print(f"--- [GA2M 测试模式] ---")

    if not os.path.exists(model_load_path):
        print(f"错误：找不到模型文件 {model_load_path}")
        return

    # 加载模型
    model = joblib.load(model_load_path)
    X_test, y_test = load_and_preprocess(test_csv)

    # 预测
    print("开始执行推理...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 计算评估指标
    auc = float(roc_auc_score(y_test, y_prob))
    # 获取分类报告中的详细指标 (针对洗钱类 '1')
    report = classification_report(y_test, y_pred, output_dict=True)

    # 整理结果至统一模板
    result_entry = {
        "model_name": "GA2M",
        "start_time": start_time or "N/A",
        "end_time": end_time or "N/A",
        "threshold": model['threshold'] if 'threshold' in model else 0.5,
        "precision": round(report['1']['precision'], 4),
        "recall": round(report['1']['recall'], 4),
        "f1_score": round(report['1']['f1-score'], 4),
        "auc": round(auc, 4)
    }

    # 写入/追加到全局结果 CSV
    res_df = pd.DataFrame([result_entry])
    header = not os.path.exists(result_csv)
    res_df.to_csv(result_csv, mode='a', index=False, header=header)

    print(f"测试集 AUC: {auc:.4f}")
    print(f"测试结果已追加至: {result_csv}")


# --- 独立运行入口 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA2M 统一调度接口")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--data", required=True, help="输入 CSV 路径")
    parser.add_argument("--model_file", required=True, help="joblib 路径")
    parser.add_argument("--output", default="global_results.csv", help="结果保存路径")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_save(args.data, args.model_file)
    else:
        load_and_test(args.data, args.model_file, args.output)