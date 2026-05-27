import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import time


# --- 保留你原有的辅助功能函数 ---
def load_data(filepath, target_col='is_laundering'):
    """加载数据，排除非特征列"""
    data = pd.read_csv(filepath)
    # 保持与你源码一致的过滤逻辑
    drop_cols = ['Timestamp', 'From_Account', 'To_Account', target_col]
    features = data.drop(columns=[c for c in drop_cols if c in data.columns])
    labels = data[target_col]
    return features, labels


def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))


def find_best_threshold(y_true, y_proba, thresholds):
    best_threshold = 0.0
    best_f1 = 0.0
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold, best_f1


# --- 新增：统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：训练并保存 (包含动态阈值计算)"""
    features, labels = load_data(train_csv)

    # 划分训练集和验证集（用于早停和寻找阈值）
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    class_weights = get_class_weights(y_train)

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        early_stopping_rounds=100,
        class_weights=class_weights,
        verbose=100,
        random_seed=42
    )

    print(f"开始训练 CatBoost...")
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    # 计算最佳阈值
    y_proba_val = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 100)
    best_threshold, _ = find_best_threshold(y_val, y_proba_val, thresholds)

    # 【关键】将模型和阈值打包保存
    save_data = {
        'model': model,
        'threshold': best_threshold,
        'features': features.columns.tolist()  # 顺便保存特征列名，防止推理时特征错位
    }
    joblib.dump(save_data, model_save_path)
    print(f"模型及阈值({best_threshold:.4f})已保存至: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载并测试 (输出至统一模板)"""
    if not os.path.exists(model_load_path):
        print(f"错误：找不到模型文件 {model_load_path}")
        return

    # 加载打包数据
    checkpoint = joblib.load(model_load_path)
    model = checkpoint['model']
    threshold = checkpoint['threshold']

    X_test, y_test = load_data(test_csv)

    # 执行预测
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 计算评估指标
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = float(roc_auc_score(y_test, y_proba))

    # 按照统一模板准备数据
    result_entry = {
        "model_name": "CatBoost",
        "start_time": start_time or "N/A",
        "end_time": end_time or "N/A",
        "threshold": round(threshold, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc": round(auc, 4)
    }

    # 写入 CSV 模板
    res_df = pd.DataFrame([result_entry])
    header = not os.path.exists(result_csv)
    res_df.to_csv(result_csv, mode='a', index=False, header=header)
    print(f"测试完成，结果已追加至: {result_csv}")


# --- 调度入口 ---
if __name__ == "__main__":
    # 这里可以保留作为单脚本运行的逻辑，或者配合我之前给你的 main_runner.py 使用
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model_file", default="catboost_model.joblib")
    parser.add_argument("--output", default="results_summary.csv")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_save(args.data, args.model_file)
    else:
        load_and_test(args.data, args.model_file, args.output)