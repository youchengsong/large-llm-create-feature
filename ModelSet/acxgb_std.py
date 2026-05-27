import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
import time
import os
import argparse
import warnings
from scipy.stats import entropy
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# 导入 SHAP 逻辑
import shap

warnings.filterwarnings("ignore")


# --- 辅助功能函数 (保留并优化原源码逻辑) ---

def get_feature_columns(df):
    """提取特征列，排除标签和无关 ID"""
    exclude = ['is_laundering', 'id', 'Unnamed: 0', 'Timestamp',
               'account_is_laundering_From', 'account_is_laundering_To']
    return [col for col in df.columns if col not in exclude]


def measure_uncertainty(probs):
    """熵作为不确定性度量"""
    return entropy(probs.T)


def perform_shap_analysis_mini(model, X_data, feature_names, save_dir):
    """执行精简版 SHAP 分析并保存图表"""
    os.makedirs(save_dir, exist_ok=True)
    # 采样以防内存溢出
    X_sample = X_data.sample(min(500, len(X_data)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 保存汇总图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(save_dir, 'acxgb_shap_summary.png'), bbox_inches='tight')
    plt.close()
    return np.abs(shap_values).mean(axis=0)  # 返回特征重要性数值


# --- 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：主动学习训练并打包"""
    print(f"--- [Active XGB 训练模式] ---")
    df = pd.read_csv(train_csv)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['is_laundering']

    # 1. 初始划分
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_pool, y_train, y_pool = train_test_split(X_temp, y_temp, train_size=0.1, random_state=42,
                                                        stratify=y_temp)

    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)

    # 2. 主动学习循环 (精简为 3 次迭代示范)
    n_iterations = 3
    samples_per_iter = 100

    for i in range(n_iterations):
        print(f"主动学习迭代 {i + 1}/{n_iterations}...")
        model.fit(X_train, y_train)

        if len(X_pool) == 0: break

        # 计算不确定性并采样
        probs = model.predict_proba(X_pool)
        uncertainties = measure_uncertainty(probs)
        idx = np.argsort(uncertainties)[-samples_per_iter:]  # 选熵最大的

        X_train = pd.concat([X_train, X_pool.iloc[idx]])
        y_train = pd.concat([y_train, y_pool.iloc[idx]])
        X_pool = X_pool.drop(X_pool.index[idx])
        y_pool = y_pool.drop(y_pool.index[idx])

    # 3. 最终训练
    model.fit(X_train, y_train)

    # 4. 打包保存
    bundle = {
        'model': model,
        'feature_names': feature_cols,
        'model_name': 'Active_XGBoost'
    }
    joblib.dump(bundle, model_save_path)
    print(f"主动学习模型已保存至: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载模型、执行预测及 SHAP 分析"""
    print(f"--- [Active XGB 测试模式] ---")
    if not os.path.exists(model_load_path): return

    bundle = joblib.load(model_load_path)
    model = bundle['model']
    saved_features = bundle['feature_names']

    df_test = pd.read_csv(test_csv)
    # 特征对齐逻辑
    X_test = pd.DataFrame()
    for col in saved_features:
        X_test[col] = df_test[col] if col in df_test.columns else 0
    X_test = X_test[saved_features]  # 保证顺序一致
    y_test = df_test['is_laundering']

    # 1. 预测
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 2. 计算指标
    auc = float(roc_auc_score(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True)

    # 3. 集成 SHAP 分析 (可选，输出到特定目录)
    shap_dir = os.path.join(os.path.dirname(result_csv), 'shap_plots')
    perform_shap_analysis_mini(model, X_test, saved_features, shap_dir)

    # 4. 写入统一结果模板
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

    res_df = pd.DataFrame([result_entry])
    header = not os.path.exists(result_csv)
    res_df.to_csv(result_csv, mode='a', index=False, header=header)
    print(f"测试完成，结果已存入 {result_csv}，SHAP图表位于 {shap_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--output", default="global_results.csv")
    args = parser.parse_args()

    if args.mode == "train":
        train_and_save(args.data, args.model_file)
    else:
        load_and_test(args.data, args.model_file, args.output)