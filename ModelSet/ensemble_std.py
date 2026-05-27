import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, \
    confusion_matrix
import joblib
import time
import os
import argparse
import warnings
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# --- 1. 辅助功能函数 (保留并优化原源码逻辑) ---

def get_feature_columns(df):
    """获取特征列名，排除标签和无关 ID"""
    exclude = ['is_laundering', 'id', 'Unnamed: 0', 'Timestamp',
               'account_is_laundering_From', 'account_is_laundering_To']
    return [col for col in df.columns if col in df.columns and col not in exclude]


def perform_shap_analysis_mini(model, X_data, feature_names, save_dir):
    """执行精简版 SHAP 分析"""
    os.makedirs(save_dir, exist_ok=True)
    # 集成模型通常使用 KernelExplainer 或对子模型使用 TreeExplainer
    # 这里我们对 X_data 进行采样以保证性能
    X_sample = X_data.sample(min(100, len(X_data)), random_state=42)

    print("正在生成集成模型 SHAP 解释图...")
    try:
        # 尝试使用通用的 KernelExplainer (适用于 Stacking)
        background = shap.sample(X_data, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)

        # 兼容二分类输出维度
        if isinstance(shap_values, list):
            shap_v = shap_values[1]
        else:
            shap_v = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_v, X_sample, feature_names=feature_names, show=False)
        plt.savefig(os.path.join(save_dir, 'ensemble_shap_summary.png'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP 分析跳过: {e}")


# --- 2. 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：训练 Stacking 集成模型并打包"""
    print(f"--- [Stacking Ensemble 训练模式] ---")
    df = pd.read_csv(train_csv)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['is_laundering']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 定义基础模型 (保持原源码配置)
    xgb_base = XGBClassifier(
        learning_rate=0.5, n_estimators=200, max_depth=8,
        n_jobs=-1, random_state=42, eval_metric='logloss'
    )
    dt_base = DecisionTreeClassifier(
        criterion='entropy', class_weight='balanced', random_state=42
    )

    # 创建 StackingClassifier
    ensemble_model = StackingClassifier(
        estimators=[('xgb', xgb_base), ('dt', dt_base)],
        final_estimator=LogisticRegression(random_state=42),
        n_jobs=-1
    )

    print("开始训练集成模型 (XGB + DT -> LR)...")
    start_time = time.time()
    ensemble_model.fit(X_train, y_train)
    print(f"训练完成，耗时: {time.time() - start_time:.2f}s")

    # 【关键】打包保存
    bundle = {
        'model': ensemble_model,
        'feature_names': feature_cols,
        'model_name': 'Stacking_XGB'
    }
    joblib.dump(bundle, model_save_path)
    print(f"集成模型已打包保存至: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载并测试"""
    print(f"--- [Stacking Ensemble 测试模式] ---")
    if not os.path.exists(model_load_path): return

    bundle = joblib.load(model_load_path)
    model = bundle['model']
    saved_features = bundle['feature_names']

    df_test = pd.read_csv(test_csv)

    # 特征对齐
    X_test = pd.DataFrame()
    for col in saved_features:
        X_test[col] = df_test[col] if col in df_test.columns else 0
    X_test = X_test[saved_features]
    y_test = df_test['is_laundering']

    # 预测
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 指标计算
    auc = float(roc_auc_score(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True)

    # 可选：SHAP 分析 (输出至模型同级目录下的 plots 文件夹)
    plot_dir = os.path.join(os.path.dirname(result_csv), 'ensemble_plots')
    perform_shap_analysis_mini(model, X_test, saved_features, plot_dir)

    # 整理结果至统一模板
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
    print(f"测试完成，结果已写入: {result_csv}")


# --- 命令行调度 ---

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