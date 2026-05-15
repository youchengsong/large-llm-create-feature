import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import platform
import joblib
import shap
import time
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

# --- 环境适配：解决 macOS/Linux/Windows 绘图字体问题 ---
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
elif platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux 默认
plt.rcParams['axes.unicode_minus'] = False


def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))


def find_best_threshold(y_true, y_proba, thresholds):
    best_threshold, best_f1 = 0.0, 0.0
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold, best_f1


def main():
    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description='CatBoost 训练与 SHAP 分析脚本')
    parser.add_argument('--train', '-tr', type=str, required=True, help='训练集 CSV 文件路径')
    parser.add_argument('--test', '-te', type=str, required=True, help='测试集 CSV 文件路径')
    parser.add_argument('--sample_shap', type=int, default=1000, help='SHAP 分析使用的样本量 (默认: 1000)')

    args = parser.parse_args()

    # 1. 初始化配置
    DROP_COLS = ['Timestamp', 'From_Account', 'To_Account', 'is_laundering']
    LABEL_COL = 'is_laundering'

    # 2. 加载数据
    if not os.path.exists(args.train) or not os.path.exists(args.test):
        print(f" 错误: 找不到输入文件。请检查路径: \n训练集: {args.train}\n测试集: {args.test}")
        return

    print(f" 正在加载数据...")
    df_train_full = pd.read_csv(args.train)
    df_test_original = pd.read_csv(args.test)

    X_trainval_full = df_train_full.drop(columns=DROP_COLS, errors='ignore')
    y_trainval_full = df_train_full[LABEL_COL]
    X_test = df_test_original.drop(columns=DROP_COLS, errors='ignore')
    y_test = df_test_original[LABEL_COL]

    print(f" 训练集: {len(df_train_full)} | 测试集: {len(df_test_original)}")

    # 3. 准备训练
    class_weights = get_class_weights(y_trainval_full)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval_full, y_trainval_full, test_size=0.25, random_state=42, stratify=y_trainval_full
    )

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        early_stopping_rounds=100,
        class_weights=class_weights,
        verbose=100,
        random_seed=42,
        loss_function='Logloss'
    )

    print("\n 开始训练 CatBoost...")
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    # 4. 寻找阈值
    y_proba_val = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f1_val = find_best_threshold(y_val, y_proba_val, np.linspace(0.01, 0.99, 100))
    print(f"\n 验证集最佳阈值: {best_threshold:.4f} (F1: {best_f1_val:.4f})")

    # 5. 测试集评估
    start_time = time.time()
    y_proba_test = model.predict_proba(X_test)[:, 1]
    adjusted_pred = (y_proba_test >= best_threshold).astype(int)
    end_time = time.time()

    total_prediction_time = end_time - start_time
    num_samples = len(X_test)

    # Calculate performance metrics
    throughput = num_samples / total_prediction_time if total_prediction_time > 0 else 0
    # Time per 1000 samples (in milliseconds)
    time_per_1k = (total_prediction_time / num_samples) * 1000 if num_samples > 0 else 0

    print("\n" + "=" * 50)
    print(" 测试集性能指标:")
    print(f"Precision: {precision_score(y_test, adjusted_pred, zero_division=0):.6f}")
    print(f"Recall:    {recall_score(y_test, adjusted_pred, zero_division=0):.6f}")
    print(f"F1-Score:  {f1_score(y_test, adjusted_pred, zero_division=0):.6f}")
    print(f"预测延迟 (每千样本耗时): {time_per_1k:.4f} 毫秒")
    print("-" * 50)
    print(classification_report(y_test, adjusted_pred, target_names=['正常', '洗钱'], zero_division=0))

    # 6. SHAP 分析
    print("\n 正在生成 SHAP 分析图 (样本量: {})...".format(args.sample_shap))
    shap_subset = X_test.sample(n=min(args.sample_shap, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_subset)

    # 保存 Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_subset, show=False)
    plt.title("SHAP 全局特征贡献分析")
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print(" 已保存 SHAP 图形至: shap_summary.png")

    # 7. 保存结果
    joblib.dump(model, 'catboost_model.joblib')
    print("\n 模型与预测逻辑已就绪。")


if __name__ == "__main__":
    main()

#  python catboost_dian.py --train ./split_data_20260119_233901/train_merchant.csv --test ./split_data_20260119_233901/test_merchant.csv
