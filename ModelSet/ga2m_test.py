import joblib
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, classification_report

def evaluate_model(pkl_file, test_csv):
    # 1. 加载模型
    print(f"正在读取模型权重: {pkl_file}")
    model = joblib.load(pkl_file)

    # 2. 加载测试数据
    print(f"正在加载测试数据: {test_csv}")
    df_test = pd.read_csv(test_csv)
    LABEL_COL = "is_laundering"
    X_test = df_test.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).fillna(0)
    y_test = df_test[LABEL_COL].astype(int)

    # 3. 预测
    print(f"开始预测...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 4. 输出结果
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n测试集 AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA2M 训练脚本")
    parser.add_argument('--dataset_path', type=str, required=True, help='训练集 csv 路径')
    parser.add_argument('--pkl_path', type=str, required=True, help='模型保存文件夹路径')
    args = parser.parse_args()

    evaluate_model(args.pkl_path, args.dataset_path)

    pass