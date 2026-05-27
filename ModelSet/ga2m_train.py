import os
import argparse
import joblib
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier


def train():
    parser = argparse.ArgumentParser(description="GA2M 训练脚本")
    parser.add_argument('--dataset_path', type=str, required=True, help='训练集 csv 路径')
    parser.add_argument('--pkl_path', type=str, required=True, help='模型保存文件夹路径')
    args = parser.parse_args()

    # 1. 准备路径
    model_save_path = args.pkl_path

    # 2. 加载数据
    print(f"正在加载数据: {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)

    # 假设标签列名为 'is_laundering'
    LABEL_COL = "is_laundering"
    X_train = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).fillna(0)
    y_train = df[LABEL_COL].astype(int)

    # 3. 初始化 EBM (根据你之前的配置)
    ebm = ExplainableBoostingClassifier(
        interactions=100,
        learning_rate=0.05,
        max_leaves=100,
        min_samples_leaf=40,
        outer_bags=20,
        n_jobs=-1,  # 训练时使用全部核心
        random_state=42
    )

    # 4. 训练
    print("开始训练 GA2M 模型...")
    ebm.fit(X_train, y_train)

    # 5. 保存模型为 pkl
    print(f"训练完成，正在保存模型至: {model_save_path}")
    joblib.dump(ebm, model_save_path)
    print("保存成功！")


if __name__ == "__main__":
    train()