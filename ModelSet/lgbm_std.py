import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import time
import argparse
import warnings
from imblearn.over_sampling import ADASYN
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

warnings.filterwarnings("ignore")


# --- 1. 辅助功能函数 (保持原源码逻辑) ---

def get_feature_columns(df):
    """提取特征列，剔除标签和业务无关的 ID/账号列"""
    drop_list = [
        'is_laundering', 'id', 'account_is_laundering_From',
        'account_is_laundering_To', 'src_bank_account_number',
        'dst_bank_account_number', 'Timestamp'
    ]
    actual_drop = [c for c in drop_list if c in df.columns]
    return [col for col in df.columns if col not in actual_drop]


# --- 2. 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：训练、执行数据增强并打包"""
    print(f"--- [LightGBM 训练模式] ---")
    df = pd.read_csv(train_csv)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['is_laundering']

    # --- 新增：处理缺失值 ---
    if X.isnull().values.any():
        print("检测到缺失值，正在执行填充（使用 0 填充）...")
        # 也可以使用 X.fillna(X.median(), inplace=True) 使用中位数填充
        X = X.fillna(0)
    # -----------------------

    # 计算正负样本比例
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    spw = neg_count / pos_count if pos_count > 0 else 1

    # 执行 ADASYN 数据增强
    print(f"执行 ADASYN 过采样... 原始分布: {np.bincount(y)}")
    # 注意：如果数据量极小或邻居太少，ADASYN 可能会报错，确保数据量足够
    adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    print(f"重采样后分布: {np.bincount(y_resampled)}")

    # 初始化模型 (保持原源码高维参数)
    model = lgb.LGBMClassifier(
        learning_rate=0.5,
        n_estimators=500,
        max_depth=8,
        num_leaves=255,
        importance_type='gain',
        scale_pos_weight=spw,
        random_state=42,
        verbosity=-1
    )

    print("开始训练 LightGBM 模型...")
    start_time = time.time()
    model.fit(X_resampled, y_resampled)
    print(f"训练完成，耗时: {time.time() - start_time:.2f}s")

    # 【关键】打包模型和特征元数据
    bundle = {
        'model': model,
        'feature_names': feature_cols,
        'model_name': 'LightGBM'
    }

    joblib.dump(bundle, model_save_path)
    print(f"模型及特征配置已保存至: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载打包文件，对齐特征并输出结果"""
    print(f"--- [LightGBM 测试模式] ---")
    if not os.path.exists(model_load_path):
        print("错误：找不到模型文件")
        return

    # 1. 加载组件包
    bundle = joblib.load(model_load_path)
    model = bundle['model']
    saved_features = bundle['feature_names']

    # 2. 读入测试集并进行特征对齐
    df_test = pd.read_csv(test_csv)
    X_test = pd.DataFrame()
    for col in saved_features:
        if col in df_test.columns:
            X_test[col] = df_test[col]
        else:
            # 训练时有但测试集缺失的列，自动补 0
            X_test[col] = 0

    # 确保列顺序与训练时严格一致
    X_test = X_test[saved_features]
    y_test = df_test['is_laundering']

    # 3. 预测
    y_proba = model.predict_proba(X_test)[:, 1]
    # 使用 0.35 作为默认阈值 (参考原源码逻辑)
    y_pred = (y_proba > 0.35).astype(int)

    # 4. 计算指标
    auc = float(roc_auc_score(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True)

    # 5. 写入统一结果模板 (CSV 追加模式)
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
    print(f"测试完成，结果已追加至: {result_csv}")


# --- 命令行交互逻辑 ---

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