import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
import pywt
import time
import os
import argparse
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# --- 1. 小波变换核心逻辑 (保留并增强鲁棒性) ---

def apply_wavelet_transform(X, selected_columns, wavelet='coif5', level=3):
    """应用小波变换到选定列并扩展特征空间"""
    X_expanded = X.copy()
    for col in selected_columns:
        if col not in X.columns:
            continue

        d = X[col].values
        # 执行小波分解
        coeffs = pywt.wavedec(d, wavelet=wavelet, mode='constant', level=level)

        for i, coeff in enumerate(coeffs):
            col_name = f'{col}_waveletA' if i == 0 else f'{col}_waveletD{i}'
            # 确保长度对齐（处理小波分解带来的边界对齐问题）
            coeff_flattened = coeff.flatten()
            if len(coeff_flattened) > len(X_expanded):
                coeff_flattened = coeff_flattened[:len(X_expanded)]
            elif len(coeff_flattened) < len(X_expanded):
                coeff_flattened = np.pad(coeff_flattened, (0, len(X_expanded) - len(coeff_flattened)), 'constant')

            X_expanded[col_name] = coeff_flattened

    return X_expanded


def get_base_features(df):
    """获取基础特征列名，排除标签和无关ID"""
    exclude = ['is_laundering', 'id', 'Unnamed: 0', 'Timestamp',
               'account_is_laundering_From', 'account_is_laundering_To']
    return [col for col in df.columns if col not in exclude]


# --- 2. 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：训练并打包模型与变换元数据"""
    print(f"--- [Wavelet XGB 训练模式] ---")
    df = pd.read_csv(train_csv)

    # 基础特征提取
    base_features = get_base_features(df)
    X = df[base_features]
    y = df['is_laundering']

    # 指定小波变换列
    wavelet_cols = ['counts_1H_From', 'counts_1H_To', '1H_frequency_From', '1H_frequency_To']
    wavelet_cols = [c for c in wavelet_cols if c in X.columns]

    # 应用小波变换扩展特征
    print(f"正在应用小波变换 (Wavelet: coif5, Level: 3)...")
    X_expanded = apply_wavelet_transform(X, wavelet_cols)
    final_feature_names = X_expanded.columns.tolist()

    # 划分训练/验证
    X_train, X_val, y_train, y_val = train_test_split(X_expanded, y, test_size=0.2, random_state=42)

    # 模型初始化
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.5,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )

    print("开始训练 XGBoost 模型...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 【关键】全组件打包
    bundle = {
        'model': model,
        'wavelet_cols': wavelet_cols,
        'final_features': final_feature_names,
        'base_features': base_features,
        'model_name': 'Wavelet_XGB'
    }

    joblib.dump(bundle, model_save_path)
    print(f"模型及特征元数据已保存至: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载打包文件，还原小波空间并测试"""
    print(f"--- [Wavelet XGB 测试模式] ---")
    if not os.path.exists(model_load_path):
        print("错误：找不到模型文件")
        return

    # 1. 加载组件包
    bundle = joblib.load(model_load_path)
    model = bundle['model']

    # 2. 读入测试集并进行基础对齐
    df_test = pd.read_csv(test_csv)
    # 确保测试集拥有训练时的所有基础列
    X_test_base = pd.DataFrame()
    for col in bundle['base_features']:
        X_test_base[col] = df_test[col] if col in df_test.columns else 0

    y_test = df_test['is_laundering']

    # 3. 还原小波变换空间
    print("正在还原小波特征空间...")
    X_test_expanded = apply_wavelet_transform(X_test_base, bundle['wavelet_cols'])

    # 确保最终特征列顺序与训练时完全一致
    X_test_final = X_test_expanded[bundle['final_features']]

    # 4. 预测
    y_proba = model.predict_proba(X_test_final)[:, 1]
    y_pred = model.predict(X_test_final)

    # 5. 计算指标
    auc = float(roc_auc_score(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True)

    # 6. 写入统一结果模板
    result_entry = {
        "model_name": bundle['model_name'],
        "start_time": start_time or "N/A",
        "end_time": end_time or "N/A",
        "threshold": model['threshold'] if 'threshold' in model else 0.5,
        "precision": round(report['1']['precision'], 4),
        "recall": round(report['1']['recall'], 4),
        "f1_score": round(report['1']['f1-score'], 4),
        "auc": round(auc, 4)
    }

    res_df = pd.DataFrame([result_entry])
    header = not os.path.exists(result_csv)
    res_df.to_csv(result_csv, mode='a', index=False, header=header)
    print(f"测试完成，结果已追加至: {result_csv}")


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