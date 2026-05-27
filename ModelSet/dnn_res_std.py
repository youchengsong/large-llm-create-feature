import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import time
import os
import argparse
import warnings
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, \
    precision_recall_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# --- 1. 模型架构定义 ---

class ResidualNet(nn.Module):
    """DNN 概率修正网络：输入特征 + GBDT 预测概率 -> 修正后的概率"""

    def __init__(self, input_dim):
        super(ResidualNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, p_gbdt):
        x_input = torch.cat([x, p_gbdt], dim=1)
        return self.net(x_input)


# --- 2. 辅助功能函数 ---

def load_and_clean(filepath, target_col='is_laundering'):
    """统一数据加载、清洗与缺失值处理"""
    df = pd.read_csv(filepath)
    drop_cols = ['id', 'account_is_laundering_From', 'account_is_laundering_To',
                 'src_bank_account_number', 'dst_bank_account_number', 'Timestamp']
    actual_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=actual_drop + [target_col], axis=1)
    y = df[target_col].astype(int)

    # 修复：处理 NaN，防止 GBDT/DNN 崩溃
    if X.isnull().values.any():
        X = X.fillna(X.median()).fillna(0)

    return X, y, X.columns.tolist()


def find_best_threshold(y_true, y_score):
    """根据 F1 分数在验证集寻找最佳阈值"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    return thresholds[np.argmax(f1_scores)]


# --- 3. 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：训练 Scaler -> LGBM -> DNN -> 打包"""
    print(f"--- [DNN-Res 训练模式] ---")
    X, y, feature_names = load_and_clean(train_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. 训练初级模型 (LightGBM)
    print("正在训练基础 LGBM 模型...")
    gbm = LGBMClassifier(n_estimators=1000, learning_rate=0.05, objective='binary', verbosity=-1)
    gbm.fit(X_train_scaled, y_train)

    # 获取 GBDT 概率作为 DNN 的额外输入
    p_train_gbm = gbm.predict_proba(X_train_scaled)[:, 1:]
    p_val_gbm = gbm.predict_proba(X_val_scaled)[:, 1:]

    # 4. 训练修正模型 (PyTorch DNN)
    print("正在训练 DNN 概率修正网络...")
    dnn = ResidualNet(X_train.shape[1]).to(device)
    optimizer = optim.Adam(dnn.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.BCELoss()

    # 转换为张量
    X_t = torch.FloatTensor(X_train_scaled).to(device)
    P_t = torch.FloatTensor(p_train_gbm).to(device)
    Y_t = torch.FloatTensor(y_train.values).view(-1, 1).to(device)

    dnn.train()
    for epoch in range(500):  # 示例缩短 epoch
        optimizer.zero_grad()
        outputs = dnn(X_t, P_t)
        loss = criterion(outputs, Y_t)
        loss.backward()
        optimizer.step()

    # 5. 验证集确定阈值
    dnn.eval()
    with torch.no_grad():
        v_outputs = dnn(torch.FloatTensor(X_val_scaled).to(device), torch.FloatTensor(p_val_gbm).to(device))
        best_threshold = find_best_threshold(y_val, v_outputs.cpu().numpy().flatten())

    # 6. 【核心】打包所有组件
    bundle = {
        'scaler': scaler,
        'gbm': gbm,
        'dnn_state_dict': dnn.state_dict(),
        'input_dim': X_train.shape[1],
        'base_features': feature_names,
        'threshold': best_threshold,
        'model_name': 'DNN_Res_Correction'
    }
    joblib.dump(bundle, model_save_path)
    print(f"全组件已保存至: {model_save_path}，最优阈值: {best_threshold:.4f}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载组件，还原残差链路预测，输出结果"""
    print(f"--- [DNN-Res 测试模式] ---")
    if not os.path.exists(model_load_path): return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = joblib.load(model_load_path)

    # 还原组件
    scaler = bundle['scaler']
    gbm = bundle['gbm']
    dnn = ResidualNet(bundle['input_dim']).to(device)
    dnn.load_state_dict(bundle['dnn_state_dict'])
    dnn.eval()

    # 加载与对齐
    X, y, _ = load_and_clean(test_csv)
    X_aligned = pd.DataFrame()
    for col in bundle['base_features']:
        X_aligned[col] = X[col] if col in X.columns else 0
    X_aligned = X_aligned[bundle['base_features']]

    # 推理链路
    X_scaled = scaler.transform(X_aligned)
    p_gbm = gbm.predict_proba(X_scaled)[:, 1:]

    with torch.no_grad():
        X_t = torch.FloatTensor(X_scaled).to(device)
        P_t = torch.FloatTensor(p_gbm).to(device)
        y_proba = dnn(X_t, P_t).cpu().numpy().flatten()

    threshold = bundle['threshold']
    y_pred = (y_proba >= threshold).astype(int)

    # 计算指标
    auc = roc_auc_score(y, y_proba)
    report = classification_report(y, y_pred, output_dict=True)

    # 写入统一结果模板
    result_entry = {
        "model_name": bundle['model_name'],
        "start_time": start_time or "N/A",
        "end_time": end_time or "N/A",
        "threshold": round(threshold, 4),
        "precision": round(report['1']['precision'], 4),
        "recall": round(report['1']['recall'], 4),
        "f1_score": round(report['1']['f1-score'], 4),
        "auc": round(auc, 4)
    }

    res_df = pd.DataFrame([result_entry])
    header = not os.path.exists(result_csv)
    res_df.to_csv(result_csv, mode='a', index=False, header=header)
    print(f"测试完成，结果已写入: {result_csv}")


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