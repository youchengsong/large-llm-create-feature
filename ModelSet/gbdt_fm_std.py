import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import time
import os
import argparse
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from scipy.sparse import hstack

warnings.filterwarnings('ignore')


# --- 1. 模型架构定义 (必须保留以供 joblib 反序列化) ---

class FactorizationMachine(nn.Module):
    """
    因子分解机模型
    数学表达: $y = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle v_i, v_j \rangle x_i x_j$
    """

    def __init__(self, n_features, k=16):
        super(FactorizationMachine, self).__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(n_features))
        self.V = nn.Parameter(torch.randn(n_features, k) * 0.01)

    def forward(self, x):
        linear = self.w0 + torch.matmul(x, self.w)
        xv = torch.matmul(x, self.V)  # [B, k]
        xv2 = torch.matmul(x * x, self.V * self.V)  # [B, k]
        interaction = 0.5 * torch.sum(xv * xv - xv2, dim=1)
        return linear + interaction


class FMDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = np.asarray(y).astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_row = self.X[idx].toarray().squeeze(0)
        return torch.FloatTensor(x_row), torch.FloatTensor([self.y[idx]])


# --- 2. 辅助功能函数 ---

def load_and_clean(filepath, target_col='is_laundering'):
    """统一数据清洗"""
    df = pd.read_csv(filepath)
    drop_cols = ['id', 'account_is_laundering_From', 'account_is_laundering_To',
                 'src_bank_account_number', 'dst_bank_account_number', 'Timestamp']
    actual_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=actual_drop + [target_col], axis=1)
    y = df[target_col].astype(int)

    if X.isnull().values.any():
        print(f"[数据检查] 发现缺失值，正在执行中位数填充...")
        # 仅对数值型列进行填充
        X = X.fillna(X.median())
        # 如果仍有缺失（例如全空的列），则填充 0
        X = X.fillna(0)

    return X, y, X.columns.tolist()


# --- 3. 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：GBDT 转换 + FM 训练 + 全组件打包"""
    print(f"--- [GBDT-FM 训练模式] ---")
    X, y, feature_names = load_and_clean(train_csv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. GBDT 特征工程
    print("正在训练 GBDT 并执行特征离散化...")
    gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gbdt.fit(X, y)

    # 提取叶节点索引并 One-Hot 编码
    train_leaf = gbdt.apply(X)[:, :, 0]
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_fm = encoder.fit_transform(train_leaf)

    # 2. FM 模型训练 (PyTorch)
    batch_size = 128
    train_loader = DataLoader(FMDataset(X_fm, y), batch_size=batch_size, shuffle=True)

    n_features = X_fm.shape[1]
    fm_model = FactorizationMachine(n_features, k=16).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(fm_model.parameters(), lr=0.001)

    print(f"开始 FM 训练 (输入特征维度: {n_features})...")
    fm_model.train()
    for epoch in range(10):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = fm_model(x_batch)
            loss = criterion(logits.view(-1), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/10 | Loss: {total_loss / len(train_loader):.4f}")

    # 3. 确定最佳阈值 (使用 0.8*Recall + 0.2*Precision 准则)
    fm_model.eval()
    with torch.no_grad():
        all_logits = fm_model(torch.FloatTensor(X_fm.toarray()).to(device))
        all_probs = torch.sigmoid(all_logits).cpu().numpy()

    thresholds = np.linspace(0, 1, 100)
    best_score, best_threshold = -1, 0.5
    for t in thresholds:
        y_p = (all_probs >= t).astype(int)
        score = 0.8 * recall_score(y, y_p, zero_division=0) + 0.2 * precision_score(y, y_p, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = t

    # 4. 【核心】全组件序列化
    # 注意：PyTorch 模型只存 state_dict，以便灵活加载
    bundle = {
        'gbdt': gbdt,
        'encoder': encoder,
        'fm_state_dict': fm_model.state_dict(),
        'fm_n_features': n_features,
        'base_features': feature_names,
        'threshold': best_threshold,
        'model_name': 'GBDT_FM_Hybrid'
    }
    joblib.dump(bundle, model_save_path)
    print(f"全组件已打包保存至: {model_save_path}，最优阈值: {best_threshold:.4f}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载组件，还原流水线，输出统一结果"""
    print(f"--- [GBDT-FM 测试模式] ---")
    if not os.path.exists(model_load_path): return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle = joblib.load(model_load_path)

    # 1. 还原模型与环境
    gbdt = bundle['gbdt']
    encoder = bundle['encoder']
    fm_model = FactorizationMachine(bundle['fm_n_features'], k=16).to(device)
    fm_model.load_state_dict(bundle['fm_state_dict'])
    fm_model.eval()

    # 2. 数据处理与对齐
    X, y, _ = load_and_clean(test_csv)
    # 特征对齐逻辑
    X_aligned = pd.DataFrame()
    for col in bundle['base_features']:
        X_aligned[col] = X[col] if col in X.columns else 0
    X_aligned = X_aligned[bundle['base_features']]

    # 3. 还原流水线预测
    test_leaf = gbdt.apply(X_aligned)[:, :, 0]
    X_fm_test = encoder.transform(test_leaf)

    with torch.no_grad():
        # 分批次处理以防止内存溢出
        test_loader = DataLoader(FMDataset(X_fm_test, y), batch_size=512, shuffle=False)
        probs = []
        for x_b, _ in test_loader:
            logits = fm_model(x_b.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy())

    y_proba = np.array(probs)
    threshold = bundle['threshold']
    y_pred = (y_proba >= threshold).astype(int)

    # 4. 计算指标
    auc = roc_auc_score(y, y_proba)
    report = classification_report(y, y_pred, output_dict=True)

    # 5. 写入统一结果模板
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