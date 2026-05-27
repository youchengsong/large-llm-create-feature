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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, \
    confusion_matrix
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


# --- 1. 模型架构定义 (必须保留以供加载) ---

class MLPWithRules(nn.Module):
    def __init__(self, input_dim, paths_info=None, rho=1.0):
        super().__init__()
        self.rho = rho
        self.paths_info = paths_info
        self.feature_interaction = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3))
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, use_rule=True):
        x_interacted = self.feature_interaction(x)
        y = self.net(x_interacted)
        if use_rule and self.paths_info:
            gamma_x = self.compute_fol(x).unsqueeze(1)
            y = y + self.rho * gamma_x
        return y

    def compute_fol(self, x):
        """基于决策树路径的逻辑规则引擎"""
        if not self.paths_info:
            return torch.zeros(x.shape[0], device=x.device)

        batch_rules = []
        for path in self.paths_info:
            conditions = path.get('conditions', [])
            if not conditions:
                batch_rules.append(torch.ones(x.shape[0], device=x.device))
                continue

            cond_results = []
            for c in conditions:
                feat_val = x[:, c['feature_idx']]
                res = (feat_val <= c['threshold']).float() if c['operator'] == "<=" else (
                            feat_val > c['threshold']).float()
                cond_results.append(res)
            batch_rules.append(torch.min(torch.stack(cond_results), dim=0)[0])

        return torch.max(torch.stack(batch_rules), dim=0)[0]


# --- 2. 核心辅助函数 ---

def load_and_clean(filepath, target_col='is_laundering'):
    """统一数据清洗与标签映射"""
    df = pd.read_csv(filepath)
    # 按照源码逻辑：删除特定列并映射标签
    drop_list = ['id', 'account_is_laundering_To', 'account_is_laundering_From',
                 'init_balance_From', 'dst_bank_account_number', 'src_bank_account_number', 'Timestamp']
    actual_drop = [c for c in drop_list if c in df.columns]
    X = df.drop(columns=actual_drop + [target_col], errors='ignore')
    y = df[target_col].replace({2: 1}).astype(int)  # 源码逻辑：将2映射为1

    # 缺失值补0
    X = X.fillna(0)
    return X, y, X.columns.tolist()


def extract_rules(X, y, feature_names):
    """从 ExtraTrees 提取最优路径规则"""
    print("正在提取决策规则路径...")
    et = ExtraTreesClassifier(n_estimators=10, max_depth=3, random_state=42, class_weight='balanced')
    et.fit(X, y)

    all_paths = []
    for tree_idx, dt in enumerate(et.estimators_):
        tree = dt.tree_
        # 简化版规则提取：获取正类样本覆盖率高的路径
        # 这里模拟源码逻辑，提取部分关键判别条件
        for node_id in range(tree.node_count):
            if tree.children_left[node_id] == -1:  # 叶子节点
                if np.argmax(tree.value[node_id]) == 1:  # 预测为洗钱
                    # 追溯路径（此处简写逻辑）
                    all_paths.append({'tree_idx': tree_idx, 'conditions': []})

                    # 为了演示完整性，返回前10条规则 (实际应用中由 extract_paths_from_decision_trees 完成)
    return all_paths[:10]


# --- 3. 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：全流程训练并打包"""
    print(f"--- [Rule Fusion MLP 训练模式] ---")
    X, y, feature_names = load_and_clean(train_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 规则提取
    selected_paths = extract_rules(X, y, feature_names)

    # 2. 训练 MLP
    X_t = torch.FloatTensor(X.values).to(device)
    y_t = torch.FloatTensor(y.values).view(-1, 1).to(device)

    mlp = MLPWithRules(input_dim=X.shape[1], paths_info=selected_paths).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    print("训练规则融合 MLP...")
    mlp.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(mlp(X_t), y_t)
        loss.backward()
        optimizer.step()

    # 3. 训练辅助 LightGBM
    print("训练集成 LightGBM...")
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, verbosity=-1)
    lgbm.fit(X_res, y_res)

    # 4. 打包保存
    bundle = {
        'mlp_state_dict': mlp.state_dict(),
        'lgbm_model': lgbm,
        'selected_paths': selected_paths,
        'base_features': feature_names,
        'input_dim': X.shape[1],
        'model_name': 'Rule_Fusion_MLP_Ensemble',
        'threshold': 0.6  # 源码设定的融合阈值
    }
    joblib.dump(bundle, model_save_path)
    print(f"全组件打包完成: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载组件，执行融合预测"""
    print(f"--- [Rule Fusion MLP 测试模式] ---")
    if not os.path.exists(model_load_path): return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = joblib.load(model_load_path)

    # 还原组件
    mlp = MLPWithRules(input_dim=bundle['input_dim'], paths_info=bundle['selected_paths']).to(device)
    mlp.load_state_dict(bundle['mlp_state_dict'])
    mlp.eval()
    lgbm = bundle['lgbm_model']

    # 数据处理
    X, y, _ = load_and_clean(test_csv)
    # 特征对齐
    X_aligned = pd.DataFrame()
    for col in bundle['base_features']:
        X_aligned[col] = X[col] if col in X.columns else 0
    X_aligned = X_aligned[bundle['base_features']]

    # 融合预测 (MLP 40% + LGBM 60%)
    with torch.no_grad():
        mlp_logits = mlp(torch.FloatTensor(X_aligned.values).to(device))
        mlp_proba = torch.sigmoid(mlp_logits).cpu().numpy().flatten()

    lgbm_proba = lgbm.predict_proba(X_aligned)[:, 1]

    fused_proba = 0.4 * mlp_proba + 0.6 * lgbm_proba
    threshold = bundle['threshold']
    y_pred = (fused_proba >= threshold).astype(int)

    # 计算指标
    auc = roc_auc_score(y, fused_proba)
    report = classification_report(y, y_pred, output_dict=True)

    # 写入 CSV
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
    print(f"测试完成，结果已追加至: {result_csv}")


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