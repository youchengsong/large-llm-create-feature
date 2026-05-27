import pandas as pd
import numpy as np
import joblib
import time
import os
import argparse
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, \
    classification_report
from toad.transform import Combiner, WOETransformer
from toad.selection import drop_iv

warnings.filterwarnings('ignore')


# --- 辅助函数：保持原有的数据清洗逻辑 ---

def load_and_clean(filepath, target_col='is_laundering'):
    """加载数据并剔除无关列"""
    df = pd.read_csv(filepath)
    # 结合了你 train 和 test 脚本中的 drop 逻辑
    drop_columns = [
        'id', 'account_is_laundering_From', 'account_is_laundering_To',
        'src_bank_account_number', 'dst_bank_account_number', 'Timestamp'
    ]
    # 动态剔除存在的列
    actual_drop = [c for c in drop_columns if c in df.columns]
    X = df.drop(columns=actual_drop + [target_col], axis=1)
    y = df[target_col]
    return X, y


def find_best_threshold(y_true, y_score):
    """根据 F1 分数寻找最佳阈值"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold


# --- 统一接口函数 ---

def train_and_save(train_csv, model_save_path):
    """阶段一：训练并打包保存所有中间件"""
    print(f"--- [ScoreCard 训练模式] ---")
    X, y = load_and_clean(train_csv)

    # 1. 标准化
    scaler = StandardScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 2. 特征分箱 (Chi-merge)
    print("正在进行特征分箱...")
    combiner = Combiner()
    combiner.fit(X_norm, y, method='chi')
    X_binned = combiner.transform(X_norm)

    # 3. WOE 编码
    print("正在进行 WOE 编码...")
    woe_trans = WOETransformer()
    X_woe = woe_trans.fit_transform(X_binned, y)

    # 4. 特征筛选 (IV > 0.05)
    print("正在基于 IV 进行特征筛选...")
    data_iv = drop_iv(X_woe.assign(target=y), target='target', threshold=0.05)
    selected_vars = [c for c in data_iv.columns if c != 'target']
    X_final = X_woe[selected_vars]

    # 5. 模型训练 (逻辑回归)
    print("开始训练逻辑回归模型...")
    clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    clf.fit(X_final, y)

    # 6. 确定最佳阈值 (使用训练集预测概率确定，防止测试集数据泄露)
    y_proba = clf.predict_proba(X_final)[:, 1]
    best_threshold = find_best_threshold(y, y_proba)

    # 【核心修改】打包所有状态
    model_bundle = {
        'scaler': scaler,
        'combiner': combiner,
        'woe_trans': woe_trans,
        'selected_vars': selected_vars,
        'clf': clf,
        'threshold': best_threshold,
        'model_name': 'ScoreCard_Toad'
    }

    joblib.dump(model_bundle, model_save_path)
    print(f"评分卡模型全组件已打包保存至: {model_save_path}")


def load_and_test(test_csv, model_load_path, result_csv, start_time=None, end_time=None):
    """阶段二：加载打包文件并在测试集上执行全流程变换"""
    print(f"--- [ScoreCard 测试模式] ---")

    if not os.path.exists(model_load_path):
        print(f"错误：找不到模型文件 {model_load_path}")
        return

    # 1. 加载全组件包
    bundle = joblib.load(model_load_path)
    X_test, y_test = load_and_clean(test_csv)

    # 2. 依次应用变换
    X_norm = pd.DataFrame(bundle['scaler'].transform(X_test), columns=X_test.columns)
    X_binned = bundle['combiner'].transform(X_norm)
    X_woe = bundle['woe_trans'].transform(X_binned)
    X_final = X_woe[bundle['selected_vars']]

    # 3. 预测
    y_proba = bundle['clf'].predict_proba(X_final)[:, 1]
    # 使用训练阶段确定的最佳阈值
    y_pred = (y_proba >= bundle['threshold']).astype(int)

    # 4. 计算指标
    auc = float(roc_auc_score(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True)

    # 整理结果至统一模板
    result_entry = {
        "model_name": bundle['model_name'],
        "start_time": start_time or "N/A",
        "end_time": end_time or "N/A",
        "threshold": bundle['threshold'] if 'threshold' in bundle else None,
        "precision": round(report['1']['precision'], 4),
        "recall": round(report['1']['recall'], 4),
        "f1_score": round(report['1']['f1-score'], 4),
        "auc": round(auc, 4)
    }

    # 5. 追加到全局结果 CSV
    res_df = pd.DataFrame([result_entry])
    header = not os.path.exists(result_csv)
    res_df.to_csv(result_csv, mode='a', index=False, header=header)
    print(f"测试完成，结果已追加至: {result_csv}")

# --- 独立运行入口 ---

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