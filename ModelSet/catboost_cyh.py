import pandas as pd
import numpy as np
from catboost_cyh import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath):
    """加载并预处理数据"""
    data = pd.read_csv(filepath)
    features = data.drop(columns=['Timestamp', 'From_Account', 'To_Account', 'is_laundering'])
    labels = data['is_laundering']
    return features, labels

def get_class_weights(labels):
    """计算类别权重"""
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

def find_best_threshold(y_true, y_proba, thresholds):
    """寻找最佳F1分数对应的阈值"""
    best_threshold = 0.0
    best_f1 = 0.0
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        # zero_division=0 防止除以零警告
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold, best_f1

def main():
    # 1. 加载数据
    features, labels = load_data('./df_normalized.csv')

    # 2. 计算类别权重
    class_weights = get_class_weights(labels)

    # 3. 划分训练、验证和测试集 (新的、严谨的方法)
    # 首先划分训练验证集和测试集
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # 再次划分训练验证集为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    # 此时数据比例大致为：训练集 60%，验证集 20%，测试集 20%

    # 4. 初始化CatBoost模型
    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        # eval_metric='F1:use_weights=True',  # 此处可根据需要选择，早期停止一般用 Logloss 或 AUC
        early_stopping_rounds=100,
        class_weights=class_weights,
        verbose=100,
        random_seed=42
    )

    # 5. 训练模型，使用验证集进行早停
    print("开始训练模型...")
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val), # 使用验证集进行早停
        use_best_model=True
    )
    print("模型训练完成。")

    # 6. 在验证集上预测概率，寻找最佳阈值
    y_proba_val = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 100)
    best_threshold, best_f1 = find_best_threshold(y_val, y_proba_val, thresholds)
    print("\n在验证集上找到的最佳阈值: {:.4f}".format(best_threshold))
    print("验证集上的最佳 F1 分数: {:.4f}".format(best_f1))

    # 7. 在测试集上进行最终评估，使用验证集找到的最佳阈值
    print("\n开始在测试集上进行最终评估...")
    y_proba_test = model.predict_proba(X_test)[:, 1]
    adjusted_pred = (y_proba_test >= best_threshold).astype(int)

    # 8. 输出评估结果
    print("\n测试集上的分类报告 (使用验证集上的最佳阈值):")
    print(classification_report(y_test, adjusted_pred, target_names=['正常', '洗钱']))
    print("Precision: {:.4f}".format(precision_score(y_test, adjusted_pred, zero_division=0)))
    print("Recall: {:.4f}".format(recall_score(y_test, adjusted_pred, zero_division=0)))
    print("F1-score: {:.4f}".format(f1_score(y_test, adjusted_pred, zero_division=0)))

    # 9. 保存模型
    joblib.dump(model, 'model_01_cat.joblib')
    print("模型已保存为 model_01_cat.joblib")

if __name__ == "__main__":
    main()