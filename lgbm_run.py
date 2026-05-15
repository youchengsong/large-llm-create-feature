import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import warnings
import argparse
import joblib
import os
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

def analyze_feature_importance(
    model,
    feature_names,
    save_dir='results',
    special_keywords=None,
    top_k=50
):
    """
    对 LightGBM 模型进行特征重要性分析（基于 gain）

    参数:
    - model: 训练好的 LGBMClassifier
    - feature_names: X_train.columns
    - save_dir: 结果保存目录
    - special_keywords: 特殊特征关键词列表
    - top_k: 导出前 K 个重要特征
    """

    os.makedirs(save_dir, exist_ok=True)

    # ===============================
    # 1. 所有特征的重要性
    # ===============================
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': model.feature_importances_
    }).sort_values(
        by='importance_gain',
        ascending=False
    ).reset_index(drop=True)

    feature_importance.to_csv(
        os.path.join(save_dir, 'all_feature_importance.csv'),
        index=False
    )

    print(f"[INFO] 已保存全部特征重要性: {save_dir}/all_feature_importance.csv")

    # ===============================
    # 2. 特殊标识符特征的重要性
    # ===============================
    if special_keywords is not None and len(special_keywords) > 0:
        pattern = '|'.join(special_keywords)

        special_feature_importance = feature_importance[
            feature_importance['feature'].str.contains(pattern, regex=True)
        ]

        special_feature_importance.to_csv(
            os.path.join(save_dir, 'special_feature_importance.csv'),
            index=False
        )

        print(f"[INFO] 已保存特殊特征重要性: {save_dir}/special_feature_importance.csv")

        # Top-K
        top_special = special_feature_importance.head(top_k)
        top_special.to_csv(
            os.path.join(save_dir, f'top_{top_k}_special_feature_importance.csv'),
            index=False
        )

        print(f"[INFO] 已保存 Top-{top_k} 特殊特征重要性")

    return feature_importance

def aggregate_src_dst_importance(feature_importance, base_features):
    rows = []

    for base in base_features:
        src_feat = f"{base}_src"
        dst_feat = f"{base}_dst"

        src_imp = feature_importance.loc[
            feature_importance['feature'] == src_feat,
            'importance_gain'
        ].sum()

        dst_imp = feature_importance.loc[
            feature_importance['feature'] == dst_feat,
            'importance_gain'
        ].sum()

        rows.append({
            'indicator': base,
            'src_importance': src_imp,
            'dst_importance': dst_imp,
            'total_importance': src_imp + dst_imp,
            'src_ratio': src_imp / (src_imp + dst_imp + 1e-9),
            'dst_ratio': dst_imp / (src_imp + dst_imp + 1e-9)
        })

    return pd.DataFrame(rows).sort_values(
        by='total_importance',
        ascending=False
    )

def train_lightgbm(train_data, val_data=None, random_state=52, model_save_path=None):
    print("开始LGBM训练")
    
    process = psutil.Process(os.getpid())
    print(f"训练前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # 准备训练数据
    X_train = train_data.drop(['is_laundering','account_is_laundering_From','account_is_laundering_To'], axis=1)
    y_train = train_data['is_laundering']
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 准备验证数据 (如果有)
    if val_data is not None:
        X_val = val_data.drop(['is_laundering','account_is_laundering_From','account_is_laundering_To'], axis=1)
        y_val = val_data['is_laundering']
        eval_set = [(X_val, y_val)]
    else:
        eval_set = None
    
    # 数据增强处理
    print(f"训练集原始分布: {np.bincount(y_train)}")
    # smote = SMOTE(
    #     sampling_strategy=0.3, random_state=random_state,
    # )
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

    print(f"训练集重采样后样本分布: {np.bincount(y_train_resampled)}")

    # X_train_resampled, y_train_resampled = X_train, y_train
    
    # 特征筛选
    model_tmp = lgb.LGBMClassifier(n_estimators=100, random_state=random_state)
    model_tmp.fit(X_train_resampled, y_train_resampled, callbacks=[log_evaluation(0)])
    importances = pd.Series(model_tmp.feature_importances_, index=X_train.columns)

    # 训练最终模型
    model = lgb.LGBMClassifier(
        learning_rate=0.5,
        n_estimators=500,
        max_depth=8,
        min_child_weight=1,
        subsample=1,
        colsample_bytree=0.9,
        reg_alpha=0.01,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        num_leaves=255,
        boosting_type='gbdt',
        random_state=42,
        feature_fraction=0.8,
        importance_type='gain',
        verbosity=-1
    )
    
    if eval_set:
        callbacks = [early_stopping(stopping_rounds=20), log_evaluation((100))]
        model.fit(X_train_resampled, y_train_resampled,
                  eval_set=eval_set,
                  callbacks=callbacks)
    else:
        callbacks = [log_evaluation((100))]
        model.fit(X_train_resampled, y_train_resampled,
                  callbacks=callbacks)
    
    special_keywords = [
        'S_'
    ]

    analyze_feature_importance(
        model=model,
        feature_names=X_train.columns,
        save_dir='results/feature_importance',
        special_keywords=special_keywords,
        top_k=50
    )
    
    # 保存模型
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
        joblib.dump(model, model_save_path)
        print(f"\n模型已保存到: {model_save_path}")
    
    # 训练集表现
    y_pred_train = model.predict(X_train_resampled)
    train_precision = precision_score(y_train_resampled, y_pred_train)
    train_f1 = f1_score(y_train_resampled, y_pred_train)
    train_recall = recall_score(y_train_resampled, y_pred_train)
    
    print("========== 训练集表现 ==========")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall:    {train_recall:.4f}")
    print(f"F1 Score:  {train_f1:.4f}")
    
    print(f"训练后内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    return {
        'model': model,
        # 'selected_features': selected_features,
        # 'smote': smote_enn,
        'train_metrics': {
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1
        }
    }

def test_lightgbm(test_data, model_path, threshold=0.35):
    print("开始模型测试")
    
    process = psutil.Process(os.getpid())
    print(f"测试前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # 加载模型和相关对象
    model = load_lightgbm_model(model_path)
    
    # 准备测试数据
    X_test = test_data.drop(['is_laundering','account_is_laundering_From','account_is_laundering_To'], axis=1)
    y_test = test_data['is_laundering']
    
    # 预测
    start_time = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob > threshold).astype(int)
    end_time = time.time()
    
    # 计算推理速度
    run_time = end_time - start_time
    num_dtest = len(X_test)
    per_sample_time = run_time / num_dtest if num_dtest > 0 else 0
    
    # 计算评估指标
    cm = confusion_matrix(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print("\n========== 测试集表现 ==========")
    print("混淆矩阵:\n", cm)
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"推理速度: {per_sample_time:.6f} 秒/条")
    print(f"测试后内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    return {
        'confusion_matrix': cm,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'accuracy': test_acc,
        'inference_time_per_sample': per_sample_time,
        'predictions': y_pred_test,
        'probabilities': y_prob
    }

def load_lightgbm_model(model_path):
    model = joblib.load(model_path)
    return model

def find_custom_optimal_threshold(y_true, y_prob, min_recall=0.9, min_precision=0.8):
    """
    根据业务需求寻找最优阈值
    
    参数:
    y_true: 真实标签
    y_prob: 预测概率
    min_recall: 最小召回率要求
    min_precision: 最小精确率要求
    
    返回:
    optimal_threshold: 最优阈值
    """
    thresholds = np.linspace(0.05, 0.95, 100)
    candidate_thresholds = []
    
    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # 检查是否满足最低要求
        if recall >= min_recall and precision >= min_precision:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            candidate_thresholds.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
    
    if candidate_thresholds:
        # 在满足要求的候选中选择F1最高的
        candidates_df = pd.DataFrame(candidate_thresholds)
        optimal_idx = candidates_df['f1'].idxmax()
        optimal_threshold = candidates_df.loc[optimal_idx, 'threshold']
        
        print(f"满足要求的阈值找到! 最优阈值: {optimal_threshold:.3f}")
        print(f"精确率: {candidates_df.loc[optimal_idx, 'precision']:.3f}, "
              f"召回率: {candidates_df.loc[optimal_idx, 'recall']:.3f}")
    else:
        # 如果没有满足要求的，选择最接近的
        print("没有完全满足要求的阈值，选择最接近的阈值")
        optimal_threshold = find_balanced_threshold(y_true, y_prob)
    
    return optimal_threshold

def find_balanced_threshold(y_true, y_prob):
    """寻找平衡精确率和召回率的阈值"""
    thresholds = np.linspace(0.1, 0.9, 50)
    best_balance = float('inf')
    optimal_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob > threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # 计算精确率和召回率的平衡度（差的绝对值）
        balance_score = abs(precision - recall)
        
        if balance_score < best_balance:
            best_balance = balance_score
            optimal_threshold = threshold
    
    print(f"平衡阈值: {optimal_threshold:.3f}")
    return optimal_threshold

def handle_nan_values(df, strategy_dict=None, fill_value_dict=None):
    """
    对 DataFrame 的 NaN 数据进行灵活处理
    
    参数:
    ----------
    df : pandas.DataFrame
        输入数据集
    strategy_dict : dict
        指定每列的缺失值处理策略，例如:
        {
            "col1": "mean",
            "col2": "median",
            "col3": "mode",
            "col4": "drop",
            "col5": "constant"
        }
    fill_value_dict : dict
        当 strategy="constant" 时，为每列指定填充值
        
    返回:
    ----------
    df_clean : pandas.DataFrame
        处理后的 DataFrame
    """
    
    df_clean = df.copy()
    
    # 如果用户没有指定策略，则默认全部使用 mean
    if strategy_dict is None:
        strategy_dict = {col: "mean" for col in df_clean.columns}

    for col, strategy in strategy_dict.items():

        if df_clean[col].isna().sum() == 0:
            continue  # 跳过无 NaN 的列

        if strategy == "mean":
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

        elif strategy == "median":
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        elif strategy == "mode":
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        elif strategy == "drop":
            df_clean = df_clean.dropna(subset=[col])

        elif strategy == "constant":
            if fill_value_dict and col in fill_value_dict:
                df_clean[col] = df_clean[col].fillna(fill_value_dict[col])
            else:
                raise ValueError(f"列 {col} 选择了 constant 策略，但未提供填充值")

        else:
            raise ValueError(f"未知缺失值处理策略：{strategy}")

    return df_clean

def main():
    parser = argparse.ArgumentParser(description='LGBM train and test')
    parser.add_argument('--train_path', type=str, required=True, help='训练集 CSV 文件路径')
    parser.add_argument('--test_path', type=str, required=True, help='测试集 CSV 文件路径')

    args = parser.parse_args()

    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"训练集文件不存在：{args.train_path}")
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"测试集文件不存在：{args.test_path}")
    
    # data_dir = os.path.join(os.getcwd(), 'Data/split_data_20260119_233901')
    # train_path = os.path.join(data_dir, 'train_merchant.csv')
    # test_path = os.path.join(data_dir, 'test_merchant.csv')

    # 加载数据
    train_data = pd.read_csv(args.train_path)
    train_data = train_data.drop(columns=['id', 'src_bank_account_number','dst_bank_account_number'])
    test_data = pd.read_csv(args.test_path)
    test_data = test_data.drop(columns=['id', 'src_bank_account_number','dst_bank_account_number'])

    # 训练模型
    train_results = train_lightgbm(
        train_data=train_data,
        model_save_path='models/lgbm_model_merchant_0120.pkl'
    )
    
    # 测试模型
    test_results = test_lightgbm(
        test_data=test_data,
        model_path='models/lgbm_model_merchant_0120.pkl'
    )
    
if __name__ == "__main__":
    main()
    