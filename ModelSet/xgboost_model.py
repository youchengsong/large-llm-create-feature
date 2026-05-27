import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, matthews_corrcoef, confusion_matrix, roc_curve, auc
import math
from sklearn.utils import resample
from xgboost import plot_importance
from scipy.stats import entropy
import psutil
import pywt
import os
import warnings
warnings.filterwarnings("ignore")

# 导入SHAP库
import shap
shap.initjs()

pd.set_option('display.max_columns', None)
import joblib

def apply_wavelet_transform(X, selected_columns, wavelet='coif5', level=3):
    """
    应用小波变换到选定列
    
    参数:
    - X: 输入数据框
    - selected_columns: 要应用小波变换的列名列表
    - wavelet: 小波类型，默认'coif5'
    - level: 分解层级，默认3
    
    返回:
    - 扩展后的数据框
    """
    X_expanded = X.copy()
    for col in selected_columns:
        if col not in X.columns:
            print(f"警告: 列 {col} 不在数据中，跳过")
            continue
            
        d = pd.DataFrame(X[col])
        coeffs = pywt.wavedec(d, wavelet=wavelet, mode='constant', level=level)
        
        for i, coeff in enumerate(coeffs):
            # 确保coeff是二维数组
            if coeff.ndim == 1:
                coeff = coeff.reshape(-1, 1)
            
            col_name = f'{col}_waveletA' if i == 0 else f'{col}_waveletD{i}'
            a = pd.DataFrame(coeff[:, 0], columns=[col_name])
            
            # 确保行数一致
            if len(a) != len(X_expanded):
                # 如果长度不匹配，使用填充或截断
                if len(a) > len(X_expanded):
                    a = a.iloc[:len(X_expanded)]
                else:
                    # 填充NaN
                    a = pd.concat([a, pd.DataFrame(np.nan, index=range(len(X_expanded)-len(a)), columns=[col_name])], 
                                  ignore_index=True)
            
            X_expanded = pd.concat([X_expanded, a], axis=1)
    
    return X_expanded

def get_feature_columns(df):
    """
    获取特征列名列表，排除非特征列
    
    参数:
    - df: 数据框
    
    返回:
    - 特征列名列表
    """
    # 需要排除的列
    columns_to_exclude = ['is_laundering', 'id', 'Unnamed: 0']
    
    # 可能存在的账户标签列
    possible_exclude = ['account_is_laundering_From', 'account_is_laundering_To']
    
    # 构建排除列表
    exclude_list = []
    for col in columns_to_exclude + possible_exclude:
        if col in df.columns:
            exclude_list.append(col)
    
    # 获取特征列
    feature_columns = [col for col in df.columns if col not in exclude_list]
    
    return feature_columns

def perform_shap_analysis(model, X_data, feature_names, sample_size=1000, save_dir='/home/aolc/ECNU/model-ensemble/Model/wavxgboost/shap_results'):
    """
    执行SHAP分析并生成可视化图表
    
    参数:
    - model: 训练好的模型
    - X_data: 特征数据
    - feature_names: 特征名称列表
    - sample_size: 用于SHAP分析的样本数量（为节省时间可以采样）
    - save_dir: 保存结果的目录
    
    返回:
    - SHAP分析结果字典或None
    """
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    print("开始SHAP分析...")
    
    # 采样以减少计算时间（如果数据量太大）
    if len(X_data) > sample_size:
        print(f"数据量较大，随机采样 {sample_size} 个样本进行SHAP分析")
        sample_indices = np.random.choice(len(X_data), sample_size, replace=False)
        X_sample = X_data.iloc[sample_indices].copy()
    else:
        X_sample = X_data.copy()
    
    try:
        # 根据模型类型选择合适的SHAP解释器
        print(f"模型类型: {type(model).__name__}")
        
        # 对于XGBoost模型，使用TreeExplainer
        if isinstance(model, XGBClassifier) or hasattr(model, 'get_booster'):
            print("使用TreeExplainer (XGBoost模型)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # 对于二分类XGBoost，shap_values可能是单个数组或列表
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 取正类的SHAP值
                
        else:
            # 对于其他模型，尝试使用不同的解释器
            print("尝试使用KernelExplainer (通用模型)")
            try:
                # 创建背景数据集
                background = shap.sample(X_sample, min(100, len(X_sample)))
                
                # 定义模型预测函数
                if hasattr(model, 'predict_proba'):
                    predict_func = lambda x: model.predict_proba(x)[:, 1]
                else:
                    predict_func = model.predict
                    
                explainer = shap.KernelExplainer(predict_func, background)
                shap_values = explainer.shap_values(X_sample)
                
                # 如果返回的是列表，取第一个
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                    
            except Exception as e:
                print(f"KernelExplainer失败: {e}")
                return None
                
    except Exception as e:
        print(f"SHAP解释器创建失败: {e}")
        return None
    
    # 检查shap_values的形状
    print(f"SHAP值形状: {shap_values.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 确保特征名称数量与SHAP值维度匹配
    if shap_values.shape[1] != len(feature_names):
        print(f"警告: 特征数量不匹配，SHAP值有{shap_values.shape[1]}个特征，但特征名有{len(feature_names)}个")
        # 截断特征名称以匹配SHAP值维度
        if shap_values.shape[1] < len(feature_names):
            feature_names = feature_names[:shap_values.shape[1]]
        else:
            # 如果SHAP值维度更大，创建占位符特征名
            feature_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), shap_values.shape[1])]
    
    print("SHAP计算完成，开始生成可视化...")
    
    # 1. 汇总图（蜂群图）
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'shap_summary_plot.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✓ 保存汇总图到: {os.path.join(save_dir, 'shap_summary_plot.png')}")
    except Exception as e:
        print(f"生成汇总图失败: {e}")
    
    # 2. 条形图（特征重要性）
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_bar_plot.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'shap_bar_plot.pdf'), bbox_inches='tight')
        plt.close()
        print(f"✓ 保存条形图到: {os.path.join(save_dir, 'shap_bar_plot.png')}")
    except Exception as e:
        print(f"生成条形图失败: {e}")
    
    # 3. 计算特征重要性（平均绝对SHAP值）
    try:
        shap_df = pd.DataFrame({
            'feature': feature_names[:shap_values.shape[1]],
            'shap_importance': np.abs(shap_values).mean(axis=0)
        })
        shap_df = shap_df.sort_values('shap_importance', ascending=False)
        
        # 保存特征重要性数据
        shap_df.to_csv(os.path.join(save_dir, 'shap_feature_importance.csv'), index=False)
        print(f"✓ 保存特征重要性数据到: {os.path.join(save_dir, 'shap_feature_importance.csv')}")
        
    except Exception as e:
        print(f"计算特征重要性失败: {e}")
        shap_df = pd.DataFrame()
    
    # 打印最重要的特征
    if not shap_df.empty:
        print("\n" + "="*80)
        print("SHAP特征重要性排名 (前20):")
        print("="*80)
        for i, row in shap_df.head(20).iterrows():
            print(f"{i+1:2d}. {row['feature']:40s} | 平均|SHAP|: {row['shap_importance']:.6f}")
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_importance': shap_df,
        'save_dir': save_dir
    }

def create_shap_report(shap_results, true_labels, predictions, prediction_proba=None):
    """
    创建SHAP分析报告
    
    参数:
    - shap_results: perform_shap_analysis返回的结果
    - true_labels: 真实标签
    - predictions: 预测结果
    - prediction_proba: 预测概率（可选）
    """
    if shap_results is None:
        print("警告: 没有SHAP结果可生成报告")
        return
    
    save_dir = shap_results.get('save_dir', 'shap_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建报告文件
    report_path = os.path.join(save_dir, 'shap_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SHAP分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. 模型性能概览\n")
        f.write("-"*40 + "\n")
        f.write(f"总样本数: {len(true_labels)}\n")
        f.write(f"正类样本数: {sum(true_labels == 1)}\n")
        f.write(f"负类样本数: {sum(true_labels == 0)}\n")
        f.write(f"准确率: {accuracy_score(true_labels, predictions):.4f}\n")
        f.write(f"F1分数: {f1_score(true_labels, predictions):.4f}\n")
        f.write(f"精确率: {precision_score(true_labels, predictions):.4f}\n")
        f.write(f"召回率: {recall_score(true_labels, predictions):.4f}\n\n")
        
        f.write("2. SHAP特征重要性 (前10)\n")
        f.write("-"*40 + "\n")
        
        if not shap_results['feature_importance'].empty:
            top_features = shap_results['feature_importance'].head(10)
            for i, row in top_features.iterrows():
                f.write(f"{i+1:2d}. {row['feature']:40s} | 平均|SHAP|: {row['shap_importance']:.6f}\n")
        else:
            f.write("无特征重要性数据\n")
        
        f.write("\n3. 生成的可视化文件\n")
        f.write("-"*40 + "\n")
        
        # 列出生成的文件
        if os.path.exists(save_dir):
            files = os.listdir(save_dir)
            image_files = [f for f in files if f.endswith(('.png', '.pdf', '.jpg'))]
            for img_file in image_files:
                f.write(f"- {img_file}\n")
        
        f.write("\n4. 建议\n")
        f.write("-"*40 + "\n")
        f.write("1. 查看shap_summary_plot.png了解特征的整体影响\n")
        f.write("2. 查看shap_bar_plot.png了解特征重要性排序\n")
        f.write("3. 分析特征重要性CSV文件进行深入分析\n")
        f.write("4. 根据重要特征优化特征工程\n")
    
    print(f"✓ SHAP分析报告已保存到: {report_path}")

def train(df_normalized, save_modelpath, save_feature_names_path=None, test_size=0.2, random_state=42):
    """
    训练模型
    
    参数:
    - df_normalized: 归一化后的数据框
    - save_modelpath: 模型保存路径
    - save_feature_names_path: 特征名称保存路径（可选）
    - test_size: 测试集比例
    - random_state: 随机种子
    
    返回:
    - 训练F1分数和特征列名
    """
    print("模型正在训练。")
    
    # 获取特征列名（排除标签列和非特征列）
    feature_columns = get_feature_columns(df_normalized)
    print(f"特征数量: {len(feature_columns)}")
    print(f"前10个特征: {feature_columns[:10]}")
    
    # 提取特征和标签
    X = df_normalized[feature_columns].copy()
    y = df_normalized['is_laundering']
    
    # 保存特征列名（用于测试时对齐）
    if save_feature_names_path:
        feature_names_df = pd.DataFrame({'feature_name': feature_columns})
        feature_names_df.to_csv(save_feature_names_path, index=False)
        print(f"特征名称已保存到: {save_feature_names_path}")
    
    # 选择进行小波变换的列
    selected_columns = ['counts_1H_From','counts_1H_To','1H_frequency_From','1H_frequency_To']
    
    # 确保选中的列存在
    selected_columns = [col for col in selected_columns if col in X.columns]
    if not selected_columns:
        print("警告: 没有找到小波变换的列，跳过小波变换")
        X_expanded = X
    else:
        # 对训练数据应用小波变换
        X_expanded = apply_wavelet_transform(X, selected_columns)
    
    # 获取最终的特征列名（小波变换后）
    final_feature_columns = X_expanded.columns.tolist()
    print(f"小波变换后特征数量: {len(final_feature_columns)}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_expanded, y, test_size=test_size, random_state=random_state)
            
    # 初始化模型
    ensemble_model = XGBClassifier(
        learning_rate=0.5, 
        n_estimators=200, 
        max_depth=8, 
        min_child_weight=1,
        subsample=1, 
        colsample_bytree=0.9, 
        gamma=0.01, 
        reg_alpha=0.01, 
        reg_lambda=0.1, 
        scale_pos_weight=1,
        random_state=random_state,
        n_jobs=-1  # 使用所有CPU核心
    )
    
    # 训练模型
    ensemble_model.fit(X_train, y_train)
    print("模型训练已完成。")
    
    # 保存模型
    joblib.dump(ensemble_model, save_modelpath)
    print(f"模型已保存到: {save_modelpath}")
    
    # 保存最终特征列名（小波变换后）
    if save_feature_names_path:
        final_feature_names_path = save_feature_names_path.replace('.csv', '_final.csv')
        final_feature_names_df = pd.DataFrame({'feature_name': final_feature_columns})
        final_feature_names_df.to_csv(final_feature_names_path, index=False)
        print(f"最终特征名称已保存到: {final_feature_names_path}")
    
    # 训练集预测
    y_pred_train = ensemble_model.predict(X_train)
    y_pred_train = np.round(y_pred_train)
    
    # 计算训练集指标
    trainprecision = precision_score(y_train, y_pred_train, zero_division=0)
    trainf1 = f1_score(y_train, y_pred_train, zero_division=0)
    trainrecall = recall_score(y_train, y_pred_train, zero_division=0)
    
    print('训练精确率: ', trainprecision)
    print('训练召回率: ', trainrecall)
    print('训练F1分数: ', trainf1)
    
    # 返回训练F1分数和特征列名
    return trainf1, final_feature_columns

def align_features(test_data, feature_columns):
    """
    对齐测试数据的特征，确保与训练数据一致
    
    参数:
    - test_data: 测试数据
    - feature_columns: 训练时的特征列名列表
    
    返回:
    - 对齐后的测试数据
    """
    aligned_data = pd.DataFrame()
    
    # 确保所有训练时的特征都在测试数据中
    for feature in feature_columns:
        if feature in test_data.columns:
            aligned_data[feature] = test_data[feature]
        else:
            # 如果测试数据缺少某个特征，用0填充
            print(f"警告: 测试数据缺少特征 '{feature}'，用0填充")
            aligned_data[feature] = 0
    
    # 确保列顺序与训练时一致
    aligned_data = aligned_data[feature_columns]
    
    return aligned_data

def test(test_data, test_data_label, model_path, id_index=None, run_shap=True, feature_columns_path=None):
    """
    测试模型
    
    参数:
    - test_data: 测试数据
    - test_data_label: 测试标签
    - model_path: 模型文件路径
    - id_index: ID列（可选）
    - run_shap: 是否运行SHAP分析
    - feature_columns_path: 训练时保存的特征列名文件路径（可选）
    
    返回:
    - 结果字典
    """
    print("正在预测...")
    
    # 加载保存的模型
    try:
        ensemble_model_loaded = joblib.load(model_path)
        print(f"已加载模型: {type(ensemble_model_loaded).__name__}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 加载训练时使用的特征列名
    if feature_columns_path and os.path.exists(feature_columns_path):
        try:
            # 尝试加载最终特征列名（小波变换后）
            final_feature_path = feature_columns_path.replace('.csv', '_final.csv')
            if os.path.exists(final_feature_path):
                feature_columns_df = pd.read_csv(final_feature_path)
                feature_columns = feature_columns_df['feature_name'].tolist()
                print(f"从文件加载特征列名，数量: {len(feature_columns)}")
            else:
                # 加载原始特征列名
                feature_columns_df = pd.read_csv(feature_columns_path)
                feature_columns = feature_columns_df['feature_name'].tolist()
                print(f"从文件加载原始特征列名，数量: {len(feature_columns)}")
        except Exception as e:
            print(f"加载特征列名失败: {e}")
            feature_columns = None
    else:
        feature_columns = None
    
    # 如果无法加载特征列名，则使用测试数据的列（排除标签列）
    if feature_columns is None:
        # 获取测试数据的特征列
        feature_columns = [col for col in test_data.columns 
                          if col not in ['is_laundering', 'id', 'account_is_laundering_From', 'account_is_laundering_To']]
        print(f"使用测试数据特征列名，数量: {len(feature_columns)}")
    
    # 对齐特征
    X_aligned = align_features(test_data, feature_columns)
    
    # 应用小波变换
    selected_columns = ['counts_1H_From', 'counts_1H_To', '1H_frequency_From', '1H_frequency_To']
    
    # 确保选中的列存在
    selected_columns = [col for col in selected_columns if col in X_aligned.columns]
    if not selected_columns:
        print("警告: 没有找到小波变换的列，跳过小波变换")
        test_data_expanded = X_aligned
    else:
        # 对测试数据应用小波变换
        test_data_expanded = apply_wavelet_transform(X_aligned, selected_columns)
    
    # 获取特征名称
    feature_names = test_data_expanded.columns.tolist()
    print(f"最终特征数量: {len(feature_names)}")
    
    # 检查特征数量是否与模型期望的一致
    try:
        # 对于XGBoost模型，可以获取booster的特征数量
        if hasattr(ensemble_model_loaded, 'get_booster'):
            booster = ensemble_model_loaded.get_booster()
            model_feature_count = booster.num_features()
            print(f"模型期望的特征数量: {model_feature_count}")
            
            if model_feature_count != len(feature_names):
                print(f"警告: 特征数量不匹配！模型期望{model_feature_count}，实际{len(feature_names)}")
                
                # 尝试检查特征名
                try:
                    model_feature_names = booster.feature_names
                    if model_feature_names:
                        print(f"模型特征名: {model_feature_names[:10]}...")
                        print(f"实际特征名: {feature_names[:10]}...")
                        
                        # 找出不匹配的特征
                        missing_in_model = [f for f in feature_names if f not in model_feature_names]
                        missing_in_data = [f for f in model_feature_names if f not in feature_names]
                        
                        if missing_in_model:
                            print(f"实际数据中有但模型中没有的特征: {missing_in_model[:5]}...")
                        if missing_in_data:
                            print(f"模型中有但实际数据中没有的特征: {missing_in_data[:5]}...")
                except:
                    pass
    except:
        pass
    
    # 预测结果
    try:
        predictions = ensemble_model_loaded.predict(test_data_expanded)
        print("预测完成")
        
        # 如果模型有predict_proba方法，获取预测概率
        if hasattr(ensemble_model_loaded, 'predict_proba'):
            prediction_proba = ensemble_model_loaded.predict_proba(test_data_expanded)[:, 1]
        else:
            prediction_proba = None
    except Exception as e:
        print(f"预测失败: {e}")
        
        # 尝试重新训练模型或调整特征
        print("尝试修复特征不匹配问题...")
        try:
            # 获取模型的特征名
            model_feature_names = ensemble_model_loaded.get_booster().feature_names
            print(f"模型特征名示例: {model_feature_names[:5]}")
            print(f"实际特征名示例: {feature_names[:5]}")
            
            # 重新对齐特征
            test_data_expanded = align_features(test_data_expanded, model_feature_names)
            predictions = ensemble_model_loaded.predict(test_data_expanded)
            print("修复后预测成功")
            
            if hasattr(ensemble_model_loaded, 'predict_proba'):
                prediction_proba = ensemble_model_loaded.predict_proba(test_data_expanded)[:, 1]
            else:
                prediction_proba = None
                
        except Exception as e2:
            print(f"修复失败: {e2}")
            return None
    
    # 将预测结果添加到原始数据
    test_data_copy = test_data.copy()
    test_data_copy['prediction'] = predictions
    
    if id_index is not None:
        test_data_copy = pd.concat([id_index.reset_index(drop=True), test_data_copy.reset_index(drop=True)], axis=1)
    
    # 计算准确率
    correct_num = sum(test_data_label == predictions)
    print(f"预测正确样本数: {correct_num}")
    print(f"总样本数: {len(test_data_label)}")
    print(f"准确率: {correct_num/len(test_data_label):.4f}")
    
    # 预测评价
    f1 = f1_score(test_data_label, predictions, zero_division=0)
    precision = precision_score(test_data_label, predictions, zero_division=0)
    recall = recall_score(test_data_label, predictions, zero_division=0)
    
    print("\n" + "="*80)
    print("模型性能评估:")
    print("="*80)
    print(f"F1分数: {f1:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    
    # 计算混淆矩阵
    try:
        cm = confusion_matrix(test_data_label, predictions)
        tn, fp, fn, tp = cm.ravel()
        print(f"\n混淆矩阵:")
        print(f"[[TN={tn:6d}  FP={fp:6d}]")
        print(f" [FN={fn:6d}  TP={tp:6d}]]")
    except Exception as e:
        print(f"计算混淆矩阵失败: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0
        cm = None
    
    # 执行SHAP分析
    shap_results = None
    if run_shap:
        print("\n" + "="*80)
        print("开始SHAP可解释性分析")
        print("="*80)
        
        shap_results = perform_shap_analysis(
            model=ensemble_model_loaded,
            X_data=test_data_expanded,
            feature_names=feature_names,
            sample_size=min(2000, len(test_data_expanded)),  # 限制样本数以加快速度
            save_dir='/home/aolc/ECNU/model-ensemble/Model/wavxgboost/shap_analysis_results'
        )
        
        if shap_results:
            print("\n" + "="*80)
            print("SHAP分析完成!")
            if 'save_dir' in shap_results:
                print(f"所有结果已保存到: {shap_results['save_dir']}/")
            print("="*80)
        
        # 创建SHAP分析报告
        create_shap_report(shap_results, test_data_label, predictions, prediction_proba)
    
    return {
        'predictions': predictions,
        'prediction_proba': prediction_proba,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'shap_results': shap_results
    }

if __name__ == "__main__":
    # 获取命令行传递的参数
    if len(sys.argv) < 3:
        print("用法:")
        print("训练模式: python script.py train <数据文件路径> <模型保存路径> [特征名称保存路径]")
        print("测试模式: python script.py test <数据文件路径> <模型文件路径> [特征名称文件路径] [是否运行SHAP分析]")
        print("\n示例:")
        print("训练: python script.py train df_normalized.csv model.pkl feature_names.csv")
        print("测试: python script.py test test_data.csv model.pkl feature_names.csv true")
        sys.exit(1)
    
    mode = sys.argv[1]  # 模式：train或test
    
    if mode == "train":
        # 训练模式
        if len(sys.argv) < 4:
            print("训练模式需要2个参数: <数据文件路径> <模型保存路径>")
            sys.exit(1)
            
        file_path = sys.argv[2]
        save_modelpath = sys.argv[3]
        feature_names_path = sys.argv[4] if len(sys.argv) > 4 else 'feature_names.csv'
        
        print("="*80)
        print(f"训练模式")
        print(f"加载数据: {file_path}")
        print(f"模型保存路径: {save_modelpath}")
        print(f"特征名称保存路径: {feature_names_path}")
        print("="*80)
        
        # 读取数据
        try:
            df_normalized = pd.read_csv(file_path)
            print(f"数据加载成功，形状: {df_normalized.shape}")
            print(f"数据列名: {df_normalized.columns.tolist()[:10]}...")
        except Exception as e:
            print(f"加载数据失败: {e}")
            sys.exit(1)
        
        # 运行训练
        start_time = time.time()
        train_f1, feature_columns = train(df_normalized, save_modelpath, feature_names_path)
        end_time = time.time()
        
        print(f"\n训练完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"训练F1分数: {train_f1:.4f}")
        print(f"使用的特征数量: {len(feature_columns)}")
        
    elif mode == "test":
        # 测试模式
        if len(sys.argv) < 4:
            print("测试模式需要2个参数: <数据文件路径> <模型文件路径>")
            sys.exit(1)
        
        file_path = sys.argv[2]
        model_path = sys.argv[3]
        feature_names_path = sys.argv[4] if len(sys.argv) > 4 else None
        run_shap = True if len(sys.argv) < 6 else sys.argv[5].lower() in ['true', '1', 'yes', 'y', 't']
        
        print("="*80)
        print(f"测试模式")
        print(f"加载数据: {file_path}")
        print(f"加载模型: {model_path}")
        if feature_names_path:
            print(f"特征名称文件: {feature_names_path}")
        print(f"运行SHAP分析: {run_shap}")
        print("="*80)
        
        # 读取数据
        try:
            df_normalized = pd.read_csv(file_path)
            print(f"数据加载成功，形状: {df_normalized.shape}")
            print(f"数据列名: {df_normalized.columns.tolist()[:10]}...")
        except Exception as e:
            print(f"加载数据失败: {e}")
            sys.exit(1)
        
        # 检查必要列是否存在
        required_columns = ['is_laundering']
        missing_columns = [col for col in required_columns if col not in df_normalized.columns]
        if missing_columns:
            print(f"错误: 数据文件缺少必要列: {missing_columns}")
            sys.exit(1)
        
        test_data_label = df_normalized['is_laundering']
        
        # 提取ID列（如果存在）
        id_index = df_normalized[['id']] if 'id' in df_normalized.columns else None
        
        # 准备测试数据 - 使用get_feature_columns函数确保一致性
        feature_columns = get_feature_columns(df_normalized)
        test_data = df_normalized[feature_columns].copy()
        
        print(f"测试数据形状: {test_data.shape}")
        print(f"特征数量: {len(test_data.columns)}")
        print(f"前10个特征: {test_data.columns.tolist()[:10]}")
        print(f"标签分布: 正类={sum(test_data_label==1)}，负类={sum(test_data_label==0)}")
        
        # 运行测试
        start_time = time.time()
        results = test(test_data, test_data_label, model_path, id_index, run_shap, feature_names_path)
        end_time = time.time()
        
        if results:
            runtime = end_time - start_time
            length = test_data_label.size
            
            print("\n" + "="*80)
            print("程序运行统计:")
            print("="*80)
            print(f"程序运行总时间: {runtime:.2f} 秒")
            print(f"每条数据预测时间: {runtime/length:.6f} 秒")
            print(f"数据总量: {length} 条")
            print("="*80)
    else:
        print(f"未知模式: {mode}")
        print("可用模式: train, test")
        sys.exit(1)