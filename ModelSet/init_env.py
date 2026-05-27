import os

def init_project():
    # 1. 定义需要自动创建的目录
    directories = [
        'data',           # 存放训练和测试 CSV
        'saved_models',   # 存放生成的 joblib 文件
        'results',        # 存放 global_results.csv
        'plots/shap',     # 存放 SHAP 可视化图表
        'plots/rules'     # 存放规则挖掘逻辑图
    ]

    print("--- 正在初始化项目环境 ---")
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"创建目录: {d}")
        else:
            print(f"目录已存在: {d}")

    # 2. 检查依赖项 (可选)
    try:
        import torch
        print(f"检测到 Torch 版本: {torch.__version__} (CUDA可用: {torch.cuda.is_available()})")
    except ImportError:
        print("警告: 未检测到 Torch，深度学习模型可能无法运行。")

    print("\n环境初始化完成！请执行以下命令安装依赖：")
    print("pip install -r requirements.txt")

if __name__ == "__main__":
    init_project()