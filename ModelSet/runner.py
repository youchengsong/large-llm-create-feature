import argparse
import sys
import os
import time

# --- 导入各个模型模块 (请确保这些文件在当前目录下) ---
import catboost_std as cat_module
import ga2m_std as ga2m_module
import scorecard_std as scorecard_module
import acxgb_std as acxgb_module
import wav_xgb_std as wav_xgb_module
import ensemble_std as ensemble_module
import lgbm_std as lgbm_module
import resample_boost_std as resample_boost_module
import gbdt_fm_std as gbdt_fm_module
import dnn_res_std as dnn_res_module
import rule_fusion_mlp_std as rule_fusion_mlp_module


def main():
    parser = argparse.ArgumentParser(description="反洗钱模型统一自动化调度平台")

    # 1. 核心参数：支持列表输入，例如 --model catboost acxgb 或 --model all
    parser.add_argument("--model", type=str, nargs='+', required=True,
                        help="指定模型名称(空格分隔)，或输入 'all' 运行所有模型")

    # 2. 执行模式：增加 'both' (训练+测试)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "test", "both"],
                        help="执行模式：train (仅训练), test (仅测试), both (一键训练并测试)")

    # 3. 路径参数：分离训练和测试数据
    parser.add_argument("--train_data", type=str, help="训练数据 CSV 路径 (train/both 模式必填)")
    parser.add_argument("--test_data", type=str, help="测试数据 CSV 路径 (test/both 模式必填)")
    parser.add_argument("--model_dir", type=str, default="./saved_models", help="模型 joblib 统一存放目录")
    parser.add_argument("--output", type=str, default="results/global_results.csv", help="结果保存路径")

    args = parser.parse_args()

    # 模型路由映射表
    model_router = {
        "catboost": cat_module, "ga2m": ga2m_module, "scorecard": scorecard_module,
        "acxgb": acxgb_module, "wav_xgb": wav_xgb_module, "ensemble": ensemble_module,
        "lgbm": lgbm_module, "resample_boost": resample_boost_module,
        "gbdt_fm": gbdt_fm_module, "dnn_res": dnn_res_module, "mlp": rule_fusion_mlp_module
    }

    # 处理 'all' 逻辑
    selected_models = list(model_router.keys()) if "all" in args.model else args.model

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    print(f"[{time.strftime('%H:%M:%S')}] 任务启动。待处理模型: {selected_models}")

    for model_name in selected_models:
        if model_name not in model_router:
            print(f"跳过未知模型: {model_name}")
            continue

        module = model_router[model_name]
        model_file = os.path.join(args.model_dir, f"{model_name}.joblib")

        # 记录该模型的总执行开始时间
        exec_start = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 训练阶段
            if args.mode in ["train", "both"]:
                print(f"--- [{model_name}] 正在训练... ---")
                module.train_and_save(args.train_data, model_file)

            # 测试阶段
            if args.mode in ["test", "both"]:
                print(f"--- [{model_name}] 正在测试... ---")
                exec_end = time.strftime("%Y-%m-%d %H:%M:%S")
                # 【关键修改】传递开始和结束时间给测试函数
                module.load_and_test(args.test_data, model_file, args.output, exec_start, exec_end)

        except Exception as e:
            print(f"!!! [{model_name}] 运行崩溃: {e}")

    print(f"[{time.strftime('%H:%M:%S')}] 所有任务执行完毕。")


if __name__ == "__main__":
    main()