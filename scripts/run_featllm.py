"""FeatLLM baseline wrapper: 在指定数据集上运行 FeatLLM 生成规则特征。

FeatLLM 流程:
1. LLM 从 few-shot 样本中提取规则
2. LLM 将规则转为 Python 函数
3. 函数将原始数据转为二值特征向量
4. 输出二值特征 CSV

用法:
    python scripts/run_featllm.py --dataset credit_g
    python scripts/run_featllm.py --dataset diabetes --num-query 10 --shot 8
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATLLM_DIR = PROJECT_ROOT / "baselines" / "FeatLLM"
sys.path.insert(0, str(FEATLLM_DIR))

import utils as featllm_utils

DATASET_REGISTRY = {
    "credit_g": {
        "csv": PROJECT_ROOT / "data" / "public" / "credit_g.csv",
        "target": "class",
        "task": "Does this person have good or bad credit risk? good or bad?",
        "meta": FEATLLM_DIR / "data" / "credit-g-metadata.json",
    },
    "diabetes": {
        "csv": PROJECT_ROOT / "data" / "public" / "diabetes.csv",
        "target": "class",
        "task": "Does this patient have diabetes? Yes or no?",
        "meta": FEATLLM_DIR / "data" / "diabetes-metadata.json",
    },
    "blood_transfusion": {
        "csv": PROJECT_ROOT / "data" / "public" / "blood_transfusion.csv",
        "target": "Class",
        "task": "Did the person donate blood in March 2007? Yes or no?",
        "meta": None,
    },
    "heart": {
        "csv": PROJECT_ROOT / "data" / "public" / "heart.csv",
        "target": "HeartDisease",
        "task": "Does this patient have heart disease? Yes or no?",
        "meta": None,
    },
    "bank_marketing": {
        "csv": PROJECT_ROOT / "data" / "public" / "bank_marketing.csv",
        "target": "y",
        "task": "Does this client subscribe to a term deposit? Yes or no?",
        "meta": None,
    },
    "breast_w": {
        "csv": PROJECT_ROOT / "data" / "public" / "breast_w.csv",
        "target": "Class",
        "task": "Is this tumor benign or malignant? benign or malignant?",
        "meta": None,
    },
    "adult": {
        "csv": PROJECT_ROOT / "data" / "public" / "adult.csv",
        "target": "class",
        "task": "Does this person earn more than $50K per year? Yes or no?",
        "meta": None,
    },
    "simecny": {
        "csv": None,
        "target": "is_risk",
        "task": "Is this transaction a money laundering risk? Yes or no?",
        "meta": None,
    },
}

_DIVIDER = "\n\n---DIVIDER---\n\n"
_VERSION = "\n\n---VERSION---\n\n"


def load_dataset(name: str) -> pd.DataFrame:
    reg = DATASET_REGISTRY[name]
    if name == "simecny":
        df = pd.read_parquet(PROJECT_ROOT / "outputs" / "baseline_cache.parquet")
        baseline_exclude = {"Min_Received", "Min_Sent", "S_src_counterparty_index",
                            "Timestamp_Hour", "S_dst_counterparty_index",
                            "1H_frequency_From", "1H_frequency_To",
                            "wallets_per_cert_From", "min_inactive_period_From", "Max_Sent"}
        return df.drop(columns=[c for c in baseline_exclude if c in df.columns], errors="ignore")
    return pd.read_csv(reg["csv"])


def prepare_data(df: pd.DataFrame, target: str, shot: int, seed: int):
    """Prepare train/test split and few-shot samples."""
    y = df[target].to_numpy()
    label_list = sorted(np.unique(y).tolist(), key=str)

    categorical_indicator = [
        df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])
        for c in df.columns if c != target
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1), y,
        test_size=0.2, random_state=seed, stratify=y,
    )

    X_new_train = X_train.copy()
    X_new_train[target] = y_train
    sampled_list = []
    remainder = shot % len(label_list)
    for _, grouped in X_new_train.groupby(target):
        sample_num = shot // len(label_list)
        if remainder > 0:
            sample_num += 1
            remainder -= 1
        n = min(sample_num, len(grouped))
        grouped = grouped.sample(n, random_state=seed)
        sampled_list.append(grouped)
    X_balanced = pd.concat(sampled_list)
    X_train_shot = X_balanced.drop([target], axis=1)
    y_train_shot = X_balanced[target].to_numpy()

    return X_train, X_test, y_train, y_test, X_train_shot, y_train_shot, label_list, categorical_indicator


def build_metadata(df: pd.DataFrame, target: str, meta_path=None):
    """Build or load feature metadata for prompts."""
    if meta_path and meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    meta = {}
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == "object":
            meta[col] = f"{col} (categorical variable)"
        else:
            meta[col] = f"{col} (numerical variable)"
    return meta


def main():
    parser = argparse.ArgumentParser(description="Run FeatLLM baseline")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--num-query", type=int, default=5, help="Number of rule ensembles")
    parser.add_argument("--shot", type=int, default=4, help="Number of few-shot examples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "featllm")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reg = DATASET_REGISTRY[args.dataset]

    featllm_utils.set_seed(args.seed)
    df = load_dataset(args.dataset)
    target = reg["target"]

    (X_train, X_test, y_train, y_test,
     X_train_shot, y_train_shot,
     label_list, is_cat) = prepare_data(df, target, args.shot, args.seed)

    X_all = df.drop(target, axis=1)
    print(f"Running FeatLLM on {args.dataset} ({len(df)} rows, shot={args.shot}, queries={args.num_query})")

    # Step 1: Generate prompts for rule extraction
    ask_file = FEATLLM_DIR / "templates" / "ask_llm.txt"
    meta = build_metadata(df, target, reg.get("meta"))

    # Build feature descriptions
    feature_desc = ""
    for col in X_all.columns:
        desc = meta.get(col, f"{col} (numerical variable)")
        feature_desc += f"- {desc}\n"

    # Build task description
    task_str = reg["task"]

    # Manually build prompts using FeatLLM's template
    with open(ask_file, "r") as f:
        prompt_template = f.read()

    templates = []
    for q in range(args.num_query):
        # Re-sample for diversity
        featllm_utils.set_seed(args.seed + q)
        examples_str = ""
        for idx in range(len(X_train_shot)):
            row = X_train_shot.iloc[idx]
            row_str = featllm_utils.serialize(
                pd.concat([row, pd.Series({target: y_train_shot[idx]})])
            )
            examples_str += row_str + "\n"

        prompt = prompt_template.replace("[TASK]", task_str)
        prompt = prompt.replace("[FEATURES]", feature_desc)
        prompt = prompt.replace("[EXAMPLES]", examples_str.strip())
        n_rules = 10
        label_str = ", ".join([str(l) for l in label_list])

        response_format = ""
        for label in label_list:
            response_format += f'{n_rules} different conditions for class "{label}":\n- [Condition]\n...\n\n'

        prompt = prompt.replace("[FORMAT]", response_format.strip())
        templates.append(prompt)

    # Step 2: Query LLM for rules
    api_key = os.environ.get("KNOWFEAT_API_KEY", "")
    rule_file = args.output_dir / f"rule-{args.dataset}-{args.shot}-{args.seed}.out"

    if rule_file.exists():
        print(f"Loading cached rules from {rule_file}")
        with open(rule_file, "r") as f:
            results = f.read().strip().split(_DIVIDER)
    else:
        print("Querying LLM for rules...")
        results = featllm_utils.query_gpt(
            templates, api_key,
            max_tokens=1500, temperature=0.5, model=args.model,
        )
        if results == [-1]:
            print("LLM query failed.")
            return
        with open(rule_file, "w") as f:
            f.write(_DIVIDER.join(results))
        print(f"Rules saved to {rule_file}")

    # Step 3: Parse rules
    parsed_rules = featllm_utils.parse_rules(results, label_list)

    # Step 4: Generate function code from rules
    func_file = args.output_dir / f"function-{args.dataset}-{args.shot}-{args.seed}.out"
    function_template = FEATLLM_DIR / "templates" / "ask_for_function.txt"

    if func_file.exists():
        print(f"Loading cached functions from {func_file}")
        with open(func_file, "r") as f:
            total_str = f.read().strip()
            fct_strs_all = [x.split(_DIVIDER) for x in total_str.split(_VERSION)]
    else:
        print("Querying LLM for function code...")
        fct_strs_all = []
        for parsed_rule in parsed_rules:
            fct_templates = featllm_utils.get_prompt_for_generating_function(
                parsed_rule, feature_desc, str(function_template),
            )
            fct_results = featllm_utils.query_gpt(
                fct_templates, api_key,
                max_tokens=1500, temperature=0, model=args.model,
            )
            fct_strs = []
            for fct_txt in fct_results:
                if isinstance(fct_txt, int):
                    fct_strs.append("")
                    continue
                parts = fct_txt.split("<start>")
                if len(parts) > 1:
                    fct_strs.append(parts[1].split("<end>")[0].strip())
                else:
                    fct_strs.append(fct_txt.strip())
            fct_strs_all.append(fct_strs)

        with open(func_file, "w") as f:
            total_str = _VERSION.join([_DIVIDER.join(x) for x in fct_strs_all])
            f.write(total_str)
        print(f"Functions saved to {func_file}")

    # Step 5: Convert to binary features
    fct_names = []
    fct_strs_final = []
    for fct_str_pair in fct_strs_all:
        if not fct_str_pair or "def" not in fct_str_pair[0]:
            continue
        fct_pair_name = []
        for fct_str in fct_str_pair:
            if "def" in fct_str:
                fct_pair_name.append(fct_str.split("def")[1].split("(")[0].strip())
            else:
                fct_pair_name.append("unknown")
        fct_names.append(fct_pair_name)
        fct_strs_final.append(fct_str_pair)

    if not fct_strs_final:
        print("No valid functions generated.")
        return

    # Apply functions to full dataset
    X_full = df.drop(target, axis=1)
    all_binary_features = []

    for q_idx, (fct_strs, names) in enumerate(zip(fct_strs_final, fct_names)):
        for cls_idx, (fct_str, name) in enumerate(zip(fct_strs, names)):
            try:
                local_ns = {"pd": pd, "np": np}
                exec(fct_str, local_ns)
                func = local_ns[name]
                binary_df = func(X_full)
                binary_df.columns = [
                    f"featllm_q{q_idx}_c{cls_idx}_{c}" for c in binary_df.columns
                ]
                all_binary_features.append(binary_df)
            except Exception as e:
                print(f"  Function {name} failed: {e}")

    if all_binary_features:
        features_df = pd.concat(all_binary_features, axis=1)
        out_path = args.output_dir / f"{args.dataset}_features.csv"
        features_df.to_csv(out_path, index=False)
        print(f"Generated {features_df.shape[1]} binary features → {out_path}")
    else:
        print("No features generated.")


if __name__ == "__main__":
    main()
