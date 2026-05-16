"""修复13个失败特征并重新评估组合指标。"""
import sys
sys.path.insert(0, '/root/大模型提取特征')

import json
import numpy as np
import pandas as pd
from src.evaluation.metrics import compute_cv_metrics
from src.data.loader import load_all_tables
from src.data.baseline import run_baseline
from src.evaluation.sandbox import execute_feature_code, validate_output
from src.data.merge import merge_feature_to_transactions
from openai import OpenAI

DEEPSEEK_API_KEY = "sk-59bd25d3db384cf09f63814989d7200c"

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def llm_fix_code(name: str, code: str, error: str) -> str:
    """调用 DeepSeek 修复特征代码。"""
    prompt = f"""特征 '{name}' 的代码执行失败。
错误信息: {error}

原始代码:
```python
{code}
```

请修复代码。要求：
1. 代码中可以直接使用变量：open_df, tx_df, close_df, pd, np
2. 不要导入 datetime, os, sys 等非白名单模块（只能用 pandas, numpy, scipy, sklearn）
3. 代码最后必须将结果赋值给 result 变量
4. result 必须是 DataFrame，包含 [wallet_id, feature_value] 或 [tx_id, feature_value]
5. 必须在30秒内完成（138万行数据），避免嵌套循环，用向量化操作
6. 用 pd.Timestamp 代替 datetime

只输出修复后的纯 Python 代码，不要 markdown 标记，不要解释。"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```python"):
        raw = raw[len("```python"):].strip()
    if raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return raw


print("=== 加载数据 ===")
open_df, tx_df, close_df = load_all_tables("data")
print(f"tx_df: {len(tx_df)} rows")

print("=== 运行基线特征 ===")
baseline_df = run_baseline(open_df, tx_df, close_df)
label = baseline_df["is_laundering"].astype(int)
baseline_df = baseline_df.drop(columns=["is_laundering"])
X_baseline = baseline_df.fillna(0)
print(f"Baseline features: {X_baseline.shape[1]}")

baseline_metrics = {"f1": 0.975762, "recall": 0.969225, "precision": 0.982456, "prauc": 0.996577}
print(f"Baseline: F1={baseline_metrics['f1']:.6f}, Recall={baseline_metrics['recall']:.6f}, "
      f"Precision={baseline_metrics['precision']:.6f}, PR-AUC={baseline_metrics['prauc']:.6f}")

print("\n=== 加载新特征 (带LLM修复) ===")
with open("/root/大模型提取特征/output/features.json") as f:
    features_data = json.load(f)

if isinstance(features_data, dict):
    features_data = features_data.get("features", features_data)

print(f"Total features in json: {len(features_data)}")

combined_df = X_baseline.copy()
added = 0
failed_list = []
fixed_list = []
seen_names = set()

for i, feat in enumerate(features_data):
    name = feat["feature_name"]
    if name in seen_names:
        print(f"  SKIP duplicate: {name}")
        continue
    seen_names.add(name)
    code = feat["code"]

    feat_df, error = execute_feature_code(code, open_df, tx_df, close_df)
    if error:
        valid = False
        reason = error
    else:
        valid, reason = validate_output(feat_df)
        if valid:
            reason = None

    # 如果失败，尝试LLM修复（最多2次）
    if reason:
        original_error = reason
        fixed = False
        for attempt in range(2):
            print(f"  [{i+1}] Fixing {name} (attempt {attempt+1}): {original_error[:60]}...")
            try:
                fixed_code = llm_fix_code(name, code, reason)
                feat_df, error = execute_feature_code(fixed_code, open_df, tx_df, close_df)
                if error:
                    reason = error
                    continue
                valid, reason2 = validate_output(feat_df)
                if not valid:
                    reason = reason2
                    continue
                code = fixed_code
                fixed = True
                fixed_list.append(name)
                print(f"  [{i+1}] FIXED: {name} (attempt {attempt+1})")
                break
            except Exception as e:
                reason = str(e)
                print(f"  [{i+1}] Fix attempt {attempt+1} error: {e}")

        if not fixed:
            failed_list.append((name, reason[:100] if reason else "unknown"))
            print(f"  [{i+1}] FAIL: {name} - {reason[:80] if reason else 'unknown'}")
            continue

    # 执行成功，merge
    try:
        merged = merge_feature_to_transactions(feat_df, tx_df)
        new_cols = [c for c in merged.columns if c not in tx_df.columns]
        if not new_cols:
            failed_list.append((name, "merge后无新列"))
            print(f"  [{i+1}] FAIL (merge): {name} - no new cols")
            continue
        tx_sorted = merged.sort_values("timestamp").reset_index(drop=True)
        for col in new_cols:
            combined_df[col] = tx_sorted[col].fillna(0).values[:len(combined_df)]
        added += 1
        if name not in fixed_list:
            print(f"  [{i+1}] OK: {name} (+{len(new_cols)} cols)")
    except Exception as e:
        failed_list.append((name, str(e)[:100]))
        print(f"  [{i+1}] FAIL (merge): {name} - {str(e)[:80]}")

print(f"\n=== 汇总 ===")
print(f"Added: {added}, Fixed by LLM: {len(fixed_list)}, Failed: {len(failed_list)}")
print(f"Combined features: {combined_df.shape[1]}")
if fixed_list:
    print(f"Fixed features: {fixed_list}")
if failed_list:
    print(f"Still failed:")
    for name, err in failed_list:
        print(f"  {name}: {err}")

print("\n=== 计算组合指标 (threshold on train set) ===")
combined_metrics = compute_cv_metrics(combined_df, label)
print(f"Combined F1:        {combined_metrics['f1']:.6f}")
print(f"Combined Recall:    {combined_metrics['recall']:.6f}")
print(f"Combined Precision: {combined_metrics['precision']:.6f}")
print(f"Combined PR-AUC:    {combined_metrics['prauc']:.6f}")

print(f"\n=== 增量 ===")
print(f"Delta F1:        {combined_metrics['f1'] - baseline_metrics['f1']:+.6f}")
print(f"Delta Recall:    {combined_metrics['recall'] - baseline_metrics['recall']:+.6f}")
print(f"Delta Precision: {combined_metrics['precision'] - baseline_metrics['precision']:+.6f}")
print(f"Delta PR-AUC:    {combined_metrics['prauc'] - baseline_metrics['prauc']:+.6f}")
