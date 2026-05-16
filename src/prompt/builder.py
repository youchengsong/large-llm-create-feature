"""Prompt 组装模块，构建 System Prompt 和 User Prompt。"""


def _estimate_tokens(text: str) -> int:
    return len(text) // 3


def build_system_prompt(
    schema_text: str,
    transaction_patterns: str,
    expert_knowledge: str,
    risk_indicators: str,
    baseline_features: list[str],
    distribution_diff: str = "",
    join_statistics: str = "",
    baseline_code_samples: str = "",
    constraints: str | None = None,
) -> str:
    baseline_list = "\n".join(f"- {f}" for f in baseline_features)

    if constraints is None:
        constraints = """1. 代码必须在 30 秒内执行完毕
2. 只能使用 pandas, numpy, scipy.stats, sklearn.preprocessing
3. 不要引用数据中不存在的字段
4. 不要使用标签列（is_risk, is_abnormal, is_zs_laundering, is_src_victim）作为特征
5. 特征值应有区分度（不能全0或全1，NaN不超过50%）
6. 只生成 Python 代码，不要生成一阶逻辑表达式"""

    prompt = f"""你是数字人民币反洗钱领域的特征工程师。你的任务是为反洗钱检测模型生成新特征。

## 思考过程
生成每个特征前，请先按以下步骤推理：
1. 分析下方的「分布差异」，找出正常和异常之间差异最大的维度
2. 思考哪些领域知识能解释这些差异
3. 设计一个能捕捉该差异的特征，确保它与已有特征不冗余
4. 编写代码实现

## 输出要求
每个特征必须以 JSON 格式输出，包含以下字段：
- feature_name: 特征名称（英文，有描述性）
- granularity: "wallet" 或 "transaction"
- reasoning: 你的推理过程（分析了什么差异、为什么这个特征有用、与已有特征的区别）
- rationale: 为什么创建这个特征（必须引用下方的领域知识）
- knowledge_source: 引用的具体领域知识条目
- code: Python 代码片段（不要定义函数，直接写代码）

代码中可以直接使用以下变量：open_df, tx_df, close_df, pd, np
代码最后必须将结果赋值给 result 变量，result 必须是 DataFrame，包含 [wallet_id, feature_value] 两列。
注意：
  - 必须按 wallet_id 聚合，每个 wallet_id 只能有一行（不允许重复）
  - 不接受 tx_id 级别的输出，必须聚合到账户级别

示例代码格式：
```python
stats = tx_df.groupby('src')['amount'].std().reset_index()
stats.columns = ['wallet_id', 'my_feature']
stats['my_feature'] = stats['my_feature'].fillna(0)
result = stats
```

## 数据探索（可选）
如果你需要先查看数据的分布或统计信息再决定生成什么特征，可以在输出中使用「查询」动作。
查询动作的格式：
```json
{{"action": "query", "question": "我想看什么", "code": "query_result = tx_df.groupby('src')['amount'].describe()"}}
```
- 代码中将结果赋值给 `query_result`（不是 result）
- 可用变量与特征代码相同：open_df, tx_df, close_df, pd, np
- 查询结果会在下一轮反馈中返回给你
- 你可以在同一个输出中同时包含查询和特征，也可以只输出查询
- 每轮最多 3 个查询，超时 15 秒

## 数据结构

{schema_text}

{join_statistics}

{distribution_diff}

## 三表关联
- 交易表的 src 和 dst 字段是 wallet_id，关联到开户表和销户表
- 一个钱包可以在多笔交易中出现

{transaction_patterns}

## 领域知识

{expert_knowledge}

{risk_indicators}

## 已有特征（不要重复，可以组合改进）
以下特征已经由人工实现：
{baseline_list}

{baseline_code_samples}

## 约束
{constraints}"""

    token_est = _estimate_tokens(prompt)
    print(f"[build_system_prompt] 估算 token 数: ~{token_est}")
    return prompt


def build_user_prompt_cold_start(
    island_name: str,
    island_fields: list[str],
    seed_examples: str,
    n_candidates: int = 3,
) -> str:
    fields_str = ", ".join(island_fields)

    prompt = f"""第 1 轮迭代（首轮）

## 种子示例
以下是已有的检测规则转换为 Python 代码的示例，展示期望的代码风格：

{seed_examples}

## 请生成 {n_candidates} 个新特征
你负责的维度是：{island_name}。
重点探索的数据字段：{fields_str}
参考上面的领域知识和交易模式说明，围绕该维度从你认为最有价值的角度出发。输出 JSON 数组格式。"""

    token_est = _estimate_tokens(prompt)
    print(f"[build_user_prompt_cold_start] 估算 token 数: ~{token_est}")
    return prompt


def build_user_prompt(
    island_name: str,
    island_fields: list[str],
    shap_feedback: str,
    top_features: list[dict],
    failed_features: list[dict],
    n_candidates: int = 3,
) -> str:
    fields_str = ", ".join(island_fields)

    top_lines = []
    for f in top_features:
        top_lines.append(
            f"- {f.get('feature_name', 'unknown')}: "
            f"score={f.get('score', 0):.4f}, "
            f"SHAP排名={f.get('shap_rank', 'N/A')}, "
            f"MI={f.get('mi', 0):.4f}"
        )
    top_text = "\n".join(top_lines) if top_lines else "（暂无）"

    fail_lines = []
    for f in failed_features:
        fail_lines.append(
            f"- {f.get('feature_name', 'unknown')}: {f.get('failure_reason', '未知原因')}"
        )
    fail_text = "\n".join(fail_lines) if fail_lines else "（暂无）"

    prompt = f"""## 本岛屿探索方向
你当前负责的维度是：{island_name}
重点探索的数据字段：{fields_str}
请围绕该维度生成特征，不要偏离方向。

## 上轮最佳特征（记忆库中的高分示例）
{top_text}

## 上轮 SHAP 分析
{shap_feedback}

## 上轮失败的特征（不要重复类似的）
{fail_text}

## 请生成 {n_candidates} 个新特征
围绕你负责的维度（{island_name}），生成 {n_candidates} 个不同角度的特征。输出 JSON 数组格式。"""

    token_est = _estimate_tokens(prompt)
    print(f"[build_user_prompt] 估算 token 数: ~{token_est}")
    return prompt
