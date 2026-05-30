"""测试知识溯源可解释性：让 LLM 生成一个特征，检查 provenance card 质量。"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from knowfeat.config import KnowFeatConfig, PathConfig
from knowfeat.prompt.domain_knowledge import load_domain_knowledge, format_domain_knowledge, resolve_knowledge_ids
from knowfeat.agent.tools import GENERATE_FEATURE_SPEC

# 1. 加载领域知识
cfg = KnowFeatConfig(stage="stage2")
dk = load_domain_knowledge(cfg)
knowledge_text = format_domain_knowledge(dk, stage="stage2")

print(f"领域知识长度: {len(knowledge_text)} chars")
print("=" * 60)

# 2. 截取知识文本的前 2000 字符看看 ID 格式
print("知识文本预览（前 2000 字符）:")
print(knowledge_text[:2000])
print("..." if len(knowledge_text) > 2000 else "")
print("=" * 60)

# 3. 构造 LLM 请求
system_prompt = f"""你是一个数字人民币反洗钱特征工程专家。

请基于以下领域知识，在 95 个基线特征的基础上，设计 1 个新的特征。

## 要求
- 特征函数签名: def feature_xxx(open_df, tx_df, close_df) -> pd.DataFrame
- 输出必须含 wallet_id 列
- **溯源要求**：必须在 knowledge_ids 中填写具体的知识资产ID（如 RULE_002, IND_05, EXP_03, PAT_A02, RF_S2_100_001），不要用笼统描述
- **计算逻辑**：必须在 computation_logic 中说明用了哪些字段、经过什么变换得到最终值

## 可用数据
- open_df: 账户信息表（wallet_id, wallet_type, wallet_level, region_code, wallet_open_date 等）
- tx_df: 交易表（tx_id, src, dst, amount, timestamp, hour, transaction_mode, is_risk 等）
- close_df: 注销表（wallet_id, zs_timestamp 等）

{knowledge_text}
"""

user_msg = """请设计 1 个针对「金额频次」维度的特征。使用 generate_feature 工具提交。"""

client = OpenAI(
    api_key=os.environ.get("KNOWFEAT_API_KEY", "sk-b484e757948249a7b36f3864818806e4"),
    base_url="https://api.deepseek.com/v1",
)

print("调用 DeepSeek API...")
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": GENERATE_FEATURE_SPEC.name,
                "description": GENERATE_FEATURE_SPEC.description,
                "parameters": GENERATE_FEATURE_SPEC.parameters,
            },
        }],
        tool_choice={"type": "function", "function": {"name": "generate_feature"}},
        temperature=0.7,
    )
except Exception as e:
    print(f"API 调用失败: {e}")
    sys.exit(1)

# 4. 解析结果
msg = response.choices[0].message
if not msg.tool_calls:
    print("LLM 没有调用工具，返回纯文本:")
    print(msg.content)
    sys.exit(1)

tool_call = msg.tool_calls[0]
args = json.loads(tool_call.function.arguments)

print("=" * 60)
print("LLM 生成的特征 Provenance Card:")
print("=" * 60)

print(f"\n特征名称: {args.get('feature_name', 'N/A')}")
print(f"维度: {args.get('dimension', [])}")
print(f"\n--- 知识溯源ID（含原始内容） ---")
knowledge_ids = args.get('knowledge_ids', [])
if knowledge_ids:
    resolved = resolve_knowledge_ids(knowledge_ids, dk.knowledge_registry)
    for item in resolved:
        print(f"  - [{item['id']}] {item['content']}")
else:
    print("  (空！LLM 未填写)")

print(f"\n--- 计算逻辑 ---")
computation_logic = args.get('computation_logic', '')
print(f"  {computation_logic if computation_logic else '(空！LLM 未填写)'}")

print(f"\n--- 使用的数据字段 ---")
data_fields = args.get('data_fields_used', [])
for f in data_fields:
    print(f"  - {f}")

print(f"\n--- 知识来源（自然语言） ---")
print(f"  {args.get('knowledge_source', 'N/A')}")

print(f"\n--- 特征解释 ---")
print(f"  {args.get('explanation', 'N/A')}")

print(f"\n--- 设计推理 ---")
print(f"  {args.get('reasoning', 'N/A')}")

print(f"\n--- 特征代码 ---")
print(args.get('code', 'N/A'))

# 5. 质量评估
print("\n" + "=" * 60)
print("溯源质量评估:")
print("=" * 60)

checks = []
if knowledge_ids:
    checks.append(f"✅ knowledge_ids 已填写 ({len(knowledge_ids)} 个)")
    valid_prefixes = ("RULE_", "IND_", "EXP_", "PAT_", "RF_S1_", "RF_S2_")
    valid = [k for k in knowledge_ids if k.startswith(valid_prefixes)]
    invalid = [k for k in knowledge_ids if not k.startswith(valid_prefixes)]
    if valid:
        checks.append(f"  ✅ 有效ID: {', '.join(valid)}")
    if invalid:
        checks.append(f"  ❌ 无效ID: {', '.join(invalid)}")
else:
    checks.append("❌ knowledge_ids 为空")

if computation_logic:
    checks.append(f"✅ computation_logic 已填写 ({len(computation_logic)} 字符)")
else:
    checks.append("❌ computation_logic 为空")

if data_fields:
    checks.append(f"✅ data_fields_used 已填写 ({len(data_fields)} 个字段)")
else:
    checks.append("❌ data_fields_used 为空")

for c in checks:
    print(c)
