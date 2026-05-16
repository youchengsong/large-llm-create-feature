"""生成Agent运行的可视化HTML报告。"""

import json
from pathlib import Path


def generate_agent_report(
    result: dict,
    eval_results: dict,
    elapsed: float,
    model: str,
    tx_count: int,
    wallet_count: int,
    baseline_feature_count: int,
    positive_rate: float,
    output_path: str = "outputs/single_agent_run/report.html",
):
    """生成完整的交互式HTML报告。"""

    accepted = result.get("accepted_features", [])
    failed = result.get("failed_features", [])
    step_log = result.get("step_log", [])

    baseline = eval_results.get("baseline", {})
    combined = eval_results.get("combined", {})
    delta = eval_results.get("delta", {})

    # 特征表格
    features_html = ""
    for i, feat in enumerate(accepted, 1):
        l2 = feat.get("l2_metrics", {})
        l3 = feat.get("l3_metrics", {})
        features_html += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{feat.get('name','')}</strong></td>
            <td><span class="badge">{feat.get('dimension','')}</span></td>
            <td title="{feat.get('explanation','')}">{feat.get('explanation','')[:50]}</td>
            <td>{l2.get('iv',0):.4f}</td>
            <td>{l2.get('ks',0):.4f}</td>
            <td>{l2.get('redundancy',0):.3f}</td>
            <td class="{'positive' if l3.get('delta_f1',0)>0 else 'negative'}">{l3.get('delta_f1',0):.6f}</td>
            <td class="{'positive' if l3.get('delta_recall',0)>0 else 'negative'}">{l3.get('delta_recall',0):.6f}</td>
        </tr>"""

    # 失败表格
    failed_html = ""
    for feat in failed[:30]:
        failed_html += f"""
        <tr>
            <td>{feat.get('name','unknown')}</td>
            <td class="error">{feat.get('error','')[:120]}</td>
        </tr>"""

    # Agent步骤
    steps_html = ""
    for step in step_log[:60]:
        tools = step.get("tool_calls", [])
        tool_str = ""
        for tc in tools:
            obs = tc.get("observation", "")[:60]
            tool_str += f'<span class="tool">{tc["tool"]}</span> '
        thought = (step.get("thought", "") or "")[:80]
        steps_html += f"""
        <tr>
            <td>{step.get('step','')}</td>
            <td class="mono">{step.get('timestamp','')}</td>
            <td>{thought}</td>
            <td>{tool_str or '-'}</td>
        </tr>"""

    # 维度分布
    dim_counts = {}
    for feat in accepted:
        d = feat.get("dimension", "未知")
        dim_counts[d] = dim_counts.get(d, 0) + 1
    dims_html = "".join(f'<div class="dim-item"><span class="dim-name">{d}</span><span class="dim-count">{c}</span></div>' for d, c in dim_counts.items())

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>单Agent特征工程 - 运行报告</title>
<style>
:root {{
    --bg: #f0f4f8; --card: #fff; --border: #e2e8f0;
    --text: #1a202c; --muted: #718096;
    --blue: #4299e1; --green: #48bb78; --red: #fc8181;
    --purple: #9f7aea; --orange: #ed8936;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
.container {{ max-width:1280px; margin:0 auto; padding:32px 24px; }}
header {{ margin-bottom:32px; }}
header h1 {{ font-size:26px; margin-bottom:4px; }}
header p {{ color:var(--muted); font-size:14px; }}

.grid {{ display:grid; gap:16px; margin-bottom:32px; }}
.grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
.grid-2 {{ grid-template-columns: 1fr 1fr; }}

.card {{ background:var(--card); border-radius:12px; padding:20px; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
.card-sm {{ text-align:center; }}
.card-sm .num {{ font-size:36px; font-weight:800; line-height:1.2; }}
.card-sm .label {{ font-size:12px; color:var(--muted); margin-top:4px; }}
.num.blue {{ color:var(--blue); }}
.num.green {{ color:var(--green); }}
.num.red {{ color:var(--red); }}
.num.purple {{ color:var(--purple); }}
.num.orange {{ color:var(--orange); }}

section {{ margin-bottom:36px; }}
section h2 {{ font-size:18px; margin-bottom:14px; padding-bottom:8px; border-bottom:2px solid var(--blue); }}

/* Metric bars */
.metric-row {{ display:flex; align-items:center; margin-bottom:12px; gap:12px; }}
.metric-label {{ width:80px; font-size:13px; font-weight:600; }}
.metric-bars {{ flex:1; }}
.bar-wrap {{ display:flex; align-items:center; margin-bottom:3px; gap:8px; }}
.bar-tag {{ font-size:11px; width:36px; color:var(--muted); }}
.bar {{ height:22px; border-radius:4px; display:flex; align-items:center; padding:0 8px; font-size:11px; color:#fff; font-weight:600; min-width:60px; transition:width .8s ease; }}
.bar-bl {{ background:var(--blue); }}
.bar-cb {{ background:var(--green); }}
.metric-delta {{ font-size:12px; font-weight:600; margin-left:8px; }}
.positive {{ color:#2f855a; }}
.negative {{ color:#c53030; }}

/* Table */
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--border); }}
th {{ background:#f7fafc; font-weight:600; position:sticky; top:0; }}
tr:hover {{ background:#f7fafc; }}
.error {{ color:var(--red); font-size:12px; }}
.mono {{ font-family:monospace; font-size:12px; }}

.badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; background:#ebf4ff; color:#2b6cb0; }}
.tool {{ display:inline-block; padding:1px 6px; border-radius:4px; font-size:10px; background:#ebf8ff; color:#2c5282; font-family:monospace; margin:1px; }}

/* Dimensions */
.dim-grid {{ display:flex; flex-wrap:wrap; gap:8px; }}
.dim-item {{ display:flex; align-items:center; gap:6px; padding:6px 12px; background:var(--card); border-radius:8px; border:1px solid var(--border); }}
.dim-name {{ font-size:13px; }}
.dim-count {{ background:var(--blue); color:#fff; border-radius:50%; width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700; }}

/* Collapsible */
details {{ margin-bottom:12px; }}
details summary {{ cursor:pointer; font-weight:600; font-size:14px; padding:8px 0; }}
details[open] summary {{ color:var(--blue); }}

.scroll-table {{ max-height:500px; overflow-y:auto; border-radius:8px; border:1px solid var(--border); }}

@media(max-width:768px) {{
    .grid-4 {{ grid-template-columns: 1fr 1fr; }}
    .grid-2 {{ grid-template-columns: 1fr; }}
    .container {{ padding:16px; }}
}}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>单Agent特征工程 — 运行报告</h1>
    <p>模型: {model} | 耗时: {elapsed:.0f}秒 | 数据: {tx_count:,}条交易, {wallet_count:,}个钱包 | 正样本率: {positive_rate*100:.3f}%</p>
</header>

<!-- Summary Cards -->
<div class="grid grid-4">
    <div class="card card-sm"><div class="num blue">{len(accepted)}</div><div class="label">接受特征</div></div>
    <div class="card card-sm"><div class="num red">{len(failed)}</div><div class="label">失败特征</div></div>
    <div class="card card-sm"><div class="num purple">{result.get('total_steps',0)}</div><div class="label">Agent步数</div></div>
    <div class="card card-sm"><div class="num orange">{len(result.get('dimensions_explored',[]))}</div><div class="label">探索维度</div></div>
    <div class="card card-sm"><div class="num green">{eval_results.get('new_features_count',0)}</div><div class="label">有效新特征</div></div>
</div>

<!-- Performance Comparison -->
<section>
<h2>性能对比 (5-Fold Stratified CV)</h2>
<div class="card">
    <div class="metric-row">
        <div class="metric-label">F1</div>
        <div class="metric-bars">
            <div class="bar-wrap"><span class="bar-tag">基线</span><div class="bar bar-bl" style="width:{baseline.get('f1',0)*100:.1f}%">{baseline.get('f1',0):.4f}</div></div>
            <div class="bar-wrap"><span class="bar-tag">组合</span><div class="bar bar-cb" style="width:{combined.get('f1',0)*100:.1f}%">{combined.get('f1',0):.4f}</div><span class="metric-delta {'positive' if delta.get('f1',0)>=0 else 'negative'}">({delta.get('f1',0):+.5f})</span></div>
        </div>
    </div>
    <div class="metric-row">
        <div class="metric-label">Recall</div>
        <div class="metric-bars">
            <div class="bar-wrap"><span class="bar-tag">基线</span><div class="bar bar-bl" style="width:{baseline.get('recall',0)*100:.1f}%">{baseline.get('recall',0):.4f}</div></div>
            <div class="bar-wrap"><span class="bar-tag">组合</span><div class="bar bar-cb" style="width:{combined.get('recall',0)*100:.1f}%">{combined.get('recall',0):.4f}</div><span class="metric-delta {'positive' if delta.get('recall',0)>=0 else 'negative'}">({delta.get('recall',0):+.5f})</span></div>
        </div>
    </div>
    <div class="metric-row">
        <div class="metric-label">Precision</div>
        <div class="metric-bars">
            <div class="bar-wrap"><span class="bar-tag">基线</span><div class="bar bar-bl" style="width:{baseline.get('precision',0)*100:.1f}%">{baseline.get('precision',0):.4f}</div></div>
            <div class="bar-wrap"><span class="bar-tag">组合</span><div class="bar bar-cb" style="width:{combined.get('precision',0)*100:.1f}%">{combined.get('precision',0):.4f}</div><span class="metric-delta {'positive' if delta.get('precision',0)>=0 else 'negative'}">({delta.get('precision',0):+.5f})</span></div>
        </div>
    </div>
    <div class="metric-row">
        <div class="metric-label">PR-AUC</div>
        <div class="metric-bars">
            <div class="bar-wrap"><span class="bar-tag">基线</span><div class="bar bar-bl" style="width:{baseline.get('pr_auc',0)*100:.1f}%">{baseline.get('pr_auc',0):.4f}</div></div>
            <div class="bar-wrap"><span class="bar-tag">组合</span><div class="bar bar-cb" style="width:{combined.get('pr_auc',0)*100:.1f}%">{combined.get('pr_auc',0):.4f}</div><span class="metric-delta {'positive' if delta.get('pr_auc',0)>=0 else 'negative'}">({delta.get('pr_auc',0):+.5f})</span></div>
        </div>
    </div>
</div>
</section>

<!-- Dimensions -->
<section>
<h2>特征维度分布</h2>
<div class="dim-grid">{dims_html}</div>
</section>

<!-- Accepted Features Table -->
<section>
<h2>接受的特征 ({len(accepted)})</h2>
<div class="scroll-table">
<table>
    <thead><tr><th>#</th><th>名称</th><th>维度</th><th>说明</th><th>IV</th><th>KS</th><th>冗余</th><th>dF1</th><th>dRecall</th></tr></thead>
    <tbody>{features_html}</tbody>
</table>
</div>
</section>

<!-- Feature Code Details (collapsible) -->
<section>
<h2>特征代码详情</h2>
{"".join(f'''<details>
<summary>{feat.get("name","")} — {feat.get("explanation","")[:40]}</summary>
<pre style="background:#1a202c;color:#e2e8f0;padding:12px;border-radius:8px;overflow-x:auto;font-size:12px;margin-top:8px;">{feat.get("code","").replace("<","&lt;").replace(">","&gt;")}</pre>
<p style="margin-top:6px;font-size:12px;color:var(--muted);">推理: {feat.get("reasoning","")}</p>
</details>''' for feat in accepted)}
</section>

<!-- Failed Features -->
<section>
<details>
<summary>失败的特征 ({len(failed)}个，点击展开)</summary>
<div class="scroll-table" style="margin-top:8px;">
<table>
    <thead><tr><th>名称</th><th>错误原因</th></tr></thead>
    <tbody>{failed_html}</tbody>
</table>
</div>
</details>
</section>

<!-- Agent Steps -->
<section>
<details>
<summary>Agent执行轨迹 (前60步，点击展开)</summary>
<div class="scroll-table" style="margin-top:8px;">
<table>
    <thead><tr><th>步骤</th><th>时间</th><th>思考</th><th>工具调用</th></tr></thead>
    <tbody>{steps_html}</tbody>
</table>
</div>
</details>
</section>

<!-- Run Info -->
<section>
<h2>运行配置</h2>
<div class="card">
<table>
    <tr><td><strong>基线特征数</strong></td><td>{baseline_feature_count}</td><td><strong>新增有效特征</strong></td><td>{eval_results.get('new_features_count',0)}</td></tr>
    <tr><td><strong>Agent总步数</strong></td><td>{result.get('total_steps',0)}</td><td><strong>总尝试次数</strong></td><td>{result.get('total_attempts',0)}</td></tr>
    <tr><td><strong>结束原因</strong></td><td colspan="3">{result.get('finish_reason','')}</td></tr>
</table>
</div>
</section>

</div>

<script>
window.addEventListener('load',()=>{{document.querySelectorAll('.bar').forEach(b=>{{const w=b.style.width;b.style.width='0%';setTimeout(()=>b.style.width=w,200)}})}});
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
