"""从 features.json 生成可视化 HTML 报告。"""

import json
import html
from pathlib import Path


def generate_report_html(features_json_path: str, output_html_path: str,
                         baseline_metrics: dict | None = None,
                         combined_metrics: dict | None = None) -> str:
    """读取 features.json，生成可视化 HTML 报告，返回输出路径。

    如果 features.json 是新格式（含 summary），自动提取 baseline/combined。
    也可通过参数显式传入 baseline_metrics 和 combined_metrics。
    """
    with open(features_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        summary = data.get("summary", {})
        features = data.get("features", [])
    else:
        features = data
        summary = {}

    baseline = baseline_metrics or summary.get("baseline_metrics", {})
    combined = combined_metrics or summary.get("combined_metrics", {})
    delta = summary.get("delta_metrics", {})

    if not baseline and features:
        f0 = features[0].get("l3_metrics", {})
        baseline = {
            "f1": f0.get("base_f1", 0),
            "recall": f0.get("base_recall", 0),
            "precision": f0.get("base_precision", 0),
            "prauc": 0,
        }

    if not delta and combined and baseline:
        delta = {k: combined.get(k, 0) - baseline.get(k, 0) for k in combined}

    features_sorted = sorted(features, key=lambda x: x.get("l3_metrics", {}).get("delta_f1", 0), reverse=True)

    feature_rows = ""
    for i, feat in enumerate(features_sorted, 1):
        name = html.escape(feat.get("feature_name", ""))
        explanation = html.escape(feat.get("explanation", "")[:80])
        l2 = feat.get("l2_metrics", {})
        l3 = feat.get("l3_metrics", {})
        score = feat.get("score", 0)
        accepted_via = feat.get("accepted_via", "direct")

        delta_f1 = l3.get("delta_f1", 0)
        delta_recall = l3.get("delta_recall", 0)
        delta_precision = l3.get("delta_precision", 0)
        iv = l2.get("iv", 0)
        ks = l2.get("ks", 0)
        redundancy = l2.get("redundancy", 0)
        shap_rank = l3.get("shap_rank", 0)

        f1_class = "positive" if delta_f1 > 0 else "negative" if delta_f1 < 0 else ""
        recall_class = "positive" if delta_recall > 0 else "negative" if delta_recall < 0 else ""
        prec_class = "positive" if delta_precision > 0 else "negative" if delta_precision < 0 else ""

        feature_rows += f"""
        <tr>
            <td>{i}</td>
            <td class="feature-name">{name}</td>
            <td>{explanation}</td>
            <td>{score:.3f}</td>
            <td>{iv:.4f}</td>
            <td>{ks:.4f}</td>
            <td>{redundancy:.3f}</td>
            <td class="{f1_class}">{delta_f1:+.4f}</td>
            <td class="{recall_class}">{delta_recall:+.4f}</td>
            <td class="{prec_class}">{delta_precision:+.4f}</td>
            <td>{shap_rank}</td>
            <td>{accepted_via}</td>
        </tr>"""

    top_f1 = features_sorted[:10]
    top_recall = sorted(features, key=lambda x: x.get("l3_metrics", {}).get("delta_recall", 0), reverse=True)[:10]

    chart_f1_names = json.dumps([f.get("feature_name", "")[:25] for f in top_f1], ensure_ascii=False)
    chart_f1_values = json.dumps([round(f.get("l3_metrics", {}).get("delta_f1", 0) * 10000, 1) for f in top_f1])

    chart_recall_names = json.dumps([f.get("feature_name", "")[:25] for f in top_recall], ensure_ascii=False)
    chart_recall_values = json.dumps([round(f.get("l3_metrics", {}).get("delta_recall", 0) * 10000, 1) for f in top_recall])

    iv_values = [f.get("l2_metrics", {}).get("iv", 0) for f in features]
    ks_values = [f.get("l2_metrics", {}).get("ks", 0) for f in features]
    redundancy_values = [f.get("l2_metrics", {}).get("redundancy", 0) for f in features]

    island_stats = {}
    for feat in features:
        exp = feat.get("explainability", {})
        name = feat.get("feature_name", "")
        if "amount" in name or "surge" in name or "dispersion" in name:
            island = "金额频次"
        elif "night" in name or "hour" in name or "region" in name or "time" in name or "weekend" in name:
            island = "时间地域"
        elif "mode" in name or "interval" in name or "sequence" in name or "burst" in name:
            island = "行为序列"
        else:
            island = "主体关联"
        island_stats.setdefault(island, {"count": 0, "avg_delta_f1": 0})
        island_stats[island]["count"] += 1
        island_stats[island]["avg_delta_f1"] += feat.get("l3_metrics", {}).get("delta_f1", 0)

    for k in island_stats:
        if island_stats[k]["count"] > 0:
            island_stats[k]["avg_delta_f1"] /= island_stats[k]["count"]

    island_names = json.dumps(list(island_stats.keys()), ensure_ascii=False)
    island_counts = json.dumps([island_stats[k]["count"] for k in island_stats])

    baseline_f1 = baseline.get("f1", 0)
    baseline_recall = baseline.get("recall", 0)
    baseline_precision = baseline.get("precision", 0)
    baseline_prauc = baseline.get("prauc", 0)
    combined_f1 = combined.get("f1", 0) if combined else 0
    combined_recall = combined.get("recall", 0) if combined else 0
    combined_precision = combined.get("precision", 0) if combined else 0
    combined_prauc = combined.get("prauc", 0) if combined else 0
    delta_f1_total = delta.get("f1", 0) if delta else 0
    delta_recall_total = delta.get("recall", 0) if delta else 0
    delta_precision_total = delta.get("precision", 0) if delta else 0
    delta_prauc_total = delta.get("prauc", 0) if delta else 0

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM 特征搜索结果报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; margin: 30px 0; color: #1a1a2e; font-size: 28px; }}
        h2 {{ color: #16213e; margin: 25px 0 15px; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .summary-card h3 {{ font-size: 14px; color: #64748b; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .summary-card .value {{ font-size: 32px; font-weight: 700; color: #1e293b; }}
        .summary-card .delta {{ font-size: 16px; margin-top: 4px; }}
        .positive {{ color: #16a34a; }}
        .negative {{ color: #dc2626; }}
        .metrics-comparison {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
        .metric-box {{ background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .metric-box .label {{ font-size: 13px; color: #64748b; margin-bottom: 6px; }}
        .metric-box .baseline {{ font-size: 20px; color: #475569; }}
        .metric-box .combined {{ font-size: 24px; font-weight: 700; color: #1e293b; }}
        .metric-box .change {{ font-size: 14px; margin-top: 4px; font-weight: 600; }}
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .chart-container {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .chart-container h3 {{ margin-bottom: 15px; font-size: 16px; color: #334155; }}
        .bar-chart {{ width: 100%; }}
        .bar-row {{ display: flex; align-items: center; margin: 6px 0; }}
        .bar-label {{ width: 180px; font-size: 12px; color: #475569; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .bar-track {{ flex: 1; height: 22px; background: #f1f5f9; border-radius: 4px; overflow: hidden; position: relative; }}
        .bar-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 11px; color: white; font-weight: 600; min-width: 35px; }}
        .bar-fill.pos {{ background: linear-gradient(90deg, #22c55e, #16a34a); }}
        .bar-fill.neg {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
        .table-container {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #f8fafc; padding: 12px 8px; text-align: left; font-weight: 600; color: #475569; border-bottom: 2px solid #e2e8f0; position: sticky; top: 0; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #f1f5f9; }}
        tr:hover {{ background: #f8fafc; }}
        .feature-name {{ font-weight: 600; color: #1e293b; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .explainability-section {{ margin-top: 30px; }}
        .feat-detail {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .feat-detail h4 {{ color: #1e293b; margin-bottom: 10px; }}
        .feat-detail .meta {{ display: flex; gap: 15px; margin-bottom: 10px; flex-wrap: wrap; }}
        .feat-detail .meta span {{ font-size: 12px; background: #f1f5f9; padding: 3px 8px; border-radius: 4px; }}
        .feat-detail .explain-content {{ font-size: 13px; color: #475569; }}
        .feat-detail .explain-content dt {{ font-weight: 600; margin-top: 8px; color: #334155; }}
        .feat-detail .explain-content dd {{ margin-left: 16px; margin-bottom: 4px; }}
        footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #94a3b8; font-size: 13px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>LLM 特征搜索结果报告</h1>

    <div class="summary-grid">
        <div class="summary-card">
            <h3>通过验证的特征</h3>
            <div class="value">{len(features)}</div>
            <div class="delta">{summary.get('total_rounds', 'N/A')} 轮搜索</div>
        </div>
        <div class="summary-card">
            <h3>组合 F1 增量</h3>
            <div class="value positive">{delta_f1_total:+.4f}</div>
            <div class="delta">{baseline_f1:.4f} → {combined_f1:.4f}</div>
        </div>
        <div class="summary-card">
            <h3>组合 Recall 增量</h3>
            <div class="value positive">{delta_recall_total:+.4f}</div>
            <div class="delta">{baseline_recall:.4f} → {combined_recall:.4f}</div>
        </div>
        <div class="summary-card">
            <h3>组合 Precision 增量</h3>
            <div class="value positive">{delta_precision_total:+.4f}</div>
            <div class="delta">{baseline_precision:.4f} → {combined_precision:.4f}</div>
        </div>
    </div>

    <h2>基线 vs 组合 指标对比</h2>
    <div class="metrics-comparison">
        <div class="metric-box">
            <div class="label">F1 Score</div>
            <div class="baseline">基线: {baseline_f1:.4f}</div>
            <div class="combined">{combined_f1:.4f}</div>
            <div class="change positive">{delta_f1_total:+.6f}</div>
        </div>
        <div class="metric-box">
            <div class="label">Recall</div>
            <div class="baseline">基线: {baseline_recall:.4f}</div>
            <div class="combined">{combined_recall:.4f}</div>
            <div class="change positive">{delta_recall_total:+.6f}</div>
        </div>
        <div class="metric-box">
            <div class="label">Precision</div>
            <div class="baseline">基线: {baseline_precision:.4f}</div>
            <div class="combined">{combined_precision:.4f}</div>
            <div class="change positive">{delta_precision_total:+.6f}</div>
        </div>
        <div class="metric-box">
            <div class="label">PR-AUC</div>
            <div class="baseline">基线: {baseline_prauc:.4f}</div>
            <div class="combined">{combined_prauc:.4f}</div>
            <div class="change positive">{delta_prauc_total:+.6f}</div>
        </div>
    </div>

    <h2>Top 10 特征（按 Delta F1 排序）</h2>
    <div class="charts-grid">
        <div class="chart-container">
            <h3>Delta F1 (×10⁴)</h3>
            <div class="bar-chart" id="chart-f1"></div>
        </div>
        <div class="chart-container">
            <h3>Delta Recall (×10⁴)</h3>
            <div class="bar-chart" id="chart-recall"></div>
        </div>
    </div>

    <h2>维度岛屿贡献</h2>
    <div class="charts-grid">
        <div class="chart-container">
            <h3>各岛屿产出特征数</h3>
            <div class="bar-chart" id="chart-island"></div>
        </div>
        <div class="chart-container">
            <h3>IV vs KS 分布</h3>
            <div style="padding: 20px; font-size: 13px; color: #475569;">
                <p>IV 均值: {sum(iv_values)/len(iv_values):.4f} | 中位数: {sorted(iv_values)[len(iv_values)//2]:.4f}</p>
                <p>KS 均值: {sum(ks_values)/len(ks_values):.4f} | 中位数: {sorted(ks_values)[len(ks_values)//2]:.4f}</p>
                <p>Redundancy 均值: {sum(redundancy_values)/len(redundancy_values):.3f}</p>
                <p style="margin-top: 10px;">IV > 0.1 的特征: {sum(1 for v in iv_values if v > 0.1)} 个</p>
                <p>KS > 0.1 的特征: {sum(1 for v in ks_values if v > 0.1)} 个</p>
            </div>
        </div>
    </div>

    <h2>全部特征详情</h2>
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>特征名</th>
                    <th>说明</th>
                    <th>Score</th>
                    <th>IV</th>
                    <th>KS</th>
                    <th>Redund.</th>
                    <th>ΔF1</th>
                    <th>ΔRecall</th>
                    <th>ΔPrecis.</th>
                    <th>SHAP</th>
                    <th>来源</th>
                </tr>
            </thead>
            <tbody>
                {feature_rows}
            </tbody>
        </table>
    </div>

    <h2 class="explainability-section">Top 10 特征可解释性</h2>
    {"".join(_render_feature_detail(f, i+1) for i, f in enumerate(features_sorted[:10]))}

    <footer>
        由 LLM 特征搜索引擎自动生成 | 特征数: {len(features)} | 搜索轮数: {summary.get('total_rounds', 'N/A')}
    </footer>
</div>

<script>
function renderBarChart(containerId, labels, values) {{
    const container = document.getElementById(containerId);
    const maxAbs = Math.max(...values.map(Math.abs), 1);
    let html = '';
    for (let i = 0; i < labels.length; i++) {{
        const pct = Math.abs(values[i]) / maxAbs * 80;
        const cls = values[i] >= 0 ? 'pos' : 'neg';
        html += `<div class="bar-row">
            <span class="bar-label">${{labels[i]}}</span>
            <div class="bar-track">
                <div class="bar-fill ${{cls}}" style="width: ${{pct}}%">${{values[i]}}</div>
            </div>
        </div>`;
    }}
    container.innerHTML = html;
}}

renderBarChart('chart-f1', {chart_f1_names}, {chart_f1_values});
renderBarChart('chart-recall', {chart_recall_names}, {chart_recall_values});
renderBarChart('chart-island', {island_names}, {island_counts});
</script>
</body>
</html>"""

    Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path


def _render_feature_detail(feat: dict, rank: int) -> str:
    """渲染单个特征的可解释性卡片。"""
    name = html.escape(feat.get("feature_name", ""))
    explanation = html.escape(feat.get("explanation", ""))
    l2 = feat.get("l2_metrics", {})
    l3 = feat.get("l3_metrics", {})
    exp = feat.get("explainability", {})

    if not exp or not isinstance(exp, dict):
        return ""

    meta_html = f"""
    <span>IV={l2.get('iv', 0):.4f}</span>
    <span>KS={l2.get('ks', 0):.4f}</span>
    <span>ΔF1={l3.get('delta_f1', 0):+.4f}</span>
    <span>ΔRecall={l3.get('delta_recall', 0):+.4f}</span>
    <span>SHAP#{l3.get('shap_rank', 0)}</span>
    """

    explain_html = ""
    if exp.get("value_meaning"):
        vm = exp["value_meaning"]
        if isinstance(vm, dict):
            items = "".join(f"<dd><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</dd>" for k, v in vm.items())
            explain_html += f"<dt>取值含义</dt>{items}"
        else:
            explain_html += f"<dt>取值含义</dt><dd>{html.escape(str(vm))}</dd>"

    if exp.get("risk_threshold"):
        explain_html += f"<dt>风险阈值</dt><dd>{html.escape(str(exp['risk_threshold']))}</dd>"

    if exp.get("synergy"):
        synergy = exp["synergy"]
        if isinstance(synergy, list):
            items = ", ".join(
                html.escape(s.get("feature", str(s)) if isinstance(s, dict) else str(s))
                for s in synergy[:3]
            )
            explain_html += f"<dt>协同特征</dt><dd>{items}</dd>"
        else:
            explain_html += f"<dt>协同特征</dt><dd>{html.escape(str(synergy))}</dd>"

    if exp.get("false_positive_cases"):
        explain_html += f"<dt>误判场景</dt><dd>{html.escape(str(exp['false_positive_cases'])[:200])}</dd>"

    return f"""
    <div class="feat-detail">
        <h4>#{rank} {name}</h4>
        <p style="font-size: 13px; color: #64748b; margin-bottom: 8px;">{explanation}</p>
        <div class="meta">{meta_html}</div>
        <dl class="explain-content">{explain_html}</dl>
    </div>"""


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "output/features.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/feature_report.html"
    result = generate_report_html(input_path, output_path)
    print(f"Report generated: {result}")
