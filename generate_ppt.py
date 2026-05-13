from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Color palette ──
DARK_BG = RGBColor(0x1B, 0x1B, 0x2F)
ACCENT_BLUE = RGBColor(0x4F, 0x8E, 0xF7)
ACCENT_GREEN = RGBColor(0x2E, 0xCC, 0x71)
ACCENT_ORANGE = RGBColor(0xF3, 0x9C, 0x12)
ACCENT_RED = RGBColor(0xE7, 0x4C, 0x3C)
ACCENT_PURPLE = RGBColor(0x9B, 0x59, 0xB6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
MID_GRAY = RGBColor(0x99, 0x99, 0x99)
DARK_GRAY = RGBColor(0x2C, 0x2C, 0x3E)
CARD_BG = RGBColor(0x24, 0x24, 0x3E)
LIGHT_BG = RGBColor(0xF5, 0xF6, 0xFA)
TEXT_DARK = RGBColor(0x2C, 0x3E, 0x50)
CARD_LIGHT = RGBColor(0xFF, 0xFF, 0xFF)
BORDER_LIGHT = RGBColor(0xDE, 0xE2, 0xE6)
SECTION_BG = RGBColor(0xEB, 0xF5, 0xFB)


def add_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, left, top, width, height, text, font_size=18, color=TEXT_DARK, bold=False, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = alignment
    return txBox


def add_bullet_list(slide, left, top, width, height, items, font_size=16, color=TEXT_DARK, spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = spacing
        p.level = 0
    return txBox


def add_left_accent_bar(slide, top, height, color=ACCENT_BLUE):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), top, Inches(0.08), height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()


def add_page_number(slide, num, total):
    add_text(slide, Inches(12.3), Inches(7.05), Inches(1), Inches(0.4),
             f"{num}/{total}", font_size=11, color=MID_GRAY, alignment=PP_ALIGN.RIGHT)


def add_section_header(slide, title, subtitle=""):
    add_bg(slide, LIGHT_BG)
    add_rect(slide, Inches(0), Inches(0), prs.slide_width, Inches(1.2), ACCENT_BLUE)
    add_text(slide, Inches(0.8), Inches(0.15), Inches(11), Inches(0.7),
             title, font_size=32, color=WHITE, bold=True)
    if subtitle:
        add_text(slide, Inches(0.8), Inches(0.75), Inches(11), Inches(0.4),
                 subtitle, font_size=16, color=RGBColor(0xBB, 0xDD, 0xFF))


TOTAL_SLIDES = 20

# ============================================================
# SLIDE 1: Title
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK_BG)

add_rect(slide, Inches(0), Inches(0), prs.slide_width, Inches(0.06), ACCENT_BLUE)

add_text(slide, Inches(1.5), Inches(1.5), Inches(10), Inches(1.2),
         "基于大语言模型的自动特征工程方案",
         font_size=42, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

add_text(slide, Inches(1.5), Inches(2.6), Inches(10), Inches(0.8),
         "数字人民币反洗钱检测场景",
         font_size=28, color=ACCENT_BLUE, alignment=PP_ALIGN.CENTER)

add_rect(slide, Inches(5), Inches(3.8), Inches(3.333), Inches(0.03), ACCENT_BLUE)

add_text(slide, Inches(1.5), Inches(4.3), Inches(10), Inches(0.5),
         "让 LLM 成为特征工程师：自动发现、生成、验证反洗钱特征",
         font_size=18, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)

add_text(slide, Inches(1.5), Inches(5.5), Inches(10), Inches(0.5),
         "2026年5月",
         font_size=16, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

add_page_number(slide, 1, TOTAL_SLIDES)

# ============================================================
# SLIDE 2: Outline
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "汇报提纲")

sections = [
    ("01", "问题定义与研究目标", "我们要解决什么问题，为什么需要 LLM"),
    ("02", "数据与领域知识", "SimECNY 三表数据、专家知识、人工基线"),
    ("03", "参考文献与技术借鉴", "6 篇论文各借鉴了什么"),
    ("04", "系统架构设计", "五大组件的融合方案"),
    ("05", "核心流程详解", "Phase 0-3 完整执行流程"),
    ("06", "Prompt 设计与 LLM 接口", "System/User Prompt、可插拔模型"),
    ("07", "四级验证框架", "L1 代码 → L2 统计 → L3 模型 → L4 稳定性"),
    ("08", "创新点总结", "5 个核心创新贡献"),
    ("09", "可行性评估与风险", "三档风险分析、实施建议"),
    ("10", "实施路线图", "四阶段时间规划"),
]

for i, (num, title, desc) in enumerate(sections):
    row = i // 2
    col = i % 2
    x = Inches(0.8 + col * 6.2)
    y = Inches(1.6 + row * 1.1)
    card = add_shape(slide, x, y, Inches(5.8), Inches(0.95), CARD_LIGHT, BORDER_LIGHT)

    num_box = add_rect(slide, x + Inches(0.15), y + Inches(0.18), Inches(0.55), Inches(0.55), ACCENT_BLUE)
    num_box.line.fill.background()
    add_text(slide, x + Inches(0.15), y + Inches(0.2), Inches(0.55), Inches(0.55),
             num, font_size=18, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.9), y + Inches(0.12), Inches(4.5), Inches(0.4),
             title, font_size=17, color=TEXT_DARK, bold=True)
    add_text(slide, x + Inches(0.9), y + Inches(0.5), Inches(4.5), Inches(0.35),
             desc, font_size=13, color=MID_GRAY)

add_page_number(slide, 2, TOTAL_SLIDES)

# ============================================================
# SLIDE 3: Problem Definition
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "01 问题定义与研究目标", "为什么需要 LLM 自动特征工程")

add_text(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(0.5),
         "当前现状", font_size=22, color=TEXT_DARK, bold=True)
add_left_accent_bar(slide, Inches(1.5), Inches(2.5), ACCENT_ORANGE)
card = add_shape(slide, Inches(0.8), Inches(2.0), Inches(5.5), Inches(2.5), CARD_LIGHT, BORDER_LIGHT)
add_bullet_list(slide, Inches(1.1), Inches(2.1), Inches(5.0), Inches(2.3), [
    "已有人工设计的 40+ 个反洗钱检测特征",
    "存在明确缺口：transaction_motif/mode 未用、设备/IP 未用、",
    "  销户模块完全未实现、时间窗口仅 1H、无图拓扑特征",
    "人工逐一设计新特征 → 效率低、容易遗漏",
    "之前尝试的 458 条规则 → 全部失败（P/R/F1=0）",
], font_size=15, color=TEXT_DARK)

add_text(slide, Inches(7.0), Inches(1.5), Inches(5.5), Inches(0.5),
         "研究目标", font_size=22, color=TEXT_DARK, bold=True)
add_left_accent_bar(slide, Inches(1.5), Inches(2.5), ACCENT_GREEN)
card = add_shape(slide, Inches(7.0), Inches(2.0), Inches(5.5), Inches(2.5), CARD_LIGHT, BORDER_LIGHT)
add_bullet_list(slide, Inches(7.3), Inches(2.1), Inches(5.0), Inches(2.3), [
    "用 LLM 自动发现人工未覆盖的新特征",
    "每个新特征必须满足三个要求：",
    "  1. 生成可直接执行的 Python 代码",
    "  2. 附带可解释性说明（领域依据）",
    "  3. 经过验证确认有效（增量贡献）",
    "预测目标：交易级二分类（is_risk, ~5%正样本）",
], font_size=15, color=TEXT_DARK)

add_text(slide, Inches(0.8), Inches(4.8), Inches(11.5), Inches(0.5),
         "核心思路", font_size=22, color=TEXT_DARK, bold=True)
card = add_shape(slide, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.2), SECTION_BG, ACCENT_BLUE)
add_text(slide, Inches(1.3), Inches(5.5), Inches(11.0), Inches(0.9),
         "构建一个系统，让 LLM 自动发现新特征并生成可执行代码，同时为每个特征提供领域依据的可解释性说明。\n人不去设计特征，LLM 是特征工程师。",
         font_size=17, color=ACCENT_BLUE, bold=True)

add_page_number(slide, 3, TOTAL_SLIDES)

# ============================================================
# SLIDE 4: Data - Three Tables
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "02 数据来源：SimECNY 三表结构", "数字人民币模拟器生成的关联数据")

tables_info = [
    ("开户表", "16 字段 / 100K 钱包", ACCENT_BLUE,
     ["wallet_id — 钱包唯一标识", "wallet_type — 1企业/2个人/3商户",
      "wallet_level — 1-4类（限额不同）", "wallet_open_cert — 身份证号",
      "open_device / open_ip — 设备MAC/IP",
      "is_abnormal — 是否异常（~5%）"]),
    ("交易表（主表）", "20 字段 / 预测目标", ACCENT_GREEN,
     ["tx_id / timestamp / amount", "src / dst — 发送/接收方 wallet_id",
      "transaction_motif — 7种交易动机", "transaction_mode — 7种结构模式",
      "is_risk — 预测标签（~5%风险）",
      "src_device / src_ip / dst_device / dst_ip"]),
    ("销户表", "6 字段 / 辅助", ACCENT_ORANGE,
     ["wallet_id — 关联钱包", "zs_channel — 销户渠道",
      "zs_timestamp — 销户时间",
      "is_zs_laundering — 是否洗钱销户"]),
]

for i, (name, desc, color, fields) in enumerate(tables_info):
    x = Inches(0.5 + i * 4.2)
    header = add_rect(slide, x, Inches(1.5), Inches(3.9), Inches(0.7), color)
    add_text(slide, x, Inches(1.52), Inches(3.9), Inches(0.35),
             name, font_size=20, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, x, Inches(1.88), Inches(3.9), Inches(0.3),
             desc, font_size=13, color=RGBColor(0xEE, 0xEE, 0xEE), alignment=PP_ALIGN.CENTER)
    card = add_shape(slide, x, Inches(2.2), Inches(3.9), Inches(3.0), CARD_LIGHT, BORDER_LIGHT)
    add_bullet_list(slide, x + Inches(0.2), Inches(2.35), Inches(3.5), Inches(2.7),
                    fields, font_size=14, color=TEXT_DARK, spacing=Pt(5))

# Relation diagram
add_text(slide, Inches(0.5), Inches(5.5), Inches(12), Inches(0.5),
         "三表关联关系", font_size=18, color=TEXT_DARK, bold=True)
card = add_shape(slide, Inches(0.5), Inches(5.95), Inches(12.3), Inches(1.0), SECTION_BG, ACCENT_BLUE)
add_text(slide, Inches(0.9), Inches(6.05), Inches(11.5), Inches(0.8),
         "交易表.src (wallet_id)  →  开户表.wallet_id  →  获取发送方账户属性  |  交易表.dst (wallet_id)  →  开户表.wallet_id  →  获取接收方账户属性\n"
         "wallet_id 同时关联销户表，一个钱包可出现在多笔交易中（一对多关系）。LLM 生成的钱包级特征自动 merge 到 src/dst 两侧。",
         font_size=14, color=TEXT_DARK)

add_page_number(slide, 4, TOTAL_SLIDES)

# ============================================================
# SLIDE 5: 8 Transaction Patterns (injected into LLM prompt)
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "02 数据集 8 种交易模式（注入 LLM Prompt）", "浦发提供的数据集说明文档，作为领域知识注入")

# Normal patterns
add_text(slide, Inches(0.5), Inches(1.4), Inches(6), Inches(0.4),
         "正常交易模式（3 种）", font_size=18, color=ACCENT_GREEN, bold=True)
normal_patterns = [
    ("single_transaction", "金额 1-5000 元\n高峰(9-11,14-16,19-21点)\n70%单笔 30%转发", ACCENT_GREEN),
    ("normal_small_high_freq", "金额 1-100 元\n日常消费\n各模式各 20%", ACCENT_GREEN),
    ("regular_large_low_freq", "金额 100-1000 元\n间隔 1H-3天\n偏向商户钱包", ACCENT_GREEN),
]
for i, (name, desc, color) in enumerate(normal_patterns):
    x = Inches(0.5 + i * 4.2)
    card = add_shape(slide, x, Inches(1.8), Inches(3.9), Inches(1.5), CARD_LIGHT, color)
    add_text(slide, x + Inches(0.15), Inches(1.85), Inches(3.6), Inches(0.35),
             name, font_size=14, color=color, bold=True)
    add_text(slide, x + Inches(0.15), Inches(2.25), Inches(3.6), Inches(0.95),
             desc, font_size=13, color=TEXT_DARK)

# Risk patterns
add_text(slide, Inches(0.5), Inches(3.5), Inches(6), Inches(0.4),
         "风险交易模式（5 种）", font_size=18, color=ACCENT_RED, bold=True)
risk_patterns = [
    ("small_amount_testing", "金额 0.01-0.1 元，深夜\n休眠账户(>=90天)突然激活\n40%扇入 60%扇出", ACCENT_RED),
    ("merchant_laundering", "金额 10-1000 元，仅商户钱包\n跨行90%，同地区80%\n佣金 7-9%", ACCENT_RED),
    ("class4_laundering", "金额 100-2000 元，深夜\n仅4类钱包\n日限额 5000 元", ACCENT_ORANGE),
    ("online_laundering", "金额 10-1000 元\n深夜(20-1点)\n交易次数多", ACCENT_ORANGE),
    ("victim_laundering", "诱导: 设备IP与开户一致\n盗用: 设备IP与开户不一致\n两种子模式", ACCENT_PURPLE),
]
for i, (name, desc, color) in enumerate(risk_patterns):
    row = i // 3
    col = i % 3
    x = Inches(0.5 + col * 4.2)
    y = Inches(3.9 + row * 1.7)
    card = add_shape(slide, x, y, Inches(3.9), Inches(1.5), CARD_LIGHT, color)
    add_text(slide, x + Inches(0.15), y + Inches(0.05), Inches(3.6), Inches(0.35),
             name, font_size=14, color=color, bold=True)
    add_text(slide, x + Inches(0.15), y + Inches(0.42), Inches(3.6), Inches(1.0),
             desc, font_size=13, color=TEXT_DARK)

add_shape(slide, Inches(0.5), Inches(7.0) - Inches(0.7), Inches(12.3), Inches(0.55), SECTION_BG, ACCENT_BLUE)
add_text(slide, Inches(0.9), Inches(7.0) - Inches(0.65), Inches(11.5), Inches(0.5),
         "这些模式说明作为浦发数据集文档注入 System Prompt（~1200 token），帮助 LLM 理解数据生成逻辑，精准构造特征。",
         font_size=14, color=ACCENT_BLUE, bold=True)

add_page_number(slide, 5, TOTAL_SLIDES)

# ============================================================
# SLIDE 6: Domain Knowledge
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "02 领域知识与人工基线", "我们独有的数据资产")

# Domain knowledge cards
dk_items = [
    ("8种交易模式说明", "浦发数据集文档：3正常+5风险模式\n金额/时间/交易模式/特殊条件", ACCENT_RED, "~1200 token"),
    ("专家访谈", "4 类洗钱模式详细描述\n薅羊毛/线下跑分/线上跑分/商户收款码", ACCENT_BLUE, "~800 token"),
    ("可疑交易识别点", "12 类 200+ 个识别点\n压缩为每类 top-3（36 条）", ACCENT_GREEN, "~1000 token"),
    ("风险分类表+检测规则", "38 个风险指标(4维度) + 15 条种子规则\n→ Python 函数示例", ACCENT_ORANGE, "~1000 token"),
]

add_text(slide, Inches(0.8), Inches(1.5), Inches(5), Inches(0.5),
         "领域知识资产（注入 System Prompt）", font_size=20, color=TEXT_DARK, bold=True)

for i, (name, desc, color, tokens) in enumerate(dk_items):
    y = Inches(2.1 + i * 1.15)
    card = add_shape(slide, Inches(0.8), y, Inches(5.5), Inches(1.0), CARD_LIGHT, color)
    badge = add_rect(slide, Inches(5.1), y + Inches(0.05), Inches(1.1), Inches(0.35), color)
    badge.line.fill.background()
    add_text(slide, Inches(5.1), y + Inches(0.05), Inches(1.1), Inches(0.35),
             tokens, font_size=11, color=WHITE, alignment=PP_ALIGN.CENTER)
    add_text(slide, Inches(1.1), y + Inches(0.1), Inches(3.8), Inches(0.35),
             name, font_size=16, color=color, bold=True)
    add_text(slide, Inches(1.1), y + Inches(0.5), Inches(5.0), Inches(0.5),
             desc, font_size=13, color=TEXT_DARK)

# Baseline gaps
add_text(slide, Inches(7.0), Inches(1.5), Inches(5.5), Inches(0.5),
         "人工基线 40+ 特征的关键缺口", font_size=20, color=TEXT_DARK, bold=True)

gaps = [
    "transaction_motif / transaction_mode 完全未使用",
    "设备/IP 交易级特征未使用（洗钱强信号）",
    "图拓扑高阶特征缺失（无 PageRank 等）",
    "销户模块完全未实现（快速销户是关键信号）",
    "时间窗口仅 1H（无日/周/月维度）",
    "领域知识特征缺失（整数金额、限额利用率）",
    "行为序列模式缺失（沉睡-爆发等）",
    "对手风险传播缺失",
]

card = add_shape(slide, Inches(7.0), Inches(2.1), Inches(5.5), Inches(4.6), CARD_LIGHT, ACCENT_RED)
add_bullet_list(slide, Inches(7.3), Inches(2.2), Inches(5.0), Inches(4.4),
                [f"  {g}" for g in gaps], font_size=14, color=TEXT_DARK, spacing=Pt(6))

add_text(slide, Inches(7.3), Inches(6.3), Inches(5.0), Inches(0.4),
         "注意：以上缺口不告诉 LLM，让 LLM 自己发现",
         font_size=13, color=ACCENT_RED, bold=True)

add_page_number(slide, 6, TOTAL_SLIDES)

# ============================================================
# SLIDE 7: Literature Overview
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "03 参考文献总览", "6 篇论文各借鉴特定组件")

papers = [
    ("LLM-FE", "ICLR'26 WS", "搜索引擎骨架", "岛屿模型进化搜索\nBoltzmann 采样\n排名 1.54（第一）", ACCENT_BLUE),
    ("MedFeat", "arXiv'26", "SHAP 导航系统", "SHAP 引导搜索方向\n失败记忆机制\n去掉 SHAP→F1 降 9-26%", ACCENT_GREEN),
    ("CAAFE", "NeurIPS'23", "代码安全与反馈", "安全白名单\n错误反馈循环\n文本解释", ACCENT_ORANGE),
    ("OCTree", "NeurIPS'24", "可解释性增强", "决策树规则反馈\n语义增强\n可选组件", ACCENT_PURPLE),
    ("CoFEH", "ASPLOS'26", "V2 升级方向", "MCTS + 贝叶斯优化\n实现复杂度最高\n排名 1.75", MID_GRAY),
    ("Balek'25", "ML Journal", "统计验证方法", "Bootstrap 卡方检验\nCramer's V\n人工评估协议", ACCENT_RED),
]

for i, (name, venue, role, desc, color) in enumerate(papers):
    row = i // 3
    col = i % 3
    x = Inches(0.5 + col * 4.2)
    y = Inches(1.5 + row * 2.7)

    header = add_rect(slide, x, y, Inches(3.9), Inches(0.65), color)
    header.line.fill.background()
    add_text(slide, x + Inches(0.15), y + Inches(0.02), Inches(2.2), Inches(0.35),
             name, font_size=18, color=WHITE, bold=True)
    add_text(slide, x + Inches(0.15), y + Inches(0.33), Inches(3.6), Inches(0.3),
             f"{venue}  |  {role}", font_size=12, color=RGBColor(0xEE, 0xEE, 0xEE))

    card = add_shape(slide, x, y + Inches(0.65), Inches(3.9), Inches(1.8), CARD_LIGHT, BORDER_LIGHT)
    add_text(slide, x + Inches(0.2), y + Inches(0.8), Inches(3.5), Inches(1.5),
             desc, font_size=14, color=TEXT_DARK)

add_page_number(slide, 7, TOTAL_SLIDES)

# ============================================================
# SLIDE 8: LLM-FE + MedFeat deep dive
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "03 核心借鉴：LLM-FE + MedFeat", "搜索效率 + 搜索方向的融合")

# LLM-FE side
add_text(slide, Inches(0.5), Inches(1.5), Inches(6), Inches(0.5),
         "LLM-FE：岛屿模型进化搜索", font_size=20, color=ACCENT_BLUE, bold=True)
card = add_shape(slide, Inches(0.5), Inches(2.0), Inches(6), Inches(4.5), CARD_LIGHT, ACCENT_BLUE)

algo_text = (
    "算法流程：\n"
    "1. 维护 m=3 个岛屿（独立种群）\n"
    "2. 每轮从岛屿中 Boltzmann 采样选 k=2 示例\n"
    "   P_i = exp(score_i / τ) / Σ exp(score / τ)\n"
    "3. 示例作为 in-context 输入给 LLM\n"
    "4. LLM 生成 b=3 个候选特征代码\n"
    "5. 执行、评估、放回岛屿\n"
    "6. 岛屿间迁移最优特征\n\n"
    "实验数据：\n"
    "• 27 个 OpenML 数据集排名 1.54（第一）\n"
    "• 去掉领域知识 → 准确率 -26%\n"
    "• 去掉进化精炼 → 准确率 -31%\n"
    "• Llama-3.1-8B 和 GPT-3.5 均有效"
)
add_text(slide, Inches(0.8), Inches(2.15), Inches(5.5), Inches(4.2),
         algo_text, font_size=14, color=TEXT_DARK)

# MedFeat side
add_text(slide, Inches(6.8), Inches(1.5), Inches(6), Inches(0.5),
         "MedFeat：SHAP 引导的特征生成", font_size=20, color=ACCENT_GREEN, bold=True)
card = add_shape(slide, Inches(6.8), Inches(2.0), Inches(6), Inches(4.5), CARD_LIGHT, ACCENT_GREEN)

algo_text2 = (
    "算法流程：\n"
    "1. 每轮训练 XGBoost，计算 SHAP 值\n"
    "   s_i = mean(|SHAP_values[:, i]|)\n"
    "2. 归一化为概率分布\n"
    "3. 构建反馈：top-10 特征 + 弱势维度\n"
    "4. LLM 根据 SHAP 反馈生成新特征\n"
    "5. 接受条件：ΔLoss ≥ baseline + β(0.01)\n"
    "6. 失败记忆 → 防止重复生成\n\n"
    "实验数据：\n"
    "• 去掉 SHAP 引导 → F1 下降 9-26%\n"
    "• 去掉模型感知 → F1 最高下降 35.9%\n"
    "• SHAP 引导是最关键组件"
)
add_text(slide, Inches(7.1), Inches(2.15), Inches(5.5), Inches(4.2),
         algo_text2, font_size=14, color=TEXT_DARK)

# Bottom: Our fusion
card = add_shape(slide, Inches(0.5), Inches(6.65), Inches(12.3), Inches(0.55), SECTION_BG, ACCENT_BLUE)
add_text(slide, Inches(0.9), Inches(6.7), Inches(11.5), Inches(0.5),
         "我们的融合：LLM-FE 的维度岛屿并行 + MedFeat 的 SHAP 引导 + 交易模式说明 = 高效 + 有方向 + 有分工的搜索",
         font_size=15, color=ACCENT_BLUE, bold=True)

add_page_number(slide, 8, TOTAL_SLIDES)

# ============================================================
# SLIDE 9: System Architecture - 5 Components
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "04 系统架构：五大组件", "各组件如何组合")

components = [
    ("1", "领域知识+\n交易模式说明", "专家访谈 + 200+识别点\n+ 8种交易模式(浦发)\n→ 注入 System Prompt", ACCENT_RED),
    ("2", "维度岛屿搜索\n（改进LLM-FE）", "4维度岛屿并行搜索\n金额/时间/行为/主体\nBoltzmann + 失败记忆", ACCENT_BLUE),
    ("3", "代码执行沙盒\n（CAAFE）", "安全白名单\n语法检查 → 执行\n错误反馈给 LLM", ACCENT_GREEN),
    ("4", "SHAP 导航\n（MedFeat）", "XGBoost + SHAP 重要性\n接受条件：ΔPR-AUC ≥ β\n反馈有价值/冗余维度", ACCENT_ORANGE),
    ("5", "统计验证层\n（Balek）", "Bootstrap 卡方检验\nCramer's V + 互信息\n五篇论文无一使用", ACCENT_PURPLE),
]

for i, (num, name, desc, color) in enumerate(components):
    x = Inches(0.3 + i * 2.58)
    y = Inches(1.6)

    num_circle = add_shape(slide, x + Inches(0.85), y, Inches(0.55), Inches(0.55), color)
    num_circle.line.fill.background()
    add_text(slide, x + Inches(0.85), y + Inches(0.05), Inches(0.55), Inches(0.45),
             num, font_size=20, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

    card = add_shape(slide, x, y + Inches(0.7), Inches(2.35), Inches(3.2), CARD_LIGHT, color)
    add_text(slide, x + Inches(0.15), y + Inches(0.85), Inches(2.1), Inches(0.9),
             name, font_size=15, color=color, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.15), y + Inches(1.8), Inches(2.1), Inches(1.8),
             desc, font_size=13, color=TEXT_DARK, alignment=PP_ALIGN.CENTER)

    if i < 4:
        add_text(slide, x + Inches(2.35), y + Inches(1.8), Inches(0.25), Inches(0.5),
                 "→", font_size=22, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

# Combination logic
add_text(slide, Inches(0.5), Inches(5.2), Inches(12), Inches(0.5),
         "组合逻辑", font_size=20, color=TEXT_DARK, bold=True)

logic_items = [
    ("LLM-FE 原论文：岛屿随机分组，搜索效率高但方向盲目 → 我们改为按风险维度划分 4 岛屿", ACCENT_BLUE),
    ("MedFeat 提供 SHAP 引导，弱势维度天然对应到具体岛屿应该加强", ACCENT_GREEN),
    ("8 种交易模式说明（浦发文档）让 LLM 理解数据生成逻辑，大幅降低猜测成本", ACCENT_RED),
    ("CAAFE 保障代码安全执行 + Balek 填补统计验证缺口（五篇论文无一使用）", ACCENT_ORANGE),
]

for i, (text, color) in enumerate(logic_items):
    y = Inches(5.7 + i * 0.33)
    bar = add_rect(slide, Inches(0.5), y, Inches(0.08), Inches(0.25), color)
    bar.line.fill.background()
    add_text(slide, Inches(0.8), y - Inches(0.02), Inches(11.5), Inches(0.3),
             text, font_size=13, color=TEXT_DARK)

add_page_number(slide, 9, TOTAL_SLIDES)

# ============================================================
# SLIDE 10: Core Process - Phase 0
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "05 核心流程：Phase 0 输入准备", "一次性准备工作")

steps = [
    ("Step 1", "运行 SimECNY\n生成三表 CSV", "开户/交易/销户\n100K 钱包\n~5% 异常率", ACCENT_BLUE),
    ("Step 2", "提取 Schema\n+ 统计摘要", "字段名、类型、取值范围\n不发原始数据给 LLM\n隐私保护", ACCENT_GREEN),
    ("Step 3", "结构化\n领域知识", "8种交易模式说明(~1200)\n专家访谈+识别点(~1800)\n总计 ~4000 token", ACCENT_ORANGE),
    ("Step 4", "提取人工\n基线清单", "40+ 特征名称列表\n告诉 LLM 不要重复\n明确已有覆盖", ACCENT_PURPLE),
    ("Step 5", "种子规则\n→ Python 示例", "15 条检测规则\n转为函数示例\n用于冷启动", ACCENT_RED),
    ("Step 6", "组装\nSystem Prompt", "~5100 token\n分层结构\n16K+ 模型宽裕", ACCENT_BLUE),
]

for i, (step, title, desc, color) in enumerate(steps):
    row = i // 3
    col = i % 3
    x = Inches(0.5 + col * 4.2)
    y = Inches(1.5 + row * 2.7)

    header = add_rect(slide, x, y, Inches(3.9), Inches(0.55), color)
    header.line.fill.background()
    add_text(slide, x + Inches(0.15), y + Inches(0.08), Inches(3.6), Inches(0.4),
             f"{step}: {title.split(chr(10))[0]}", font_size=16, color=WHITE, bold=True)

    card = add_shape(slide, x, y + Inches(0.55), Inches(3.9), Inches(1.9), CARD_LIGHT, BORDER_LIGHT)
    add_text(slide, x + Inches(0.2), y + Inches(0.7), Inches(3.5), Inches(0.7),
             title, font_size=15, color=color, bold=True)
    add_text(slide, x + Inches(0.2), y + Inches(1.4), Inches(3.5), Inches(0.9),
             desc, font_size=13, color=TEXT_DARK)

add_page_number(slide, 10, TOTAL_SLIDES)

# ============================================================
# SLIDE 11: Core Process - Phase 1-2 Search Loop
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "05 核心流程：Phase 1-2 维度岛屿搜索循环", "4 维度岛屿 × 3 候选 × 最多 10 轮")

steps_data = [
    ("Step 1\n生成候选", ACCENT_BLUE,
     "4 个维度岛屿各自：\n"
     "• 金额/时间/行为/主体\n"
     "• Boltzmann采样 + 维度引导\n"
     "• 调用 LLM 生成 b=3 候选\n"
     "→ 本轮共 12 个候选特征"),
    ("Step 2\n执行与安全", ACCENT_GREEN,
     "对每个候选：\n"
     "• 安全白名单检查\n"
     "• 沙盒执行(30s/2GB)\n"
     "• 输出检查：非空/非全0/\n"
     "  非全1/NaN<50%\n"
     "• 失败→反馈LLM重试(2次)"),
    ("Step 3\n质量评估", ACCENT_ORANGE,
     "L2 统计验证：\n"
     "• MI > 0.01 / |corr| < 0.9\n"
     "L3 模型验证：\n"
     "• XGBoost 5折CV\n"
     "• ΔPR-AUC ≥ 0.005\n"
     "综合打分(归一化后加权)"),
    ("Step 4\n记忆更新", ACCENT_PURPLE,
     "接受 → 岛屿成功记忆库\n"
     "拒绝 → 失败记忆库\n"
     "岛屿间迁移：\n"
     "• 最优特征广播所有岛屿\n"
     "SHAP 反馈（给下轮）：\n"
     "• top-10 + 弱势维度提示"),
]

for i, (title, color, desc) in enumerate(steps_data):
    x = Inches(0.3 + i * 3.25)
    header = add_rect(slide, x, Inches(1.5), Inches(3.0), Inches(0.55), color)
    header.line.fill.background()
    add_text(slide, x + Inches(0.1), Inches(1.52), Inches(2.8), Inches(0.5),
             title.replace('\n', ' '), font_size=15, color=WHITE, bold=True)

    card = add_shape(slide, x, Inches(2.05), Inches(3.0), Inches(3.6), CARD_LIGHT, BORDER_LIGHT)
    add_text(slide, x + Inches(0.15), Inches(2.15), Inches(2.7), Inches(3.4),
             desc, font_size=13, color=TEXT_DARK)

    if i < 3:
        add_text(slide, x + Inches(3.0), Inches(3.5), Inches(0.3), Inches(0.5),
                 "→", font_size=22, color=MID_GRAY)

# Convergence
card = add_shape(slide, Inches(0.3), Inches(5.9), Inches(12.7), Inches(0.5), SECTION_BG, ACCENT_BLUE)
add_text(slide, Inches(0.8), Inches(5.95), Inches(12), Inches(0.4),
         "收敛判断：连续 2 轮无新特征通过 → 停止    |    SHAP 全局共享 + 维度引导岛屿专属 + Boltzmann 岛屿独立",
         font_size=14, color=ACCENT_BLUE, bold=True)

# Key numbers
add_text(slide, Inches(0.3), Inches(6.6), Inches(12.7), Inches(0.5),
         "关键参数：m=4 维度岛屿  |  b=3 候选/岛/轮  |  k=2 示例  |  T<=10 轮  |  最多 120 次 LLM 调用  |  beta=0.01",
         font_size=14, color=MID_GRAY)

add_page_number(slide, 11, TOTAL_SLIDES)

# ============================================================
# SLIDE 12: SHAP + Island Fusion Detail
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "05 维度岛屿与 SHAP 融合细节", "维度分工 + 全局引导 + 局部多样性")

# Left: flow
add_text(slide, Inches(0.5), Inches(1.5), Inches(6), Inches(0.5),
         "每轮迭代的执行流程", font_size=20, color=TEXT_DARK, bold=True)

flow_steps = [
    ("1. SHAP 分析（全局共享）", "训练 XGBoost → 计算 SHAP → 生成 top-10 + 弱势维度\n所有岛屿共享 SHAP 反馈，弱势维度对应到具体岛屿", ACCENT_BLUE),
    ("2. 维度引导 + 岛屿独立采样", "每个岛屿附加维度引导词(如'重点探索金额频次')\nBoltzmann 独立采样 k=2 示例 → 岛屿 prompt 不同", ACCENT_GREEN),
    ("3. Prompt = 维度引导(专属) + SHAP(共享) + 示例(独立)", "User Prompt 三部分组合\n维度引导让 LLM 聚焦，SHAP 提供全局方向", ACCENT_ORANGE),
    ("4. 跨维度迁移", "本轮最高分特征 → 复制到所有岛屿记忆库\n好经验跨维度传播，如金额发现启发主体关联", ACCENT_PURPLE),
]

for i, (title, desc, color) in enumerate(flow_steps):
    y = Inches(2.1 + i * 1.25)
    bar = add_rect(slide, Inches(0.5), y, Inches(0.08), Inches(1.0), color)
    bar.line.fill.background()
    add_text(slide, Inches(0.8), y, Inches(6), Inches(0.35),
             title, font_size=15, color=color, bold=True)
    add_text(slide, Inches(0.8), y + Inches(0.35), Inches(6), Inches(0.6),
             desc, font_size=13, color=TEXT_DARK)

# Right: why this design
add_text(slide, Inches(7.2), Inches(1.5), Inches(5.5), Inches(0.5),
         "为什么这样设计", font_size=20, color=TEXT_DARK, bold=True)

card = add_shape(slide, Inches(7.2), Inches(2.1), Inches(5.5), Inches(4.5), CARD_LIGHT, BORDER_LIGHT)

reasons = (
    "三层改进：\n\n"
    "1. 维度划分（我们的改进）\n"
    "   原论文岛屿随机分组 → 我们按风险维度分\n"
    "   金额频次 / 时间地域 / 行为序列 / 主体关联\n\n"
    "2. SHAP 全局引导（MedFeat）\n"
    "   弱势维度 → 对应岛屿加强探索\n\n"
    "3. 交易模式说明（浦发文档）\n"
    "   LLM 知道每种风险模式的参数\n"
    "   → 精准构造交叉特征\n\n"
    "= 高效 + 有方向 + 有分工"
)
add_text(slide, Inches(7.5), Inches(2.2), Inches(5.0), Inches(4.2),
         reasons, font_size=14, color=TEXT_DARK)

add_page_number(slide, 12, TOTAL_SLIDES)

# ============================================================
# SLIDE 13: Prompt Design
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "06 Prompt 设计", "System Prompt + User Prompt 分层结构")

# System prompt
add_text(slide, Inches(0.5), Inches(1.5), Inches(6.2), Inches(0.5),
         "System Prompt（固定, ~5100 token）", font_size=18, color=ACCENT_BLUE, bold=True)
card = add_shape(slide, Inches(0.5), Inches(2.0), Inches(6.2), Inches(4.8), CARD_LIGHT, ACCENT_BLUE)

sp_text = (
    "## 角色定义\n"
    "你是反洗钱领域的特征工程专家...\n\n"
    "## 数据结构（Schema + 统计摘要）\n"
    "开户表 16 字段 | 交易表 20 字段 | 销户表 6 字段\n"
    "三表关联关系说明\n\n"
    "## 交易模式说明（浦发，~1200 token）\n"
    "3 种正常模式 + 5 种风险模式的详细参数\n\n"
    "## 领域知识\n"
    "4 类洗钱模式（~800 token）\n"
    "38 个风险指标（~600 token）\n"
    "36 条核心识别点（~1000 token）\n\n"
    "## 已有特征清单（40+, 不要重复）\n\n"
    "## 约束\n"
    "30s 执行 / 白名单模块 / 禁用标签列"
)
add_text(slide, Inches(0.7), Inches(2.1), Inches(5.8), Inches(4.6),
         sp_text, font_size=13, color=TEXT_DARK)

# User prompt
add_text(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(0.5),
         "User Prompt（每轮动态, ~1500 token）", font_size=18, color=ACCENT_GREEN, bold=True)
card = add_shape(slide, Inches(7.0), Inches(2.0), Inches(5.8), Inches(2.3), CARD_LIGHT, ACCENT_GREEN)
up_text = (
    "常规轮（第 2+ 轮）：\n"
    "• 维度引导词(岛屿专属方向)\n"
    "• 上轮最佳特征 top-3\n"
    "• SHAP：top-10 + 弱势维度\n"
    "• 上轮失败特征 + 原因\n"
    "• 生成 3 个本维度新特征"
)
add_text(slide, Inches(7.2), Inches(2.1), Inches(5.4), Inches(2.1),
         up_text, font_size=14, color=TEXT_DARK)

card = add_shape(slide, Inches(7.0), Inches(4.5), Inches(5.8), Inches(2.3), CARD_LIGHT, ACCENT_ORANGE)
cs_text = (
    "冷启动（第 1 轮）：\n"
    "• 无 SHAP 反馈（首轮）\n"
    "• 维度引导词 + 种子规则示例\n"
    "• 参考交易模式说明 + 洗钱模式\n"
    "• 围绕本岛屿维度出发"
)
add_text(slide, Inches(7.2), Inches(4.6), Inches(5.4), Inches(2.1),
         cs_text, font_size=14, color=TEXT_DARK)

# Token budget
add_text(slide, Inches(0.5), Inches(7.0) - Inches(0.15), Inches(12.3), Inches(0.4),
         "总计 ~6600 token  →  16K+ context 宽裕  |  8K 可压缩识别点(36->18条)降至 ~5600  |  32K+ 可放入更多原文",
         font_size=14, color=MID_GRAY)

add_page_number(slide, 13, TOTAL_SLIDES)

# ============================================================
# SLIDE 14: LLM Output Format + Pluggable Interface
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "06 LLM 输出格式与可插拔接口", "标准化输出 + 模型无关设计")

# Output format
add_text(slide, Inches(0.5), Inches(1.5), Inches(6.2), Inches(0.5),
         "LLM 输出 JSON 格式", font_size=20, color=TEXT_DARK, bold=True)
card = add_shape(slide, Inches(0.5), Inches(2.0), Inches(6.2), Inches(4.5), CARD_LIGHT, ACCENT_BLUE)

json_text = (
    '[\n'
    '  {\n'
    '    "feature_name": "night_large_fan_out_ratio",\n'
    '    "description": "夜间大额扇出交易占比",\n'
    '    "code": "def gen(tx, open_df, close_df):\\n...",\n'
    '    "explanation": "线上跑分模式特征：...",\n'
    '    "domain_source": "专家访谈-模式3-线上跑分",\n'
    '    "expected_signal": "洗钱钱包此比值显著高"\n'
    '  },\n'
    '  { ... },\n'
    '  { ... }\n'
    ']\n\n'
    "每个特征包含 6 个字段：\n"
    "• feature_name — 唯一标识\n"
    "• code — 可直接执行的 Python 函数\n"
    "• explanation — 可解释性说明（WHY）\n"
    "• domain_source — 领域知识来源引用"
)
add_text(slide, Inches(0.7), Inches(2.1), Inches(5.8), Inches(4.3),
         json_text, font_size=13, color=TEXT_DARK)

# Pluggable interface
add_text(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(0.5),
         "LLM 可插拔接口", font_size=20, color=TEXT_DARK, bold=True)
card = add_shape(slide, Inches(7.0), Inches(2.0), Inches(5.8), Inches(2.5), CARD_LIGHT, ACCENT_GREEN)
interface_text = (
    "class LLMInterface:\n"
    "    def generate(self, system_prompt,\n"
    "                 user_prompt) -> str\n\n"
    "class OpenAILLM(LLMInterface):  # API\n"
    "class LocalLLM(LLMInterface):   # 本地"
)
add_text(slide, Inches(7.2), Inches(2.1), Inches(5.4), Inches(2.3),
         interface_text, font_size=14, color=TEXT_DARK)

add_text(slide, Inches(7.0), Inches(4.8), Inches(5.8), Inches(0.5),
         "模型选择依据", font_size=18, color=TEXT_DARK, bold=True)
card = add_shape(slide, Inches(7.0), Inches(5.3), Inches(5.8), Inches(1.7), CARD_LIGHT, ACCENT_ORANGE)
model_text = (
    "LLM-FE 论文验证：\n"
    "• Llama-3.1-8B 和 GPT-3.5-Turbo 均有效\n"
    "• 框架对底层 LLM 不敏感\n"
    "• 搜索智能来自进化框架，不是 LLM 本身\n"
    "→ 具体模型根据实际可用资源确定"
)
add_text(slide, Inches(7.2), Inches(5.4), Inches(5.4), Inches(1.5),
         model_text, font_size=14, color=TEXT_DARK)

add_page_number(slide, 14, TOTAL_SLIDES)

# ============================================================
# SLIDE 15: Validation L1-L2
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "07 四级验证框架（上）：L1 代码 + L2 统计", "逐级过滤，只保留高质量特征")

# L1
add_text(slide, Inches(0.5), Inches(1.5), Inches(6.2), Inches(0.5),
         "L1: 代码验证（来自 CAAFE）", font_size=20, color=ACCENT_BLUE, bold=True)
card = add_shape(slide, Inches(0.5), Inches(2.0), Inches(6.2), Inches(4.5), CARD_LIGHT, ACCENT_BLUE)

l1_checks = [
    ("安全白名单", "仅允许 pandas/numpy/scipy/sklearn\n禁止 os/sys/subprocess/open/exec/eval/网络"),
    ("执行成功", "30 秒内无异常退出\n失败 → 错误信息反馈 LLM 重试（最多 2 次）"),
    ("输出格式", "返回 DataFrame\n含 wallet_id 或 tx_id 列"),
    ("数据质量", "非全 0、非全 1\nNaN 不超过 50%"),
]

for i, (check, desc) in enumerate(l1_checks):
    y = Inches(2.15 + i * 1.05)
    add_text(slide, Inches(0.8), y, Inches(1.8), Inches(0.35),
             check, font_size=15, color=ACCENT_BLUE, bold=True)
    add_text(slide, Inches(2.6), y, Inches(3.8), Inches(0.9),
             desc, font_size=13, color=TEXT_DARK)
    if i < 3:
        add_rect(slide, Inches(0.8), y + Inches(0.9), Inches(5.6), Inches(0.01), BORDER_LIGHT)

# L2
add_text(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(0.5),
         "L2: 统计验证（来自 Balek）", font_size=20, color=ACCENT_GREEN, bold=True)
card = add_shape(slide, Inches(7.0), Inches(2.0), Inches(5.8), Inches(4.5), CARD_LIGHT, ACCENT_GREEN)

l2_text = (
    "互信息检验\n"
    "MI(feature, label) > 0.01\n"
    "→ 特征与标签有最低限度的信息关联\n\n"
    "冗余度检验\n"
    "与每个已有特征 |corr| < 0.9\n"
    "→ 避免重复特征，保证增量信息\n\n"
    "Cramer's V（分类特征）\n"
    "V > 0.05\n"
    "→ 不受样本量影响的效应大小度量\n\n"
    "Bootstrap 卡方检验\n"
    "2500 次重采样，mean(p) < 0.05\n"
    "→ 统计显著性的稳健估计"
)
add_text(slide, Inches(7.2), Inches(2.1), Inches(5.4), Inches(4.3),
         l2_text, font_size=14, color=TEXT_DARK)

# Bottom note
card = add_shape(slide, Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.5), SECTION_BG, ACCENT_RED)
add_text(slide, Inches(0.9), Inches(6.75), Inches(11.5), Inches(0.4),
         "关键发现：五篇参考论文（LLM-FE, CAAFE, MedFeat, OCTree, CoFEH）无一使用统计检验。我们用 Balek 的方法填补这个缺口。",
         font_size=14, color=ACCENT_RED, bold=True)

add_page_number(slide, 15, TOTAL_SLIDES)

# ============================================================
# SLIDE 16: Validation L3-L4
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "07 四级验证框架（下）：L3 模型 + L4 稳定性", "PR-AUC 增量 + Bootstrap 稳定性")

# L3
add_text(slide, Inches(0.5), Inches(1.5), Inches(6.2), Inches(0.5),
         "L3: 模型验证（来自 MedFeat）", font_size=20, color=ACCENT_ORANGE, bold=True)
card = add_shape(slide, Inches(0.5), Inches(2.0), Inches(6.2), Inches(3.5), CARD_LIGHT, ACCENT_ORANGE)

l3_text = (
    "流程：\n"
    "1. 基线：40 人工特征 → XGBoost → 5 折 CV\n"
    "   → PR-AUC_baseline\n\n"
    "2. 增量：40 人工 + 新特征 → XGBoost → 5 折 CV\n"
    "   → PR-AUC_augmented\n\n"
    "3. 接受条件：\n"
    "   ΔPR-AUC = augmented - baseline ≥ 0.005\n\n"
    "XGBoost 参数：\n"
    "  scale_pos_weight=19（正样本~5%, 19:1）\n"
    "  eval_metric='aucpr'"
)
add_text(slide, Inches(0.7), Inches(2.1), Inches(5.8), Inches(3.3),
         l3_text, font_size=14, color=TEXT_DARK)

# L4
add_text(slide, Inches(7.0), Inches(1.5), Inches(5.8), Inches(0.5),
         "L4: 稳定性验证（最终阶段）", font_size=20, color=ACCENT_PURPLE, bold=True)
card = add_shape(slide, Inches(7.0), Inches(2.0), Inches(5.8), Inches(3.5), CARD_LIGHT, ACCENT_PURPLE)

l4_text = (
    "流程：\n"
    "1. 10 次 Bootstrap 重采样\n"
    "2. 每次：训练 XGBoost → 计算 SHAP\n"
    "   → 记录 top-20 特征排名\n\n"
    "3. 计算 Nogueira 稳定性指数\n"
    "   衡量 10 次结果中特征排名一致性\n\n"
    "4. 要求：stability_index > 0.8\n\n"
    "注意：L4 不在搜索循环中执行\n"
    "（成本太高），仅在最终特征集确定后"
)
add_text(slide, Inches(7.2), Inches(2.1), Inches(5.4), Inches(3.3),
         l4_text, font_size=14, color=TEXT_DARK)

# Why PR-AUC
add_text(slide, Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.5),
         "为什么用 PR-AUC 而非 ROC-AUC", font_size=18, color=TEXT_DARK, bold=True)
card = add_shape(slide, Inches(0.5), Inches(6.3), Inches(12.3), Inches(0.9), CARD_LIGHT, BORDER_LIGHT)
add_text(slide, Inches(0.8), Inches(6.35), Inches(11.7), Inches(0.8),
         "AML 数据正样本 ~5%，ROC-AUC 会过度乐观。五篇论文用的 Accuracy/ROC-AUC/错误率 在不平衡场景下都有问题。\n"
         "PR-AUC 直接衡量「在找出的可疑交易中，有多少是真正可疑的」—— 这才是反洗钱场景的核心关注点。",
         font_size=14, color=TEXT_DARK)

add_page_number(slide, 16, TOTAL_SLIDES)

# ============================================================
# SLIDE 17: Final Experiments
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "07 最终增量贡献实验设计", "5 组对比实验验证 LLM 特征的真实价值")

experiments = [
    ("实验 1", "基线", "40 个人工特征\n→ XGBoost\n→ PR-AUC_baseline", ACCENT_BLUE),
    ("实验 2", "增量", "40 人工 + N 个 LLM 特征\n→ XGBoost\n→ PR-AUC_augmented", ACCENT_GREEN),
    ("实验 3", "独立", "仅 N 个 LLM 特征\n→ XGBoost\n→ PR-AUC_llm_only", ACCENT_ORANGE),
    ("实验 4", "随机对照", "40 人工 + N 个随机特征\n→ XGBoost\n→ PR-AUC_random", MID_GRAY),
    ("实验 5", "消融", "逐个移除 LLM 特征\n→ 观察 PR-AUC 变化\n→ 识别最关键特征", ACCENT_PURPLE),
]

for i, (num, name, desc, color) in enumerate(experiments):
    x = Inches(0.3 + i * 2.58)
    header = add_rect(slide, x, Inches(1.5), Inches(2.35), Inches(0.85), color)
    header.line.fill.background()
    add_text(slide, x + Inches(0.1), Inches(1.52), Inches(2.15), Inches(0.4),
             num, font_size=14, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.1), Inches(1.9), Inches(2.15), Inches(0.4),
             name, font_size=18, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

    card = add_shape(slide, x, Inches(2.35), Inches(2.35), Inches(2.0), CARD_LIGHT, BORDER_LIGHT)
    add_text(slide, x + Inches(0.15), Inches(2.5), Inches(2.1), Inches(1.7),
             desc, font_size=14, color=TEXT_DARK, alignment=PP_ALIGN.CENTER)

# Expected results
add_text(slide, Inches(0.5), Inches(4.8), Inches(12), Inches(0.5),
         "报告指标", font_size=20, color=TEXT_DARK, bold=True)

card = add_shape(slide, Inches(0.5), Inches(5.3), Inches(12.3), Inches(1.8), CARD_LIGHT, BORDER_LIGHT)
metrics_text = (
    "• 增量 PR-AUC = PR-AUC_augmented - PR-AUC_baseline（应 > 0，核心指标）\n"
    "• 独立 PR-AUC = PR-AUC_llm_only（LLM 特征独立的预测能力）\n"
    "• 随机对照 = PR-AUC_random - PR-AUC_baseline（应 ≈ 0，排除过拟合假象）\n"
    "• 各 LLM 特征的 SHAP 排名（在完整模型中的重要性）\n"
    "• 每个 LLM 特征的消融贡献（移除后 PR-AUC 下降幅度）\n"
    "• 所有实验使用 5 折分层 CV，报告均值 ± 标准差"
)
add_text(slide, Inches(0.8), Inches(5.4), Inches(11.7), Inches(1.6),
         metrics_text, font_size=14, color=TEXT_DARK)

add_page_number(slide, 17, TOTAL_SLIDES)

# ============================================================
# SLIDE 18: Innovation Points
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "08 核心创新点", "5 个论文贡献")

innovations = [
    ("1", "领域知识+交易模式\n增强的LLM特征工程",
     "首次将专家知识 + 数据集交易模式说明（8种模式详细参数）系统性注入 LLM 特征生成。\n现有方法：LLM-FE 仅靠特征名、MedFeat 无领域知识、CAAFE 仅靠数据集描述。",
     ACCENT_BLUE),
    ("2", "多表关联\n特征发现",
     "首次将 LLM 特征工程扩展到多表关系型数据（开户 + 交易 + 销户三表关联）。\n所有 6 篇参考论文均仅支持单表输入。",
     ACCENT_GREEN),
    ("3", "维度岛屿\n进化搜索",
     "改进 LLM-FE 的随机岛屿为按风险维度划分的 4 岛屿（金额/时间/行为/主体），\n结合 SHAP 引导，实现高效 + 有方向 + 有分工的搜索。原论文岛屿无差异化。",
     ACCENT_ORANGE),
    ("4", "四级特征\n验证框架",
     "代码 → 统计 → 模型 → 稳定性，超越现有论文的验证深度。\n关键：五篇论文无一使用统计检验，我们用 Balek 方法填补。",
     ACCENT_PURPLE),
    ("5", "金融领域\n可解释特征",
     "每个 LLM 特征附带领域依据说明和知识来源引用，\n满足金融监管的可解释性要求。",
     ACCENT_RED),
]

for i, (num, title, desc, color) in enumerate(innovations):
    y = Inches(1.5 + i * 1.15)
    num_box = add_rect(slide, Inches(0.5), y, Inches(0.5), Inches(0.5), color)
    num_box.line.fill.background()
    add_text(slide, Inches(0.5), y + Inches(0.05), Inches(0.5), Inches(0.45),
             num, font_size=20, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, Inches(1.2), y + Inches(0.03), Inches(2.5), Inches(1.0),
             title, font_size=15, color=color, bold=True)
    add_text(slide, Inches(3.8), y + Inches(0.05), Inches(9.0), Inches(1.0),
             desc, font_size=14, color=TEXT_DARK)

    if i < 4:
        add_rect(slide, Inches(0.5), y + Inches(1.05), Inches(12.3), Inches(0.01), BORDER_LIGHT)

add_page_number(slide, 18, TOTAL_SLIDES)

# ============================================================
# SLIDE 19: Feasibility Assessment
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "09 可行性评估与风险分析", "总体判断：方案可行，3 个中等风险需关注")

# High feasibility
add_text(slide, Inches(0.5), Inches(1.5), Inches(4.0), Inches(0.5),
         "高可行性", font_size=18, color=ACCENT_GREEN, bold=True)
card = add_shape(slide, Inches(0.5), Inches(1.95), Inches(4.0), Inches(2.5), CARD_LIGHT, ACCENT_GREEN)
hi_items = [
    "领域知识注入：数据齐全",
    "代码执行沙盒：成熟方案",
    "L1-L2 验证：标准库实现",
    "三表 merge：代码已写好",
    "可解释性：LLM 天然擅长",
]
add_bullet_list(slide, Inches(0.7), Inches(2.05), Inches(3.6), Inches(2.3),
                hi_items, font_size=13, color=TEXT_DARK, spacing=Pt(5))

# Medium risks
add_text(slide, Inches(4.8), Inches(1.5), Inches(8.0), Inches(0.5),
         "中等风险（需重点关注）", font_size=18, color=ACCENT_ORANGE, bold=True)

risks = [
    ("风险 1: LLM 代码生成质量", "三表比单表复杂，6 篇论文仅验证单表\n建议：第 1 轮手动测试 3-5 个候选", ACCENT_RED),
    ("风险 2: PR-AUC 增量阈值", "取决于基线预测能力，当前阈值可能过严或过松\n建议：先跑基线 PR-AUC 再定阈值", ACCENT_ORANGE),
    ("风险 3: 打分公式量纲", "SHAP_rank(整数) vs MI(小数) 量纲不同\n建议：min-max 归一化到 [0,1] 后再加权", ACCENT_PURPLE),
]

for i, (title, desc, color) in enumerate(risks):
    y = Inches(1.95 + i * 0.85)
    card = add_shape(slide, Inches(4.8), y, Inches(8.0), Inches(0.75), CARD_LIGHT, color)
    add_text(slide, Inches(5.0), y + Inches(0.05), Inches(7.5), Inches(0.3),
             title, font_size=14, color=color, bold=True)
    add_text(slide, Inches(5.0), y + Inches(0.35), Inches(7.5), Inches(0.4),
             desc, font_size=12, color=TEXT_DARK)

# Low risk
add_text(slide, Inches(0.5), Inches(4.7), Inches(12.3), Inches(0.5),
         "低风险", font_size=18, color=MID_GRAY, bold=True)
card = add_shape(slide, Inches(0.5), Inches(5.15), Inches(12.3), Inches(0.8), CARD_LIGHT, BORDER_LIGHT)
add_text(slide, Inches(0.8), Inches(5.2), Inches(11.7), Inches(0.7),
         "• SimECNY 数据生成：销户模块需修复硬编码路径（已有方案）\n"
         "• LLM 调用成本：最多 120 次调用，API $7-20 / 本地 ~60min，整个流程 1-2 小时可完成",
         font_size=14, color=TEXT_DARK)

# Three must-do
add_text(slide, Inches(0.5), Inches(6.2), Inches(12.3), Inches(0.5),
         "实施前必须完成的三件事", font_size=18, color=ACCENT_RED, bold=True)
card = add_shape(slide, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.65), SECTION_BG, ACCENT_RED)
add_text(slide, Inches(0.8), Inches(6.65), Inches(11.7), Inches(0.55),
         "1. 先生成数据、跑基线 PR-AUC（决定阈值）   2. 第 1 轮手动测试 LLM 输出质量（验证 prompt）   3. 补充打分归一化实现（避免量纲问题）",
         font_size=14, color=ACCENT_RED, bold=True)

add_page_number(slide, 19, TOTAL_SLIDES)

# ============================================================
# SLIDE 20: Roadmap
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_section_header(slide, "10 实施路线图", "四阶段共约 7.5 天")

phases = [
    ("Phase 0", "数据生成", "0.5 天", ACCENT_BLUE,
     ["运行 SimECNY 三模块", "修复销户硬编码路径", "验证字段/标签/关联"]),
    ("Phase 1", "基础设施", "2 天", ACCENT_GREEN,
     ["数据加载 + 三表关联", "领域知识结构化", "种子规则转换", "沙盒 + 评估模块", "LLM 接口 + Prompt"]),
    ("Phase 2", "核心搜索", "3 天", ACCENT_ORANGE,
     ["4维度岛屿结构", "Boltzmann + 维度引导", "搜索循环主程序", "冷启动 + 跨维度迁移", "收敛判断"]),
    ("Phase 3", "验证产出", "2 天", ACCENT_PURPLE,
     ["L4 稳定性验证", "5 组增量实验", "特征字典生成", "结果报告 + 可视化"]),
]

for i, (name, desc, duration, color, tasks) in enumerate(phases):
    x = Inches(0.3 + i * 3.25)

    header = add_rect(slide, x, Inches(1.5), Inches(3.0), Inches(0.9), color)
    header.line.fill.background()
    add_text(slide, x + Inches(0.1), Inches(1.5), Inches(2.8), Inches(0.45),
             f"{name}: {desc}", font_size=17, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.1), Inches(1.95), Inches(2.8), Inches(0.4),
             duration, font_size=14, color=RGBColor(0xEE, 0xEE, 0xEE), alignment=PP_ALIGN.CENTER)

    card = add_shape(slide, x, Inches(2.4), Inches(3.0), Inches(3.0), CARD_LIGHT, BORDER_LIGHT)
    add_bullet_list(slide, x + Inches(0.2), Inches(2.55), Inches(2.6), Inches(2.8),
                    tasks, font_size=13, color=TEXT_DARK, spacing=Pt(6))

# Expected output
add_text(slide, Inches(0.5), Inches(5.7), Inches(12.3), Inches(0.5),
         "预期产出", font_size=20, color=TEXT_DARK, bold=True)

outputs = [
    ("15-30 个", "经过四级验证的\n高质量新特征", ACCENT_BLUE),
    ("特征字典", "代码+解释+SHAP\n+统计指标", ACCENT_GREEN),
    ("增量报告", "PR-AUC/F1/Recall\n5 组对比实验", ACCENT_ORANGE),
    ("验证报告", "四级验证结果\n稳定性分析", ACCENT_PURPLE),
    ("可复用框架", "可插拔 LLM\n可换数据集", ACCENT_RED),
]

for i, (title, desc, color) in enumerate(outputs):
    x = Inches(0.3 + i * 2.58)
    card = add_shape(slide, x, Inches(6.15), Inches(2.35), Inches(1.1), color, color)
    add_text(slide, x + Inches(0.1), Inches(6.18), Inches(2.15), Inches(0.4),
             title, font_size=15, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text(slide, x + Inches(0.1), Inches(6.55), Inches(2.15), Inches(0.6),
             desc, font_size=12, color=WHITE, alignment=PP_ALIGN.CENTER)

add_page_number(slide, 20, TOTAL_SLIDES)


# ── Save ──
output_path = "/home/ubuntu/Projects/大模型提取特征/LLM_Feature_Engineering_Report.pptx"
prs.save(output_path)
print(f"PPT saved to {output_path}")
