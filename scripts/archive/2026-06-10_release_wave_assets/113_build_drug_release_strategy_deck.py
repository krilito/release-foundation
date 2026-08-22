"""
113 - Build a Chinese strategy deck for the unified drug-release positioning work.

Purpose:
    Turn the existing Chinese slide packs and rendered strategy figures into
    a real PPTX draft that can be opened and presented immediately.

Consumes:
    docs/drug_release_slide_content_pack_zh_2026-05-29.md
    outputs/91_release_paper_figures/figure1/figure1_problem_framing.png
    outputs/91_release_paper_figures/figure2/figure2_plga_core.png
    outputs/91_release_paper_figures/figure4/figure4_liposome_bridge.png
    outputs/92_release_strategy_figures/stack_occupancy/stack_occupancy_panel.png
    outputs/92_release_strategy_figures/external_methods_venues/external_methods_venues_panel.png

Produces:
    outputs/113_drug_release_strategy_deck/drug_release_strategy_deck_zh_2026-05-29.pptx
    outputs/113_drug_release_strategy_deck/summary.txt
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Inches, Pt


OUTDIR = Path("outputs/113_drug_release_strategy_deck")
OUTFILE = OUTDIR / "drug_release_strategy_deck_zh_2026-05-29.pptx"

FIG1 = Path("outputs/91_release_paper_figures/figure1/figure1_problem_framing.png")
FIG2 = Path("outputs/91_release_paper_figures/figure2/figure2_plga_core.png")
FIG4 = Path("outputs/91_release_paper_figures/figure4/figure4_liposome_bridge.png")
STACK = Path("outputs/92_release_strategy_figures/stack_occupancy/stack_occupancy_panel.png")
EXTERNAL = Path("outputs/92_release_strategy_figures/external_methods_venues/external_methods_venues_panel.png")
FULL_BRIDGE = Path("outputs/92_release_strategy_figures/full_cell_bridge/full_cell_bridge_panel.png")
BOUNDARY = Path("outputs/92_release_strategy_figures/unification_boundary/unification_boundary_panel.png")
COMPETITOR = Path("outputs/92_release_strategy_figures/competitor_response/competitor_response_panel.png")


BG_DARK = RGBColor(24, 32, 61)
BG_LIGHT = RGBColor(245, 246, 248)
TEXT_DARK = RGBColor(30, 41, 59)
TEXT_LIGHT = RGBColor(250, 250, 252)
ACCENT = RGBColor(0, 140, 149)
ACCENT2 = RGBColor(202, 92, 48)
GREEN = RGBColor(38, 120, 88)
PURPLE = RGBColor(102, 77, 138)
SOFT_BLUE = RGBColor(227, 240, 248)
SOFT_ORANGE = RGBColor(247, 238, 228)
SOFT_GREEN = RGBColor(231, 244, 237)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_bg(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, x, y, w, h, text, *, font_size=20, bold=False, color=TEXT_DARK, font_name="Calibri", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name
    return box


def add_bullets(slide, x, y, w, h, items, *, font_size=18, color=TEXT_DARK):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.clear()
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.bullet = True
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
    return box


def add_header(slide, title: str, subtitle: str | None = None, *, dark=False) -> None:
    title_color = TEXT_LIGHT if dark else TEXT_DARK
    sub_color = RGBColor(226, 232, 240) if dark else RGBColor(85, 95, 120)
    add_textbox(slide, Inches(0.55), Inches(0.28), Inches(8.7), Inches(0.55), title, font_size=28, bold=True, color=title_color, font_name="Aptos Display")
    if subtitle:
        add_textbox(slide, Inches(0.58), Inches(0.78), Inches(8.2), Inches(0.42), subtitle, font_size=13, color=sub_color)


def add_footer_note(slide, text: str, *, dark=False) -> None:
    color = RGBColor(210, 216, 226) if dark else RGBColor(95, 105, 125)
    add_textbox(slide, Inches(0.58), Inches(5.10), Inches(8.7), Inches(0.28), text, font_size=9.5, color=color)


def add_card(slide, x, y, w, h, title: str, bullets: list[str], fill_rgb: RGBColor) -> None:
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    shape.line.color.rgb = RGBColor(215, 220, 228)
    add_textbox(slide, x + Inches(0.12), y + Inches(0.10), w - Inches(0.24), Inches(0.32), title, font_size=16, bold=True)
    add_bullets(slide, x + Inches(0.12), y + Inches(0.45), w - Inches(0.24), h - Inches(0.55), bullets, font_size=13)


def add_image(slide, path: Path, x, y, w, h):
    if path.exists():
        slide.shapes.add_picture(str(path), x, y, w, h)


def build_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Slide 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_DARK)
    add_header(slide, "我们不该再把这件事讲成一个更强的 PLGA predictor", "更值得占的位置，是 shared release-intelligence stack", dark=True)
    add_textbox(slide, Inches(0.72), Inches(1.45), Inches(5.2), Inches(0.6), "核心判断", font_size=16, bold=True, color=RGBColor(171, 216, 214))
    add_textbox(slide, Inches(0.72), Inches(1.88), Inches(5.8), Inches(1.0), "现在最该占的位置，不是“最好单模型”，而是\nshared release-intelligence stack。", font_size=25, bold=True, color=TEXT_LIGHT)
    add_card(slide, Inches(0.78), Inches(3.10), Inches(2.75), Inches(1.6), "不是", ["best single-material regressor", "universal solved model"], RGBColor(45, 55, 85))
    add_card(slide, Inches(3.72), Inches(3.10), Inches(2.95), Inches(1.6), "而是", ["shared partially observed benchmark", "shared posterior-family object"], RGBColor(33, 86, 92))
    add_image(slide, FIG1, Inches(7.15), Inches(1.25), Inches(5.45), Inches(4.85))
    add_footer_note(slide, "先改问题框架，再谈谁的模型更强。", dark=True)

    # Slide 2
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_LIGHT)
    add_header(slide, "外面现在不是一个对手，而是三种压力", "benchmark / platform / infrastructure")
    add_image(slide, EXTERNAL, Inches(0.55), Inches(1.10), Inches(7.3), Inches(5.65))
    add_card(slide, Inches(8.15), Inches(1.28), Inches(4.45), Inches(1.55), "Benchmark rival", ["NC / Bannigan", "explainable LAI few-shot", "更大规模 PLGA predictor"], SOFT_BLUE)
    add_card(slide, Inches(8.15), Inches(2.98), Inches(4.45), Inches(1.55), "Platform threat", ["FormulationLAI", "FormulationAI", "optimization systems"], SOFT_ORANGE)
    add_card(slide, Inches(8.15), Inches(4.68), Inches(4.45), Inches(1.55), "Infrastructure / workflow", ["liposome IVR workflow", "Scientific Data / AI-ready release data", "computational pharmaceutics"], SOFT_GREEN)
    add_footer_note(slide, "这页的作用是先拆战场，不急着说我们已经赢谁。")

    # Slide 3
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, RGBColor(250, 251, 252))
    add_header(slide, "我们当前最硬的 moat 在 L2-L3，不在平台层", "object/benchmark 层领先，L5 仍然保守")
    add_image(slide, STACK, Inches(0.55), Inches(1.18), Inches(8.0), Inches(5.55))
    add_card(slide, Inches(8.82), Inches(1.50), Inches(3.95), Inches(1.45), "最硬领先层", ["L2 partial-observation benchmark", "L3 shared posterior-family object"], SOFT_BLUE)
    add_card(slide, Inches(8.82), Inches(3.05), Inches(3.95), Inches(1.35), "相对最强外部", ["L2 +0.50", "L3 +1.00", "L5 -0.25"], SOFT_GREEN)
    add_card(slide, Inches(8.82), Inches(4.55), Inches(3.95), Inches(1.45), "一句话", ["我们赢的是 object/benchmark 层", "不是 platform/reporting 层"], SOFT_ORANGE)
    add_footer_note(slide, "不要泛泛说“更完整”，要说清楚更完整在哪一层。")

    # Slide 4
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_LIGHT)
    add_header(slide, "PLGA audited core：早期释放是真主信号，middle object 不是装饰", "当前 audited core 的最硬科学地基")
    add_image(slide, FIG2, Inches(0.52), Inches(1.18), Inches(8.65), Inches(5.75))
    add_card(slide, Inches(9.38), Inches(1.48), Inches(3.4), Inches(1.25), "关键数", ["formulation-only OOD best median R²: 0.525–0.672", "early-only OOD best median R²: 0.907–0.949"], SOFT_BLUE)
    add_card(slide, Inches(9.38), Inches(2.98), Inches(3.4), Inches(1.25), "middle object", ["theta-minus-direct OOD gain: +0.002 to +0.121", "不是 narrative add-on"], SOFT_GREEN)
    add_card(slide, Inches(9.38), Inches(4.48), Inches(3.4), Inches(1.25), "解释", ["纯配方 harder OOD 不稳", "早期释放是真主信号"], SOFT_ORANGE)
    add_footer_note(slide, "这是整套故事的科学地基，必须讲得最稳。")

    # Slide 5
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, RGBColor(249, 250, 251))
    add_header(slide, "Liposome bridge：这已经不是一个 PLGA-only 故事", "shared interface survives a first non-PLGA bridge")
    add_image(slide, FIG4, Inches(0.55), Inches(1.18), Inches(8.9), Inches(5.45))
    add_card(slide, Inches(9.72), Inches(1.46), Inches(2.9), Inches(1.35), "group_by_API", ["mechanism 0.903", "direct 0.864"], SOFT_BLUE)
    add_card(slide, Inches(9.72), Inches(3.00), Inches(2.9), Inches(1.35), "group_by_method", ["mechanism 0.903", "direct 0.457"], SOFT_GREEN)
    add_card(slide, Inches(9.72), Inches(4.54), Inches(2.9), Inches(1.35), "caveat", ["route superiority 仍 target-dependent", "不是 complete cross-mechanism validation"], SOFT_ORANGE)
    add_footer_note(slide, "这页的任务不是吹赢所有指标，而是证明 unified interface 已经越出 PLGA。")

    # Slide 6
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_LIGHT)
    add_header(slide, "统一可以，但只能统一到接口 / 对象 / 报告层", "统一的是 stack，不是所有机制共用一个 raw physical latent")
    add_image(slide, BOUNDARY, Inches(0.55), Inches(1.18), Inches(8.75), Inches(5.80))
    add_card(slide, Inches(9.55), Inches(1.42), Inches(3.05), Inches(1.40), "现在可以大胆讲", [
        "shared release-intelligence stack",
        "shared posterior-family object",
        "first non-PLGA bridge",
    ], SOFT_GREEN)
    add_card(slide, Inches(9.55), Inches(3.05), Inches(3.05), Inches(1.40), "现在还不能讲", [
        "universal solved model",
        "universal t80",
        "benchmark-wide duration UQ",
    ], SOFT_ORANGE)
    add_card(slide, Inches(9.55), Inches(4.68), Inches(3.05), Inches(1.20), "下一跳", [
        "route-consistent uncertainty-aware duration / shape",
        "completed chitosan reveal",
    ], SOFT_BLUE)
    add_footer_note(slide, "这页是“大胆但不越界”的核心页。")

    # Slide 7
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, RGBColor(250, 251, 252))
    add_header(slide, "现在最不能乱吹的是 universal t80", "当前更成熟的是 shape-aware target layer")
    add_image(slide, FULL_BRIDGE, Inches(0.55), Inches(1.25), Inches(8.45), Inches(5.75))
    add_card(slide, Inches(9.18), Inches(1.45), Inches(3.42), Inches(1.55), "Timing 现实", [
        "global t10 / t50 / t80 = 0.574 / 0.320 / 0.074",
        "deep-threshold timing 仍明显更弱",
    ], SOFT_ORANGE)
    add_card(slide, Inches(9.18), Inches(3.15), Inches(3.42), Inches(1.55), "这张图真正说明了什么", [
        "shape-aware uncertainty 明显更成熟",
        "当前已经推进到 full 9-cell cell panel",
        "但 coverage 结构仍是 mixed",
    ], SOFT_BLUE)
    add_card(slide, Inches(9.18), Inches(4.85), Inches(3.42), Inches(1.15), "边界", [
        "liposome + internal181 已经 full matched-curve",
        "只剩 cross321 三个 cell 还在 sampled",
    ], SOFT_GREEN)
    add_footer_note(slide, "这页是防夸张的关键页，要让听众接受“我们现在没统一到 t80”。")

    # Slide 8
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_LIGHT)
    add_header(slide, "外部对象怎么处理：打 / 借 / 防", "不同外部对象，不该用同一套应对策略")
    add_image(slide, COMPETITOR, Inches(0.55), Inches(1.20), Inches(8.80), Inches(5.70))
    add_card(slide, Inches(9.58), Inches(1.40), Inches(3.00), Inches(1.38), "Fight", [
        "NC / Bannigan",
        "Explainable LAI",
        "更大规模 PLGA predictor",
    ], SOFT_ORANGE)
    add_card(slide, Inches(9.58), Inches(3.00), Inches(3.00), Inches(1.38), "Borrow", [
        "Scientific Data / AI-ready data",
        "liposome workflow",
        "computational pharmaceutics",
    ], SOFT_GREEN)
    add_card(slide, Inches(9.58), Inches(4.60), Inches(3.00), Inches(1.38), "Defend", [
        "FormulationLAI / FormulationAI",
        "optimization systems",
        "L5 platform narrative",
    ], SOFT_BLUE)
    add_footer_note(slide, "这页不是文献综述页，而是资源分配页。")

    # Slide 9
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, RGBColor(248, 249, 251))
    add_header(slide, "现在最值钱的下一步不是再多一个 PLGA cell", "最值钱的是把项目从强 benchmark paper 推到 release decision-support stack")
    add_card(slide, Inches(0.95), Inches(1.55), Inches(3.55), Inches(3.9), "优先级 1", [
        "route-consistent uncertainty-aware duration / shape",
        "这是当前最大的系统缺口",
        "也是从 benchmark paper 走向 decision-support 的关键台阶",
    ], SOFT_ORANGE)
    add_card(slide, Inches(4.88), Inches(1.55), Inches(3.55), Inches(3.9), "优先级 2", [
        "completed chitosan reveal",
        "把 prospective discipline 变成真正 reveal 结果",
        "让 credibility jump 发生",
    ], SOFT_GREEN)
    add_card(slide, Inches(8.80), Inches(1.55), Inches(3.55), Inches(3.9), "优先级 3", [
        "further cross-mechanism evidence beyond liposome",
        "继续扩大 non-PLGA coverage",
        "增强 shared stack 的跨机制说服力",
    ], SOFT_BLUE)
    add_footer_note(slide, "这页要管资源投向，不是做结果回顾。")

    # Slide 10
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_DARK)
    add_header(slide, "最后一句：我们到底在占什么位置", "现在最值得占的位置，是 shared release-intelligence stack", dark=True)
    add_textbox(slide, Inches(0.82), Inches(1.70), Inches(11.5), Inches(1.1), "我们不该再把自己缩成一个更强的 PLGA predictor，\n也不能提前吹 universal solved model。", font_size=24, bold=True, color=TEXT_LIGHT)
    add_card(slide, Inches(0.95), Inches(3.20), Inches(3.65), Inches(1.75), "我们已经有", ["shared benchmark", "posterior-family object", "first non-PLGA bridge"], RGBColor(43, 62, 97))
    add_card(slide, Inches(4.83), Inches(3.20), Inches(3.65), Inches(1.75), "我们最硬的 moat", ["L2 partial-observation benchmark", "L3 shared posterior-family object"], RGBColor(33, 86, 92))
    add_card(slide, Inches(8.72), Inches(3.20), Inches(3.65), Inches(1.75), "下一跳", ["route-consistent duration/UQ", "decision-support layer"], RGBColor(111, 66, 54))
    add_footer_note(slide, "让听众带着“怎么把 stack 接到 decision-support”这个问题离场。", dark=True)

    return prs


def write_summary() -> None:
    lines = [
        "=== 113 -- drug release strategy deck zh ===",
        "",
        f"PPTX: {OUTFILE.as_posix()}",
        "",
        "Slides",
        "  1. Positioning reset",
        "  2. External pressure types",
        "  3. Current moat at L2-L3",
        "  4. PLGA audited core",
        "  5. Liposome bridge",
        "  6. What can and cannot be unified",
        "  7. Why universal t80 is not legal now",
        "  8. Fight / borrow / defend",
        "  9. Highest-value next moves",
        "  10. Closing position statement",
    ]
    (OUTDIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTDIR)
    prs = build_deck()
    prs.save(OUTFILE)
    write_summary()


if __name__ == "__main__":
    main()
