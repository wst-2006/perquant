from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.opc.constants import CONTENT_TYPE as CT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.parts.numbering import NumberingPart
from docx.shared import Pt, RGBColor


BLUE = RGBColor(20, 78, 140)
DARK = RGBColor(31, 45, 61)
MUTED = RGBColor(92, 107, 126)


def set_run(run, *, size: float = 10.5, bold: bool = False, color: RGBColor = DARK) -> None:
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = color


def add_paragraph(document: Document, text: str, *, size: float = 10.5, bold: bool = False, color: RGBColor = DARK, after: float = 5):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.25
    run = paragraph.add_run(text)
    set_run(run, size=size, bold=bold, color=color)
    return paragraph


def add_heading(document: Document, text: str) -> None:
    paragraph = add_paragraph(document, text, size=13, bold=True, color=BLUE, after=7)
    paragraph.paragraph_format.keep_with_next = True


def create_bullet_numbering(document: Document) -> int:
    try:
        numbering_part = document.part.part_related_by(RT.NUMBERING)
    except KeyError:
        numbering_part = NumberingPart(
            PackURI("/word/numbering.xml"),
            CT.WML_NUMBERING,
            parse_xml(f"<w:numbering {nsdecls('w')}></w:numbering>"),
            document.part.package,
        )
        document.part.relate_to(numbering_part, RT.NUMBERING)
    numbering = numbering_part.element
    abstract_ids = [int(node.get(qn("w:abstractNumId"))) for node in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(node.get(qn("w:numId"))) for node in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=-1) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    number_format = OxmlElement("w:numFmt")
    number_format.set(qn("w:val"), "bullet")
    level_text = OxmlElement("w:lvlText")
    level_text.set(qn("w:val"), "•")
    alignment = OxmlElement("w:lvlJc")
    alignment.set(qn("w:val"), "left")
    paragraph_properties = OxmlElement("w:pPr")
    indent = OxmlElement("w:ind")
    indent.set(qn("w:left"), "540")
    indent.set(qn("w:hanging"), "260")
    paragraph_properties.append(indent)
    level.extend([start, number_format, level_text, alignment, paragraph_properties])
    abstract.append(level)
    numbering.append(abstract)

    number = OxmlElement("w:num")
    number.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    number.append(abstract_ref)
    numbering.append(number)
    return num_id


def add_bullet(document: Document, text: str, num_id: int) -> None:
    paragraph = document.add_paragraph()
    properties = paragraph._p.get_or_add_pPr()
    number_properties = OxmlElement("w:numPr")
    level = OxmlElement("w:ilvl")
    level.set(qn("w:val"), "0")
    number = OxmlElement("w:numId")
    number.set(qn("w:val"), str(num_id))
    number_properties.extend([level, number])
    properties.append(number_properties)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.2
    for run in paragraph.runs:
        set_run(run)
    if not paragraph.runs:
        set_run(paragraph.add_run(text))
    else:
        paragraph.runs[0].text = text


def shade(paragraph, fill: str) -> None:
    properties = paragraph._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)
    paragraph.paragraph_format.left_indent = Pt(8)
    paragraph.paragraph_format.right_indent = Pt(8)
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(10)


def update(source: Path, output: Path) -> None:
    document = Document(source)
    bullet_num_id = create_bullet_numbering(document)
    page_break = document.add_paragraph()
    page_break.add_run().add_break(WD_BREAK.PAGE)

    add_paragraph(document, "2026-07-14 v8 Top5 机会检索执行结果", size=18, bold=True, color=BLUE, after=3)
    add_paragraph(document, "真实 Tushare 全市场 · 时间隔离 OOS · 诊断工件，不允许上线", size=10.5, color=MUTED, after=12)

    conclusion = add_paragraph(
        document,
        "结论：v8 工程计划已完成，项目综合完成度约 78%。当前最好模型在最后 100 个严格隔离 OOS 交易日实现 Top5 平均 5 日净收益 +0.8767%，但回撤、全亏日率和短期 Hit@5 未同时过线，状态仍为 observe_only。",
        size=11,
        bold=True,
        after=8,
    )
    shade(conclusion, "EAF2F8")

    add_heading(document, "本轮完成")
    for text in (
        "2,806,095 行、523 个交易日的日期分区 Parquet；1/3/5/10/20 日 T+1 安全标签。",
        "10,000 行独立标签重建，比较 630,000 个标签值，差异为 0。",
        "5,515 只股票的训练股票池无 UNKNOWN、隔离资产或非 A 股污染。",
        "LightGBM/XGBoost 全市场日分组 GPU Ranker、四独立头、模型共识、失败窗口和置换诊断已完成。",
        "Web 已增加 v8 状态、评分、结算、Top5 证据和人工保留/排除。",
        "forecast、express、dividend 公告事件已按 ann_date 严格滞后一日接入；缺失不模拟。",
    ):
        add_bullet(document, text, bullet_num_id)

    add_heading(document, "三轮真实结果")
    for text in (
        "v8 固定合成：5 日净收益 -0.3900%；正收益折率 60%；全亏日率 13%；最大回撤 -30.78%；下跌状态 -4.55%。",
        "v8.1 独立头与共识：5 日净收益 +0.8767%；正收益折率 80%；全亏日率 11%；最大回撤 -24.82%；+8%/20 日 Hit@5 为 100%。",
        "v8 events：5 日净收益 +0.5971%；正收益折率 60%；全亏日率 6%；最大回撤 -18.38%；下跌状态 -0.2819%，regime guard 首次通过。",
        "最强简单基线为 20 日动量：5 日净收益 +0.6976%、全亏日率 4%。三轮均未同时超过全部硬基线。",
    ):
        add_bullet(document, text, bullet_num_id)

    add_heading(document, "市场准入与当前实时结果")
    add_paragraph(
        document,
        "三段隔离准入研究的最佳方案为 Logistic、训练目标覆盖 40%。最终覆盖 22.5%，准入日均收益 +2.7556%，全亏日率 0%，回撤 -14.97%；但覆盖不足 25%，全日历收益未超过不准入模型，因此仍为 observe_only。",
    )
    add_paragraph(
        document,
        "当前诊断工件为 v8_top5_20260601_533ca3ca。2026-07-10 扫描 5,521 只股票，准入结果为 diagnostic_no_signal。诊断 Top5：迈为股份、龙图光罩、立航科技、协创数据、方邦股份。该名单不是买入建议。",
    )

    add_heading(document, "硬限制与停止条件")
    for text in (
        "CatBoost GPU Ranker 单 query 最多 1,023 行，无法容纳约 5,300 只 A 股的完整日分组；不拆碎股票池，CatBoost 不进入正式门禁。",
        "同一 100 日最终段已经用于三轮研发反馈，停止继续在该段筛方案或调参数，避免研究者过拟合。",
        "任何模型在 candidate 前只显示于研究页面；不输出仓位、金额或自动交易指令。",
    ):
        add_bullet(document, text, bullet_num_id)

    add_heading(document, "下一步")
    for text in (
        "按交易日自动同步真实 Tushare 数据、运行门禁、生成诊断 Top5、保存人工选择。",
        "用 2026-06-01 之后的新日期形成独立 lockbox；至少 40 个已结算有效信号日后再讨论 usable。",
        "v9 只新增信息或结构：龙虎榜/机构席位、回购、股东增减持、限售解禁、涨停开板与停复牌；先同步审计，再进入特征。",
        "暂不租云 GPU。本机 RTX 4070 已完成三轮全量研究。",
    ):
        add_bullet(document, text, bullet_num_id)

    add_heading(document, "关键工件")
    for text in (
        "当前模型：reports/ml_lab/v8_artifact/artifact.json",
        "v8.1 报告：reports/ml_lab/v8_1_research_gpu_full/v8_research_report.json",
        "事件版报告：reports/ml_lab/v8_events_research_gpu_full/v8_research_report.json",
        "准入报告：reports/ml_lab/v8_1_market_gate/v8_market_gate_report.json",
        "详细状态：docs/PROJECT_V8_EXECUTION_STATUS_20260714.md",
    ):
        add_bullet(document, text, bullet_num_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    update(args.source, args.output)


if __name__ == "__main__":
    main()
