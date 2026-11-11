from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK

from update_project_map_v8 import BLUE, MUTED, add_bullet, add_heading, add_paragraph, create_bullet_numbering, shade


MARKER = "2026-07-13 v8 盘后全链路与断点恢复"


def update(source: Path, output: Path) -> None:
    document = Document(source)
    if any(paragraph.text.strip() == MARKER for paragraph in document.paragraphs):
        if source.resolve() != output.resolve():
            document.save(output)
        return

    bullet_num_id = create_bullet_numbering(document)
    page_break = document.add_paragraph()
    page_break.add_run().add_break(WD_BREAK.PAGE)

    add_paragraph(document, MARKER, size=18, bold=True, color=BLUE, after=3)
    add_paragraph(document, "真实 Tushare 全市场 · 一键盘后流水线 · 严重漂移阻断纸面准入", size=10.5, color=MUTED, after=12)

    conclusion = add_paragraph(
        document,
        "结论：7 月 13 日完整链路已经跑通，但结果仍为 diagnostic_no_signal。数据和股票池门禁通过，模型扫描 5,524 只股票；同时 11.76% 特征发生严重漂移，因此该日只保留诊断榜，不产生纸面可参与信号。",
        size=11,
        bold=True,
        after=8,
    )
    shade(conclusion, "EAF2F8")

    add_heading(document, "新增工程闭环")
    for text in (
        "新增 daily-run 主入口：全市场同步、股票池审计、指定日快照门禁、生产地基、factor_lab_v2 快照、结算和评分固定串联。",
        "首次运行在缺少 7 月 13 日因子快照时被正确阻断并保存失败日报；补齐自动快照阶段后使用 --skip-sync 从数据库断点恢复。",
        "每次运行生成独立 JSON 日报、SHA256 侧车和 latest.json 索引；同日重跑使用数据库 upsert，不生成仓位、金额或下单动作。",
        "纸面准入增加第三道条件：研究门禁、市场准入、实时数据漂移必须同时通过。",
    ):
        add_bullet(document, text, bullet_num_id)

    add_heading(document, "2026-07-13 真实结果")
    for text in (
        "同步：行情 5,524；daily_basic 5,602；真实 moneyflow 5,197；复权 5,540；涨跌停 7,695；核心指数 7。",
        "factor_lab_v2 全市场快照 5,524 行；市场状态 red。",
        "市场准入分数 0.147406，阈值 0.587663；结果 diagnostic_no_signal，纸面候选 0。",
        "诊断 Top5：京仪装备、立航科技、中富电路、同大股份、迈为股份。该名单不是买入建议。",
        "102 个特征中 12 个严重漂移，比例 11.76%；主要集中于多周期资金流衍生项和跌停计数。",
    ):
        add_bullet(document, text, bullet_num_id)

    add_heading(document, "下一步决策")
    for text in (
        "继续按新交易日运行 daily-run，先判断漂移是持续分布变化还是单日市场状态，不用一个新日期反向调参。",
        "当前仍为 observe_only，40 个 usable 纸面信号日尚未开始计数；先用 v9 新信息或结构变化达到 candidate。",
        "v9 优先接入并审计龙虎榜/机构席位、回购、股东增减持、限售解禁、涨停开板和停复牌。",
        "停止使用相同 100 日最终审核段继续筛模型；无事件 v8.1 与事件版只在新 lockbox 上比较。",
    ):
        add_bullet(document, text, bullet_num_id)

    add_heading(document, "日常唯一命令")
    add_paragraph(document, "python scripts/run_v8.py daily-run --trade-date YYYYMMDD", bold=True)
    add_paragraph(document, "断点恢复：python scripts/run_v8.py daily-run --trade-date YYYYMMDD --skip-sync")
    add_paragraph(document, "最新日报：reports/ml_lab/v8_daily/latest.json")

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
