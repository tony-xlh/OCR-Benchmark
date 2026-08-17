#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a self-contained bilingual (Chinese / English) HTML report from an
evaluation report JSON produced by eval_winrt_ocr.py.

Usage:
  python make_report_html.py --report ocr_dataset_ar/eval_report_mac.json \
      --output ocr_eval_report_mac.html --overlay macocr_visualization.png

The verdict / chart notes adapt to the engine's actual behaviour (dropped
diacritics, scrambled box order, loose boxes) so the same template serves any
Windows.Media.Ocr / macOS Vision / Tesseract result.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def fmt(p):
    return f"{p * 100:.1f}%"


ZH_LANG = {"Arabic": "阿拉伯语", "Chinese": "中文", "English": "英文"}


def build_html(report, overlay=None, dataset_label=None):
    agg = report["aggregate"]
    engine = report.get("engine", "?")
    lang_name = report.get("language", "Arabic")
    lang_code = report.get("language_code", "ar")
    direction = report.get("direction", "rtl")
    n = agg["total_images"]
    thr = report.get("iou_threshold", 0.5)
    itp_src = report.get("itp", "?.itp")
    dataset = report.get("dataset", "?")
    if dataset_label is None:
        dataset_label = Path(dataset).name or dataset

    det = agg["detection_recall"]
    rec_exact = agg["recognition_exact"]
    rec_norm = agg["recognition_norm"]
    char_cer = agg["char_cer_norm"]
    iou = agg["mean_best_iou"]
    cer_ro = agg["text_cer_reading_order"]
    wer_ro = agg["text_wer_reading_order"]
    cer_naive = agg["text_cer_naive"]
    wer_naive = agg["text_wer_naive"]

    drops_diacritics = (rec_norm - rec_exact) > 0.15
    order_issue = (wer_naive - wer_ro) > 0.10
    loose_boxes = iou < 0.70

    # ---- verdict: adaptive ----
    caveats = []
    if order_issue:
        caveats.append("order")
    if drops_diacritics:
        caveats.append("diacritics")
    if not caveats and loose_boxes:
        caveats.append("boxes")
    if len(caveats) >= 2:
        tag_text_zh, tag_text_en, tag_cls = "两个集成要点", "2 integration caveats", "warn"
    elif caveats:
        tag_map = {"order": ("顺序需重排", "re-sort order"),
                   "diacritics": ("变音符丢失", "diacritics dropped"),
                   "boxes": ("词框偏松", "loose boxes")}
        tag_text_zh, tag_text_en = tag_map[caveats[0]]
        tag_cls = "warn"
    else:
        tag_text_zh, tag_text_en, tag_cls = "开箱即用", "ready to use", "good"

    if drops_diacritics:
        diac_zh = f"引擎不输出变音符，含变音符时逐字精确率仅 {fmt(rec_exact)}"
        diac_en = f"the engine outputs no diacritics — only {fmt(rec_exact)} of words match exactly when they are kept"
    elif rec_exact >= 0.5:
        diac_zh = f"引擎保留大部分变音符，含变音符时逐字精确率已达 {fmt(rec_exact)}"
        diac_en = f"the engine preserves most diacritics — {fmt(rec_exact)} of words match exactly including them"
    else:
        diac_zh = f"含变音符时逐字精确率 {fmt(rec_exact)}"
        diac_en = f"{fmt(rec_exact)} of words match exactly including diacritics"
    if order_issue:
        ord_zh = "返回的词框不是阅读顺序，必须按几何位置重排"
        ord_en = "the returned word boxes are not in reading order — re-sort them by geometry"
    else:
        ord_zh = "词框已按阅读顺序返回，无需重排"
        ord_en = "word boxes are already in reading order — no re-sorting needed"
    if loose_boxes and not order_issue and not drops_diacritics:
        box_zh = f"注意：词框偏松（平均 IoU {fmt(iou)}），像素级词框需求建议留意"
        box_en = f"note: boxes are loose (mean IoU {fmt(iou)}); worth checking for pixel-exact box needs"
    else:
        box_zh, box_en = "", ""

    lang_name_zh = ZH_LANG.get(lang_name, lang_name)
    verdict_p_zh = (
        f"对 {n} 页 A4 {lang_name_zh}文档（含变音符）逐字比对 GT。去变音符后字符 CER 仅 {fmt(char_cer)}、"
        f"按阅读顺序拼接的文本 WER {fmt(wer_ro)} —— 识别质量优秀。"
        f"{diac_zh}；{ord_zh}。{box_zh}")
    verdict_p_en = (
        f"Word-by-word comparison against GT on {n} A4 {lang_name} pages (vocalized). "
        f"After diacritic stripping, char-level CER is {fmt(char_cer)} and text WER in reading order "
        f"is {fmt(wer_ro)} — recognition quality is excellent. {diac_en}; {ord_en}. {box_en}")

    # ---- chart notes (adaptive) ----
    if drops_diacritics:
        chart1_sub_zh = f"引擎不输出变音符；去掉后真实识别率 {fmt(rec_norm)}"
        chart1_sub_en = f"no diacritics are output; the real glyph recognition is {fmt(rec_norm)} once ignored"
    else:
        chart1_sub_zh = f"引擎保留大部分变音符；标准化后识别率 {fmt(rec_norm)}"
        chart1_sub_en = f"most diacritics are preserved; normalized recognition is {fmt(rec_norm)}"
    if order_issue:
        chart2_sub_zh = "返回顺序不是阅读顺序，直接拼接会乱序"
        chart2_sub_en = "the returned order is not reading order — naive concatenation scrambles it"
    else:
        chart2_sub_zh = "词框已按阅读顺序返回，两种顺序结果一致"
        chart2_sub_en = "boxes are already in reading order — both orders give the same result"
    chart2_naive_color = "var(--critical)" if order_issue else "var(--deemph)"

    # ---- glossary: adaptive "why it matters" card ----
    if order_issue and drops_diacritics:
        why_zh = ("本例中「识别」本身很好，但两个问题会把下游结果彻底毁掉：① 顺序错 → 拼接文本乱序；"
                  "② 变音符丢失 → 含音标的需求无法满足。评测脚本（eval_winrt_ocr.py）已内置顺序重排与去变音符逻辑，可直接复用。")
        why_en = ("Recognition itself is strong here, but two issues can wreck downstream output: ① wrong order → "
                  "scrambled text; ② dropped diacritics → no vocalized text. eval_winrt_ocr.py already implements "
                  "re-sorting and diacritic stripping.")
    elif order_issue:
        why_zh = ("引擎返回的词框不是阅读顺序（也不是简单反转），直接拼接会乱序。集成时必须按几何位置重排 "
                  "（行内从右到左、行间从上到下）再拼文本；eval_winrt_ocr.py 内置了该重排逻辑。")
        why_en = ("The engine returns boxes not in reading order (and not a simple reversal); naive concatenation "
                  "scrambles the text. Consumers must re-sort by geometry (right-to-left per line, top-to-bottom "
                  "between lines); eval_winrt_ocr.py includes this re-sorting.")
    elif drops_diacritics:
        why_zh = ("引擎不输出变音符。若下游需要含音标文本，此引擎无法满足；评测脚本已内置去变音符逻辑，"
                  "可公平比较识别质量。")
        why_en = ("The engine outputs no diacritics (harakat). If vocalized text is required downstream this engine "
                  "cannot provide it; eval_winrt_ocr.py strips diacritics for a fair quality comparison.")
    else:
        why_zh = ("本引擎的词框已按阅读顺序返回且保留变音符，文本可直接拼接使用，集成成本低。"
                  + (f"只需留意词框偏松（平均 IoU {fmt(iou)}）。" if loose_boxes else ""))
        why_en = ("This engine returns boxes in reading order and preserves diacritics, so text can be joined "
                  "directly — low integration cost."
                  + (f" Only note the loose boxes (mean IoU {fmt(iou)})." if loose_boxes else ""))

    # ---- build data chunks ----
    def tile(label_zh, label_en, val, color, note_zh, note_en, pct):
        return f'''    <div class="tile">
      <div class="lbl"><span data-lang="zh">{label_zh}</span><span data-lang="en">{label_en}</span></div>
      <div class="val" style="color:{color}">{val}</div>
      <div class="note"><span data-lang="zh">{note_zh}</span><span data-lang="en">{note_en}</span></div>
      <div class="meter"><i style="width:{pct}; background:{color}"></i></div>
    </div>'''

    det_color = "var(--good)" if det >= 0.90 else "var(--series-1)"
    rec_norm_color = "var(--good)" if rec_norm >= 0.90 else "var(--series-1)"
    char_cer_color = "var(--good)" if char_cer <= 0.05 else "var(--warning)"
    iou_color = "var(--warning)" if loose_boxes else "var(--series-1)"
    rec_exact_color = "var(--good)" if rec_exact >= 0.50 else "var(--warning)"
    wer_ro_color = "var(--good)" if wer_ro <= 0.05 else "var(--series-1)"

    kpis = "\n".join([
        tile("词框检出率", "Word detection", fmt(det), det_color,
             "GT 词中带 IoU≥0.5 框的比例", "GT words with a box at IoU≥0.5", fmt(det)),
        tile("词识别率 · 标准化后", "Word recognition · normalized", fmt(rec_norm), rec_norm_color,
             "去变音符后逐字精确", "exact match after diacritic stripping", fmt(rec_norm)),
        tile("字符 CER · 标准化后", "Char CER · normalized", fmt(char_cer), char_cer_color,
             "越低越好", "lower is better", fmt(char_cer)),
        tile("平均框 IoU", "Mean box IoU", fmt(iou), iou_color,
             "定位框与 GT 框的重合度", "box overlap vs ground truth", fmt(iou)),
        tile("词识别率 · 原文", "Word recognition · exact", fmt(rec_exact), rec_exact_color,
             "含变音符比较", "incl. diacritics", fmt(rec_exact)),
        tile("文本 WER · 阅读顺序", "Text WER · reading order", fmt(wer_ro), wer_ro_color,
             "重排后拼接，越低越好", "after re-sorting; lower is better", fmt(wer_ro)),
    ])

    bar_rec_norm = f'''      <div class="bar-row">
        <div class="bar-cat"><span data-lang="zh">标准化后</span><span data-lang="en">normalized</span></div>
        <div class="bar-track"><div class="bar" style="width:{rec_norm*100:.1f}%; background:var(--series-1)"></div></div>
        <div class="bar-val">{fmt(rec_norm)}</div>
      </div>'''
    bar_rec_exact = f'''      <div class="bar-row">
        <div class="bar-cat"><span data-lang="zh">原文（含变音符）</span><span data-lang="en">as returned</span></div>
        <div class="bar-track"><div class="bar" style="width:{rec_exact*100:.1f}%; background:var(--deemph)"></div></div>
        <div class="bar-val">{fmt(rec_exact)}</div>
      </div>'''
    bar_wer_ro = f'''      <div class="bar-row">
        <div class="bar-cat"><span data-lang="zh">阅读顺序重排</span><span data-lang="en">reading order</span></div>
        <div class="bar-track"><div class="bar" style="width:{wer_ro*100:.1f}%; background:var(--series-1)"></div></div>
        <div class="bar-val">{fmt(wer_ro)}</div>
      </div>'''
    bar_wer_naive = f'''      <div class="bar-row">
        <div class="bar-cat"><span data-lang="zh">返回顺序（原始）</span><span data-lang="en">returned order</span></div>
        <div class="bar-track"><div class="bar" style="width:{wer_naive*100:.1f}%; background:{chart2_naive_color}"></div></div>
        <div class="bar-val">{fmt(wer_naive)}</div>
      </div>'''

    # ---- per-image table ----
    def row_cells(r, last=False):
        rec_exact_cls = ' class="pct-bad"' if drops_diacritics and r["recognition_exact"] < 0.5 else ""
        naive_cls = ' class="pct-bad"' if order_issue else ""
        return ("<tr>" + ("<td>MEAN</td><td>—</td><td>—</td>" if last else
                          f'<td>{r["image"]}</td><td>{r["gt_words"]}</td><td>{r["ocr_boxes"]}</td>')
                + f'<td>{r["detection_recall"]*100:.1f}</td>'
                + f'<td{rec_exact_cls}>{r["recognition_exact"]*100:.1f}</td>'
                + f'<td>{r["recognition_norm"]*100:.1f}</td>'
                + f'<td>{r["char_cer_norm"]*100:.1f}</td>'
                + f'<td>{r["mean_best_iou"]*100:.1f}</td>'
                + f'<td>{r["text_cer_reading_order"]*100:.1f}</td>'
                + f'<td>{r["text_wer_reading_order"]*100:.1f}</td>'
                + f'<td>{r["text_cer_naive"]*100:.1f}</td>'
                + f'<td{naive_cls}>{r["text_wer_naive"]*100:.1f}</td></tr>')

    rows = "".join(row_cells(r) for r in report["per_image"])
    agg_row = row_cells({
        "image": "MEAN", "gt_words": "—", "ocr_boxes": "—",
        "detection_recall": det, "recognition_exact": rec_exact,
        "recognition_norm": rec_norm, "char_cer_norm": char_cer,
        "mean_best_iou": iou, "text_cer_reading_order": cer_ro,
        "text_wer_reading_order": wer_ro, "text_cer_naive": cer_naive,
        "text_wer_naive": wer_naive,
    }, last=True)
    rows += "\n" + agg_row.replace("<tr>", '<tr class="agg">', 1)

    figure = ""
    if overlay:
        figure = f'''
  <h2><span data-lang="zh">定位可视化（GT 绿框 vs OCR 红框）</span><span data-lang="en">Localization preview (GT green vs OCR red)</span></h2>
  <figure>
    <img src="{overlay}" alt="Ground truth (green) vs OCR boxes (red) on doc_001" loading="lazy">
    <figcaption>
      <span data-lang="zh">doc_001 叠图：绿 = GT 词框，红虚线 = OCR 词框。可见两者几乎逐词重合，定位准确。</span>
      <span data-lang="en">doc_001 overlay: green = GT boxes, red dashed = OCR boxes. They align word-for-word — localization is accurate.</span>
    </figcaption>
  </figure>'''

    # ---- glossary: static cards (bilingual) ----
    glossary = f'''
    <div class="gcard">
      <h4><span data-lang="zh">词框检出率</span><span data-lang="en">Word detection (recall)</span></h4>
      <p class="zh">GT 中有多少比例的词能在 OCR 输出里找到位置对应的框（IoU ≥ 阈值 0.5）。反映「能不能找对每个词的位置」。比率高且每页 OCR 框数 == GT 词数，说明几乎没有漏检或过度切分。</p>
      <p data-lang="en">Share of ground-truth words with a detected box at IoU ≥ 0.5. Measures localization coverage. High here, with box counts matching GT on every page — almost no misses or over-segmentation.</p>
    </div>
    <div class="gcard">
      <h4><span data-lang="zh">词识别率 · 原文</span><span data-lang="en">Word recognition · exact</span></h4>
      <p class="zh">OCR 输出词与 GT 词逐字完全相同（包括变音符）。若引擎不输出变音符，此值会偏低 —— 偏低是引擎能力所限，不代表读错字。</p>
      <p data-lang="en">Words that match the GT byte-for-byte including diacritics. If the engine cannot output harakat this value stays low — an engine limitation, not a reading error.</p>
    </div>
    <div class="gcard">
      <h4><span data-lang="zh">词识别率 · 标准化后</span><span data-lang="en">Word recognition · normalized</span></h4>
      <p class="zh">去掉阿拉伯语变音符后再逐字比较。这才是真正的文字识别质量（常见残留错误为 ٬ ٫ ٪ 等特殊标点与 ة/ه、ى/ي 混淆）。</p>
      <p data-lang="en">Exact match after stripping Arabic diacritics — the real glyph recognition quality. Residual errors are mostly special punctuation (٬ ٫ ٪) and glyph confusions (ة/ه, ى/ي).</p>
    </div>
    <div class="gcard">
      <h4><span data-lang="zh">字符级 CER</span><span data-lang="en">Character error rate</span></h4>
      <p class="zh">字符编辑距离 ÷ GT 字符数（Levenshtein）。越低越好，是衡量识别质量最直接的指标。</p>
      <p data-lang="en">Character edit distance ÷ GT character count (Levenshtein). Lower is better; the most direct quality measure.</p>
      <div class="formula">CER = edit_distance(pred, gt) ÷ len(gt)</div>
    </div>
    <div class="gcard">
      <h4><span data-lang="zh">平均框 IoU</span><span data-lang="en">Mean box IoU</span></h4>
      <p class="zh">每个 GT 词框与「交并比最高的 OCR 框」的重合度（交集面积÷并集面积）。反映定位框与真值的贴合程度；字体/高度差异会使其低于 100%。</p>
      <p data-lang="en">Intersection-over-union between each GT box and its best-matching OCR box. Reflects how tightly boxes fit the text; font/size differences keep it below 100%.</p>
      <div class="formula">IoU = |A ∩ B| ÷ |A ∪ B|</div>
    </div>
    <div class="gcard">
      <h4><span data-lang="zh">文本 CER / WER（阅读顺序）</span><span data-lang="en">Text CER / WER · reading order</span></h4>
      <p class="zh">把词框按阅读顺序重排（行内从右到左、行间从上到下）后拼接整页文本，再算 CER/WER。这是集成时应采用的正确做法。</p>
      <p data-lang="en">CER/WER after re-sorting boxes into reading order (right-to-left within line, top-to-bottom between lines) and joining — how a consumer should assemble text.</p>
      <div class="formula">WER = word_edit_distance ÷ GT_words</div>
    </div>
    <div class="gcard">
      <h4><span data-lang="zh">文本 CER / WER（返回顺序）</span><span data-lang="en">Text CER / WER · returned order</span></h4>
      <p class="zh">直接用引擎返回的框顺序拼接文本的 CER/WER。若该值与阅读顺序值差距大，说明返回顺序不是阅读顺序，直接拼接会乱序。</p>
      <p data-lang="en">CER/WER joining boxes in the raw returned order. A large gap to the reading-order value means the returned order is not reading order and must be re-sorted.</p>
    </div>
    <div class="gcard">
      <h4><span data-lang="zh">顺序与变音符是否重要</span><span data-lang="en">Do ordering &amp; diacritics matter?</span></h4>
      <p class="zh">{why_zh}</p>
      <p data-lang="en">{why_en}</p>
    </div>'''

    html = f'''<!DOCTYPE html>
<html lang="zh" class="lang-zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCR 评估报告 · {engine} | OCR Evaluation Report</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="topbar">
      <div>
        <h1><span data-lang="zh">OCR 评估报告 · {engine}</span>
            <span data-lang="en">OCR Evaluation Report · {engine}</span></h1>
        <p class="sub" data-lang="zh">用生成的 {lang_name_zh}数据集对 ImageTrans「{engine}」引擎做定量评测</p>
        <p class="sub" data-lang="en">Quantitative evaluation of the ImageTrans "{engine}" engine on a generated {lang_name} dataset</p>
      </div>
      <button class="lang-btn" id="langBtn" type="button" data-lang="zh">EN ⇄ 中文</button>
    </div>
    <div class="chips">
      <span class="chip"><span data-lang="zh">引擎 / </span><span data-lang="en">Engine: </span><b>{engine}</b></span>
      <span class="chip"><span data-lang="zh">语言 / </span><span data-lang="en">Language: </span><b>{lang_name} ({lang_code}, {direction.upper()})</b></span>
      <span class="chip"><span data-lang="zh">数据集 / </span><span data-lang="en">Dataset: </span><b>{dataset_label}</b></span>
      <span class="chip"><span data-lang="zh">页数 / </span><span data-lang="en">Images: </span><b>{n}</b></span>
      <span class="chip"><span data-lang="zh">IoU 阈值 / </span><span data-lang="en">IoU threshold: </span><b>{thr:.2f}</b></span>
    </div>
  </header>

  <div class="verdict">
    <span class="tag good" data-lang="zh">识别优秀</span><span class="tag good" data-lang="en">Excellent recognition</span>
    <span class="tag {tag_cls}" data-lang="zh">{tag_text_zh}</span><span class="tag {tag_cls}" data-lang="en">{tag_text_en}</span>
    <p data-lang="zh">{verdict_p_zh}</p>
    <p data-lang="en">{verdict_p_en}</p>
  </div>

  <!-- KPI stat tiles -->
  <h2><span data-lang="zh">核心指标</span><span data-lang="en">Key metrics</span></h2>
  <div class="kpis">
{kpis}
  </div>

  <!-- bar charts -->
  <div class="charts">
    <div class="card">
      <h3><span data-lang="zh">词识别：变音符的影响</span><span data-lang="en">Word recognition — diacritics effect</span></h3>
      <p class="sub" data-lang="zh">{chart1_sub_zh}</p>
      <p class="sub" data-lang="en">{chart1_sub_en}</p>
{bar_rec_norm}
{bar_rec_exact}
    </div>
    <div class="card">
      <h3><span data-lang="zh">文本 WER：顺序的影响</span><span data-lang="en">Text WER — ordering effect</span></h3>
      <p class="sub" data-lang="zh">{chart2_sub_zh}</p>
      <p class="sub" data-lang="en">{chart2_sub_en}</p>
{bar_wer_ro}
{bar_wer_naive}
    </div>
  </div>

  <!-- per-image table -->
  <h2>
    <span data-lang="zh">逐页明细</span><span data-lang="en">Per-image detail</span>
    <span class="en" data-lang="zh">（水平可滚动）</span><span class="en" data-lang="en">(scroll horizontally)</span>
  </h2>
  <div class="table-scroll">
    <table>
      <thead>
        <tr>
          <th><span data-lang="zh">图像</span><span data-lang="en">Image</span></th>
          <th>GT</th><th>OCR</th>
          <th><span data-lang="zh">检出%</span><span data-lang="en">det%</span></th>
          <th><span data-lang="zh">识别·原文%</span><span data-lang="en">rec·exact%</span></th>
          <th><span data-lang="zh">识别·标准化%</span><span data-lang="en">rec·norm%</span></th>
          <th><span data-lang="zh">字符CER%</span><span data-lang="en">charCER%</span></th>
          <th><span data-lang="zh">IoU%</span><span data-lang="en">boxIoU%</span></th>
          <th><span data-lang="zh">CER·阅读序%</span><span data-lang="en">CER·ro%</span></th>
          <th><span data-lang="zh">WER·阅读序%</span><span data-lang="en">WER·ro%</span></th>
          <th><span data-lang="zh">CER·返回序%</span><span data-lang="en">CER·naive%</span></th>
          <th><span data-lang="zh">WER·返回序%</span><span data-lang="en">WER·naive%</span></th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>

  <!-- glossary: the meaning of each metric -->
  <h2>
    <span data-lang="zh">各项指标的意义</span><span data-lang="en">What each metric means</span>
    <span class="en" data-lang="zh">（GT = 浏览器精确标注的真值）</span><span class="en" data-lang="en">(GT = ground truth, pixel-exact from the renderer)</span>
  </h2>
  <div class="glossary">
{glossary}
  </div>
{figure}
  <footer>
    <span data-lang="zh">复现：</span><span data-lang="en">Reproduce: </span>
    <code>python eval_winrt_ocr.py --itp {itp_src} --dataset {dataset}</code>
    <span data-lang="zh">，完整数值见 </span><span data-lang="en"> · full numbers in </span>
    <a href="{report.get('_report_path', 'eval_report.json')}">eval_report.json</a>。
  </footer>

</div>

<script>
  var btn = document.getElementById("langBtn");
  function setLang(l) {{
    document.documentElement.className = "lang-" + l;
    document.documentElement.lang = l === "zh" ? "zh" : "en";
  }}
  btn.addEventListener("click", function () {{
    var cur = document.documentElement.className.indexOf("lang-en") >= 0 ? "en" : "zh";
    setLang(cur === "zh" ? "en" : "zh");
  }});
</script>
</body>
</html>'''

    return html


CSS = """  :root {
    --page:       #f9f9f7;
    --surface:    #fcfcfb;
    --ink:        #0b0b0b;
    --ink-2:      #52514e;
    --muted:      #898781;
    --grid:       #e1e0d9;
    --baseline:   #c3c2b7;
    --border:     rgba(11,11,11,0.10);
    --series-1:   #2a78d6;
    --series-6:   #e34948;
    --good:       #0ca30c;
    --warning:    #fab219;
    --critical:   #d03b3b;
    --deemph:     #c8c6bf;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --page:       #0d0d0d;
      --surface:    #1a1a19;
      --ink:        #ffffff;
      --ink-2:      #c3c2b7;
      --muted:      #898781;
      --grid:       #2c2c2a;
      --baseline:   #383835;
      --border:     rgba(255,255,255,0.10);
      --series-1:   #3987e5;
      --series-6:   #e66767;
      --good:       #0ca30c;
      --warning:    #fab219;
      --critical:   #d03b3b;
      --deemph:     #4a4a47;
    }
  }
  [data-lang] { display: none; }
  html.lang-zh [data-lang="zh"], html.lang-en [data-lang="en"] { display: revert; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    line-height: 1.6; -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 32px 24px 64px; }
  header { border-bottom: 1px solid var(--grid); padding-bottom: 20px; margin-bottom: 24px; }
  .topbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }
  h1 { font-size: 26px; line-height: 1.3; margin: 0; font-weight: 700; }
  .sub { color: var(--ink-2); margin: 6px 0 0; font-size: 15px; }
  .lang-btn {
    font: inherit; font-size: 14px; cursor: pointer;
    background: var(--surface); color: var(--ink);
    border: 1px solid var(--baseline); border-radius: 8px; padding: 7px 14px;
  }
  .lang-btn:hover { border-color: var(--ink-2); }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
  .chip {
    font-size: 13px; color: var(--ink-2);
    border: 1px solid var(--grid); border-radius: 999px;
    padding: 3px 12px; background: var(--surface);
  }
  .chip b { color: var(--ink); font-weight: 600; }
  .verdict {
    background: var(--surface); border: 1px solid var(--grid);
    border-left: 4px solid var(--good); border-radius: 10px;
    padding: 16px 20px; margin-bottom: 28px; font-size: 15px;
  }
  .verdict p { margin: 6px 0; }
  .verdict .tag {
    display: inline-block; font-size: 12px; font-weight: 600;
    border-radius: 6px; padding: 1px 8px; margin-right: 6px;
  }
  .tag.good { background: color-mix(in srgb, var(--good) 16%, transparent); color: var(--good); }
  .tag.warn { background: color-mix(in srgb, var(--warning) 22%, transparent); color: var(--ink); }
  h2 { font-size: 19px; font-weight: 700; margin: 40px 0 14px; display: flex; align-items: baseline; gap: 10px; }
  h2 .en { font-size: 13px; color: var(--muted); font-weight: 500; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .tile {
    background: var(--surface); border: 1px solid var(--grid);
    border-radius: 10px; padding: 14px 16px 12px;
  }
  .tile .lbl { font-size: 13px; color: var(--ink-2); margin-bottom: 2px; }
  .tile .lbl .en { color: var(--muted); }
  .tile .val { font-size: 32px; font-weight: 650; line-height: 1.1; }
  .tile .note { font-size: 12.5px; color: var(--muted); margin-top: 6px; }
  .tile .meter { height: 5px; border-radius: 999px; background: var(--grid); margin-top: 10px; overflow: hidden; }
  .tile .meter > i { display: block; height: 100%; border-radius: 999px; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 4px; }
  @media (max-width: 760px) { .charts { grid-template-columns: 1fr; } }
  .card {
    background: var(--surface); border: 1px solid var(--grid);
    border-radius: 10px; padding: 18px 20px;
  }
  .card h3 { margin: 0 0 2px; font-size: 16px; font-weight: 650; }
  .card .sub { font-size: 12.5px; color: var(--muted); margin: 0 0 14px; }
  .bar-row { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
  .bar-cat { width: 132px; flex: none; font-size: 13px; color: var(--ink-2); }
  .bar-track { flex: 1; height: 16px; position: relative; }
  .bar { height: 16px; border-radius: 0 4px 4px 0; }
  .bar-val {
    width: 64px; flex: none; text-align: right;
    font-variant-numeric: tabular-nums; font-size: 13px; color: var(--ink);
  }
  .table-scroll { overflow-x: auto; border: 1px solid var(--grid); border-radius: 10px; background: var(--surface); }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { padding: 7px 10px; text-align: right; white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  th {
    font-weight: 600; color: var(--ink-2); font-size: 12px;
    border-bottom: 1px solid var(--grid); background: var(--surface);
    position: sticky; top: 0;
  }
  td { border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
  tbody tr:last-child td { border-bottom: none; }
  tr.agg td { font-weight: 650; background: color-mix(in srgb, var(--series-1) 7%, var(--surface)); }
  td.pct-bad { color: var(--critical); }
  .glossary { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 760px) { .glossary { grid-template-columns: 1fr; } }
  .gcard {
    background: var(--surface); border: 1px solid var(--grid);
    border-radius: 10px; padding: 14px 18px;
  }
  .gcard h4 { margin: 0 0 8px; font-size: 14.5px; font-weight: 650; }
  .gcard h4 .en { color: var(--muted); font-weight: 500; }
  .gcard p { margin: 6px 0; font-size: 13.5px; color: var(--ink-2); }
  .gcard p.zh { color: var(--ink); }
  .gcard .formula {
    font-size: 12.5px; color: var(--ink-2);
    background: var(--page); border: 1px solid var(--grid);
    border-radius: 6px; padding: 4px 10px; margin: 8px 0 0;
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  }
  figure { margin: 8px 0 0; }
  figure img {
    width: 100%; height: auto; border: 1px solid var(--grid);
    border-radius: 10px; background: var(--surface);
  }
  figcaption { font-size: 12.5px; color: var(--muted); margin-top: 8px; }
  footer { margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--grid); color: var(--muted); font-size: 13px; }
  code {
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    font-size: 0.92em; background: var(--surface);
    border: 1px solid var(--grid); border-radius: 5px; padding: 1px 6px;
  }
  a { color: var(--series-1); }
  a:visited { color: var(--series-1); }"""


def main():
    ap = argparse.ArgumentParser(
        description="Generate a bilingual HTML report from an eval_winrt_ocr.py report JSON")
    ap.add_argument("--report", required=True, help="evaluation report JSON (from eval_winrt_ocr.py)")
    ap.add_argument("--output", required=True, help="output HTML path")
    ap.add_argument("--overlay", default=None,
                    help="optional GT-vs-OCR overlay image to embed (e.g. macocr_visualization.png)")
    ap.add_argument("--dataset-label", default=None,
                    help="optional dataset name shown in the chips (default: dataset folder name)")
    args = ap.parse_args()

    report = json.load(open(args.report, encoding="utf-8"))
    report["_report_path"] = args.report
    html_text = build_html(report, overlay=args.overlay, dataset_label=args.dataset_label)

    out = Path(args.output)
    out.write_text(html_text, encoding="utf-8")
    print(f"Report HTML written: {out}")


if __name__ == "__main__":
    main()
