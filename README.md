# OCR Dataset Generator (Arabic / Chinese / English)

Generate **OCR datasets** in three languages — **Arabic** (default), **Chinese** and
**English**: real document images plus precise ground truth (text and word-level
bounding boxes), for evaluating OCR engines. Select the language with `--lang`.

Images are rendered in a **headless browser (Playwright)** using the system
Edge/Chrome, so Arabic shaping, ligatures, diacritics (harakat) and RTL layout
are rendered natively by the browser — the same way a real reader sees them.
Word/character positions are measured with DOM `Range.getBoundingClientRect()`,
which avoids the unreliable text layer that Word/LibreOffice PDFs produce for RTL
Arabic (scrambled runs, lost diacritics, misplaced punctuation). CJK and Latin
text use the same exact-pixel measurement.

## Features

- **A4 document pages** (default): 2480×3508 px (A4 @ 300 DPI), a centered title
  plus 8–11 paragraphs per page. Clean, no noise, no rotation.
- **Word cards** (optional): short words/phrases/sentences with optional
  augmentation (gaussian / salt & pepper / blur / rotation noise). Rotation
  transforms the GT boxes to match the rotated image.
- **Ground truth** per sample:
  - `text` — full page/sample text (logical order)
  - `words` — word-level boxes `{text, bbox:{x,y,width,height}}` in pixel coords.
    For Chinese, where there are no word separators, each CJK character is a
    token (consecutive Latin/digit runs stay grouped).
  - `characters` — per-character boxes
- **Language-aware rendering**: `dir` (rtl/ltr), alignment, fonts and line-height
  are chosen per language. Font lists use Windows built-ins
  (SimSun/SimHei/Microsoft YaHei/KaiTi/FangSong for Chinese, Arial/Times New
  Roman/Georgia/… for English).
- **Evaluation**: run a real OCR engine (Tesseract) and get CER / WER / word-box
  IoU. The language is auto-detected from the dataset metadata
  (`ara` / `chi_sim` / `eng`).

## Project layout

```
.
├── ocr_web_generator.py          # multi-language dataset generator (--lang ar|zh|en)
├── arabic_ocr_web_generator.py   # backward-compatible shim (old name, still works)
├── evaluate_ocr.py               # CER / WER / IoU evaluator
├── run_eval_tesseract.py         # end-to-end Tesseract evaluation runner
├── eval_winrt_ocr.py             # evaluate an ImageTrans .itp OCR result (WinRT / macOS Vision)
├── make_report_html.py           # build a bilingual HTML report from an eval report JSON
└── ocr_dataset_<lang>/           # generated dataset (gitignored), one dir per language
    ├── images/                   #   PNG pages
    ├── ground_truth/             #   JSON ground truth
    ├── metadata.json
    └── eval_report.json          #   produced by run_eval_tesseract.py
```

## Requirements

- Python 3.9+
- [Playwright](https://playwright.dev/python/) — uses the **system Edge or Chrome**
  automatically, so **no browser download is needed**. If you must download
  Chromium, use a mirror in China:
  `PLAYWRIGHT_DOWNLOAD_HOST=https://registry.npmmirror.com/-/binary/playwright python -m playwright install chromium`
- For evaluation: [Tesseract](https://github.com/tesseract-ocr/tesseract) with
  the matching `*.traineddata` (`ara`, `chi_sim`, `eng`), plus `editdistance`.

Install Python dependencies:

```bash
pip install playwright pillow numpy opencv-python editdistance matplotlib
```

> On a Chinese network you can use the Tsinghua PyPI mirror:
> `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple playwright pillow numpy opencv-python editdistance matplotlib`

## Usage

### 1. Generate the dataset (choose the language with `--lang`)

```bash
# Arabic (default), 5 clean A4 pages
python ocr_web_generator.py --lang ar --mode document --num 5

# Chinese, 5 clean A4 pages
python ocr_web_generator.py --lang zh --mode document --num 5

# English word cards with optional noise
python ocr_web_generator.py --lang en --mode card --num 20
```

The default output directory is language-aware: `./ocr_dataset_ar`,
`./ocr_dataset_zh`, `./ocr_dataset_en` — each language keeps its own directory so
`metadata.json` is never overwritten by another language. Override with `--output`.

Other options:

```bash
# different font / body size (pt); default font is language-specific
python ocr_web_generator.py --lang zh --font "Microsoft YaHei" --font-size 16

# explicit output directory
python ocr_web_generator.py --lang en --mode document --num 10 --output ./my_english_docs
```

> The old command name still works — `arabic_ocr_web_generator.py` is a shim
> that forwards to `ocr_web_generator.py` (e.g.
> `python arabic_ocr_web_generator.py --lang zh --mode document --num 5`).

### 2. Evaluate with Tesseract

```bash
python run_eval_tesseract.py --dataset ./ocr_dataset_ar
python run_eval_tesseract.py --dataset ./ocr_dataset_zh
python run_eval_tesseract.py --dataset ./ocr_dataset_en
```

The OCR language is auto-detected from each dataset's `metadata.json`
(`ara` / `chi_sim` / `eng`); override with `--lang ar|zh|en`.

Adjust the Tesseract path if it is not at the default location:

```bash
python run_eval_tesseract.py --tesseract "C:/path/to/tesseract.exe" --psm 6
```

The report is written to `<dataset>/eval_report.json`.

> Note: for Chinese, CER is the meaningful metric — WER splits on spaces, and
> Chinese text has no spaces between words.

### 3. Evaluate an OCR engine result (ImageTrans .itp project)

If you ran OCR on the generated images inside ImageTrans — with a
Windows.Media.Ocr plugin (engine `word level (WinRT)`) or the macOS Vision
plugin (engine `word level (mac)`) — save the project and evaluate it:

```bash
# Windows.Media.Ocr result
python eval_winrt_ocr.py --itp winrt.itp --dataset ./ocr_dataset_ar
# macOS Vision result
python eval_winrt_ocr.py --itp macocr.itp --dataset ./ocr_dataset_ar \
    --report ./ocr_dataset_ar/eval_report_mac.json
```

The script re-sorts the OCR word boxes into reading order from their geometry
(required for Arabic RTL), strips Arabic diacritics (harakat) and reports word
detection, word/char recognition accuracy and text CER/WER — per image,
aggregated, and as a JSON report. Language is auto-detected from the dataset
metadata (`--lang ar|zh|en` to override). Findings so far:

- **Windows.Media.Ocr** recognizes words accurately (char CER ~1.5%) but
  **drops all diacritics** and returns the word boxes **not in reading order**
  (naive WER ~90%) — consumers must re-sort the boxes by geometry.
- **macOS Vision** is slightly more accurate (char CER ~0.4%), **preserves most
  diacritics** and returns boxes **already in reading order**; its boxes are a
  little looser (mean IoU ~65% vs ~77%).

### 4. Generate a bilingual HTML report

Turn an evaluation report JSON into a self-contained bilingual (中文/EN) HTML
page with KPI tiles, bar charts, a per-image table and a plain-language
explanation of every metric. The verdict text adapts to the engine's behaviour
(dropped diacritics / scrambled order / loose boxes).

```bash
python make_report_html.py \
    --report ocr_dataset_ar/eval_report_mac.json \
    --output ocr_eval_report_mac.html \
    --overlay macocr_visualization.png
```

Existing examples: `ocr_eval_report.html` (WinRT) and `ocr_eval_report_mac.html`
(macOS Vision) in the repo root. Open them directly in a browser; a toggle in the
top-right switches 中文 ⇄ English and the page follows the OS light/dark theme.

## Ground truth format

Each `ground_truth/*.json`:

```json
{
  "image": "doc_001.png",
  "text": "人工智能正在深刻改变人们的生活与工作方式...",
  "words": [
    {"text": "人", "bbox": {"x": 180, "y": 216, "width": 74, "height": 74}},
    ...
  ],
  "characters": [{"index": 0, "char": "人", "x": ..., "y": ..., "width": ..., "height": ...}, ...],
  "metadata": {"language": "Chinese", "language_code": "zh", "direction": "ltr",
               "character_count": ..., "word_count": ...}
}
```

- Coordinates are **image pixels** (top-left origin), measured from the actual
  rendered image, so they align with the pixels exactly (verified ~99% ink
  coverage).
- `words` are in **reading order** (top-to-bottom, right-to-left per line for
  Arabic RTL; left-to-right for Chinese/English).
- `metadata` records `language`, `language_code` and `direction` so downstream
  tools (e.g. the evaluator) can configure the OCR engine automatically.
