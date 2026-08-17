# Arabic OCR Dataset Generator

Generate **Arabic OCR datasets**: real document images plus precise ground truth
(text and word-level bounding boxes), for evaluating OCR engines.

Images are rendered in a **headless browser (Playwright)** using the system
Edge/Chrome, so Arabic shaping, ligatures, diacritics (harakat) and RTL layout
are all rendered natively by the browser — the same way a real reader sees them.
Word/character positions are measured with DOM `Range.getBoundingClientRect()`,
which avoids the unreliable text layer that Word/LibreOffice PDFs produce for RTL
Arabic (scrambled runs, lost diacritics, misplaced punctuation).

## Features

- **A4 document pages** (default): 2480×3508 px (A4 @ 300 DPI), a centered title
  plus 8–11 vocalized Arabic paragraphs per page. Clean, no noise, no rotation.
- **Word cards** (optional): short words/phrases/sentences with optional
  augmentation (gaussian / salt & pepper / blur / rotation noise). Rotation
  transforms the GT boxes to match the rotated image.
- **Ground truth** per sample:
  - `text` — full page/sample text (logical order, with diacritics)
  - `words` — word-level boxes `{text, bbox:{x,y,width,height}}` in pixel coords,
    in RTL reading order
  - `characters` — per-character boxes
- **Evaluation**: run a real OCR engine (Tesseract with `ara`) and get
  CER / WER / word-box IoU.

## Project layout

```
.
├── arabic_ocr_web_generator.py   # dataset generator
├── evaluate_ocr.py               # CER / WER / IoU evaluator
├── run_eval_tesseract.py         # end-to-end Tesseract evaluation runner
└── arabic_ocr_dataset/           # generated dataset (gitignored)
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
  `ara.traineddata`, plus `editdistance`.

Install Python dependencies:

```bash
pip install playwright pillow numpy opencv-python editdistance matplotlib
```

> On a Chinese network you can use the Tsinghua PyPI mirror:
> `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple playwright pillow numpy opencv-python editdistance matplotlib`

## Usage

### 1. Generate the dataset (5 clean A4 pages by default)

```bash
python arabic_ocr_web_generator.py --mode document --num 5 --output ./arabic_ocr_dataset
```

Other options:

```bash
# different font / body size (pt)
python arabic_ocr_web_generator.py --font "Simplified Arabic" --font-size 16

# word cards with optional noise
python arabic_ocr_web_generator.py --mode card --num 20
```

### 2. Evaluate with Tesseract

```bash
python run_eval_tesseract.py --dataset ./arabic_ocr_dataset
```

Adjust the Tesseract path if it is not at the default location:

```bash
python run_eval_tesseract.py --tesseract "C:/path/to/tesseract.exe" --psm 6
```

The report is written to `arabic_ocr_dataset/eval_report.json`. Example result on
clean A4 pages (Tesseract `ara`, PSM 6):

```
Mean CER       : 0.168
Mean WER       : 0.771
Mean box IoU   : 0.735
```

> WER is high because Tesseract's Arabic model drops diacritics (harakat),
> misreads some Arabic-Indic digits, and makes occasional letter errors — the
> evaluation is designed to surface exactly such OCR weaknesses.

## Ground truth format

Each `ground_truth/*.json`:

```json
{
  "image": "doc_001.png",
  "text": "التقريرُ السنويُّ للعامِ ٢٠٢٦م\nالرياضُ عاصمةُ ...",
  "words": [
    {"text": "التقريرُ", "bbox": {"x": 1609.8, "y": 216, "width": 308.3, "height": 125}},
    ...
  ],
  "characters": [{"index": 0, "char": "ا", "x": ..., "y": ..., "width": ..., "height": ...}, ...],
  "metadata": {"language": "Arabic", "direction": "rtl", "character_count": ..., "word_count": ...}
}
```

- Coordinates are **image pixels** (top-left origin), measured from the actual
  rendered image, so they align with the pixels exactly (verified ~99% ink
  coverage).
- `words` are in **RTL reading order** (top-to-bottom, right-to-left per line).
