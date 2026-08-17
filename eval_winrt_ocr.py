#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a Windows.Media.Ocr result stored in an ImageTrans .itp project
against the ground truth produced by ocr_web_generator.py.

Why this exists: the "word level (WinRT)" engine recognizes each Arabic/Chinese/
English word accurately but (for Arabic RTL) returns the word boxes in a
scrambled order and drops diacritics. This script therefore:

  1) reads every image's OCR boxes (text + geometry) from the .itp project,
  2) re-sorts the boxes into reading order from their geometry
     (top-to-bottom, right-to-left for Arabic / left-to-right for Chinese/English),
  3) strips Arabic diacritics (harakat) before comparing text,
  4) compares against the dataset ground truth: word detection (IoU),
     word recognition accuracy, char-level CER and text CER/WER,
  5) prints a per-image table + aggregate summary and writes a JSON report.

Usage:
  python eval_winrt_ocr.py --itp winrt.itp --dataset ./ocr_dataset_ar
  python eval_winrt_ocr.py --itp winrt.itp --dataset ./ocr_dataset_zh --report ./my_report.json

Dependencies:
  pip install editdistance        # CER/WER
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import editdistance

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- language table (mirrors ocr_web_generator.LANGUAGES) ----
LANG_INFO = {
    "ar": {"name": "Arabic", "direction": "rtl"},
    "zh": {"name": "Chinese", "direction": "ltr"},
    "en": {"name": "English", "direction": "ltr"},
}
LANG_BY_NAME = {v["name"]: k for k, v in LANG_INFO.items()}

# Arabic combining diacritics / harakat range (U+064B..U+065F) + superscript alef (U+0670)
AR_DIACRITICS = re.compile("[\\u064B-\\u065F\\u0670]")


def detect_lang(dataset_dir, itp, lang_arg):
    """Resolve the language: explicit --lang > dataset metadata.json > itp settings."""
    if lang_arg in LANG_INFO:
        return lang_arg
    meta = Path(dataset_dir) / "metadata.json"
    if meta.exists():
        try:
            info = json.load(open(meta, encoding="utf-8")).get("dataset_info", {})
            code = info.get("language_code") or LANG_BY_NAME.get(info.get("language"))
            if code in LANG_INFO:
                return code
        except Exception:
            pass
    src = itp.get("settings", {}).get("sourceLang")
    if src in LANG_INFO:
        return src
    raise SystemExit(
        "[error] cannot detect language; pass --lang ar|zh|en explicitly")


def normalize_text(text, lang):
    """Remove Arabic diacritics (harakat) that the WinRT engine cannot output."""
    return AR_DIACRITICS.sub("", text) if lang == "ar" else text


def box_geometry(b):
    """Normalize an itp box geometry dict (accepts WinRT-style X/Y or x/y keys)."""
    g = b["geometry"]
    return {"x": g.get("X", g.get("x")), "y": g.get("Y", g.get("y")),
            "width": g.get("width"), "height": g.get("height")}


def iou(a, b):
    x1 = max(a["x"], b["x"]); y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def cer(pred, gt):
    if not gt:
        return 1.0 if pred else 0.0
    return editdistance.eval(pred, gt) / len(gt)


def wer(pred_words, gt_words):
    if not gt_words:
        return 1.0 if pred_words else 0.0
    return editdistance.eval(pred_words, gt_words) / len(gt_words)


def reading_order(boxes, direction):
    """Return indices of boxes in reading order: top-to-bottom, and within each
    line right-to-left (rtl) or left-to-right (ltr). Groups boxes into lines by
    vertical center proximity."""
    if not boxes:
        return []
    geos = [b["bbox"] if "bbox" in b else b for b in boxes]
    heights = [g["height"] for g in geos]
    med_h = statistics.median(heights) if heights else 20.0
    gap = 0.6 * med_h
    centers = [g["y"] + g["height"] / 2 for g in geos]
    idx = sorted(range(len(geos)), key=lambda i: centers[i])

    lines = []
    cur = [idx[0]]
    cur_mean = centers[idx[0]]
    for i in idx[1:]:
        if centers[i] - cur_mean < gap:
            cur.append(i)
            cur_mean = (cur_mean * (len(cur) - 1) + centers[i]) / len(cur)
        else:
            lines.append(cur)
            cur = [i]
            cur_mean = centers[i]
    lines.append(cur)

    order = []
    for line in lines:
        line.sort(key=lambda i: geos[i]["x"], reverse=(direction == "rtl"))
        order.extend(line)
    return order


def evaluate_image(gt, ocr_boxes, lang, iou_thr=0.5):
    """Compare one image's OCR boxes against its ground truth."""
    direction = LANG_INFO[lang]["direction"]
    gt_words = gt["words"]
    gt_tokens = [w["text"] for w in gt_words]
    gt_norm = [normalize_text(t, lang) for t in gt_tokens]
    n_gt = len(gt_words)
    n_ocr = len(ocr_boxes)

    # detection + recognition via best-IoU matching (order-independent)
    best_iou_gt, matches = [], []
    for gi, gw in enumerate(gt_words):
        best, bidx = 0.0, -1
        for oi, ow in enumerate(ocr_boxes):
            v = iou(gw["bbox"], ow["bbox"])
            if v > best:
                best, bidx = v, oi
        best_iou_gt.append(best)
        if best >= iou_thr:
            matches.append((gi, bidx, best))

    recog_exact = recog_norm = char_err_norm = char_total_norm = 0
    for gi, oi, _ in matches:
        o, g = ocr_boxes[oi]["text"], gt_words[gi]["text"]
        if o == g:
            recog_exact += 1
        on, gn = normalize_text(o, lang), normalize_text(g, lang)
        if on == gn:
            recog_norm += 1
        char_err_norm += editdistance.eval(on, gn)
        char_total_norm += len(gn)

    # text metrics in reading order (as a consumer would assemble the text)
    order = reading_order(ocr_boxes, direction)
    seq_ro = [normalize_text(ocr_boxes[i]["text"], lang) for i in order]
    # text metrics in the raw returned order (to surface the ordering problem)
    seq_naive = [normalize_text(b["text"], lang) for b in ocr_boxes]

    return {
        "image": gt.get("image"),
        "gt_words": n_gt,
        "ocr_boxes": n_ocr,
        "word_alignment_ok": n_ocr == n_gt,
        "detected": len(matches),
        "detection_recall": len(matches) / n_gt if n_gt else 0.0,
        "recognition_exact": recog_exact / n_gt if n_gt else 0.0,
        "recognition_norm": recog_norm / n_gt if n_gt else 0.0,
        "char_cer_norm": char_err_norm / char_total_norm if char_total_norm else 0.0,
        "mean_best_iou": statistics.mean(best_iou_gt) if best_iou_gt else 0.0,
        "text_cer_reading_order": cer(" ".join(seq_ro), " ".join(gt_norm)),
        "text_wer_reading_order": wer(seq_ro, gt_norm),
        "text_cer_naive": cer(" ".join(seq_naive), " ".join(gt_norm)),
        "text_wer_naive": wer(seq_naive, gt_norm),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a Windows.Media.Ocr result (ImageTrans .itp) against "
                    "the ground truth from ocr_web_generator.py")
    ap.add_argument("--itp", required=True, help="path to the ImageTrans .itp project file")
    ap.add_argument("--dataset", required=True,
                    help="path to the generated dataset (contains ground_truth/ and metadata.json)")
    ap.add_argument("--lang", choices=["auto", "ar", "zh", "en"], default="auto",
                    help="language: auto (from dataset metadata or itp, default), ar/zh/en")
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU threshold for a detected word (default 0.5)")
    ap.add_argument("--report", default=None,
                    help="output JSON report path (default: <dataset>/eval_report_winrt.json)")
    args = ap.parse_args()

    itp = json.load(open(args.itp, encoding="utf-8"))
    lang = detect_lang(args.dataset, itp, args.lang)
    info = LANG_INFO[lang]
    engine = itp.get("previousOCR", {}).get("engineName", "word level (WinRT)")

    dataset = Path(args.dataset)
    gt_dir = dataset / "ground_truth"
    iou_thr = args.iou_threshold

    per_image = []
    for img, data in sorted(itp["images"].items()):
        gt_file = gt_dir / (Path(img).stem + ".json")
        if not gt_file.exists():
            print(f"[skip] {img}: no ground truth in {gt_dir}")
            continue
        gt = json.load(open(gt_file, encoding="utf-8"))
        ocr_boxes = []
        for b in data.get("boxes", []):
            geo = box_geometry(b)
            if None in (geo["x"], geo["y"], geo["width"], geo["height"]):
                continue
            ocr_boxes.append({"text": b.get("text", ""), "bbox": geo})
        per_image.append(evaluate_image(gt, ocr_boxes, lang, iou_thr))

    if not per_image:
        raise SystemExit("[error] no matching images found between the itp and the dataset")

    def avg(k):
        return statistics.mean(r[k] for r in per_image)

    n = len(per_image)
    print(f"\nLanguage: {info['name']} ({lang}, {info['direction']})  |  "
          f"Engine: {engine}  |  IoU threshold: {iou_thr:.2f}")
    print(f"Images evaluated: {n}\n")

    hdr = (f"{'image':<14}{'GT':>4}{'OCR':>5} | "
           f"{'det%':>5} {'rec_exact%':>10} {'rec_norm%':>10} {'charCER%':>8} {'boxIoU%':>7} | "
           f"{'CER(ro)%':>8} {'WER(ro)%':>8} {'CER(naive)%':>10} {'WER(naive)%':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in per_image:
        print(f"{r['image']:<14}{r['gt_words']:>4}{r['ocr_boxes']:>5} | "
              f"{r['detection_recall']*100:>5.1f} {r['recognition_exact']*100:>9.1f}% "
              f"{r['recognition_norm']*100:>9.1f}% {r['char_cer_norm']*100:>7.1f}% "
              f"{r['mean_best_iou']*100:>6.1f}% | "
              f"{r['text_cer_reading_order']*100:>7.1f}% {r['text_wer_reading_order']*100:>7.1f}% "
              f"{r['text_cer_naive']*100:>9.1f}% {r['text_wer_naive']*100:>9.1f}%")

    agg = {
        "total_images": n,
        "detection_recall": avg("detection_recall"),
        "recognition_exact": avg("recognition_exact"),
        "recognition_norm": avg("recognition_norm"),
        "char_cer_norm": avg("char_cer_norm"),
        "mean_best_iou": avg("mean_best_iou"),
        "text_cer_reading_order": avg("text_cer_reading_order"),
        "text_wer_reading_order": avg("text_wer_reading_order"),
        "text_cer_naive": avg("text_cer_naive"),
        "text_wer_naive": avg("text_wer_naive"),
    }
    print("-" * len(hdr))
    print(f"{'MEAN':<14}{'':>9} | "
          f"{agg['detection_recall']*100:>5.1f} {agg['recognition_exact']*100:>9.1f}% "
          f"{agg['recognition_norm']*100:>9.1f}% {agg['char_cer_norm']*100:>7.1f}% "
          f"{agg['mean_best_iou']*100:>6.1f}% | "
          f"{agg['text_cer_reading_order']*100:>7.1f}% {agg['text_wer_reading_order']*100:>7.1f}% "
          f"{agg['text_cer_naive']*100:>9.1f}% {agg['text_wer_naive']*100:>9.1f}%")

    diacritics_note = ("  (low: the engine drops Arabic diacritics)"
                       if agg['recognition_norm'] - agg['recognition_exact'] > 0.15 else "")
    order_note = ("  (large gap = boxes must be re-sorted into reading order)"
                  if agg['text_wer_naive'] - agg['text_wer_reading_order'] > 0.10 else "")

    print("\n===== Summary =====")
    print(f"Word detection (GT word with a box at IoU>={iou_thr}): "
          f"{agg['detection_recall']*100:.1f}%")
    print(f"Word recognition, exact as returned:            {agg['recognition_exact']*100:.1f}%"
          + diacritics_note)
    print(f"Word recognition, after diacritic normalization:{agg['recognition_norm']*100:.1f}%")
    print(f"Char-level CER on matched words (normalized):   {agg['char_cer_norm']*100:.1f}%")
    print(f"Mean best box IoU:                              {agg['mean_best_iou']*100:.1f}%")
    print(f"Text CER / WER in reading order (normalized):   "
          f"{agg['text_cer_reading_order']*100:.1f}% / {agg['text_wer_reading_order']*100:.1f}%")
    print(f"Text CER / WER in returned order (normalized):  "
          f"{agg['text_cer_naive']*100:.1f}% / {agg['text_wer_naive']*100:.1f}%"
          + order_note)

    report = {
        "engine": engine,
        "language": info["name"],
        "language_code": lang,
        "direction": info["direction"],
        "iou_threshold": iou_thr,
        "itp": args.itp,
        "dataset": str(dataset),
        "aggregate": agg,
        "per_image": per_image,
    }
    report_path = Path(args.report) if args.report else dataset / "eval_report_winrt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
