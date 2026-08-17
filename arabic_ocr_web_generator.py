#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web-based Arabic OCR dataset generator.
Renders HTML with Playwright and extracts exact character/word positions
via DOM Range.getBoundingClientRect(), so Arabic shaping, ligatures and
diacritics are rendered natively by the browser (no corrupted PDF text layer).
"""

import os
import sys
import json
import uuid
import random
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from playwright.sync_api import sync_playwright
import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid GBK console errors when printing Arabic/unicode on Windows
except Exception:
    pass

# ---- A4 page geometry (300 DPI) ----
A4_W = 2480            # 210mm @ 300dpi
A4_H = 3508            # 297mm @ 300dpi
A4_MARGIN = 180        # page margin ~0.6 inch
PT2PX = 300.0 / 72.0   # 1pt = 4.1667 CSS px @ 300dpi

# ---- Arabic paragraph corpus (Modern Standard Arabic, with diacritics/numbers/punctuation) ----
ARABIC_PARAGRAPHS = [
    "تُعَدُّ اللغةُ العربيةُ من أقدمِ اللغاتِ الساميةِ وأوسعِها انتشاراً، ويتحدثُ بها أكثرُ من ٤٢٢ مليونَ إنسانٍ حولَ العالمِ.",
    "بدأتِ الثورةُ الصناعيةُ في بريطانيا خلالَ القرنِ الثامنِ عشر، وشهدَت تحوّلاتٍ اقتصاديةً واجتماعيةً هائلة.",
    "بلغَ عددُ سكانِ القاهرةِ حوالي ٢١ مليونَ نسمةٍ في عامِ ٢٠٢٤م، وهي المدينةُ الأكبرُ في إفريقيا والشرقِ الأوسطِ.",
    "قالَ الكاتبُ: «إنَّ العلمَ نورٌ»، ثمّ أضافَ: «والجهلُ ظلامٌ». هل توافقُ على ذلكَ؟",
    "تتكوّنُ الخليةُ الحيةُ من النواةِ والسيتوبلازمِ والغشاءِ البلازميِّ، وتؤدي كلٌّ من هذه المكوّناتِ وظيفةً محددةً.",
    "يبلغُ إجماليُّ الإنفاقِ على التعليمِ ١٥٫٥ مليارَ دولارٍ سنوياً، بزيادةٍ قدرُها ٣٫٨٪ مقارنةً بالعامِ الماضي.",
    "أعلنَ البنكُ المركزيُّ خفضَ سعرِ الفائدةِ إلى ٩٫٢٥٪ اعتباراً من شهرِ مارسَ، بهدفِ تحفيزِ الاستثمارِ في قطاعِ الصناعةِ.",
    "وافقَ مجلسُ الإدارةِ على الميزانيةِ الجديدةِ في اجتماعٍ عُقِدَ يومَ الاثنينِ الموافقِ ١٧ أغسطسَ ٢٠٢٦م، بأغلبيةِ ٨٥٪ من الأصواتِ.",
    "الرياضُ عاصمةُ المملكةِ العربيةِ السعوديةِ، وتقعُ في الجزءِ الشرقيِّ من شبهِ الجزيرةِ العربيةِ، على ارتفاعٍ يبلغُ ٦٠٠ مترٍ فوقَ سطحِ البحرِ.",
    "دراسةٌ حديثةٌ تشيرُ إلى أنّ التمرينَ المنتظمَ لمدّةِ ٣٠ دقيقةً يومياً يقلّلُ خطرَ الإصابةِ بأمراضِ القلبِ بنسبةِ ٢٥٪.",
    "أنشأتِ الدولةُ خمسينَ مدرسةً جديدةً هذا العام، منها عشرونَ مدرسةً في المناطقِ الريفيةِ وثلاثونَ في المدنِ الكبرى.",
    "تبلغُ مساحةُ المملكةِ حوالي ٢٬١٤٩٬٦٩٠ كيلومتراً مربعاً، وتحتضنُ تراثاً ثقافياً عريقاً يمتدُّ لآلافِ السنين.",
    "يُعدُّ الذكاءُ الاصطناعيُّ من أبرزِ التقنياتِ التي تُغيّرُ ملامحَ القرنِ الحادي والعشرين، وتتسارعُ تطبيقاتُهُ في مجالاتِ الطبِّ والتعليمِ والصناعةِ.",
    "تعتمدُ أنظمةُ التعرفِ الضوئيِّ على تحويلِ النصوصِ المطبوعةِ أو المكتوبةِ بخطِّ اليدِ إلى نصوصٍ رقميةٍ قابلةٍ للبحثِ والمعالجةِ الآليةِ.",
    "تتفاوتُ جودةُ صورِ المستنداتِ الواردةِ من الماسحاتِ الضوئيةِ، وقد تؤدي الظلالُ والإضاءةُ غيرُ المنتظمةِ إلى تقليلِ دقةِ نتائجِ القراءةِ الآليةِ.",
    "تشملُ اللغةُ العربيةُ تنوعاً واسعاً من الخطوطِ واللهجاتِ، ويُعدُّ الحفاظُ على التراثِ العربيِّ المخطوطِ من أولوياتِ المراكزِ الثقافيةِ والبحثيةِ.",
    "بلغَ إنتاجُ المملكةِ من القمحِ في الموسمِ الماضي ما يقاربُ مليونَ طنٍ، بزيادةٍ بلغتْ ١٥٪ عن العامِ السابقِ، وفقَ البياناتِ الرسميةِ.",
    "أكدَ الخبراءُ أنّ الحفاظَ على البيئةِ يتطلبُ تعاوناً دولياً مشتركاً، وتقليلَ انبعاثاتِ الكربونِ بنسبةِ ٤٥٪ بحلولِ عامِ ٢٠٣٠م.",
    "تُعقدُ القمةُ العربيةُ هذا العامَ في العاصمةِ، بمشاركةِ وفودٍ من اثنتينِ وعشرينَ دولةً، لمناقشةِ قضايا التنميةِ والتعليمِ والطاقةِ.",
    "يستمرُّ العملُ في مشروعِ توسعةِ المسجدِ الحرامِ، الذي سيسعُ عندَ اكتمالِهِ لأكثرَ من ثلاثةِ ملايينَ مصلٍّ في وقتٍ واحدٍ.",
]

PAGE_TITLES = [
    "التقريرُ السنويُّ للعامِ ٢٠٢٦م",
    "نشرةُ الأخبارِ الثقافيةِ والاقتصاديةِ",
    "مقالةٌ في اللغةِ والأدبِ العربيِّ",
    "ملخصُ البحثِ العلميِّ",
    "الصفحةُ الأولى من المجلةِ",
]


class ArabicOCRWebGenerator:
    """Web-based Arabic OCR data generator."""

    def __init__(self, output_dir="./ocr_dataset_web"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.gt_dir = self.output_dir / "ground_truth"
        self.html_dir = self.output_dir / "html_templates"

        # create directories
        for dir_path in [self.images_dir, self.gt_dir, self.html_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Arabic text corpus (varied length and complexity)
        self.arabic_corpus = {
            "words": [
                "مرحبا", "السلام", "شكرا", "جزيلا", "اللغة",
                "العربية", "جميل", "عالم", "ذكاء", "اصطناعي",
                "تعلم", "آلة", "معالجة", "نصوص", "تقنية",
                "تطوير", "برمجة", "بيانات", "خوارزميات", "حوسبة",
                "سحابية", "أمن", "سيبراني", "إنترنت", "أشياء",
                "واقع", "افتراضي", "ذكاء", "تنافس", "ابتكار"
            ],
            "phrases": [
                "مرحبا بك في عالم الذكاء الاصطناعي",
                "اللغة العربية من أجمل اللغات في العالم",
                "تعلم الآلة هو فرع من فروع الذكاء الاصطناعي",
                "معالجة اللغات الطبيعية تساعد في فهم النصوص",
                "تقنيات OCR تستخدم لتحويل الصور إلى نصوص",
                "بسم الله الرحمن الرحيم",
                "الحمد لله رب العالمين",
                "السلام عليكم ورحمة الله وبركاته",
                "شكرا جزيلا على مساعدتكم",
                "التكنولوجيا تغير العالم بشكل سريع",
                "الذكاء الاصطناعي يفتح آفاقا جديدة",
                "تعلم اللغة العربية مهم للفهم الثقافي"
            ],
            "sentences": [
                "في العصر الحديث، أصبح الذكاء الاصطناعي جزءاً لا يتجزأ من حياتنا اليومية",
                "تعد معالجة اللغة العربية من التحديات الكبيرة في مجال التعلم الآلي",
                "تطبيقات التعرف الضوئي على الحروف تساعد في رقمنة المخطوطات العربية",
                "الخط العربي يتميز بجماله وتنوعه، وهو فن يحظى بتقدير عالمي",
                "تساهم التقنيات الحديثة في تسهيل عملية تعلم اللغة العربية للناطقين بغيرها"
            ]
        }

        # font configs
        self.font_configs = [
            {"family": "Arial", "size": 20, "weight": "normal"},
            {"family": "Times New Roman", "size": 24, "weight": "normal"},
            {"family": "Tahoma", "size": 22, "weight": "normal"},
            {"family": "Simplified Arabic", "size": 26, "weight": "normal"},
            {"family": "Arial", "size": 18, "weight": "bold"},
            {"family": "Traditional Arabic", "size": 28, "weight": "normal"},
        ]

        # background color configs
        self.background_configs = [
            {"bg": "#FFFFFF", "fg": "#000000"},
            {"bg": "#FFF8F0", "fg": "#1A1A1A"},
            {"bg": "#F5F5F5", "fg": "#000000"},
            {"bg": "#FFFFF0", "fg": "#2C2C2C"},
            {"bg": "#FAFAFA", "fg": "#0A0A0A"},
        ]

        self.metadata = {
            "dataset_info": {
                "name": "Arabic OCR Web Dataset",
                "language": "Arabic",
                "created": datetime.now().isoformat(),
                "total_samples": 0
            },
            "samples": []
        }

    def generate_html(self, text, font_config, bg_config, width=1200, height=300):
        """Generate HTML containing Arabic text (single text node)."""
        html_template = f'''<!DOCTYPE html>
<html dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background-color: {bg_config["bg"]};
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: "{font_config['family']}", Arial, sans-serif;
        }}
        .text-container {{
            width: {width - 40}px;
            padding: 20px;
            background-color: {bg_config["bg"]};
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .arabic-text {{
            font-size: {font_config['size']}px;
            font-weight: {font_config['weight']};
            color: {bg_config["fg"]};
            text-align: right;
            line-height: 1.8;
            direction: rtl;
            unicode-bidi: embed;
            word-spacing: 3px;
            letter-spacing: 0.5px;
        }}
        /* NOTE: do NOT wrap characters/words in spans (especially display:inline-block);
           that breaks Arabic letter shaping. Positions are measured with
           Range.getBoundingClientRect() instead. */
    </style>
</head>
<body>
    <div class="text-container">
        <div class="arabic-text" id="arabic-text">
            {text}
        </div>
    </div>
</body>
</html>'''
        return html_template

    def _launch_browser(self, p):
        """Prefer a system-installed Edge/Chrome to avoid downloading Chromium; fall back to the bundled engine."""
        for channel in ("msedge", "chrome"):
            try:
                return p.chromium.launch(headless=True, channel=channel)
            except Exception:
                continue
        return p.chromium.launch(headless=True)

    def generate_html_page(self, title, paragraphs, font_config, bg_config=None,
                           page_w=A4_W, page_h=A4_H, margin=A4_MARGIN):
        """Generate an A4 document-page HTML: title + multiple Arabic paragraphs.
        page_w/page_h are real pixel sizes (default A4 @ 300 DPI = 2480x3508).
        Font size is converted pt -> px @ 300dpi to keep real document layout."""
        if bg_config is None:
            bg_config = {"bg": "#FFFFFF", "fg": "#000000"}
        body_px = round(font_config['size'] * PT2PX)
        title_px = round(body_px * 1.8)
        para_gap = round(body_px * 0.8)
        paras_html = "\n".join(f"<p>{p}</p>" for p in paragraphs)
        return f'''<!DOCTYPE html>
<html dir="rtl">
<head>
<meta charset="UTF-8">
<style>
    body {{ margin: 0; padding: 0; background: {bg_config['bg']}; }}
    #arabic-page {{
        width: {page_w}px; height: {page_h}px; box-sizing: border-box;
        padding: {margin}px; background: {bg_config['bg']}; color: {bg_config['fg']};
        direction: rtl; text-align: right; overflow: hidden;
        font-family: "{font_config['family']}", Tahoma, sans-serif;
        font-size: {body_px}px; font-weight: {font_config['weight']};
        line-height: 1.9;
    }}
    #arabic-page h1 {{
        text-align: center; font-size: {title_px}px; font-weight: bold;
        margin: 0 0 {para_gap}px;
    }}
    #arabic-page p {{ margin: 0 0 {para_gap}px; text-align: justify; }}
</style>
</head>
<body>
<div id="arabic-page">
    <h1>{title}</h1>
    {paras_html}
</div>
</body>
</html>'''

    def render_and_extract_positions(self, html_content, output_path,
                                     root_id="arabic-text", viewport=(1200, 800)):
        """Render HTML and extract word/character positions.

        Key: keep the text as a plain text node — do NOT wrap it in spans
        (otherwise Arabic shaping/joining is broken). Positions are measured
        with Range.getBoundingClientRect(); coordinates are absolute viewport
        pixels which equal image pixels for a full_page screenshot at scroll 0.
        root_id: id of the element holding the text; default "arabic-text",
        use "arabic-page" for document pages.
        """
        with sync_playwright() as p:
            browser = self._launch_browser(p)
            page = browser.new_page(viewport={'width': viewport[0], 'height': viewport[1]})

            page.set_content(html_content)
            page.wait_for_timeout(300)

            # screenshot (screenshot origin == image origin)
            page.screenshot(path=str(output_path), full_page=True)

            # measure char/word boxes with Range (walk all text nodes under the root,
            # preserving document/reading order)
            positions = page.evaluate('''
                (rootId) => {
                    const root = document.getElementById(rootId);
                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
                    const nodes = [];
                    let n;
                    while ((n = walker.nextNode())) nodes.push(n);

                    const measure = (node, start, end) => {
                        const r = document.createRange();
                        r.setStart(node, start);
                        r.setEnd(node, end);
                        const b = r.getBoundingClientRect();
                        return { x: b.left, y: b.top, width: b.width, height: b.height };
                    };

                    const characters = [];
                    const words = [];
                    for (const node of nodes) {
                        const data = node.data;
                        for (let i = 0; i < data.length; i++) {
                            const m = measure(node, i, i + 1);
                            characters.push({
                                index: i,
                                char: data[i],
                                x: m.x, y: m.y, width: m.width, height: m.height,
                                center_x: m.x + m.width / 2,
                                center_y: m.y + m.height / 2
                            });
                        }
                        const re = /\\S+/g;
                        let mm;
                        while ((mm = re.exec(data))) {
                            const b = measure(node, mm.index, mm.index + mm[0].length);
                            words.push({
                                text: mm[0],
                                bbox: { x: b.x, y: b.y, width: b.width, height: b.height },
                                center: { x: b.x + b.width / 2, y: b.y + b.height / 2 }
                            });
                        }
                    }

                    return { characters: characters, words: words };
                }
            ''', root_id)

            browser.close()
            return positions

    def add_noise(self, image_path, noise_type="gaussian", intensity=1.0, boxes=None):
        """Add noise to the image. When boxes is given, rotation noise also
        transforms the coordinates; other noise types keep boxes valid because
        they do not move pixels."""
        import math
        img = Image.open(image_path)

        if noise_type == "gaussian":
            img_array = np.array(img)
            noise = np.random.normal(0, intensity * 10, img_array.shape)
            noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(noisy_array)

        elif noise_type == "salt_pepper":
            img_array = np.array(img)
            salt_pepper = np.random.random(img_array.shape[:2])
            img_array[salt_pepper < intensity * 0.01] = 0
            img_array[salt_pepper > 1 - intensity * 0.01] = 255
            img = Image.fromarray(img_array)

        elif noise_type == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=intensity * 0.5))

        elif noise_type == "rotation":
            angle = random.uniform(-intensity * 2, intensity * 2)
            # rotate around the center keeping canvas size (matches the coordinate transform), white fill
            img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
            if boxes is not None:
                self._transform_boxes(angle, img.width, img.height, boxes)

        img.save(image_path)

    def _transform_boxes(self, angle, w, h, boxes):
        """Apply the same coordinate transform as PIL rotate to the boxes.
        Supports both {'bbox':{x,y,width,height}} (word) and {x,y,width,height} (char)."""
        import math
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        cx, cy = w / 2.0, h / 2.0

        def rot_point(x, y):
            return (cx + (x - cx) * c + (y - cy) * s,
                    cy - (x - cx) * s + (y - cy) * c)

        for item in boxes:
            b = item['bbox'] if isinstance(item, dict) and 'bbox' in item else item
            if not isinstance(b, dict) or 'width' not in b or 'height' not in b:
                continue
            corners = [(b['x'], b['y']),
                       (b['x'] + b['width'], b['y']),
                       (b['x'], b['y'] + b['height']),
                       (b['x'] + b['width'], b['y'] + b['height'])]
            xs = [rot_point(x, y)[0] for x, y in corners]
            ys = [rot_point(x, y)[1] for x, y in corners]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            b['x'], b['y'] = x0, y0
            b['width'], b['height'] = x1 - x0, y1 - y0

    def generate_word_boxes(self, char_positions, min_chars=1):
        """Return the word boxes measured in the browser (text + bbox).
        min_chars is kept for compatibility only — an OCR GT must not drop short words."""
        if char_positions and 'words' in char_positions:
            return char_positions['words']
        if not char_positions or 'characters' not in char_positions:
            return []

        chars = char_positions['characters']
        if not chars:
            return []

        words = []
        current_word = []
        arabic_space = False

        for i, char_data in enumerate(chars):
            char = char_data['char']

            # detect whitespace (Arabic space or regular space)
            if char in [' ', '‌', '‍']:  # zero-width joiners etc.
                if current_word and len(current_word) >= min_chars:
                    words.append(self._merge_chars_to_word(current_word))
                current_word = []
                continue

            current_word.append(char_data)

        # handle the last word
        if current_word and len(current_word) >= min_chars:
            words.append(self._merge_chars_to_word(current_word))

        return words

    def _merge_chars_to_word(self, chars):
        """Merge a list of characters into a word."""
        if not chars:
            return None

        min_x = min(c['x'] for c in chars)
        min_y = min(c['y'] for c in chars)
        max_x = max(c['x'] + c['width'] for c in chars)
        max_y = max(c['y'] + c['height'] for c in chars)

        text = ''.join(c['char'] for c in chars)

        return {
            "text": text,
            "bbox": {
                "x": min_x,
                "y": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y
            },
            "center": {
                "x": (min_x + max_x) / 2,
                "y": (min_y + max_y) / 2
            }
        }

    def generate_ground_truth(self, text, char_positions, word_boxes, image_filename):
        """Build the ground-truth JSON dict."""
        characters = char_positions.get('characters', []) if isinstance(char_positions, dict) else []
        gt = {
            "image": image_filename,
            "text": text,
            "characters": characters,
            "words": word_boxes,
            "metadata": {
                "language": "Arabic",
                "direction": "rtl",
                "character_count": len(text.replace(' ', '').replace('‌', '').replace('‍', '')),
                "word_count": len(word_boxes)
            }
        }
        return gt

    def generate_dataset(self, num_samples=50):
        """Generate word-card samples (may include noise)."""
        print(f"Generating {num_samples} samples...")

        for sample_id in range(num_samples):
            try:
                # pick a random text
                text_type = random.choice(['words', 'phrases', 'sentences'])
                text = random.choice(self.arabic_corpus[text_type])

                # pick a random style
                font_config = random.choice(self.font_configs)
                bg_config = random.choice(self.background_configs)

                # randomly decide whether to add noise
                add_noise = random.random() < 0.3  # 30% chance of noise
                noise_types = ['gaussian', 'salt_pepper', 'blur', 'rotation']
                noise_type = random.choice(noise_types) if add_noise else None
                noise_intensity = random.uniform(0.3, 1.5) if add_noise else 0

                # generate HTML
                html = self.generate_html(text, font_config, bg_config)

                # save HTML
                html_filename = f"sample_{sample_id:04d}.html"
                html_path = self.html_dir / html_filename
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html)

                # render and extract positions
                image_filename = f"sample_{sample_id:04d}.png"
                image_path = self.images_dir / image_filename

                char_positions = self.render_and_extract_positions(html, image_path)

                # word-level annotations (from the original render; noise does not change text)
                word_boxes = self.generate_word_boxes(char_positions)

                # add noise (rotation transforms word/char coordinates; others keep boxes)
                if add_noise and noise_type:
                    boxes = word_boxes + char_positions.get('characters', [])
                    self.add_noise(image_path, noise_type, noise_intensity, boxes=boxes)

                # generate ground truth
                gt = self.generate_ground_truth(
                    text, char_positions, word_boxes, image_filename
                )

                # save ground truth
                gt_filename = f"sample_{sample_id:04d}.json"
                gt_path = self.gt_dir / gt_filename
                with open(gt_path, 'w', encoding='utf-8') as f:
                    json.dump(gt, f, ensure_ascii=False, indent=2)

                # update metadata
                self.metadata["samples"].append({
                    "id": sample_id,
                    "image": image_filename,
                    "ground_truth": gt_filename,
                    "text": text,
                    "font": font_config,
                    "has_noise": add_noise,
                    "noise_type": noise_type if add_noise else None
                })

                print(f"OK sample {sample_id + 1}/{num_samples}: {text[:20]}...")

            except Exception as e:
                print(f"FAIL sample {sample_id}: {str(e)}")
                continue

        # update metadata
        self.metadata["dataset_info"]["total_samples"] = len(self.metadata["samples"])

        # save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"\nDone! {len(self.metadata['samples'])} samples generated")
        print(f"Output directory: {self.output_dir}")
        print(f"Images: {self.images_dir}")
        print(f"Ground truth: {self.gt_dir}")

        return self.metadata

    def generate_document_dataset(self, num_pages=5, font_pt=14, font_family="Tahoma",
                                  seed=42, prefix="doc"):
        """Generate num_pages A4 Arabic document pages (title + multiple paragraphs),
        clean, without noise or rotation.
        Outputs: images/{prefix}_NNN.png + ground_truth/{prefix}_NNN.json"""
        rng = random.Random(seed)
        font_config = {"family": font_family, "size": font_pt, "weight": "normal"}
        bg_config = {"bg": "#FFFFFF", "fg": "#000000"}
        viewport = (A4_W, A4_H)

        for i in range(num_pages):
            try:
                title = rng.choice(PAGE_TITLES)
                n_paras = rng.randint(8, 11)
                paras = rng.sample(ARABIC_PARAGRAPHS,
                                   k=min(n_paras, len(ARABIC_PARAGRAPHS)))

                html = self.generate_html_page(title, paras, font_config, bg_config)

                image_filename = f"{prefix}_{i + 1:03d}.png"
                image_path = self.images_dir / image_filename
                pos = self.render_and_extract_positions(html, image_path,
                                                        root_id="arabic-page",
                                                        viewport=viewport)
                word_boxes = self.generate_word_boxes(pos)
                full_text = "\n".join([title] + paras)

                gt = self.generate_ground_truth(full_text, pos, word_boxes, image_filename)
                gt_filename = f"{prefix}_{i + 1:03d}.json"
                with open(self.gt_dir / gt_filename, "w", encoding="utf-8") as f:
                    json.dump(gt, f, ensure_ascii=False, indent=2)

                self.metadata["samples"].append({
                    "id": i, "image": image_filename, "ground_truth": gt_filename,
                    "text": full_text, "font": font_config, "has_noise": False,
                    "page_size": [A4_W, A4_H],
                })
                print(f"OK document {i + 1}/{num_pages}: {title[:24]} ({len(word_boxes)} words)")
            except Exception as e:
                print(f"FAIL document {i}: {str(e)}")

        self.metadata["dataset_info"]["total_samples"] = len(self.metadata["samples"])
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"\nDone: {len(self.metadata['samples'])} A4 pages -> {self.output_dir}")
        return self.metadata

    def visualize_ground_truth(self, image_path, gt_data, output_path=None):
        """Draw ground-truth word boxes over the image."""
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # draw word bounding boxes
        for word in gt_data.get('words', []):
            bbox = word['bbox']
            draw.rectangle(
                [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']],
                outline=(255, 0, 0),
                width=2
            )
            # draw the word text
            draw.text(
                (bbox['x'], bbox['y'] - 20),
                word['text'],
                fill=(255, 0, 0)
            )

        if output_path:
            img.save(output_path)
        return img


def main():
    """Entry point."""
    import argparse
    ap = argparse.ArgumentParser(description="Arabic OCR dataset generator")
    ap.add_argument("--mode", choices=["document", "card"], default="document",
                    help="document=A4 multi-paragraph pages (default, no noise); card=word cards (may add noise)")
    ap.add_argument("--num", type=int, default=5, help="number of samples (default 5)")
    ap.add_argument("--output", default="./arabic_ocr_dataset", help="output directory")
    ap.add_argument("--font", default="Tahoma", help="font family (must support Arabic)")
    ap.add_argument("--font-size", type=int, default=14, help="body font size in pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    generator = ArabicOCRWebGenerator(args.output)

    if args.mode == "document":
        metadata = generator.generate_document_dataset(
            num_pages=args.num, font_pt=args.font_size,
            font_family=args.font, seed=args.seed)
    else:
        metadata = generator.generate_dataset(num_samples=args.num)

    # visualize the first sample (boxes overlaid, to check alignment)
    if metadata['samples']:
        first = metadata['samples'][0]
        with open(generator.gt_dir / first['ground_truth'], 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        vis_path = generator.output_dir / "visualization_example.png"
        generator.visualize_ground_truth(generator.images_dir / first['image'],
                                         gt_data, vis_path)
        print(f"Preview visualization: {vis_path}")

    print("\nDone! Use run_eval_tesseract.py for evaluation.")


if __name__ == "__main__":
    # deps: pip install playwright pillow numpy opencv-python editdistance
    # browser: the script auto-uses the system Edge/Chrome (no chromium download needed)
    main()
