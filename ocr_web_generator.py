#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web-based multi-language OCR dataset generator (Arabic / Chinese / English).

Renders HTML with Playwright and extracts exact character/word positions
via DOM Range.getBoundingClientRect(), so Arabic shaping, ligatures and
diacritics (and CJK / Latin layout) are rendered natively by the browser
(no corrupted PDF text layer). Pass --lang ar|zh|en to choose the language.

Selecting a language:
  python ocr_web_generator.py --lang ar --mode document --num 5
  python ocr_web_generator.py --lang zh --mode document --num 5
  python ocr_web_generator.py --lang en --mode card --num 20
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
    sys.stdout.reconfigure(encoding="utf-8")  # avoid GBK console errors when printing Arabic/Chinese/unicode on Windows
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

ARABIC_PAGE_TITLES = [
    "التقريرُ السنويُّ للعامِ ٢٠٢٦م",
    "نشرةُ الأخبارِ الثقافيةِ والاقتصاديةِ",
    "مقالةٌ في اللغةِ والأدبِ العربيِّ",
    "ملخصُ البحثِ العلميِّ",
    "الصفحةُ الأولى من المجلةِ",
]

ARABIC_CORPUS = {
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

# ---- Chinese paragraph corpus (Simplified Chinese, with numbers/percentages/dates/punctuation) ----
CHINESE_PARAGRAPHS = [
    "人工智能正在深刻改变人们的生活与工作方式，从智能语音助手到自动驾驶汽车，各类应用层出不穷。",
    "据统计，2025年中国数字经济规模已超过六十万亿元人民币，占国内生产总值的比重持续上升。",
    "汉字是世界上使用历史最悠久的文字之一，从甲骨文到现代简体字，历经数千年演变仍充满活力。",
    "阅读不仅是获取知识的途径，更是培养思维能力和审美情趣的重要方式，应当从小养成习惯。",
    "科学家们认为，全球气候变暖是一个需要各国共同应对的严峻挑战，减少碳排放刻不容缓。",
    "教育部宣布，从今年秋季学期起，中小学将全面开设人工智能相关课程，培养创新人才。",
    "本月二十三日，市图书馆新馆正式对外开放，藏书量超过三百万册，设有电子阅览室和少儿专区。",
    "研究表明，每周进行一百五十分钟的中等强度运动，可以有效降低患心血管疾病的风险。",
    "五千年前，中华先民在黄河流域创造了灿烂的农业文明，农耕技术逐渐传播到世界各地。",
    "这家公司成立于二零一零年，如今已发展成为拥有两万多名员工的行业龙头企业，业务遍布全球。",
    "春天来了，公园里的桃花和杏花竞相开放，游客们纷纷驻足拍照，记录下这美好的时光。",
    "根据最新报告，智能手机的普及率已达到百分之九十，移动支付成为人们日常消费的主要方式。",
    "敦煌莫高窟保存着大量精美的壁画和彩塑，是研究古代丝绸之路文化的珍贵资料。",
    "为了提高教学质量，学校引入了先进的多媒体设备，并定期组织教师参加专业培训。",
    "中国的长城全长超过两万一千公里，横跨十五个省份，是世界文化遗产中的璀璨明珠。",
    "医生建议，成年人每天应保证七到八小时的睡眠，同时注意饮食均衡，多吃蔬菜水果。",
    "奥运会是全世界运动员展示体育精神的最高舞台，每四年举办一届，吸引数十亿观众关注。",
    "随着互联网技术的发展，远程办公和在线教育逐渐成为常态，改变了传统的生产和学习模式。",
    "环保组织呼吁公众减少使用一次性塑料制品，共同保护海洋生态，让地球家园更加美好。",
    "这本小说讲述了一位年轻医生在乡村工作的故事，语言朴实生动，深受读者喜爱，销量突破百万册。",
]

CHINESE_PAGE_TITLES = [
    "二〇二六年度工作总结报告",
    "文化科技与经济新闻简讯",
    "中国语言文字研究专题",
    "科学研究成果摘要",
    "杂志首刊目录与序言",
]

CHINESE_CORPUS = {
    "words": [
        "人工智能", "机器学习", "计算机", "深度学习", "神经网络",
        "数据科学", "自然语言", "图像识别", "语音识别", "机器人",
        "云计算", "大数据", "算法", "编程", "互联网",
        "网络安全", "创新发展", "科学研究", "高等教育", "智能终端"
    ],
    "phrases": [
        "你好，欢迎来到人工智能世界",
        "汉字是世界上使用历史最悠久的文字",
        "机器学习是人工智能的重要分支",
        "图像识别技术在医疗领域应用广泛",
        "语音助手已经成为人们生活的好帮手",
        "科技改变生活，创新引领未来",
        "今天天气晴朗，阳光明媚",
        "我们一起去公园散步吧",
        "谢谢你的帮助和关心",
        "读书使人进步，学习使人充实",
        "人工智能的发展日新月异",
        "保护环境，人人有责"
    ],
    "sentences": [
        "在现代社会，人工智能已经渗透到生活的方方面面，为人们带来了极大的便利",
        "中文信息处理是自然语言处理领域的重要研究方向，面临许多独特的挑战",
        "光学字符识别技术能够将印刷或手写的文字图像转化为可编辑的电子文本",
        "中国书法具有独特的艺术魅力，是中华文化的重要组成部分，深受世界人民喜爱",
        "随着科学技术的飞速发展，人们的学习方式和知识获取途径正在发生深刻的变化"
    ]
}

# ---- English paragraph corpus (with numbers/percentages/dates/punctuation) ----
ENGLISH_PARAGRAPHS = [
    "Artificial intelligence is transforming the way people live and work, from smart voice assistants to self-driving cars.",
    "According to the latest report, the global economy grew by 3.2 percent in 2025, driven largely by the technology sector.",
    "The English language has borrowed words from many sources, making it one of the most diverse and adaptable languages in the world.",
    "Reading is not only a way to gain knowledge but also an excellent habit that broadens the mind and enriches the imagination.",
    "Scientists warn that climate change poses a serious threat to the planet, and urgent action is needed to reduce carbon emissions.",
    "The university announced that it will open a new research center for artificial intelligence next September, with an initial budget of 50 million dollars.",
    "Founded in 1955, the company grew from a small workshop into a global manufacturer employing over 20,000 workers in more than 30 countries.",
    "Studies show that walking for 30 minutes a day can significantly reduce the risk of heart disease and improve overall health.",
    "The Great Wall of China, stretching more than 13,000 miles, was built over several dynasties to protect the northern border of the empire.",
    "Digital cameras capture images as arrays of pixels, each pixel storing color and brightness information that OCR software can analyze.",
    "Optical character recognition converts scanned documents, photographs, and screenshots into machine-readable text that can be edited and searched.",
    "The library's new wing houses over two million volumes, including rare manuscripts, ancient maps, and a dedicated section for children's books.",
    "As of August 2026, the smartphone market has reached 90 percent penetration, and mobile payments now account for the majority of retail transactions.",
    "Historians believe that the invention of the printing press in the fifteenth century revolutionized the spread of knowledge across Europe.",
    "Regular exercise, a balanced diet, and adequate sleep are the three pillars of a healthy lifestyle, according to medical experts.",
    "The Olympic Games bring together athletes from more than two hundred countries to compete in the spirit of fair play and friendship.",
    "With the rapid growth of remote work, companies are investing heavily in cloud computing, video conferencing, and digital collaboration tools.",
    "Environmental groups are urging consumers to reduce the use of single-use plastics in order to protect marine ecosystems and wildlife.",
    "The novelist spent five years writing his latest book, a gripping story about a young doctor who moves to a remote mountain village.",
    "Artificial intelligence is expected to contribute up to 15 trillion dollars to the global economy by 2030, according to a recent study.",
]

ENGLISH_PAGE_TITLES = [
    "Annual Report for the Year 2026",
    "Science, Technology and Culture News",
    "An Essay on the English Language",
    "Summary of Scientific Research",
    "First Issue of the Monthly Magazine",
]

ENGLISH_CORPUS = {
    "words": [
        "artificial", "intelligence", "machine", "learning", "computer",
        "vision", "natural", "language", "processing", "algorithm",
        "data", "science", "network", "robot", "cloud",
        "security", "innovation", "research", "education", "technology"
    ],
    "phrases": [
        "Welcome to the world of artificial intelligence",
        "Machine learning is a branch of artificial intelligence",
        "OCR technology converts images into searchable text",
        "The quick brown fox jumps over the lazy dog",
        "Technology changes the world at an incredible speed",
        "Artificial intelligence opens new possibilities",
        "Reading is the key to lifelong learning",
        "Practice makes perfect",
        "Data is the new oil",
        "Innovation drives progress and growth",
        "Communication is the foundation of cooperation",
        "Never stop learning and exploring"
    ],
    "sentences": [
        "In the modern era, artificial intelligence has become an inseparable part of our daily lives",
        "Optical character recognition helps digitize historical documents and manuscripts for future generations",
        "Natural language processing enables computers to understand and generate human language",
        "The development of deep learning has revolutionized the field of computer vision",
        "Cloud computing provides scalable resources that power everything from websites to scientific simulations"
    ]
}

# ---- shared background color configs (language-independent) ----
BACKGROUND_CONFIGS = [
    {"bg": "#FFFFFF", "fg": "#000000"},
    {"bg": "#FFF8F0", "fg": "#1A1A1A"},
    {"bg": "#F5F5F5", "fg": "#000000"},
    {"bg": "#FFFFF0", "fg": "#2C2C2C"},
    {"bg": "#FAFAFA", "fg": "#0A0A0A"},
]

# ---- per-language font lists (must be available on the system; Windows built-ins) ----
AR_FONTS = [
    {"family": "Arial", "size": 20, "weight": "normal"},
    {"family": "Times New Roman", "size": 24, "weight": "normal"},
    {"family": "Tahoma", "size": 22, "weight": "normal"},
    {"family": "Simplified Arabic", "size": 26, "weight": "normal"},
    {"family": "Arial", "size": 18, "weight": "bold"},
    {"family": "Traditional Arabic", "size": 28, "weight": "normal"},
]

ZH_FONTS = [
    {"family": "SimSun", "size": 22, "weight": "normal"},
    {"family": "SimHei", "size": 24, "weight": "normal"},
    {"family": "Microsoft YaHei", "size": 20, "weight": "normal"},
    {"family": "KaiTi", "size": 26, "weight": "normal"},
    {"family": "FangSong", "size": 22, "weight": "normal"},
    {"family": "SimHei", "size": 20, "weight": "bold"},
]

EN_FONTS = [
    {"family": "Arial", "size": 20, "weight": "normal"},
    {"family": "Times New Roman", "size": 24, "weight": "normal"},
    {"family": "Georgia", "size": 22, "weight": "normal"},
    {"family": "Courier New", "size": 22, "weight": "normal"},
    {"family": "Verdana", "size": 18, "weight": "normal"},
    {"family": "Arial", "size": 18, "weight": "bold"},
]

FONT_FALLBACKS = {"ar": "Tahoma", "zh": "SimSun", "en": "Arial"}

# ---- language registry ----
LANGUAGES = {
    "ar": {
        "name": "Arabic",
        "direction": "rtl",
        "tesseract": "ara",
        "default_font": "Tahoma",
        "line_height_card": 1.8,
        "line_height_doc": 1.9,
        "font_configs": AR_FONTS,
        "paragraphs": ARABIC_PARAGRAPHS,
        "page_titles": ARABIC_PAGE_TITLES,
        "corpus": ARABIC_CORPUS,
    },
    "zh": {
        "name": "Chinese",
        "direction": "ltr",
        "tesseract": "chi_sim",
        "default_font": "SimSun",
        "line_height_card": 1.7,
        "line_height_doc": 1.7,
        "font_configs": ZH_FONTS,
        "paragraphs": CHINESE_PARAGRAPHS,
        "page_titles": CHINESE_PAGE_TITLES,
        "corpus": CHINESE_CORPUS,
    },
    "en": {
        "name": "English",
        "direction": "ltr",
        "tesseract": "eng",
        "default_font": "Times New Roman",
        "line_height_card": 1.5,
        "line_height_doc": 1.5,
        "font_configs": EN_FONTS,
        "paragraphs": ENGLISH_PARAGRAPHS,
        "page_titles": ENGLISH_PAGE_TITLES,
        "corpus": ENGLISH_CORPUS,
    },
}


class OCRWebGenerator:
    """Web-based multi-language OCR data generator (Arabic / Chinese / English)."""

    def __init__(self, output_dir=None, language="ar"):
        if language not in LANGUAGES:
            raise ValueError(
                f"Unsupported language {language!r}; choose from {sorted(LANGUAGES)}")
        self.lang = language
        self.lang_info = LANGUAGES[language]

        if output_dir is None:
            output_dir = f"./ocr_dataset_{language}"
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.gt_dir = self.output_dir / "ground_truth"
        self.html_dir = self.output_dir / "html_templates"

        # create directories
        for dir_path in [self.images_dir, self.gt_dir, self.html_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # language-specific text corpus (varied length and complexity)
        self.corpus = self.lang_info["corpus"]
        self.font_configs = self.lang_info["font_configs"]
        self.background_configs = BACKGROUND_CONFIGS
        self.paragraphs = self.lang_info["paragraphs"]
        self.page_titles = self.lang_info["page_titles"]

        self.metadata = {
            "dataset_info": {
                "name": f"{self.lang_info['name']} OCR Web Dataset",
                "language": self.lang_info["name"],
                "language_code": self.lang,
                "direction": self.lang_info["direction"],
                "created": datetime.now().isoformat(),
                "total_samples": 0
            },
            "samples": []
        }

    def _font_stack(self, family):
        """CSS font-family stack for the current language (fallback guarantees glyph coverage)."""
        return f'"{family}", {FONT_FALLBACKS.get(self.lang, "sans-serif")}, sans-serif'

    def generate_html(self, text, font_config, bg_config, width=1200, height=300):
        """Generate HTML containing text in the selected language (single text node)."""
        direction = self.lang_info["direction"]
        dir_css = "direction: rtl; unicode-bidi: embed;" if direction == "rtl" else ""
        spacing_css = ("word-spacing: 3px;\n            letter-spacing: 0.5px;"
                       if self.lang == "ar" else "")
        text_align = "right" if direction == "rtl" else "left"
        html_template = f'''<!DOCTYPE html>
<html dir="{direction}">
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
            font-family: {self._font_stack(font_config['family'])};
        }}
        .text-container {{
            width: {width - 40}px;
            padding: 20px;
            background-color: {bg_config["bg"]};
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .doc-text {{
            font-size: {font_config['size']}px;
            font-weight: {font_config['weight']};
            color: {bg_config["fg"]};
            text-align: {text_align};
            line-height: {self.lang_info["line_height_card"]};
            {dir_css}
            {spacing_css}
        }}
        /* NOTE: do NOT wrap characters/words in spans (especially display:inline-block);
           that breaks Arabic letter shaping. Positions are measured with
           Range.getBoundingClientRect() instead. */
    </style>
</head>
<body>
    <div class="text-container">
        <div class="doc-text" id="doc-text">
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
        """Generate an A4 document-page HTML: title + multiple paragraphs.
        page_w/page_h are real pixel sizes (default A4 @ 300 DPI = 2480x3508).
        Font size is converted pt -> px @ 300dpi to keep real document layout."""
        if bg_config is None:
            bg_config = {"bg": "#FFFFFF", "fg": "#000000"}
        direction = self.lang_info["direction"]
        body_px = round(font_config['size'] * PT2PX)
        title_px = round(body_px * 1.8)
        para_gap = round(body_px * 0.8)
        paras_html = "\n".join(f"<p>{p}</p>" for p in paragraphs)
        # RTL pages right-align the title/body; LTR pages keep left alignment
        doc_align = "direction: rtl; text-align: right;" if direction == "rtl" else ""
        return f'''<!DOCTYPE html>
<html dir="{direction}">
<head>
<meta charset="UTF-8">
<style>
    body {{ margin: 0; padding: 0; background: {bg_config['bg']}; }}
    #doc-page {{
        width: {page_w}px; height: {page_h}px; box-sizing: border-box;
        padding: {margin}px; background: {bg_config['bg']}; color: {bg_config['fg']};
        {doc_align}
        font-family: {self._font_stack(font_config['family'])};
        font-size: {body_px}px; font-weight: {font_config['weight']};
        line-height: {self.lang_info["line_height_doc"]};
    }}
    #doc-page h1 {{
        text-align: center; font-size: {title_px}px; font-weight: bold;
        margin: 0 0 {para_gap}px;
    }}
    #doc-page p {{ margin: 0 0 {para_gap}px; text-align: justify; }}
</style>
</head>
<body>
<div id="doc-page">
    <h1>{title}</h1>
    {paras_html}
</div>
</body>
</html>'''

    def render_and_extract_positions(self, html_content, output_path,
                                     root_id="doc-text", viewport=(1200, 800)):
        """Render HTML and extract word/character positions.

        Key: keep the text as a plain text node — do NOT wrap it in spans
        (otherwise Arabic shaping/joining is broken). Positions are measured
        with Range.getBoundingClientRect(); coordinates are absolute viewport
        pixels which equal image pixels for a full_page screenshot at scroll 0.
        root_id: id of the element holding the text; default "doc-text",
        use "doc-page" for document pages.
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
            positions = page.evaluate(r'''
                ({ rootId, lang }) => {
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

                    // word tokenizer: for Chinese there are no spaces between words,
                    // so a plain \S+ run would span a whole line; instead split CJK
                    // and punctuation per character and group consecutive Latin
                    // letters / digits (ASCII + full-width) into one token.
                    const wordRe = (lang === 'zh')
                        ? /[0-9A-Za-z０-９Ａ-Ｚａ-ｚ]+|[^\s0-9A-Za-z０-９Ａ-Ｚａ-ｚ]/g
                        : /\S+/g;

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
                        let mm;
                        while ((mm = wordRe.exec(data))) {
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
            ''', {"rootId": root_id, "lang": self.lang})

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
                "language": self.lang_info["name"],
                "language_code": self.lang,
                "direction": self.lang_info["direction"],
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
                text = random.choice(self.corpus[text_type])

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

    def generate_document_dataset(self, num_pages=5, font_pt=14, font_family=None,
                                  seed=42, prefix="doc"):
        """Generate num_pages A4 document pages (title + multiple paragraphs),
        clean, without noise or rotation.
        Outputs: images/{prefix}_NNN.png + ground_truth/{prefix}_NNN.json"""
        if font_family is None:
            font_family = self.lang_info["default_font"]
        rng = random.Random(seed)
        font_config = {"family": font_family, "size": font_pt, "weight": "normal"}
        bg_config = {"bg": "#FFFFFF", "fg": "#000000"}
        viewport = (A4_W, A4_H)

        for i in range(num_pages):
            try:
                title = rng.choice(self.page_titles)
                n_paras = rng.randint(8, 11)
                paras = rng.sample(self.paragraphs,
                                   k=min(n_paras, len(self.paragraphs)))

                html = self.generate_html_page(title, paras, font_config, bg_config)

                image_filename = f"{prefix}_{i + 1:03d}.png"
                image_path = self.images_dir / image_filename
                pos = self.render_and_extract_positions(html, image_path,
                                                        root_id="doc-page",
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


# backward-compatible alias (the class was previously named ArabicOCRWebGenerator)
ArabicOCRWebGenerator = OCRWebGenerator


def main():
    """Entry point."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Multi-language OCR dataset generator (Arabic / Chinese / English)")
    ap.add_argument("--lang", choices=["ar", "zh", "en"], default="ar",
                    help="language of the generated text (default: ar=Arabic)")
    ap.add_argument("--mode", choices=["document", "card"], default="document",
                    help="document=A4 multi-paragraph pages (default, no noise); card=word cards (may add noise)")
    ap.add_argument("--num", type=int, default=5, help="number of samples (default 5)")
    ap.add_argument("--output", default=None,
                    help="output directory (default: ./ocr_dataset_<lang>, e.g. ./ocr_dataset_ar)")
    ap.add_argument("--font", default=None,
                    help="font family (default: language-specific, e.g. Tahoma/SimSun/Times New Roman)")
    ap.add_argument("--font-size", type=int, default=14, help="body font size in pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    generator = OCRWebGenerator(args.output, language=args.lang)

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
