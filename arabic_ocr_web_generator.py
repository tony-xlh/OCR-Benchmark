#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backward-compatible shim for the OCR dataset generator.

The generator now lives in ocr_web_generator.py and supports Arabic, Chinese
and English (select with --lang ar|zh|en). This file keeps the old command
`python arabic_ocr_web_generator.py ...` working unchanged.
"""
from ocr_web_generator import main

if __name__ == "__main__":
    main()
