#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR evaluation script.
Compares OCR output against ground truth (CER / WER / word-box IoU).
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import editdistance
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class OCREvaluator:
    """OCR evaluator."""

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.gt_dir = self.dataset_dir / "ground_truth"

    def load_ground_truth(self, gt_file):
        """Load the ground truth JSON."""
        with open(gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calculate_cer(self, pred_text, gt_text):
        """Compute the Character Error Rate (CER)."""
        if not gt_text:
            return 1.0
        distance = editdistance.eval(pred_text, gt_text)
        return distance / len(gt_text)

    def calculate_wer(self, pred_text, gt_text):
        """Compute the Word Error Rate (WER)."""
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        if not gt_words:
            return 1.0
        distance = editdistance.eval(pred_words, gt_words)
        return distance / len(gt_words)

    def iou_score(self, pred_bbox, gt_bbox):
        """Compute the Intersection-over-Union (IoU) score."""
        # compute the intersection
        x1 = max(pred_bbox['x'], gt_bbox['x'])
        y1 = max(pred_bbox['y'], gt_bbox['y'])
        x2 = min(pred_bbox['x'] + pred_bbox['width'], gt_bbox['x'] + gt_bbox['width'])
        y2 = min(pred_bbox['y'] + pred_bbox['height'], gt_bbox['y'] + gt_bbox['height'])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        pred_area = pred_bbox['width'] * pred_bbox['height']
        gt_area = gt_bbox['width'] * gt_bbox['height']
        union = pred_area + gt_area - intersection

        return intersection / union if union > 0 else 0.0

    def evaluate_sample(self, gt_file, ocr_output):
        """Evaluate a single sample."""
        gt_data = self.load_ground_truth(gt_file)

        results = {
            "cer": self.calculate_cer(ocr_output['text'], gt_data['text']),
            "wer": self.calculate_wer(ocr_output['text'], gt_data['text']),
            "iou_scores": []
        }

        # compute IoU for every word
        for gt_word in gt_data.get('words', []):
            # assumes the OCR output contains boxes for the same words;
            # in practice the OCR output needs to be matched to the GT
            best_iou = 0
            for pred_word in ocr_output.get('words', []):
                iou = self.iou_score(pred_word['bbox'], gt_word['bbox'])
                best_iou = max(best_iou, iou)
            results['iou_scores'].append(best_iou)

        results['mean_iou'] = np.mean(results['iou_scores']) if results['iou_scores'] else 0

        return results

    def generate_report(self, ocr_results):
        """Generate an evaluation report."""
        report = {
            "total_samples": len(ocr_results),
            "mean_cer": np.mean([r['cer'] for r in ocr_results]),
            "mean_wer": np.mean([r['wer'] for r in ocr_results]),
            "mean_iou": np.mean([r['mean_iou'] for r in ocr_results]),
            "detailed_results": ocr_results
        }
        return report

    def visualize_evaluation(self, image_path, gt_data, pred_data, output_path):
        """Visualize the evaluation result."""
        img = Image.open(image_path)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(img)

        # draw GT (green)
        for word in gt_data.get('words', []):
            bbox = word['bbox']
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(bbox['x'], bbox['y'] - 5, word['text'],
                    color='green', fontsize=8, fontweight='bold')

        # draw predictions (red)
        for word in pred_data.get('words', []):
            bbox = word['bbox']
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(bbox['x'], bbox['y'] + bbox['height'] + 5, word['text'],
                    color='red', fontsize=8, fontweight='bold')

        ax.set_title('OCR Evaluation: Green=GT, Red=Prediction')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Entry point (demo with a mock OCR output)."""
    evaluator = OCREvaluator("./arabic_ocr_dataset")

    # an actual OCR output is required here
    # example: mock OCR output
    mock_ocr_output = {
        "text": "مرحبا بك في عالم الذكاء الاصطناعي",
        "words": [
            {"text": "مرحبا", "bbox": {"x": 100, "y": 50, "width": 80, "height": 30}},
            {"text": "بك", "bbox": {"x": 190, "y": 50, "width": 40, "height": 30}},
        ]
    }

    # evaluate the first sample
    gt_files = list(evaluator.gt_dir.glob("*.json"))
    if gt_files:
        results = evaluator.evaluate_sample(gt_files[0], mock_ocr_output)
        print(f"Evaluation result: {results}")


if __name__ == "__main__":
    main()
