# Training Log

This document records the human-detection model training experiments.

## Dataset

**Dataset:** VisDrone DET

**Task:** Single-class person detection

**Classes:**
```text
0: person
```

## MT-003 — 960-Pixel RGB Human Detection

YOLO26n was trained on the VisDrone person-only dataset for 50 epochs at 960 × 960 input resolution with a batch size of 4.

### Configuration

| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Dataset | VisDrone Person |
| Input size | 960 × 960 |
| Epochs | 50 |
| Batch size | 4 |
| GPU | NVIDIA RTX 4050 Laptop GPU |

### Validation Results

| Metric | Result |
|---|---:|
| Precision | 0.716 |
| Recall | 0.548 |
| mAP@50 | 0.618 |
| mAP@50–95 | 0.274 |

**Validation set:** 548 images
**Person instances:** 13,969

**Training time:** 13.247 hours

**Inference speed:** 5.7 ms/image during validation

**Status:** Completed

### Observation

Increasing the training resolution from 640 × 640 to 960 × 960 resulted in higher validation precision, recall, mAP@50, and mAP@50–95.

The remaining limitation is recall, particularly for small aerial persons. This motivated further investigation of small-object detection performance and higher-resolution
## MT-004 — 1280-Pixel RGB Human Detection

YOLO26n was trained on the same VisDrone person-only dataset at 1280 × 1280 input resolution to investigate whether increased spatial resolution improves detection of small aerial persons.

### Configuration

| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Dataset | VisDrone Person |
| Input size | 1280 × 1280 |
| Epochs | 50 |
| Batch size | 2 |
| GPU | NVIDIA RTX 4050 Laptop GPU |
| AMP | Enabled |

### Validation Results

| Metric | Result |
|---|---:|
| Precision | 0.738 |
| Recall | 0.563 |
| mAP@50 | 0.653 |
| mAP@50–95 | 0.297 |

**Validation set:** 548 images
**Person instances:** 13,969

**Status:** Completed

### Comparison with MT-003

| Metric | MT-003 (960) | MT-004 (1280) | Change |
|---|---:|---:|---:|
| Precision | 0.716 | 0.738 | +0.022 |
| Recall | 0.548 | 0.563 | +0.015 |
| mAP@50 | 0.618 | 0.653 | +0.035 |
| mAP@50–95 | 0.274 | 0.297 | +0.023 |

### Observation

Increasing the input resolution from 960 × 960 to 1280 × 1280 improved all four reported validation metrics. The improvement in recall was smaller than the improvement in precision and mAP, indicating that higher resolution alone does not completely solve the small-object detection problem.

## Confidence Threshold Study

The MT-004 model was evaluated on the same VisDrone validation set using different confidence thresholds.

| Confidence | Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---:|---:|---:|---:|
| 0.15 | 0.744 | 0.569 | 0.596 | 0.278 |
| 0.25 | 0.714 | 0.585 | 0.540 | 0.258 |
| 0.35 | 0.838 | 0.505 | 0.477 | 0.234 |
| 0.50 | 0.935 | 0.342 | 0.334 | 0.176 |

### Observation

Increasing the confidence threshold increased reported precision while reducing recall. The 0.50 threshold produced substantially fewer detections but a higher precision value, whereas lower thresholds retained more candidate detections.

The confidence threshold is therefore a deployment parameter that should be selected according to the required balance between missed persons and false detections.

## Small-Object Error Analysis

A post-training analysis was performed on the VisDrone validation predictions using an IoU threshold of 0.50.

The analysis categorized ground-truth persons using bounding-box area:

- Small: area < 32 × 32 pixels
- Medium: 32 × 32 ≤ area < 96 × 96 pixels
- Large: area ≥ 96 × 96 pixels

For MT-003, the analysis showed:

| Size | Ground Truth | Matched | Matched Ratio |
|---|---:|---:|---:|
| Small | 12,377 | 6,723 | 0.543 |
| Medium | 1,564 | 1,249 | 0.799 |
| Large | 28 | 26 | 0.929 |

**Total ground-truth persons:** 13,969
**Matched detections:** 7,998
**Overall matched ratio:** 0.573

The analysis indicates that detection performance decreases substantially as person bounding-box size becomes smaller.

## Missed-Person Analysis

The final MT-004 validation predictions were examined to identify complete detection failures.

There were 21 validation images without prediction label files. Of these, 14 images contained no ground-truth persons, while 7 images contained a total of 14 ground-truth persons.

The 14 missed persons had the following bounding-box characteristics:

| Measurement | Value |
|---|---:|
| Average width | 10.0 px |
| Average height | 22.9 px |
| Minimum width | 5 px |
| Minimum height | 7 px |
| Maximum width | 26 px |
| Maximum height | 54 px |

**11 of the 14 missed persons had bounding-box widths of 10 pixels or less.**

The missed detections were therefore predominantly associated with extremely small persons in aerial imagery. This supports the conclusion that small-object visibility is a primary limitation of the current RGB detector.

## Tiled-Inference Experiment

A custom tiled-inference pipeline was tested with MT-004 to determine whether processing smaller image regions could improve detection of small persons.

The tiled pipeline was corrected to use the required `[x, y, width, height]` format for OpenCV NMS before evaluating the results.

### Standard vs Tiled Results

Both methods were evaluated on the 548-image VisDrone validation set using an IoU threshold of 0.50 and the same size definitions.

| Metric | Standard | Tiled |
|---|---:|---:|
| Ground-truth persons | 13,969 | 13,969 |
| Predicted boxes | 11,509 | 11,024 |
| Matched persons | 8,199 | 8,112 |
| Overall matched ratio | 0.5869 | 0.5807 |
| Zero-detection images | 21 | 21 |

### Size-wise Results

| Size | Standard Recall | Tiled Recall |
|---|---:|---:|
| Small | 0.5809 | 0.5769 |
| Medium | 0.7724 | 0.6989 |
| Large | 0.8000 | 0.6000 |

The tested tiled configuration did not improve the current MT-004 detector. It produced 87 fewer matched persons and 485 fewer predicted boxes than standard inference.

One of the 14 previously identified missed persons was recovered during tiled inference, but the overall validation comparison did not show an improvement.

**Conclusion:** Tiled inference was investigated as a possible small-object solution but was not selected as the next optimization direction for the current model.

## Current RGB Model Status

The RGB detection experiments established that increasing input resolution from 640 × 640 to 960 × 960 and then to 1280 × 1280 improves overall validation performance. However, error analysis shows that very small aerial persons remain difficult to detect.

The next development stage is therefore to introduce thermal infrared data using the HIT-UAV dataset and establish a separate thermal detection baseline before investigating RGB–thermal fusion.training.
