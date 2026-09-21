# Training Log

This document records the human-detection model training experiments for the
eVTOL UAV human-detection subsystem.

Experiments are identified as `MT-xxx` and are listed in the order they were
performed.

---

## 1. Development Environment

| Parameter    | Value                              |
| ------------ | ---------------------------------- |
| OS           | Windows                            |
| Python       | 3.12.5                             |
| Ultralytics  | 8.4.152                            |
| PyTorch      | 2.14.0+cu132                       |
| CUDA         | 13.2                               |
| GPU          | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM         | 6140 MiB                           |
| Model family | YOLO26n                            |

The development environment uses the NVIDIA GPU for accelerated training and
inference.

---

## 2. Experiment Index

| ID     | Modality | Dataset                       | Input size | Epochs | Status    |
| ------ | -------- | ----------------------------- | ---------- | ------ | --------- |
| MT-001 | RGB      | VisDrone Person               | 640        | 3      | Completed |
| MT-002 | RGB      | VisDrone Person               | 640        | 50     | Completed |
| MT-003 | RGB      | VisDrone Person               | 960        | 50     | Completed |
| MT-004 | RGB      | VisDrone Person               | 1280       | 50     | Completed |
| MT-005 | Thermal  | HIT-UAV Person                | 640        | 50     | Completed |
| MT-006 | Thermal  | HIT-UAV Person                | 960        | 50     | Completed |
| MT-007 | Thermal  | HIT-UAV Person + 256 px crops | 640        | 50     | Completed |

---

# 3. RGB Detection Experiments - VisDrone

## Dataset

**Dataset:** VisDrone DET

**Task:** Single-class person detection

The original VisDrone `pedestrian` and `people` categories were combined into
one class:

```text
0: person
```

### Dataset statistics

| Split      | Images | Images with persons | Person annotations |
| ---------- | -----: | ------------------: | -----------------: |
| Train      |   6471 |                5684 |             106396 |
| Validation |    548 |                 531 |              13969 |

The converted dataset was visually inspected using bounding-box visualization
before training.

---

## MT-001 - RGB Pilot Training

A short 3-epoch pilot run was performed to verify the complete training
pipeline.

### Configuration

| Parameter  | Value                      |
| ---------- | -------------------------- |
| Model      | YOLO26n                    |
| Dataset    | VisDrone Person            |
| Input size | 640 x 640                  |
| Epochs     | 3                          |
| Batch size | 8                          |
| GPU        | NVIDIA RTX 4050 Laptop GPU |
| AMP        | Enabled                    |

### Results

| Metric    | Result |
| --------- | ------ |
| Precision | 0.498  |
| Recall    | 0.358  |
| mAP@50    | 0.366  |
| mAP@50-95 | 0.132  |

**Status:** Completed - pilot only.

The run confirmed that the dataset, model, GPU and training pipeline were
functioning correctly. It was not treated as a final benchmark because of the
very short training duration.

---

## MT-002 - RGB Baseline Training

A 50-epoch baseline was trained at 640 x 640 input resolution.

### Configuration

| Parameter  | Value                      |
| ---------- | -------------------------- |
| Model      | YOLO26n                    |
| Dataset    | VisDrone Person            |
| Input size | 640 x 640                  |
| Epochs     | 50                         |
| Batch size | 8                          |
| GPU        | NVIDIA RTX 4050 Laptop GPU |
| AMP        | Enabled                    |

### Results

Approximate final validation performance:

| Metric    | Result |
| --------- | ------ |
| Precision | ~0.63  |
| Recall    | ~0.44  |
| mAP@50    | ~0.50  |
| mAP@50-95 | ~0.198 |

**Status:** Completed.

### Error analysis

The confusion matrix indicated a significant number of missed person
detections.

The principal limitation identified was low recall, particularly for small
aerial persons. This motivated controlled experiments using increased input
resolution.

---

## MT-003 - 960-Pixel RGB Human Detection

YOLO26n was trained on the VisDrone person-only dataset for 50 epochs at
960 x 960 input resolution with a batch size of 4.

### Configuration

| Parameter  | Value                      |
| ---------- | -------------------------- |
| Model      | YOLO26n                    |
| Dataset    | VisDrone Person            |
| Input size | 960 x 960                  |
| Epochs     | 50                         |
| Batch size | 4                          |
| GPU        | NVIDIA RTX 4050 Laptop GPU |
| AMP        | Enabled                    |

### Validation results

| Metric    | Result |
| --------- | ------ |
| Precision | 0.716  |
| Recall    | 0.548  |
| mAP@50    | 0.618  |
| mAP@50-95 | 0.274  |

- Validation set: 548 images
- Person instances: 13,969
- Training time: 13.247 hours
- Inference speed: 5.7 ms/image during validation

**Status:** Completed.

### Observation

Increasing the training resolution from 640 x 640 to 960 x 960 improved
validation precision, recall, mAP@50 and mAP@50-95.

However, recall remained the main limitation, particularly for small aerial
persons.

---

## MT-004 - 1280-Pixel RGB Human Detection

A controlled higher-resolution experiment was performed using the same VisDrone
split and YOLO26n model.

### Configuration

| Parameter  | Value                      |
| ---------- | -------------------------- |
| Model      | YOLO26n                    |
| Dataset    | VisDrone Person            |
| Input size | 1280 x 1280                |
| Epochs     | 50                         |
| Batch size | 2                          |
| GPU        | NVIDIA RTX 4050 Laptop GPU |
| AMP        | Enabled                    |

### Validation results

| Metric    | Result |
| --------- | ------ |
| Precision | 0.738  |
| Recall    | 0.563  |
| mAP@50    | 0.653  |
| mAP@50-95 | 0.297  |

- Validation set: 548 images
- Person instances: 13,969

**Status:** Completed.

### Comparison with MT-003

| Metric    | MT-003 (960) | MT-004 (1280) | Change |
| --------- | ------------ | ------------- | ------ |
| Precision | 0.716        | 0.738         | +0.022 |
| Recall    | 0.548        | 0.563         | +0.015 |
| mAP@50    | 0.618        | 0.653         | +0.035 |
| mAP@50-95 | 0.274        | 0.297         | +0.023 |

### Observation

Increasing input resolution from 960 x 960 to 1280 x 1280 improved all four
reported validation metrics.

The recall improvement was relatively modest, indicating that increased
resolution alone does not completely solve the small-object detection problem.

MT-004 is the selected RGB reference model.

---

## 4. RGB Confidence Threshold Study

The MT-004 model was evaluated on the same VisDrone validation set using
different confidence thresholds.

| Confidence | Precision | Recall | mAP@50 | mAP@50-95 |
| ---------- | --------- | ------ | ------ | --------- |
| 0.15       | 0.744     | 0.569  | 0.596  | 0.278     |
| 0.25       | 0.714     | 0.585  | 0.540  | 0.258     |
| 0.35       | 0.838     | 0.505  | 0.477  | 0.234     |
| 0.50       | 0.935     | 0.342  | 0.334  | 0.176     |

### Observation

Increasing the confidence threshold reduced the number of accepted detections
and generally increased precision while reducing recall.

The confidence threshold is therefore an operational/deployment parameter that
must be selected according to the required balance between missed persons and
false detections.

The mAP values in this threshold study must not be interpreted as conventional
threshold-independent AP comparisons, because a raised confidence threshold
truncates the precision-recall curve.

---

## 5. RGB Small-Object Error Analysis

Post-training analysis was performed on the VisDrone validation predictions
using an IoU threshold of 0.50.

An area-based analysis was performed using:

- Small: area < 32 x 32 pixels
- Medium: 32 x 32 <= area < 96 x 96 pixels
- Large: area >= 96 x 96 pixels

### MT-003 analysis

| Size   | Ground Truth | Matched | Matched Ratio |
| ------ | -----------: | ------: | ------------: |
| Small  |       12,377 |   6,723 |         0.543 |
| Medium |        1,564 |   1,249 |         0.799 |
| Large  |           28 |      26 |         0.929 |

- Total ground-truth persons: 13,969
- Matched detections: 7,998
- Overall matched ratio: 0.573

The analysis showed substantially lower detection performance for smaller
person bounding boxes.

---

## 6. MT-004 Missed-Person Analysis

The final MT-004 validation predictions were examined for images without
prediction label files.

There were 21 validation images without prediction label files. Of these:

- 14 contained no ground-truth persons.
- 7 contained a total of 14 ground-truth persons.

### Missed-person bounding-box characteristics

| Measurement    | Value   |
| -------------- | ------- |
| Average width  | 10.0 px |
| Average height | 22.9 px |
| Minimum width  | 5 px    |
| Minimum height | 7 px    |
| Maximum width  | 26 px   |
| Maximum height | 54 px   |

11 of the 14 missed persons had bounding-box widths of 10 pixels or less.

The missed detections were predominantly associated with very small persons in
aerial imagery.

---

## 7. RGB Tiled-Inference Experiment

A custom tiled-inference pipeline was tested with MT-004 to investigate whether
processing smaller image regions could improve small-person detection.

The NMS implementation was corrected before the final comparison because
OpenCV NMS requires boxes in `[x, y, width, height]` format.

### Standard vs tiled

Both methods were evaluated on the 548-image VisDrone validation set using
IoU >= 0.50.

| Metric                | Standard | Tiled  |
| --------------------- | -------- | ------ |
| Ground-truth persons  | 13,969   | 13,969 |
| Predicted boxes       | 11,509   | 11,024 |
| Matched persons       | 8,199    | 8,112  |
| Overall matched ratio | 0.5869   | 0.5807 |
| Zero-detection images | 21       | 21     |

### Size-wise results

| Size   | Standard Recall | Tiled Recall |
| ------ | --------------- | ------------ |
| Small  | 0.5809          | 0.5769       |
| Medium | 0.7724          | 0.6989       |
| Large  | 0.8000          | 0.6000       |

The tested tiled configuration produced 87 fewer matched persons and 485 fewer
predicted boxes.

One of the 14 previously identified missed persons was recovered during tiled
inference, but the overall validation comparison did not show an improvement.

**Conclusion:** The tested tiled configuration was not selected as the next
optimization direction. Tiling was later re-examined as a *training-time*
augmentation instead of an inference-time strategy (MT-007).

---

# 8. Thermal Detection Experiments - HIT-UAV

## 8.1 HIT-UAV Thermal Dataset Integration

A separate thermal detection baseline was introduced using the HIT-UAV Infrared
Thermal Dataset.

The dataset contains UAV thermal imagery covering day and night conditions and
multiple object categories. For this project, only the Person category was
retained.

### Original HIT-UAV category mapping

| ID  | Category     |
| --- | ------------ |
| 0   | Person       |
| 1   | Car          |
| 2   | Bicycle      |
| 3   | OtherVehicle |
| 4   | DontCare     |

The Person category is therefore mapped to:

```text
0: person
```

### Converted dataset statistics

| Split      | Images | Images with persons | Images without persons | Person annotations |
| ---------- | -----: | ------------------: | ---------------------: | -----------------: |
| Train      |  2,029 |               1,165 |                    864 |              8,533 |
| Validation |    290 |                 171 |                    119 |              1,168 |
| Test       |    579 |                 355 |                    224 |              2,611 |
| Total      |  2,898 |               1,691 |                  1,207 |             12,312 |

The COCO-style bounding boxes were converted to YOLO format.

The converted annotations were visually inspected using thermal-image
bounding-box visualization. The sampled annotations showed correct placement
over visible human thermal signatures.

The test split is held out and is used only for final evaluation.

---

## 9. MT-005 - Thermal Baseline

YOLO26n was trained on the converted HIT-UAV Person dataset.

### Configuration

| Parameter  | Value                      |
| ---------- | -------------------------- |
| Model      | YOLO26n                    |
| Dataset    | HIT-UAV Person             |
| Input size | 640 x 640                  |
| Epochs     | 50                         |
| Batch size | 8                          |
| GPU        | NVIDIA RTX 4050 Laptop GPU |
| AMP        | Enabled                    |

### Validation results

| Metric    | Result |
| --------- | ------ |
| Precision | 0.890  |
| Recall    | 0.875  |
| mAP@50    | 0.919  |
| mAP@50-95 | 0.500  |

- Validation set: 290 images
- Person instances: 1,168
- Training time: 1.599 hours
- Best epoch: 47

**Status:** Completed.

---

## 10. MT-005 Held-Out Test Evaluation

The completed MT-005 model was evaluated on the separate HIT-UAV test split.

| Metric    | Result |
| --------- | ------ |
| Precision | 0.898  |
| Recall    | 0.892  |
| mAP@50    | 0.933  |
| mAP@50-95 | 0.510  |

- Test set: 579 images
- Person instances: 2,611
- Inference speed: approximately 4.2 ms/image during evaluation

**Status:** Completed.

This held-out test result is the primary test-set benchmark for the MT-005
thermal baseline.

The thermal and RGB models must not be interpreted as showing that one modality
is inherently superior, because they were trained and evaluated on different
datasets with different imagery and annotation policies.

---

## 11. MT-005 Thermal Test Error Analysis

The MT-005 model was evaluated on the HIT-UAV test images at confidence
threshold 0.25 for detailed error analysis.

A one-to-one greedy matching procedure with IoU >= 0.50 was used for this
custom analysis.

### Confidence 0.25

| Measurement                | Result |
| -------------------------- | ------ |
| Test images                | 579    |
| Images with GT persons     | 355    |
| Images with predictions    | 369    |
| Zero-prediction images     | 210    |
| Ground-truth persons       | 2,611  |
| Predicted boxes            | 3,063  |
| Matched persons            | 2,425  |
| Missed GT persons          | 186    |
| Unmatched prediction boxes | 638    |
| Overall matched ratio      | 0.9288 |

Of the 210 zero-prediction images:

- 202 contained no persons.
- 8 contained persons.

The 8 complete-failure images contained a total of 9 ground-truth persons.

### Size analysis

Using the project-wide small-object definition of width < 32 px OR
height < 32 px:

| Size   |    GT | Matched | Recall |
| ------ | ----: | ------: | ------ |
| Small  | 2,606 |   2,422 | 0.9294 |
| Medium |     5 |       3 | 0.6000 |
| Large  |     0 |       0 | N/A    |

99.8% of the HIT-UAV test person instances were classified as small under this
definition.

---

## 12. MT-005 Confidence 0.10 Diagnostic

A second prediction run was performed at confidence threshold 0.10 to
investigate whether low-confidence detections were responsible for some of the
missed persons.

### Confidence 0.10

| Measurement                | Result |
| -------------------------- | ------ |
| Test images                | 579    |
| Images with predictions    | 384    |
| Zero-prediction images     | 195    |
| Predicted boxes            | 3,768  |
| Matched persons            | 2,468  |
| Missed GT persons          | 143    |
| Unmatched prediction boxes | 1,300  |
| Overall matched ratio      | 0.9452 |

Compared with confidence 0.25:

- 705 additional predicted boxes were generated.
- 43 additional persons were matched.
- 43 fewer persons were missed.
- 662 additional prediction boxes remained unmatched.
- Zero-prediction images decreased by 15.
- Zero-prediction images containing persons decreased by 4.

The 43 recovered persons had detection confidences between approximately
0.1034 and 0.2490, with an average of approximately 0.1747.

This indicates that some valid thermal-person detections are produced at
relatively low confidence.

---

## 13. MT-005 Remaining-Miss Analysis

The remaining misses after the confidence 0.10 diagnostic were examined
further.

The traceable per-person diagnostic identified 142 remaining missed persons.
The small difference from the aggregate 143-miss result is due to the matching
procedure used by the diagnostic.

### Remaining-miss categories

| Category                                     | Count | Percentage |
| -------------------------------------------- | ----: | ---------- |
| No overlapping prediction at confidence 0.10  |    35 | ~24.6%     |
| Prediction exists but IoU < 0.50              |   107 | ~75.4%     |
| Total traceable misses                        |   142 | 100%       |

### Remaining-miss size

| Measurement    | Value    |
| -------------- | -------- |
| Average width  | 13.30 px |
| Average height | 18.25 px |
| Minimum width  | 4 px     |
| Minimum height | 8 px     |
| Maximum width  | 87 px    |
| Maximum height | 74 px    |

The majority of the remaining missed persons had some overlapping prediction
but insufficient IoU for a match.

This suggests that the dominant remaining limitation is **localization** of
very small thermal targets, with a secondary issue of complete recognition
failures.

Lowering the confidence threshold alone therefore does not solve the problem:
it recovered 43 of the 186 misses while adding 662 unmatched prediction boxes.

This result defined the two follow-up experiments:

- **MT-006** - increase input resolution, so that small targets occupy more
  pixels at the model input.
- **MT-007** - increase the effective scale of small targets during training
  through a crop-based training set.

---

## 14. MT-006 - Thermal Resolution Experiment (960 px)

MT-006 tested whether increasing the input resolution from 640 x 640 to
960 x 960 improves thermal person detection, with all other major training
conditions held constant.

### Configuration

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Model         | YOLO26n                    |
| Dataset       | HIT-UAV Person             |
| Input size    | 960 x 960                  |
| Epochs        | 50                         |
| Batch size    | 4                          |
| GPU           | NVIDIA RTX 4050 Laptop GPU |
| AMP           | Enabled                    |
| Dataset split | Same as MT-005             |

The run was interrupted and resumed from `last.pt` at epoch 25, so the recorded
wall-clock time is not meaningful. At approximately 245 s/epoch the run
represents roughly 3.4 hours of GPU time.

### Validation results (290 images, 1,168 instances)

| Metric    | MT-005 | MT-006 | Change |
| --------- | ------ | ------ | ------ |
| Precision | 0.890  | 0.896  | +0.006 |
| Recall    | 0.875  | 0.857  | -0.018 |
| mAP@50    | 0.919  | 0.912  | -0.007 |
| mAP@50-95 | 0.500  | 0.510  | +0.010 |

Best epoch: 50.

**Status:** Completed.

### Held-out test results (579 images, 2,611 instances)

| Metric    | MT-005 | MT-006 | Change |
| --------- | ------ | ------ | ------ |
| Precision | 0.897  | 0.903  | +0.006 |
| Recall    | 0.891  | 0.880  | -0.011 |
| mAP@50    | 0.933  | 0.926  | -0.007 |
| mAP@50-95 | 0.510  | 0.526  | +0.016 |

These figures are from the unified benchmark (section 17) and were produced
with identical evaluation settings for every model.

### Observation

The resolution increase produced a **localization** improvement and a
**detection** regression:

- mAP@50-95 increased by 0.016. That metric is dominated by box quality at
  stricter IoU thresholds, which is consistent with the MT-005 remaining-miss
  analysis identifying localization as the dominant failure mode.
- mAP@50 and recall decreased slightly, meaning MT-006 finds marginally fewer
  persons overall.

This is the opposite of the RGB result, where resolution improved every metric.
The explanation is that HIT-UAV images are natively 640 x 512. Upscaling to
960 x 960 adds no new sensor information; it only changes the scale at which
the network sees the targets. The VisDrone RGB images are much larger, so a
higher input resolution there preserved real detail that was previously
discarded by downscaling.

---

## 15. MT-007 - Thermal Crop-Augmented Training

MT-007 tested whether increasing the *apparent* size of small thermal targets
during training improves detection, without changing inference resolution.

Instead of tiling at inference time (which was tested and rejected for RGB in
section 7), the tiling was moved into the training set.

### Crop dataset construction

| Parameter                | Value                                |
| ------------------------ | ------------------------------------ |
| Crop size                | 256 x 256                            |
| Crop overlap             | 25%                                  |
| Minimum visible fraction | 0.50 of the original box             |
| Empty crops per image    | at most 2                            |
| Original images retained | Yes                                  |
| Validation / test splits | Copied unchanged from HIT-UAV Person |

Keeping the validation and test splits untouched is what makes MT-005, MT-006
and MT-007 directly comparable: only the training data changed.

Generated by `scripts/create_hit_uav_crops.py`.

### Crop training set statistics

| Content                | Images     | Person annotations |
| ---------------------- | ---------: | -----------------: |
| Original full images   |      2,029 |              8,533 |
| Generated 256 px crops |      9,241 |             16,942 |
| **Total training set** | **11,270** |         **25,475** |

Of the 11,270 training images, 6,521 contain at least one person and 4,749 are
background-only.

The generated crops were visually inspected before training.

### Configuration

| Parameter     | Value                         |
| ------------- | ----------------------------- |
| Model         | YOLO26n                       |
| Dataset       | HIT-UAV Person + 256 px crops |
| Input size    | 640 x 640                     |
| Epochs        | 50                            |
| Batch size    | 8                             |
| GPU           | NVIDIA RTX 4050 Laptop GPU    |
| AMP           | Enabled                       |
| Training time | 8.146 hours                   |

Because a 256 px crop is upscaled to the 640 px model input, every person in a
crop is presented to the network at roughly 2.5x its original scale.

### Validation results (290 images, 1,168 instances)

| Metric    | MT-005 | MT-007 | Change |
| --------- | ------ | ------ | ------ |
| Precision | 0.890  | 0.890  | 0.000  |
| Recall    | 0.875  | 0.861  | -0.014 |
| mAP@50    | 0.919  | 0.910  | -0.009 |
| mAP@50-95 | 0.500  | 0.495  | -0.005 |

Best epoch: 44.

**Status:** Completed.

### Held-out test results (579 images, 2,611 instances)

| Metric    | MT-005 | MT-007 | Change |
| --------- | ------ | ------ | ------ |
| Precision | 0.897  | 0.895  | -0.002 |
| Recall    | 0.891  | 0.873  | -0.018 |
| mAP@50    | 0.933  | 0.926  | -0.007 |
| mAP@50-95 | 0.510  | 0.504  | -0.006 |

### Observation

Crop-based training did not improve thermal detection. Every test metric was
equal to or slightly below MT-005, at roughly five times the training cost.

---

## 16. MT-005 vs MT-007 Per-Person Comparison

Because the aggregate metrics for MT-005 and MT-007 are close, a per-person
comparison was performed on all 2,611 test-set ground-truth persons to check
whether the two models fail on the *same* people or on different people.

Matching used IoU >= 0.50 at confidence 0.25
(`scripts/compare_mt005_vs_mt007.py`).

| Category               | Persons | Share |
| ---------------------- | ------: | ----- |
| Matched by both models |   2,327 | 89.1% |
| Missed by both models  |     131 | 5.0%  |
| Matched by MT-005 only |      98 | 3.8%  |
| Matched by MT-007 only |      55 | 2.1%  |

### Box size by category (test-set ground truth)

| Category        | Mean width | Mean height |
| --------------- | ---------- | ----------- |
| Matched by both | 12.74 px   | 19.61 px    |
| Missed by both  | 13.94 px   | 18.92 px    |
| MT-005 only     | 13.72 px   | 19.34 px    |
| MT-007 only     | 12.62 px   | 18.58 px    |

### Localization quality on commonly matched persons

| Measurement     | MT-005 | MT-007 |
| --------------- | ------ | ------ |
| Mean IoU        | 0.784  | 0.783  |
| Mean confidence | 0.663  | 0.649  |

### Observation

Two results follow from this comparison.

1. **The models are not interchangeable.** 153 persons (5.9% of the test set)
   are found by exactly one of the two models. MT-007 recovers 55 persons that
   MT-005 misses. That is a genuine ensemble/fusion opportunity, even though
   MT-007 is worse on aggregate.

2. **The 131 persons missed by both models are not distinguished by size.**
   Their mean box dimensions are almost identical to the persons that both
   models detect successfully. Size alone therefore does not explain the
   remaining failures; they are more likely caused by thermal contrast,
   occlusion, or posture.

The second point is the important one: it means further scale-oriented
experiments (higher resolution, more aggressive cropping) are unlikely to
recover the remaining hard cases. MT-006 and MT-007 both tested scale, and
neither improved on MT-005.

---

## 17. Unified Model Benchmark

All reference models were re-evaluated with one script
(`scripts/benchmark_models.py`) under identical settings so that the reported
numbers are directly comparable.

### Benchmark settings

| Parameter       | Value                                      |
| --------------- | ------------------------------------------ |
| Confidence      | 0.001 (AP-standard, threshold-independent) |
| NMS IoU         | 0.70                                       |
| Batch size      | 1                                          |
| Evaluation size | each model's own training resolution       |
| Device          | RTX 4050 Laptop GPU                        |

Each model is evaluated in a separate process. Evaluating several models in one
process keeps earlier models resident in VRAM on a 6 GB GPU and inflates the
reported inference time.

### Results

| Experiment | Modality | Split | imgsz | Images | Instances | P     | R     | mAP@50 | mAP@50-95 | Inference |
| ---------- | -------- | ----- | ----: | -----: | --------: | ----- | ----- | ------ | --------- | --------- |
| MT-004     | RGB      | val   |  1280 |    548 |    13,969 | 0.744 | 0.570 | 0.654  | 0.300     | 24.6 ms   |
| MT-005     | Thermal  | test  |   640 |    579 |     2,611 | 0.897 | 0.891 | 0.933  | 0.510     | 25.4 ms   |
| MT-006     | Thermal  | test  |   960 |    579 |     2,611 | 0.903 | 0.880 | 0.926  | 0.526     | 25.0 ms   |
| MT-007     | Thermal  | test  |   640 |    579 |     2,611 | 0.895 | 0.873 | 0.926  | 0.504     | 25.5 ms   |

The MT-004 row differs marginally from section 3 (0.744/0.570/0.654/0.300 vs
0.738/0.563/0.653/0.297) because the benchmark applies uniform confidence and
NMS settings to every model rather than the per-run defaults.

### Interpretation

**This table must not be read as "thermal is better than RGB."** The two
modalities are evaluated on different datasets:

- VisDrone is a large, cluttered, high-density urban RGB dataset with up to
  hundreds of annotated persons per image, many of them partially occluded.
- HIT-UAV is a smaller thermal dataset in which a person's heat signature is
  high-contrast against a cool background, with far fewer persons per image.

The RGB task as posed by VisDrone is intrinsically harder. The comparison that
*is* valid is within each modality:

- Within RGB: resolution helps (MT-002 -> MT-003 -> MT-004).
- Within thermal: MT-005 at 640 px remains the best overall model. Neither the
  resolution increase (MT-006) nor the crop augmentation (MT-007) improved it,
  although MT-006 gives the best mAP@50-95, i.e. the best box localization.

The inference timings are validator throughput under identical conditions on a
laptop GPU with `batch=1` and `workers=0`. They are dominated by per-image
Python and data-loading overhead, and should not be quoted as deployment
latency. A dedicated latency benchmark is required before companion-computer
selection.

### Selected reference models

| Role                 | Model  | Reason                                           |
| -------------------- | ------ | ------------------------------------------------ |
| RGB reference        | MT-004 | Best RGB metrics                                 |
| Thermal reference    | MT-005 | Best thermal recall and mAP@50, lowest cost      |
| Thermal localization | MT-006 | Best mAP@50-95; candidate if box quality matters |

---

## 18. Custom Error-Analysis Comparison (Thermal)

The three thermal models were also compared using the project's own greedy
one-to-one matching analysis at confidence 0.25 and IoU >= 0.50 on the held-out
test split. This analysis answers an operational question ("how many actual
people were found at the deployment threshold?") rather than an AP question.

| Measurement                       | MT-005 | MT-006 | MT-007 |
| --------------------------------- | -----: | -----: | -----: |
| Ground-truth persons              |  2,611 |  2,611 |  2,611 |
| Predicted boxes                   |  3,063 |  2,933 |  2,952 |
| Matched persons                   |  2,425 |  2,404 |  2,382 |
| Missed persons                    |    186 |    207 |    229 |
| Unmatched prediction boxes        |    638 |    529 |    570 |
| Matched ratio (recall)            | 0.9288 | 0.9207 | 0.9123 |
| Matched / predicted (precision)   | 0.7917 | 0.8196 | 0.8069 |
| Images producing predictions      |    369 |    348 |    354 |
| Images with persons but no output |      8 |     19 |     16 |
| Persons in those images           |      9 |     28 |     25 |

### Observation

At the operational threshold of 0.25, MT-005 finds the most persons (2,425) and
has the fewest complete-failure images (8 images / 9 persons).

MT-006 and MT-007 are more conservative: they emit fewer boxes and therefore
have higher custom precision, but they also produce more images in which a
person is present and nothing at all is reported. For a search-and-rescue
payload a complete-failure image is the most costly error mode, which
reinforces MT-005 as the operational thermal model.

---

## 19. Conclusions from the Detection Experiments

1. **Resolution is the dominant lever for RGB.** MT-002 -> MT-004 improved
   mAP@50 from ~0.50 to 0.654, but recall remains the limiting factor (0.570).

2. **Resolution is not a lever for HIT-UAV thermal.** The source imagery is
   640 x 512, so upscaling adds no information. MT-006 improved box quality
   (mAP@50-95 +0.016) but reduced recall.

3. **Crop-based training is not a lever either.** MT-007 cost 8.1 hours and did
   not improve any test metric.

4. **Scale is no longer the bottleneck for the remaining thermal failures.**
   The 131 persons missed by both MT-005 and MT-007 have essentially the same
   box sizes as the persons that are detected correctly.

5. **MT-005 and MT-007 fail on different people.** 153 persons are found by
   exactly one of them, which makes model ensembling a more promising direction
   than further single-model scale tuning.

6. **The thermal detector is close to a usable operating point.** At confidence
   0.25 it finds 92.9% of the held-out persons, with complete failure on only
   8 of 355 person-containing images.

The next stages therefore move away from further single-modality accuracy
tuning and toward system integration: temporal tracking, movement detection, a
real-time pipeline, and embedded deployment.

---

## 20. MT-007 Confidence 0.10 Diagnostic

MT-007 was re-run at confidence 0.10 to test whether its misses are scored
below the operating threshold or absent altogether.

Comparing MT-005 at 0.25 against MT-007 at 0.10 would confound two variables,
since MT-005 itself changes substantially when its threshold is lowered. Both
models were therefore evaluated at both thresholds with identical matching
(`scripts/compare_mt005_vs_mt007_conf010.py`), holding the test set, ground
truth, input size and IoU criterion fixed.

The MT-005 figures reproduce the section 12 results exactly, which confirms the
two analyses are directly comparable.

### MT-007 at both thresholds

| Measurement                        | conf 0.25 | conf 0.10 | Change |
| ---------------------------------- | --------: | --------: | -----: |
| Ground-truth persons               |     2,611 |     2,611 |      - |
| Predicted boxes                    |     2,952 |     3,661 |   +709 |
| Matched persons                    |     2,382 |     2,453 |    +71 |
| Missed persons                     |       229 |       158 |    -71 |
| Unmatched prediction boxes         |       570 |     1,208 |   +638 |
| Matched ratio                      |    0.9123 |    0.9395 | +0.027 |
| Zero-prediction images             |       225 |       202 |    -23 |
| Zero-prediction images with people |        16 |        10 |     -6 |
| Persons in those images            |        25 |        12 |    -13 |

### Side by side with MT-005 at 0.10

| Measurement                        | MT-005 | MT-007 |
| ---------------------------------- | -----: | -----: |
| Predicted boxes                    |  3,768 |  3,661 |
| Matched persons                    |  2,468 |  2,453 |
| Missed persons                     |    143 |    158 |
| Unmatched prediction boxes         |  1,300 |  1,208 |
| Matched ratio                      | 0.9452 | 0.9395 |
| Zero-prediction images with people |      4 |     10 |

### Per-person agreement at 0.10

| Category               | Persons | Share | At 0.25 |
| ---------------------- | ------: | ----- | ------: |
| Matched by both models |   2,397 | 91.8% |   2,327 |
| Missed by both models  |      87 |  3.3% |     131 |
| Matched by MT-005 only |      71 |  2.7% |      98 |
| Matched by MT-007 only |      56 |  2.1% |      55 |

### Observation

Lowering the threshold recovers a meaningful but minority share of the shared
failures: persons missed by both models fall from 131 to 87, so **44 of the 131
(33.6%) were suppressed by the confidence threshold** and 87 were not found at
all.

MT-007 pays less for its recovery than MT-005 does - 71 persons for 638 extra
unmatched boxes, against MT-005's 43 for 662 - but it starts from a worse
position and still ends below MT-005 in absolute terms.

The cost of this route is poor regardless: about **15 extra false positives per
person recovered**. Threshold lowering is not a good answer on its own, which
motivated the failure characterisation in section 21.

---

## 21. Characterising the Shared Failures

The persons that both models miss are the ones that matter: a person missed by
only one model can be recovered by ensembling, but a person missed by both is a
capability gap.

`scripts/analyze_common_failures.py` characterises them along every axis the
dataset provides, always against the detected persons as a baseline, because a
property of the failures is only interesting if it differs from the population.

Scene metadata is parsed from the HIT-UAV filename convention
`<light>_<altitude>_<angle>_0_<id>`, verified across the test split: field 0 is
day/night, field 1 is flight altitude in metres (60-130), field 2 is camera
angle in degrees (30-90).

### Failure mode is the primary split

A miss with an overlapping prediction that fell short of IoU 0.50 is a
**localisation** failure. A miss with no overlapping prediction anywhere is a
**recognition** failure. The two require different fixes.

| Failure mode | conf 0.25  | conf 0.10  |
| ------------ | ---------: | ---------: |
| Localisation | 88 (67.2%) | 68 (78.2%) |
| Recognition  | 43 (32.8%) | 19 (21.8%) |
| Total        |        131 |         87 |

At confidence 0.10, **78% of the shared failures are localisation failures**:
the model does produce a box on the person, but not an accurate enough one.

### The localisation failures are near misses

Best-overlap IoU for the 68 localisation failures at conf 0.10:

| Best-overlap IoU | Count | Share |
| ---------------- | ----: | ----- |
| 0.00 - 0.10      |     2 | 2.9%  |
| 0.10 - 0.25      |     4 | 5.9%  |
| 0.25 - 0.40      |    25 | 36.8% |
| 0.40 - 0.50      |    37 | 54.4% |

Median 0.417. **91% sit between 0.25 and 0.50**, and over half are within 0.10
of the matching threshold.

The margin is tighter than it looks. For a typical 12 x 19 px person, a
4-pixel horizontal offset is enough to drop IoU below 0.50. These are not gross
errors.

### Size does not explain the failures

| Group    | Small | Medium or larger |
| -------- | ----: | ---------------: |
| Missed   |    86 |         1 (1.1%) |
| Detected | 2,394 |         3 (0.1%) |

Mean width is 13.6 px for missed and 12.8 px for detected - the missed persons
are marginally *wider*. Median width is identical at 12.0 px.

**Target scale is not the discriminator.** This is consistent with MT-006 and
MT-007 both failing: each addressed scale, which was never the problem.

### Crowding is the discriminator

Share of each group with a neighbouring annotated person closer than a given
distance:

| Threshold | Localisation failures | Recognition failures | Detected |
| --------- | --------------------: | -------------------: | -------: |
| < 10 px   |                 32.8% |                18.8% |     5.7% |
| < 15 px   |                 65.7% |                31.2% |    20.6% |
| < 20 px   |                 77.6% |                37.5% |    32.3% |
| < 30 px   |                 88.1% |                37.5% |    51.4% |

Median nearest-neighbour distance is about 13 px for the localisation failures
against 29 px for detected persons. They are **5.8x more likely** than detected
persons to have a neighbour within 10 px, and 88% have one within 30 px.

The box shapes agree. Median height/width is 1.33 for the failures against 1.67
for detected persons, and 38% of localisation failures have an aspect ratio
below 1.2 against 22% of detected persons. A box squarer than a standing person
seen from above is what a box spanning two adjacent people looks like.

This remained circumstantial until the mechanism itself was measured in
section 24, which confirms it: 63% of the overlapping failures are a single
prediction drawn around two annotated people, against 4.46% among successful
detections.

### Recognition failures are a daylight problem

| Group                 | Daylight share |
| --------------------- | -------------: |
| Recognition failures  |          57.9% |
| Localisation failures |          23.5% |
| Detected persons      |          19.9% |

Recognition failures are **2.9x over-represented in daylight**. In daytime
thermal imagery the background is warm, so a person's heat signature stands out
far less against it. The single largest failure in the test set - an 87 x 74 px
person, alone in the frame, with zero overlap from any model - is a daylight
image.

### Scene conditions

Over-representation of the shared failures relative to detected persons:

| Camera angle | Over-representation |
| ------------ | ------------------: |
| 40 degrees   |               x1.98 |
| 90 degrees   |               x1.72 |
| 30 degrees   |               x0.52 |

| Altitude | Over-representation |
| -------- | ------------------: |
| 110 m    |               x1.84 |
| 100 m    |               x1.81 |
| 90 m     |               x1.80 |
| 60 m     |               x0.56 |

Higher altitudes and oblique 40-degree views are harder, as expected. The
130 m figure (x0.63) runs against the trend but rests on only 4 missed persons
and should not be relied on.

### Conclusion

The remaining thermal detection problem is **not** small-object detection. It
is two separate problems:

1. **Crowding-induced localisation failure (78% of shared failures).** People
   standing close together produce boxes that are close but below IoU 0.50.
2. **Daylight thermal contrast (22%).** Warm backgrounds suppress the signature
   entirely.

Neither is addressed by higher resolution or by crop augmentation, which
explains the MT-006 and MT-007 results directly.

---

## 22. Thermal Model Ensembling

The per-person comparisons showed the three thermal models failing on different
people. Checking MT-006 against the 131 persons that MT-005 and MT-007 both
miss at confidence 0.25:

| Measurement                 | Result     |
| --------------------------- | ---------- |
| MT-006 recovers, of the 131 | 29 (22.1%) |
| Missed by all three models  | 102        |

Coverage of the 2,611 test persons:

| Model set                  | Matched | Recall |
| -------------------------- | ------: | ------ |
| MT-005 alone (best single) |   2,425 | 92.88% |
| MT-006 alone               |   2,404 | 92.07% |
| MT-007 alone               |   2,382 | 91.23% |
| MT-005 + MT-007            |   2,480 | 94.98% |
| MT-005 + MT-006            |   2,481 | 95.02% |
| All three                  |   2,509 | 96.09% |

That union is an oracle bound: it assumes fused boxes can be produced without
also inheriting every model's false positives. `scripts/ensemble_thermal.py`
measures the real system, pooling the three models' predictions, fusing
overlapping boxes, and scoring with the same person-level matching.

### Measured fusion results, all models at confidence 0.25

| Method | Min votes | Predicted | Matched | Missed | Unmatched | Recall | Precision | Blind images |
| ------ | --------: | --------: | ------: | -----: | --------: | ------ | --------- | -----------: |
| NMS    |         1 |     3,198 |   2,470 |    141 |       728 | 0.9460 | 0.7724    |            4 |
| NMS    |         2 |     2,762 |   2,393 |    218 |       369 | 0.9165 | 0.8664    |           15 |
| NMS    |         3 |     2,523 |   2,301 |    310 |       222 | 0.8813 | 0.9120    |           28 |
| WBF    |         1 |     3,198 |   2,476 |    135 |       722 | 0.9483 | 0.7742    |            4 |
| WBF    |         2 |     2,762 |   2,402 |    209 |       360 | 0.9200 | 0.8697    |           15 |
| WBF    |         3 |     2,523 |   2,309 |    302 |       214 | 0.8843 | 0.9152    |           28 |

Weighted box fusion beats plain NMS at every vote level, by 6 to 9 persons.
This is the effect predicted by section 21: averaging several independent
near-miss boxes recovers a better-centred box than any single one of them, and
over half the localisation failures sit within 0.10 IoU of the threshold.

### Ensembling versus lowering the threshold

Both routes trade false positives for recall. They are not equally priced.

| Configuration            | Matched | Missed | Unmatched | Recall | Precision | Blind images |
| ------------------------ | ------: | -----: | --------: | ------ | --------- | -----------: |
| MT-005 @ 0.25 (baseline) |   2,425 |    186 |       638 | 0.9288 | 0.7917    |            8 |
| MT-005 @ 0.10            |   2,468 |    143 |     1,300 | 0.9452 | 0.6550    |            4 |
| WBF ensemble @ 0.25      |   2,476 |    135 |       722 | 0.9483 | 0.7742    |            4 |

Cost per additional person recovered, against the MT-005 baseline:

| Route           | Persons gained | Extra unmatched boxes | Cost per person |
| --------------- | -------------: | --------------------: | --------------: |
| Lower threshold |             43 |                   662 |            15.4 |
| WBF ensemble    |             51 |                    84 |             1.6 |

**Against threshold lowering, the ensemble recovers more people at roughly a
tenth of the false-positive cost**, and beats it on every reported measure:
more matched, fewer missed, 578 fewer unmatched boxes, higher precision
(0.7742 against 0.6550), and the same number of images where a person is
present but nothing is reported.

Against the MT-005 baseline the trade is different and must not be overstated.
The ensemble improves recall and the missed-person count, but its precision is
**lower** than the baseline's - 0.7742 against 0.7917 - because it emits 84
more unmatched boxes. The ensemble buys recall and pays for it in false
positives; it is not strictly better than MT-005 on every axis.

For a search-and-rescue payload that is the right direction to trade, since a
missed person is the costlier error, but the report must state it as a trade
rather than as a free improvement.

### Cost

The ensemble runs three models per frame. Section 17 and the latency benchmark
both show the models are launch-bound at batch 1, so three sequential models
cost roughly three times the frame budget, which the current pipeline cannot
absorb at full rate. Whether this is affordable depends on exported-model
latency, which is not yet measured.

An ensemble is therefore established as the best-performing thermal
configuration, but not yet as the deployable one.

---

## 23. Where the Detector Stands

Seven training experiments and four diagnostic studies support a single
conclusion: **single-model scale tuning is exhausted, and the remaining
failures have named causes.**

| Direction tested           | Experiment | Outcome                          |
| -------------------------- | ---------- | -------------------------------- |
| Higher RGB resolution      | MT-003/004 | Worked; recall still the limit   |
| Inference-time tiling      | RGB study  | No improvement                   |
| Higher thermal resolution  | MT-006     | Localisation up, recall down     |
| Crop-augmented training    | MT-007     | No improvement, 5x training cost |
| Lower confidence threshold | Diagnostic | 15 false positives per person    |
| Model ensembling           | Section 22 | 1.6 false positives per person   |

The evidence for the next training experiment, should one be run, is now
specific rather than speculative. Any MT-008 should target crowd separation or
daylight thermal contrast, because those are the measured failure modes, and
should not target target scale, which three separate studies have now ruled
out.

---

## 24. Verifying the Crowding Mechanism

Section 21 concluded that crowding drives the shared localisation failures, but
the evidence there was circumstantial: nearest-neighbour distance is only a
proxy for visual crowding, and a correlation with some third factor would look
the same. `scripts/verify_crowding_hypothesis.py` tests the mechanism directly
by asking what the near-miss prediction actually did.

Four outcomes are distinguishable:

| Mechanism  | Signature                                                      |
| ---------- | -------------------------------------------------------------- |
| merged     | One prediction covers 2+ annotated people and is oversized      |
| stolen     | The best-overlapping prediction was matched to a neighbour      |
| undersized | Covers this person alone, but well under the annotated area     |
| offset     | Covers this person alone, right size, simply displaced          |

Only the first two are crowding. The last two would point at box regression
instead, which needs a completely different fix.

### Result, 87 persons missed by MT-005 and MT-007 at confidence 0.10

| Mechanism  | Count | Share of the 68 with an overlapping prediction |
| ---------- | ----: | ---------------------------------------------: |
| merged     |    43 |                                          63.2% |
| stolen     |     2 |                                           2.9% |
| undersized |    20 |                                          29.4% |
| offset     |     3 |                                           4.4% |
| no overlap |    19 |                             (no prediction at all) |

**Crowding accounts for 45 of 68 (66.2%).**

The merged boxes are unambiguous: median area 2.03x the person they should have
covered, and 41 of the 43 touch exactly two annotated people (the other two
touch three).

The control settles the correlation question. Among the 2,397 successfully
detected persons, only **4.46%** have a prediction covering two or more
annotated people. The failures are therefore about **14x more likely** to
involve a multi-person box than the successes are. This is a mechanism, not a
coincidence.

Visual inspection of the rendered crops confirms it directly: the merged cases
show a single prediction box drawn around two adjacent people, with each person
separately annotated inside it.

### Ordinary box-regression error is almost absent

Only 3 of 68 failures (4.4%) are simple displacement. The original section 21
wording called this group "shifted", which was wrong. Measuring width and
height separately shows what it actually is:

| Group                  | Width ratio | Height ratio | Area ratio |
| ---------------------- | ----------: | -----------: | ---------: |
| Undersized failures    |       0.651 |        0.715 |       0.48 |
| All matched detections |       0.994 |        0.986 |       ~1.0 |

These boxes are correctly centred but roughly half the annotated area. In
thermal imagery the detector locks onto the bright heat signature while the
HIT-UAV annotation covers the whole body, so the box is too small to reach
IoU 0.50 no matter how well it is placed. A box containing 48% of the
annotation's area cannot exceed IoU 0.48, which matches the observed median
best-overlap IoU of 0.44 almost exactly.

### Global box enlargement does not work

The undersizing above suggests an obvious cheap fix: scale every predicted box
up. The matched-detection ratios already argue against it - at 0.994 and 0.986
the model shows **no systematic sizing bias** on the 2,468 persons it gets
right, so enlarging boxes would damage those to rescue 20.

Measured on MT-005 at confidence 0.25:

| Box scale | Matched | Missed | Unmatched | Recall |
| --------- | ------: | -----: | --------: | ------ |
| 1.00      |   2,425 |    186 |       638 | 0.9288 |
| 1.05      |   2,434 |    177 |       629 | 0.9322 |
| 1.10      |   2,424 |    187 |       639 | 0.9284 |
| 1.15      |   2,394 |    217 |       669 | 0.9169 |
| 1.20      |   2,323 |    288 |       740 | 0.8897 |
| 1.30      |   2,025 |    586 |     1,038 | 0.7756 |

The gain at 1.05 is 9 persons and it is gone by 1.10, with sharp degradation
beyond. Two reasons not to adopt it: the effect is marginal, and this sweep was
run on the held-out test set, so selecting a scale factor from it would be
tuning on test data. Any such factor would have to be chosen on the validation
split and only then confirmed here.

**Global box-scale correction is rejected.** The undersized cases are genuine
per-instance failures, not a calibration bias.

### What this means for MT-008

The remaining thermal failures now have measured mechanisms rather than
suspected ones:

| Mechanism              | Share of the 87 | Implied direction                     |
| ---------------------- | --------------: | ------------------------------------- |
| Merged / crowding      |      45 (51.7%) | Crowd separation                      |
| Undersized boxes       |      20 (23.0%) | Box-scale learning on partial signatures |
| No detection at all    |      19 (21.8%) | Daylight thermal contrast             |
| Ordinary displacement  |        3 (3.4%) | Nothing worth targeting               |

Higher resolution and crop augmentation address none of these, which is
precisely what MT-006 and MT-007 measured.

---

## 25. MT-008 - Mosaic Ablation (Crowd Separation)

Section 24 established that 63% of the shared localisation failures are a
single prediction drawn around two adjacent people. Mosaic augmentation
composes four training images into one and downscales them, manufacturing
dense arrangements of small objects at densities that do not occur in the
source imagery. The hypothesis was that this teaches the detector to accept
crowded groups as single objects.

MT-008 tests that and nothing else. Every parameter was copied verbatim from
the MT-005 run configuration; only `mosaic` changed from 1.0 to 0.0.
`close_mosaic` was left at its MT-005 value of 10, where it is inert with
mosaic disabled, rather than changed as a second variable.

### Configuration

| Parameter     | MT-005 | MT-008 |
| ------------- | ------ | ------ |
| Model         | YOLO26n | YOLO26n |
| Dataset       | HIT-UAV Person | HIT-UAV Person |
| Input size    | 640 | 640 |
| Epochs        | 50 | 50 |
| Batch size    | 8 | 8 |
| **mosaic**    | **1.0** | **0.0** |
| Training time | 1.599 h | 1.536 h |
| Best epoch    | 47 | 47 |

### Held-out test results

| Metric    | MT-005 | MT-008 | Change |
| --------- | ------ | ------ | ------ |
| Precision | 0.897  | 0.898  | +0.001 |
| Recall    | 0.891  | 0.876  | -0.015 |
| mAP@50    | 0.933  | 0.923  | -0.010 |
| mAP@50-95 | 0.510  | 0.502  | -0.008 |

### The target metric

| Measurement              | MT-005 | MT-008 | Change |
| ------------------------ | -----: | -----: | -----: |
| **Merged-box failures**  | **57** | **70** | **+13** |
| Matched persons          |  2,425 |  2,424 |     -1 |
| Missed persons           |    186 |    187 |     +1 |
| Unmatched boxes          |    638 |    717 |    +79 |
| Custom precision         | 0.7917 | 0.7717 | -0.020 |
| Blind images             |      8 |     11 |     +3 |

### Conclusion: hypothesis refuted

Removing mosaic made merged-box failures **worse**, from 57 to 70. The
mechanism it was designed to fix moved in the wrong direction, and aggregate
precision fell as well.

Mosaic is therefore not the cause of merged detections in crowded thermal
scenes. If anything it is mildly protective, which is a plausible reading: by
manufacturing dense arrangements it may give the detector *more* practice at
separating adjacent objects, not less.

This is a clean negative result and closes the augmentation direction.

---

## 26. What Has Been Ruled Out

Six directions have now been tested against the MT-005 baseline. It is worth
stating the whole set together, because the value of the remaining options can
only be judged against what has already failed.

| Direction                   | Experiment       | Outcome                                  |
| --------------------------- | ---------------- | ---------------------------------------- |
| Higher input resolution     | MT-006           | Recall down, mAP@50 down; merged 56 vs 57 |
| Crop-augmented training     | MT-007           | All test metrics down, 5x training cost   |
| Mosaic removal              | MT-008           | Merged boxes worse, 57 -> 70              |
| Lower confidence threshold  | Diagnostic       | 15 false positives per person recovered   |
| Global box-scale correction | Sweep            | No sizing bias exists to correct          |
| Longer training             | Curve analysis   | All four runs flat over their last 5 epochs |
| Larger model                | Latency measurement | YOLO26s is 1.7-2.8 FPS on the target   |

### On model capacity

Every experiment in this project has used YOLO26n. A larger model was proposed
as early as the MT-003 planning and never run, so capacity was the one major
untested variable.

It is now ruled out on compute rather than accuracy. Measured through ONNX
Runtime on CPU at 512 x 640:

| Model    | Params | GFLOPs | Dev CPU | Target estimate | Verdict            |
| -------- | ------ | ------ | ------: | --------------- | ------------------ |
| YOLO26n  | 2.57 M |    6.2 | 39.4 ms | 5.1 - 8.5 FPS   | Meets 6.7 at 80 m+ |
| YOLO26s  | 10.0 M |   23.1 | 117.3 ms| 1.7 - 2.8 FPS   | Too slow           |
| YOLO26m  | 21.9 M |   75.6 | -       | -               | Far too slow       |

YOLO26s costs 3x the inference time of YOLO26n and falls below even the
detect-only requirement at the low end of the estimate. On a CPU-only
companion computer the accuracy budget is capped by the compute budget, and
scaling the model up is not available.

It would become available with an AI accelerator, which is the main technical
argument for buying one.

---

## 27. Two-Model Ensembling - the Configuration That Works

With single-model improvements exhausted, the remaining lever is combining
models that fail on different people. Section 22 measured a three-model
ensemble; MT-008 now provides a fourth candidate, and the pairwise
combinations are cheaper.

### Complementarity

Union coverage of the 2,611 test persons (oracle upper bound):

| Combination             | Matched | Recall | Gain over MT-005 | Compute |
| ----------------------- | ------: | ------ | ---------------: | ------- |
| MT-005 alone            |   2,425 | 0.9288 |               -  | 1x      |
| MT-005 + MT-006         |   2,481 | 0.9502 |              +56 | 2x      |
| MT-005 + MT-008         |   2,480 | 0.9498 |              +55 | 2x      |
| MT-005 + MT-007         |   2,480 | 0.9498 |              +55 | 2x      |
| MT-005 + MT-006 + MT-007|   2,509 | 0.9609 |              +84 | 3x      |
| All four                |   2,520 | 0.9651 |              +95 | 4x      |

The second model contributes most of the available gain. Going from two models
to three adds 28 more people for another 50% compute, and the fourth adds 11.

### Measured fusion, not the bound

Weighted box fusion at confidence 0.25, scored with the same person-level
matching used throughout:

| Configuration        | Predicted | Matched | Missed | Unmatched | Recall | Precision | Blind |
| -------------------- | --------: | ------: | -----: | --------: | ------ | --------- | ----: |
| MT-005 alone         |     3,063 |   2,425 |    186 |       638 | 0.9288 | 0.7917    |     8 |
| **MT-005 + MT-006**  | **3,073** | **2,456** |**155** | **617** | **0.9406** | **0.7992** | **5** |
| MT-005 + MT-008      |     3,099 |   2,455 |    156 |       644 | 0.9403 | 0.7922    |     5 |
| Three-model          |     3,198 |   2,476 |    135 |       722 | 0.9483 | 0.7742    |     4 |

### The important result

**MT-005 + MT-006 is better than MT-005 alone on every measured axis at once.**

- 31 more people found
- 31 fewer missed
- **21 fewer** unmatched boxes, not more
- Higher custom precision, 0.7992 against 0.7917
- Blind images down from 8 to 5

That is not the usual recall-for-precision trade. The three-model ensemble
bought recall by accepting 84 extra false positives and 1.75 points of
precision; this pair improves both simultaneously.

The reason is visible in section 24's mechanism analysis. Most shared failures
are near-miss boxes between IoU 0.25 and 0.50. Averaging two independent
near-misses moves the fused box across the threshold, which converts a miss
into a match *and* removes what would otherwise have been counted as an
unmatched box. Both columns improve from the same effect.

### Cost, and which pair to choose

| Configuration   | Dev CPU | Target estimate | Tracking (6.7) | Detect-only (1.34) |
| --------------- | ------: | --------------- | -------------- | ------------------ |
| MT-005 alone    | 37.1 ms | 5.4 - 9.0 FPS   | Marginal at 60 m, clears 80 m+ | Yes |
| MT-005 + MT-008 | ~74 ms  | 2.7 - 4.5 FPS   | No, clears 4.0 at 100 m top-end | Yes |
| MT-005 + MT-006 | ~114 ms | 1.7 - 2.9 FPS   | No             | Yes                |

MT-006 runs at 960 px, so the better-performing pair is also the more
expensive one. MT-005 + MT-008 pairs two 640 px models for almost the same
recall (2,455 against 2,456) at two-thirds of the fusion cost, but it does not
reproduce the precision gain - its unmatched count rises to 644.

The choice therefore depends on which requirement binds:

- **Movement classification required at 60 m:** MT-005 alone is the only
  configuration that fits.
- **Flying at 100 m with movement classification:** MT-005 + MT-008 is
  marginal at the optimistic end of the estimate.
- **Detect-and-report without movement state:** MT-005 + MT-006 fits
  comfortably and is the most accurate configuration available.

This is the first configuration in the project that improves on the frozen
baseline without a compensating regression. It should be the reported result,
with its compute cost stated alongside.

---

## 28. Concept of Operations - Detect-and-Report First

The two-model ensemble in section 27 is the most accurate configuration
measured, but at 2x inference it cannot sustain the 6.7 FPS that movement
classification needs at 60 m. That forces a choice, and for a search-and-rescue
payload the choice is not close.

### Detect-and-report is the primary requirement

Three reasons, in order of weight.

**1. The victims who most need finding are the ones who cannot move.**

A movement classifier separates moving people from stationary ones. In search
and rescue, the highest-priority casualties are unconscious, injured, trapped
or hypothermic - and therefore stationary. The classifier would label them
identically to a warm rock. Thermal search exists precisely to find people who
cannot signal for themselves, so constraining the search configuration to
preserve a feature that fails on the most critical casualties inverts the
mission.

**2. The trade costs found people.**

At 60 m with movement classification, MT-005 alone is the only configuration
that fits the frame budget. It finds 2,425 of 2,611 test persons against the
ensemble's 2,456. Accepting 31 fewer found people in exchange for a movement
label is a poor exchange when a missed person can die and a false alarm costs
a rescuer a walk.

**3. Movement state is a secondary attribute, not a detection.**

The pipeline already withholds the movement label at frame borders rather than
guessing, on the principle that a wrong movement label is worse than no label.
The same principle extends further: movement state is useful context attached
to a detection, never a precondition for reporting one.

### The aircraft removes the trade anyway

The frame-rate requirement is set by ground speed, and this airframe is an
eVTOL. It can slow down or hover. From `scripts/coverage_requirements.py`:

| Ground speed | Tracking requirement @ 60 m | @ 100 m |
| ------------ | --------------------------: | ------: |
| 20 m/s cruise |                    6.70 FPS | 4.02 FPS |
| 12 m/s        |                    4.02 FPS | 2.41 FPS |
| 8 m/s loiter  |                    2.68 FPS | 1.61 FPS |

The two-model ensemble delivers an estimated 1.7 - 2.9 FPS. At 8 m/s and 100 m
it clears the 1.61 FPS tracking requirement outright.

### Two-phase operation

```text
   SEARCH PHASE                        INVESTIGATE PHASE
   20 m/s cruise, 100 m                8 m/s loiter or hover
   MT-005 + MT-006 WBF                 same ensemble
   detect and report                   movement classification available
        |                                      ^
        |  person detected                     |
        +--> geolocate (lat/lon + error) ------+
                   |
                   v
        Telemetry / Ground Station
```

The search phase maximises people found per unit of ground covered. On a
detection, the geolocated coordinate lets the aircraft return to and loiter
over the position, where the reduced ground speed lengthens dwell time enough
that movement classification becomes available with the same models.

Nothing is given up. The mission gets maximum recall during search and the
movement attribute where it is actually useful - over a confirmed target,
where "is this person moving" informs triage.

### Consequence

**The reported configuration is MT-005 + MT-006 with weighted box fusion,
operating detect-and-report.** Movement classification is retained as a
loiter-phase capability rather than a search-phase constraint.

---

## 29. What the "False Positives" Actually Are

Section 27 showed fusion improving recall and precision simultaneously, which
is unusual. Investigating why produced the most useful single result in this
phase.

MT-005 was run over its own **training** split - never the test split, which
would leak - and every prediction that failed to match an annotated person at
IoU 0.50 was collected. There are 1,729 of them. Each was then classified by
how much it overlapped the nearest real person.

| What the "false positive" is                  | Count | Share |
| --------------------------------------------- | ----: | ----- |
| Near-miss box **on a real person**, IoU 0.25-0.50 |   877 | 50.7% |
| Touching a real person, IoU 0-0.25            |    89 |  5.1% |
| Genuine background firing, IoU = 0            |   763 | 44.1% |
| **Touching a real person at all**             | **966** | **55.9%** |

### The precision problem and the recall problem are one problem

**Over half of the detector's "false positives" are boxes on real people that
are simply not accurate enough to count.** A near-miss box is penalised twice
by the evaluation: once as a missed person, and once as an unmatched
prediction. It appears in both the recall column and the precision column as a
separate failure, when it is one failure.

This explains the section 27 result exactly. Weighted box fusion averages two
independent near-miss boxes; the fused box crosses IoU 0.50; and that single
correction simultaneously converts a miss into a match *and* deletes what the
evaluation was counting as a false positive. One mechanism, both columns. The
+31 matched and -21 unmatched are the same 31 events seen from two sides.

It also reframes the whole precision figure. A custom precision of 0.7917 does
not mean the detector hallucinates people 21% of the time. It means roughly
9% genuine background firing and roughly 12% boxes that found a person and
localised them poorly.

### Consequence for hard-negative mining

Hard-negative mining addresses background firing. It can therefore reach at
most the 44% of unmatched boxes that are genuine background, and none of the
56% that are localisation failures on real people.

That is a substantially weaker case than Lygouras et al. faced, where the
false positives were boats in otherwise-empty water - entirely the background
category. It does not make the experiment worthless, but it caps the available
gain before the run starts, and the result should be read against that cap.

---

## 30. MT-009 - Hard-Negative Mining

Applies the method from Lygouras et al. [4]: collect the regions the detector
falsely fires on and train them explicitly as background.

### Mining

Negatives are mined from the **training** split only. Harvesting the model's
mistakes on test data and training on them would leak the test set and
invalidate every number in this document.

| Stage                                      | Count |
| ------------------------------------------ | ----: |
| False positives found on the training split | 1,729 |
| Rejected: a real person fell inside the crop | 1,444 |
| **Usable negative crops**                   | **283** |

The rejection rate is high and unavoidable: 95.9% of these false positives
occur on images that already contain people, so any window around them tends
to catch one. A 256 px crop yielded only 156 usable negatives; 128 px yields
283 while still carrying roughly 7x the area of a typical 13 x 18 px target.

A crop is discarded if any annotated person overlaps it by more than 5% of
that person's area. Teaching a real person as background would be far worse
than discarding a good negative.

Each surviving crop is written with an empty label file, which is how YOLO
represents a background image.

### Configuration

| Parameter     | MT-005 | MT-009 |
| ------------- | ------ | ------ |
| Model         | YOLO26n | YOLO26n |
| Input size    | 640 | 640 |
| Epochs        | 50 | 50 |
| Batch size    | 8 | 8 |
| mosaic        | 1.0 | 1.0 |
| Training images | 2,029 | **2,312** (+283 negatives) |
| Val / test    | unchanged | unchanged |

Only the training set differs. Validation and test splits are copied byte for
byte from HIT-UAV Person, so MT-009 is directly comparable to every other
thermal experiment.

### Expectation, stated before the result

The negative ratio is 1:7 against Lygouras's 1:1, and section 29 caps the
addressable share of unmatched boxes at 44%. A large improvement would be
surprising. The honest prediction is a modest reduction in unmatched boxes
with recall approximately unchanged.

Recording the expectation in advance keeps the result interpretable either
way.

---

## 31. MT-009 Result - Hard-Negative Mining Rejected

Training completed in 1.780 hours, best epoch 46.

### Held-out test results

| Metric    | MT-005 | MT-009 | Change |
| --------- | ------ | ------ | ------ |
| Precision | 0.897  | 0.899  | +0.002 |
| Recall    | 0.891  | 0.882  | -0.009 |
| mAP@50    | 0.933  | 0.930  | -0.003 |
| mAP@50-95 | 0.510  | 0.506  | -0.004 |

### Custom error analysis, confidence 0.25

| Measurement              | MT-005 | MT-009 | Change |
| ------------------------ | -----: | -----: | -----: |
| Matched persons          |  2,425 |  2,428 |     +3 |
| Missed persons           |    186 |    183 |     -3 |
| **Unmatched boxes**      |  **638** | **637** | **-1** |
| **Blind images**         |    **8** |  **12** | **+4** |
| Custom precision         | 0.7917 | 0.7922 | +0.0005 |
| Small-person recall      | 0.9294 | 0.9302 | +0.0008 |
| Merged-box failures      |     57 |     55 |     -2 |

### Verdict against the prediction

Section 30 recorded a prediction before the run: *"a modest reduction in
unmatched boxes with recall approximately unchanged."*

Half of that held. Recall was approximately unchanged, +0.0011 on the custom
measure. But the unmatched-box reduction was **one box out of 638**, which is
not a modest improvement - it is no improvement. The prediction was too
optimistic even at its deliberately low bar.

Every difference in both tables sits inside run-to-run variation. The single
metric that moved outside noise moved the wrong way: **blind images rose from
8 to 12**, a 50% increase in the error mode that matters most for a search
payload, where a person is present in the frame and nothing at all is
reported.

**MT-009 is rejected. MT-005 remains the frozen baseline.**

### Why it failed

Section 29 capped the available gain before the run: hard-negative mining
addresses background firing, which is only 44% of unmatched boxes. The other
56% are near-miss boxes on real people, which no amount of background training
can fix.

Two further factors compressed even that 44%:

1. **Ratio.** 283 mined negatives against 2,029 training images is 1:7.
   Lygouras et al. used 1:1. The intervention was roughly seven times weaker
   than the method it copies.
2. **Availability.** 1,444 of 1,727 candidate crops were discarded because a
   real person fell inside the window. 95.9% of these false positives occur on
   images that already contain people, so the negatives that would have been
   most informative are exactly the ones that cannot be safely extracted.

The method is sound and is well evidenced in the literature. It does not
transfer to this failure profile. Lygouras's detector was firing on boats in
otherwise-empty water - entirely the background category, and freely
collectable as clean negatives. This detector fires mostly on people it has
already found and localised badly.

### What this closes

Seven directions have now been tested against the MT-005 baseline and none has
improved it:

| Direction                   | Experiment | Outcome                          |
| --------------------------- | ---------- | -------------------------------- |
| Higher input resolution     | MT-006     | Recall down                      |
| Crop-augmented training     | MT-007     | All test metrics down            |
| Mosaic removal              | MT-008     | Merged boxes 57 -> 70            |
| Hard-negative mining        | MT-009     | No change; blind images 8 -> 12  |
| Lower confidence threshold  | Diagnostic | 15 false positives per person    |
| Global box-scale correction | Sweep      | No sizing bias exists to correct |
| Longer training             | Curves     | All runs flat over last 5 epochs |
| Larger model                | Latency    | YOLO26s is 1.7-2.8 FPS on target |

Single-model training-side improvement on this dataset is exhausted. The only
intervention that has improved on the baseline is combining models that fail
on different people, and it improves both precision and recall at once because
it attacks the localisation mechanism that produces both failure types.

### The remaining untested lever

Box-loss weighting has never been changed. Every experiment from MT-001 to
MT-009 used the Ultralytics defaults, `box: 7.5` and `dfl: 1.5`.

That is the one remaining lever aimed at the measured dominant mechanism.
Section 29 showed that 56% of unmatched boxes are localisation near-misses,
and that each is counted twice - once as a missed person and once as a false
positive - so an improvement in box regression pays in both columns
simultaneously. That is precisely the effect observed when fusion corrects
near-misses.

It is worth one controlled run. It is also the last one worth making before
the constraint moves from the detector to the hardware.

---

## 32. Is IoU 0.50 the Right Criterion?

Every recall figure in this project uses IoU >= 0.50. That is the computer
vision convention, inherited from PASCAL VOC and COCO. It has never been
checked against what the payload actually delivers.

The payload does not hand a rescue team a bounding box. It hands them a
latitude and longitude with a stated uncertainty. A detection is operationally
useful if it puts the team close enough to find the person, and "close enough"
is set by the geolocation error budget, not by box overlap.

### What each threshold means on the ground

At 100 m with a 50 degree horizontal field of view and a 640 px sensor, the
ground sampling distance is 0.146 m/px and a 12 x 19 px person is
1.75 x 2.77 m. For two equal boxes offset along one axis:

| IoU  | Pixel offset | Ground error | Matched | Missed | Recall | Unmatched |
| ---- | -----------: | -----------: | ------: | -----: | ------ | --------: |
| 0.50 |          4.0 |       0.58 m |   2,425 |    186 | 0.9288 |       638 |
| 0.40 |          5.1 |       0.75 m |   2,484 |    127 | 0.9514 |       579 |
| 0.30 |          6.5 |       0.94 m |   2,509 |    102 | 0.9609 |       554 |
| 0.25 |          7.2 |       1.05 m |   2,511 |    100 | 0.9617 |       552 |
| 0.20 |          8.0 |       1.17 m |   2,514 |     97 | 0.9628 |       549 |
| 0.10 |          9.8 |       1.43 m |   2,516 |     95 | 0.9636 |       547 |

### The argument

The geolocation fix reported for each detection carries a position error of
**+/- 3.8 m** at 100 m nadir, dominated by attitude uncertainty at roughly
1.75 m per degree.

The box error that IoU 0.25 still accepts is **1.05 m**.

**The coordinate handed to the rescue team is uncertain by 3.6 times more than
the box error the criterion would reject.** A detection discarded for landing
between IoU 0.25 and 0.50 would have been reported to the same place on the
ground as one that passed. The rescue team walks to the same spot.

Under the convention the detector finds 2,425 people. Under a criterion
derived from the delivery mechanism it finds **2,511** - 86 more, and 86 fewer
missed.

Recall saturates below 0.25: dropping to 0.10 adds only five more. So 0.25 is
not an arbitrary slide toward an easier number; it is the point where "the box
is on the person" stops excluding anyone.

This also re-reads section 29 from the other side. The near-miss boxes between
IoU 0.25 and 0.50 were counted twice, once as a missed person and once as a
false positive. Under the operational criterion most of them become what they
physically are: a person, found, and localised well within the accuracy the
system can report anyway.

### How this must be reported

The temptation here is obvious and must be resisted. Loosening a threshold
until the number improves is metric-gaming, and the difference between that
and what is argued above is entirely in whether the criterion is derived from
something real.

Two rules:

1. **mAP@50 stays in the results.** It is the comparable academic metric, it
   is what Rizk, Lygouras and the other reference systems report, and removing
   it would make this work incomparable.
2. **The operational figure is reported alongside, never instead**, and always
   with the criterion named and this derivation attached. A recall figure
   without its IoU threshold is meaningless.

Stated correctly:

> MT-005 achieves 0.933 mAP@50 on the held-out thermal test split. Under the
> conventional IoU 0.50 matching criterion it recovers 92.9% of test-set
> persons. Because the payload reports geolocated coordinates with a +/- 3.8 m
> position uncertainty, a box error below roughly 1 m does not change where a
> rescue team is directed; under a mission-derived IoU 0.25 criterion the
> detector recovers 96.2%.

That is defensible. "Our recall is 96.2%" on its own is not.

### Why this matters more than another training run

No model changed. Seven training directions were tested and none improved the
baseline, while this costs nothing and recovers 86 people - more than any
experiment in this project achieved, including the two-model ensemble's 31.

The lesson is that the evaluation criterion was inherited rather than chosen,
and it was stricter than the physics of the delivery mechanism requires.

---

## 33. The Compute Budget Vetoes the P2 Head

Section 32 and `docs/related_work.md` ended with three techniques worth
adopting, ranked by how well they fit the measured failure profile. That
ranking considered accuracy and ignored cost, which is the wrong order for
this project: the companion computer is a Raspberry Pi 5 class board and the
deployed baseline already sits at an estimated 5.4-9.0 FPS against a binding
6.7 FPS requirement. There is almost no headroom to spend.

`scripts/architecture_budget.py` measures what each candidate costs before a
training run is spent finding out what it gains. Each architecture is built
from its YAML at the deployment class count, exported to ONNX at the
deployment input shape, and timed through ONNX Runtime on CPU with four
threads - the same harness as `benchmark_edge_cpu.py`, so the numbers line up
with section 3 of `deployment_target.md`.

Weights are random. Latency depends on the graph, not on weight values, so an
untrained network times identically to a trained one. **The cost is knowable
before the accuracy is**, and that is the whole point of running this first.

### Measured, 512 x 640, nc = 1

| Architecture | Params | GFLOPs | Dev CPU | Pi 5 estimate | Pi 5 FPS |
|--------------|-------:|-------:|--------:|---------------|----------|
| YOLO26n (deployed) | 2.50 M | 4.7 | 38.8 ms | 117 - 194 ms | 5.2 - 8.6 |
| YOLO26n-p2 | 2.52 M | 6.1 | 50.2 ms | 150 - 251 ms | **4.0 - 6.6** |
| YOLO26s | 9.95 M | 18.2 | 114.2 ms | 343 - 571 ms | 1.8 - 2.9 |

### Against the derived requirements

| Architecture | 6.7 FPS @ 60 m | 4.0 FPS @ 100 m | 1.34 FPS report-only |
|--------------|----------------|-----------------|----------------------|
| YOLO26n | marginal | clears | clears |
| **YOLO26n-p2** | **FAILS** | marginal | clears |
| YOLO26s | FAILS | FAILS | clears |

**The P2 head fails the binding requirement outright.** Not marginally - its
optimistic end, 6.6 FPS, still falls below the 6.7 FPS the movement classifier
needs at 60 m. It would restrict the aircraft to 100 m and above, and even
there it is only marginal.

MT-012 is therefore **cancelled before it is run**. This is a cheaper negative
result than the seven that preceded it: fifteen minutes of benchmarking
instead of a 90-minute training run followed by an evaluation that could not
have changed the answer.

### GFLOPs understate the cost of a high-resolution head

Worth recording separately, because it will mislead the next architecture
decision if it is not:

| Architecture | GFLOPs ratio | Measured latency ratio |
|--------------|-------------:|-----------------------:|
| YOLO26n-p2 | 1.30x | **1.29 - 1.50x** |
| YOLO26s | 3.87x | 2.94 - 3.03x |

The two were measured twice, once while MT-010 was training and competing for
CPU and once after; the spread above is that variance, and it is why a clean
re-run is worth doing before this table is quoted anywhere load-bearing.

The pattern survives the noise, and the two architectures scale in opposite
directions. YOLO26s costs **less** than its arithmetic predicts - a wider
backbone is dense matrix work, which is exactly what SIMD and cache are good
at. The P2 head costs **more** than its arithmetic predicts, because its extra
branch operates on a stride-4 feature map with four times the spatial area of
P3. That is bandwidth-bound rather than arithmetic-bound, and memory bandwidth
is precisely where a Pi 5 is weakest relative to an x86 development machine.

The practical consequence: **GFLOPs is not a safe proxy for cost on this
target**, and it errs in the dangerous direction for exactly the kind of
change small-object detection work keeps recommending.

### What survives, and why it is the better half

Removing P2 leaves the two techniques that cost nothing at inference:

| Technique | Inference cost | Why it is free |
|-----------|----------------|----------------|
| **MT-011** two-stage aerial pretraining | **Zero** | Identical graph; only the initial weights differ |
| **MT-013** NWD localisation loss | **Zero** | Loss is training-only; the exported graph is unchanged |
| MT-012 P2 head | 1.3 - 1.5x | Cancelled |

This is not a consolation. Section 24 measured that 56% of unmatched
predictions are near-miss boxes on real people, and section 32 showed a
4-pixel displacement is enough to fail IoU 0.50 on a 12 x 19 px target. NWD
attacks that mechanism directly by replacing a similarity that collapses
discontinuously with one that degrades smoothly - and it does so in the loss,
where it costs nothing to deploy.

The technique best matched to the measured failure is also the one that is
free. The one that breaks the budget was the one recommended on scale grounds,
and scale has already been ruled out three times: MT-006 at 960 px, MT-007
with crop augmentation, and the sensor-resolution study.

### The rule this establishes

Any future change that touches the network - a head, a backbone, an attention
module, a neck - goes through `architecture_budget.py` before it is trained.
Changes that touch only training - losses, augmentation, datasets, schedules -
are free at inference and need no gate.

Every experiment from MT-005 to MT-010 happened to be the second kind, so the
deployed cost never moved and the question never came up. It comes up now
because the literature survey's recommendations are the first that would have
changed it.

---

## 34. MT-010 - Box-Loss Weighting Rejected

The failure-mechanism analysis found that 29% of localisation failures were
**undersized** boxes - a prediction correctly centred on a person but drawn too
small to reach IoU 0.50. MT-010 tested the obvious response: double the
box-regression loss weight so the optimiser cares more about getting the
extents right.

    MT-005   box = 7.5   (Ultralytics default)
    MT-010   box = 15.0  (this run)

Everything else is copied from MT-005, so the result is attributable to the
loss weight alone.

### Result: worse on every axis

Held-out test split, confidence 0.25, IoU >= 0.50.

| Metric | MT-005 baseline | MT-010 | Delta |
|--------|----------------:|-------:|-------|
| Matched persons | 2,425 | 2,422 | **-3** |
| Missed persons | 186 | 189 | +3 |
| Unmatched boxes | 638 | 646 | +8 |
| **Blind images** | **8** | **18** | **+10** |
| Recall | 0.9288 | 0.9276 | -0.0012 |
| Custom precision | 0.7917 | 0.7894 | -0.0023 |
| Small-person recall | 0.9294 | 0.9282 | -0.0012 |
| mAP@50 | 0.9330 | 0.9246 | **-0.0084** |

Under the mission-derived IoU 0.25 criterion it is worse there too: 2,494
matched against the baseline's 2,511.

**MT-010 is rejected.** That is the eighth direction tested and the eighth
rejected.

### But the hypothesis was not wrong

This is the interesting part, and it would be lost by recording only the
headline. The mechanism it targeted did move:

| Mechanism | MT-005 | MT-010 | Change |
|-----------|-------:|-------:|--------|
| merged | 57 | 49 | **-8** |
| undersized | - | 30 | - |
| offset | - | 10 | - |
| stolen | - | 15 | - |
| no_overlap | - | 85 | - |

Weighting box regression more heavily **did** reduce merged boxes, by 8. The
run still lost overall because the cost landed somewhere the hypothesis never
considered: **blind images more than doubled, from 8 to 18**. An image where a
person is present and nothing at all is reported is the worst failure this
payload has, and MT-010 produced ten more of them.

The mechanism is a budget. Total loss is a weighted sum, so doubling the box
term halves the relative weight of the classification term. The detector
became better at drawing boxes and worse at deciding there was something
there, and for a search payload that is the wrong trade at any exchange rate.

### What this adds to the picture

Seven previous rejections all attacked target **scale** and produced no
movement in any mechanism. MT-010 is the first that moved its target and still
lost. That distinction matters for what to try next:

- Scale-directed changes do not work because scale is not the problem.
- Loss-*weighting* changes cannot win because they only move weight between
  terms that are all still needed.

What has never been tried is changing the *shape* of the localisation loss
rather than its weight - keeping the classification term untouched and making
the box term informative where it currently is not. That is exactly what NWD
does, and it is why MT-013 is the next run rather than another weight sweep.

---

## 35. INT8 Quantisation Rejected - It Costs 406 People

INT8 was the one remaining lever that makes the payload *lighter* rather than
heavier. Everything else considered either costs compute or is free; nothing
gives any back. With the deployed baseline sitting at "marginal" against the
6.7 FPS requirement, and the two-model ensemble rejected purely on its 2x cost,
a 2-3x speedup would have changed two answers at once.

It does not deliver one, and the attempt produced three findings worth keeping.

### Finding 1: the wrong activation type makes it 2.3x slower

First attempt, static QDQ quantisation with **int8** activations and
per-channel int8 weights:

| | Size | Dev CPU | Pi 5 estimate |
|---|-----:|--------:|---------------|
| FP32 | 9.76 MB | 35.5 ms | 5.6 - 9.4 FPS |
| INT8, int8 activations | 3.03 MB | **82.0 ms** | 2.4 - 4.1 FPS |

**0.43x - it got 2.3x slower.** ONNX Runtime's x86 CPU kernels are built for
the `u8s8` combination: uint8 activations against int8 weights. With int8
activations the integer path is not taken, so every convolution still runs in
FP32 *and* pays for the quantise/dequantise conversions around it. Switching
activations to uint8 recovered it to 1.17x.

This is a property of the runtime, not of the model, and it is the reason
`results/quantization/quantization.json` records the exact ONNX Runtime
version alongside the numbers.

### Finding 2: quantising the detection head silently destroys the detector

With uint8 activations the model was 1.17x faster and **found nothing at all**
- zero detections across all 579 test images, against 368 images with
detections for FP32.

The cause is visible in the raw network output on a single frame:

| | Confidence max | Confidence mean | Box coordinate range |
|---|---------------:|----------------:|----------------------|
| FP32 | 0.008748 | 0.000032 | [-74.96, 638.86] |
| INT8 | **0.000000** | **0.000000** | [-82.42, 639.47] |

**Box regression survived intact; the classification branch collapsed to
exactly zero.** Confidences occupy a very small range near zero, so mapping
them onto 256 integer levels with a scale set by the calibrated maximum rounds
the entire branch away. The boxes still looked healthy, which is precisely
what makes this dangerous: a latency benchmark on the broken model reports a
speedup, because timing does not care whether the output is meaningful.

Excluding the detection head - 79 nodes under `/model.23/` - from
quantisation and leaving it in FP32 restores detection. `quantize_int8.py`
now does this by default, and `--quantise-head` exists only to reproduce the
failure.

### Finding 3: even done correctly, it costs too much accuracy

With the head excluded, the model works. Evaluated with the project's own
matching against the frozen baseline, confidence 0.25, IoU >= 0.50:

| Metric | Baseline | FP32 (ONNX) | INT8 | Delta vs baseline |
|--------|---------:|------------:|-----:|------------------:|
| Matched persons | 2,425 | **2,425** | 2,019 | **-406** |
| Missed persons | 186 | **186** | 592 | +406 |
| Unmatched boxes | 638 | **638** | 1,124 | +486 |
| Blind images | 8 | **8** | 23 | +15 |
| Recall | 0.9288 | **0.9288** | 0.7733 | **-0.1555** |
| Custom precision | 0.7917 | **0.7917** | 0.6424 | -0.1493 |

The FP32 column is worth pausing on. Driving the exported graph directly at
its native 512 x 640 shape reproduces the frozen baseline **exactly** - every
delta zero. That validates the prediction path used for the INT8 column, so
the INT8 losses are attributable to quantisation and not to the harness.

INT8 loses **406 of 2,425 people, 16.7% of every detection the payload makes**,
and nearly triples the images where a person is present and nothing is
reported.

### What it buys, and why that settles it

| | Dev CPU | Pi 5 estimate | 6.7 FPS @ 60 m |
|---|--------:|---------------|----------------|
| FP32 | 41.8 ms | 4.8 - 8.0 FPS | marginal |
| INT8 | 35.8 ms | 5.6 - 9.3 FPS | **marginal** |

1.17x, and **the deployment verdict does not change**. Both are marginal at
60 m and both clear 100 m. Not one decision in this project would be made
differently.

So the trade is 406 people for a speedup that changes nothing. **INT8 is
rejected**, and `deployment_target.md` section 4, which listed it as "the most
promising untried option", is corrected accordingly.

### The caveat that does not rescue it

The 1.17x is an x86 measurement, and the fusion diagnostic shows why it is
low: **zero QLinearConv nodes**, so the integer kernels were never used even
with uint8 activations. ONNX Runtime's ARM64 backend has better int8 support,
so on a real Cortex-A76 the speedup could be considerably larger.

That caveat applies to the speedup only. **The accuracy loss is not hardware
-dependent** - 406 people are lost on any device, because the arithmetic that
loses them happens before anything reaches the CPU's instruction set. A larger
speedup would change the exchange rate, not the fact that the exchange is bad.

If an ARM board is ever in hand, the test to re-run is the latency half. The
accuracy half is already answered.
