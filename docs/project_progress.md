# Project Progress

## Project

**Development of a Fixed Wing eVTOL UAV for Emergency Logistics**

## Human Detection Subsystem Progress

| Stage | Activity                                                  | Status      |
| ----: | --------------------------------------------------------- | ----------- |
|     1 | Project workspace and Python environment setup            | Completed   |
|     2 | NVIDIA CUDA environment setup                             | Completed   |
|     3 | PyTorch GPU verification                                  | Completed   |
|     4 | Ultralytics/YOLO environment setup                        | Completed   |
|     5 | VisDrone RGB dataset preparation                          | Completed   |
|     6 | VisDrone pedestrian + people class conversion to `person` | Completed   |
|     7 | RGB dataset annotation visualization                      | Completed   |
|     8 | MT-001 pilot training, 3 epochs @ 640 px                  | Completed   |
|     9 | MT-002 RGB baseline training, 50 epochs @ 640 px          | Completed   |
|    10 | MT-002 result/error analysis                              | Completed   |
|    11 | MT-003 RGB training, 50 epochs @ 960 px                   | Completed   |
|    12 | MT-003 evaluation and comparison                          | Completed   |
|    13 | MT-004 RGB training, 50 epochs @ 1280 px                  | Completed   |
|    14 | RGB small-object/error analysis                           | Completed   |
|    15 | RGB confidence-threshold investigation                    | Completed   |
|    16 | RGB tiled-inference investigation                         | Completed   |
|    17 | HIT-UAV thermal dataset integration                       | Completed   |
|    18 | Thermal annotation visualization/verification             | Completed   |
|    19 | MT-005 thermal baseline, 50 epochs @ 640 px               | Completed   |
|    20 | MT-005 held-out test evaluation                           | Completed   |
|    21 | MT-005 thermal error analysis                             | Completed   |
|    22 | MT-005 confidence-threshold diagnostic                    | Completed   |
|    23 | MT-005 remaining-miss analysis                            | Completed   |
|    24 | MT-006 thermal training, 50 epochs @ 960 px               | Completed   |
|    25 | MT-006 evaluation and comparison                          | Completed   |
|    26 | MT-007 thermal crop-augmented training @ 640 px           | Completed   |
|    27 | MT-007 evaluation and per-person MT-005 comparison        | Completed   |
|    28 | Unified RGB/thermal model benchmark                       | Completed   |
|    29 | Deployment latency benchmark                              | Pending     |
|    30 | Movement detection/tracking                               | Pending     |
|    31 | Real-time inference pipeline                              | Pending     |
|    32 | Model export for embedded inference (ONNX)                | Pending     |
|    33 | RGB/thermal model ensembling investigation                | Pending     |
|    34 | Embedded companion-computer selection                     | Pending     |
|    35 | Embedded model deployment/benchmarking                    | Pending     |
|    36 | UAV payload/system integration                            | Pending     |

---

## Completed Work

### Development Environment

A Windows-based development environment has been established for the
human-detection subsystem.

The NVIDIA RTX 4050 Laptop GPU is used for accelerated model training and
inference through CUDA-enabled PyTorch.

Current software environment:

- Python 3.12.5
- PyTorch 2.14.0+cu132
- CUDA 13.2
- Ultralytics 8.4.152
- YOLO26n

### RGB Dataset

The VisDrone DET dataset was prepared for single-class human detection.

The original `pedestrian` and `people` categories were combined into:

```text
0: person
```

Dataset size:

- 6,471 training images
- 548 validation images
- 106,396 training person annotations
- 13,969 validation person annotations

The converted annotations were visually verified.

### RGB Model Development

Three main RGB training stages were completed:

- MT-002: 640 x 640 baseline
- MT-003: 960 x 960
- MT-004: 1280 x 1280

MT-004 achieved:

| Metric    | Result |
| --------- | ------ |
| Precision | 0.738  |
| Recall    | 0.563  |
| mAP@50    | 0.653  |
| mAP@50-95 | 0.297  |

Increasing resolution improved the official validation metrics, but small
aerial persons remained difficult to detect.

MT-004 is the selected RGB reference model.

### RGB Error Analysis

The RGB detector was investigated using:

- small-object analysis
- complete-failure analysis
- confidence-threshold testing
- tiled inference

The tested tiled configuration did not improve the overall custom validation
comparison, so it was not selected as the next optimization direction.

### Thermal Dataset

The HIT-UAV Infrared Thermal Dataset was integrated as a separate thermal
detection dataset.

Only the Person category was retained:

```text
0: person
```

Converted dataset:

| Split      | Images | Person annotations |
| ---------- | -----: | -----------------: |
| Train      |  2,029 |              8,533 |
| Validation |    290 |              1,168 |
| Test       |    579 |              2,611 |

The converted thermal annotations were visually inspected and verified.

### MT-005 Thermal Baseline

The first thermal baseline was trained at 640 x 640.

Validation:

| Metric    | Result |
| --------- | ------ |
| Precision | 0.890  |
| Recall    | 0.875  |
| mAP@50    | 0.919  |
| mAP@50-95 | 0.500  |

Held-out test:

| Metric    | Result |
| --------- | ------ |
| Precision | 0.898  |
| Recall    | 0.892  |
| mAP@50    | 0.933  |
| mAP@50-95 | 0.510  |

The held-out test set contains 579 images and 2,611 person instances.

### Thermal Error Analysis

At confidence 0.25, the MT-005 test predictions produced:

- 3,063 predicted boxes
- 2,425 matched persons at IoU >= 0.50
- 186 missed ground-truth persons
- 638 unmatched prediction boxes

A confidence 0.10 diagnostic recovered 43 additional persons but introduced 662
additional unmatched prediction boxes.

Further analysis showed that many remaining misses had an overlapping
prediction but insufficient IoU, indicating a localization limitation for small
thermal targets.

This produced two follow-up experiments, MT-006 and MT-007, each testing a
different way of increasing the scale at which small thermal targets are seen.

### MT-006 - Thermal Resolution Experiment

MT-006 repeated the MT-005 training at 960 x 960 with the same dataset split.

Held-out test results compared with MT-005:

| Metric    | MT-005 | MT-006 | Change |
| --------- | ------ | ------ | ------ |
| Precision | 0.897  | 0.903  | +0.006 |
| Recall    | 0.891  | 0.880  | -0.011 |
| mAP@50    | 0.933  | 0.926  | -0.007 |
| mAP@50-95 | 0.510  | 0.526  | +0.016 |

The resolution increase improved box localization (mAP@50-95) but reduced
recall. HIT-UAV imagery is natively 640 x 512, so upscaling adds no sensor
information; this is the opposite of the RGB result, where higher resolution
preserved real detail.

### MT-007 - Thermal Crop-Augmented Training

MT-007 moved tiling from inference time into the training set. A crop-augmented
training set was generated from the HIT-UAV training split:

| Content                | Images     | Person annotations |
| ---------------------- | ---------: | -----------------: |
| Original full images   |      2,029 |              8,533 |
| Generated 256 px crops |      9,241 |             16,942 |
| **Total**              | **11,270** |         **25,475** |

The validation and test splits were copied unchanged, so MT-005, MT-006 and
MT-007 remain directly comparable.

Held-out test results compared with MT-005:

| Metric    | MT-005 | MT-007 | Change |
| --------- | ------ | ------ | ------ |
| Precision | 0.897  | 0.895  | -0.002 |
| Recall    | 0.891  | 0.873  | -0.018 |
| mAP@50    | 0.933  | 0.926  | -0.007 |
| mAP@50-95 | 0.510  | 0.504  | -0.006 |

Crop-based training did not improve thermal detection, at roughly five times
the training cost of MT-005.

### MT-005 vs MT-007 Per-Person Comparison

All 2,611 test-set persons were compared between the two models:

| Category               | Persons | Share |
| ---------------------- | ------: | ----- |
| Matched by both models |   2,327 | 89.1% |
| Missed by both models  |     131 | 5.0%  |
| Matched by MT-005 only |      98 | 3.8%  |
| Matched by MT-007 only |      55 | 2.1%  |

Two findings came out of this:

1. The two models fail on different people. 153 persons (5.9%) are found by
   exactly one model, which makes ensembling a promising direction.
2. The 131 persons missed by both models have essentially the same box sizes as
   the persons detected successfully, so size alone does not explain the
   remaining failures.

Together with MT-006, this indicates that target scale is no longer the
bottleneck for thermal detection, and that further scale-oriented experiments
are unlikely to help.

### Unified Benchmark

All reference models were re-evaluated under identical settings using
`scripts/benchmark_models.py`:

| Experiment | Modality | Split | imgsz | P     | R     | mAP@50 | mAP@50-95 |
| ---------- | -------- | ----- | ----: | ----- | ----- | ------ | --------- |
| MT-004     | RGB      | val   |  1280 | 0.744 | 0.570 | 0.654  | 0.300     |
| MT-005     | Thermal  | test  |   640 | 0.897 | 0.891 | 0.933  | 0.510     |
| MT-006     | Thermal  | test  |   960 | 0.903 | 0.880 | 0.926  | 0.526     |
| MT-007     | Thermal  | test  |   640 | 0.895 | 0.873 | 0.926  | 0.504     |

The RGB and thermal rows are not comparable to each other: they use different
datasets with different imagery, annotation policies and object densities. The
valid comparisons are within each modality.

Selected models:

| Role                 | Model  |
| -------------------- | ------ |
| RGB reference        | MT-004 |
| Thermal reference    | MT-005 |
| Thermal localization | MT-006 |

---

## Current Work

Single-modality accuracy tuning has reached a point of diminishing returns for
the thermal detector. Two independent scale experiments (MT-006, MT-007) both
failed to improve on MT-005, and the remaining failures are not explained by
target size.

Work therefore moves to system integration:

1. Deployment latency benchmark, to replace the validator throughput figures
   with numbers that are usable for companion-computer selection.
2. Temporal tracking and movement detection on video.
3. A real-time inference pipeline.
4. Model export for embedded inference.

---

## Next Planned Steps

1. Measure real single-frame inference latency for MT-004, MT-005 and MT-006
   with warm-up, separate from dataloader overhead.
2. Add multi-object tracking so that a detected person keeps an identity across
   frames.
3. Derive movement detection from track histories, with UAV ego-motion taken
   into account.
4. Build the real-time inference pipeline that runs detection + tracking on a
   video or camera stream and emits structured detection output.
5. Export the selected models to ONNX and verify numerical agreement with the
   PyTorch models.
6. Investigate MT-005 + MT-007 ensembling to recover the 55 persons that only
   MT-007 finds.
7. Select the companion computer using the measured latency and export results.
8. Benchmark the exported model on the selected embedded hardware.
9. Integrate the human-detection payload with the UAV system.
