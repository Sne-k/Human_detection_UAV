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
|    24 | MT-006 thermal training, 50 epochs @ 960 px               | In progress |
|    25 | MT-006 evaluation and comparison                          | Pending     |
|    26 | RGB/thermal model benchmarking                            | Pending     |
|    27 | RGB/thermal fusion investigation                          | Pending     |
|    28 | Movement detection/tracking                               | Pending     |
|    29 | Real-time inference pipeline                              | Pending     |
|    30 | Embedded companion-computer selection                     | Pending     |
|    31 | Embedded model deployment/benchmarking                    | Pending     |
|    32 | UAV payload/system integration                            | Pending     |

---

## Completed Work

### Development Environment

A Windows-based development environment has been established for the human-detection subsystem.

The NVIDIA RTX 4050 Laptop GPU is being used for accelerated model training and inference through CUDA-enabled PyTorch.

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

Dataset size:

6,471 training images
548 validation images
106,396 training person annotations
13,969 validation person annotations

The converted annotations were visually verified.

RGB Model Development

Three main RGB training stages were completed:

MT-002: 640 x 640 baseline
MT-003: 960 x 960
MT-004: 1280 x 1280

MT-004 achieved:

Metric	Result
Precision	0.738
Recall	0.563
mAP@50	0.653
mAP@50-95	0.297

Increasing resolution improved the official validation metrics, but small aerial persons remained difficult to detect.

RGB Error Analysis

The RGB detector was investigated using:

small-object analysis
complete-failure analysis
confidence-threshold testing
tiled inference

The tested tiled configuration did not improve the overall custom validation comparison, so it was not selected as the next optimization direction.

Thermal Dataset

The HIT-UAV Infrared Thermal Dataset was integrated as a separate thermal detection dataset.

Only the Person category was retained:

0: person

Converted dataset:

Split	Images	Person annotations
Train	2,029	8,533
Validation	290	1,168
Test	579	2,611

The converted thermal annotations were visually inspected and verified.

MT-005 Thermal Baseline

The first thermal baseline was trained at 640 x 640.

Validation:

Metric	Result
Precision	0.890
Recall	0.875
mAP@50	0.919
mAP@50-95	0.500

Held-out test:

Metric	Result
Precision	0.898
Recall	0.892
mAP@50	0.933
mAP@50-95	0.510

The held-out test set contains 579 images and 2,611 person instances.

Thermal Error Analysis

At confidence 0.25, the MT-005 test predictions produced:

3,063 predicted boxes
2,425 matched persons at IoU >= 0.50
186 missed ground-truth persons
638 unmatched prediction boxes

A confidence 0.10 diagnostic recovered 43 additional persons but introduced 662 additional unmatched prediction boxes.

Further analysis showed that many remaining misses had an overlapping prediction but insufficient IoU, indicating a localization limitation for small thermal targets.

Current Work
MT-006 - Thermal Small-Object Resolution Experiment

MT-006 is currently running.

Configuration:

Model:       YOLO26n
Dataset:     HIT-UAV Person
Input size:  960 x 960
Epochs:      50
Batch size:  4
GPU:         RTX 4050 Laptop GPU
AMP:         Enabled

The experiment is designed as a controlled comparison against MT-005.

The main question is whether increased input resolution improves detection/localization of the very small thermal person targets present in HIT-UAV.

Final metrics will be recorded after training completes.

Next Planned Steps

After MT-006 completes:

Evaluate MT-006 on the validation set.
Evaluate MT-006 on the held-out HIT-UAV test set.
Compare MT-005 and MT-006 using the same evaluation methodology.
Perform thermal error analysis.
Determine whether the resolution increase provides a useful improvement.
Establish a controlled RGB-vs-thermal benchmark with appropriate dataset-specific interpretation.
Investigate RGB/thermal fusion.
Add movement detection/tracking.
Develop the real-time inference pipeline.
Select the companion computer for eventual UAV integration.
Benchmark the selected model on the embedded hardware.
Integrate the human-detection payload with the UAV system.
```
