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
|    29 | Deployment latency benchmark                              | Completed   |
|    30 | Movement detection/tracking                               | Completed   |
|    31 | Real-time inference pipeline                              | Completed   |
|    32 | Model export for embedded inference (ONNX)                | Completed   |
|    33 | MT-007 confidence-threshold diagnostic                    | Completed   |
|    34 | Shared-failure characterisation                           | Completed   |
|    35 | Thermal model ensembling investigation                    | Completed   |
|    36 | Failure-mechanism verification                            | Completed   |
|    37 | MT-005 frozen as thermal baseline                         | Completed   |
|    38 | Three-model WBF ensemble inference pipeline               | Completed   |
|    39 | MT-008 mosaic ablation (crowd separation)                 | Completed   |
|    40 | Model-capacity feasibility (YOLO26s latency)              | Completed   |
|    41 | Two-model ensemble selection                              | Completed   |
|    42 | Concept of operations decision (detect-and-report)        | Completed   |
|    43 | False-positive composition analysis                       | Completed   |
|    44 | MT-009 hard-negative mining                               | Completed   |
|    45 | Payload interface control document                        | Completed   |
|    46 | Payload hardware selection matrix                         | Completed   |
|    47 | Unified payload runtime (ensemble + tracking + geolocation) | Completed |
|    48 | MT-010 box-loss weight experiment                         | In progress |
|    49 | All-conditions day/night performance analysis             | Completed   |
|    50 | Thermal camera procurement                                | Pending     |
|    51 | Companion-computer procurement and on-hardware benchmark  | Pending     |
|    52 | Embedded model deployment/benchmarking                    | Pending     |
|    53 | UAV payload/system integration                            | Pending     |

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

### Real-Time Pipeline

A runtime pipeline was implemented that runs detection, ByteTrack identity
tracking and movement classification over a video, image sequence or camera
stream, and emits an annotated video plus a JSONL detection stream.

Movement is measured in a stabilised world frame. The pipeline estimates the
camera motion caused by the aircraft each frame using optical flow and a
partial affine fit, accumulates it, and converts track positions into world
coordinates. This matters because on a moving aircraft the apparent motion of a
stationary person is usually larger than the real motion of a walking person.

Validated on a synthesised sequence with known ground truth (a viewport panning
across a HIT-UAV thermal image containing 31 stationary persons, plus one
composited person that genuinely moves):

| Configuration               | Tracks | Flagged moving | False positives |
| --------------------------- | -----: | -------------: | --------------: |
| Ego-motion compensation on  |     43 |              1 |               0 |
| Ego-motion compensation off |     43 |              7 |               6 |

With compensation the single flagged track matched the composited person's
known trajectory across all 120 frames.

### Model Export

Both reference models were exported to ONNX and verified against the PyTorch
models on 150 real dataset images, rather than on random tensors.

Verification found that the export input shape is the critical parameter. An
ONNX graph has one fixed input shape, while the PyTorch predict path
letterboxes rectangularly. For MT-005 on a 640 x 512 thermal sensor:

| Export shape       | Boxes      | Mean IoU | Lost detections | Verdict |
| ------------------ | ---------- | -------- | --------------: | ------- |
| 640 x 640 (square) | 792 -> 778 | 0.972    |              27 | FAIL    |
| 512 x 640 (native) | 792 -> 792 | 1.000    |               0 | PASS    |

The square export silently loses 3.4% of detections. The graph itself is
faithful - raw pre-NMS outputs agree to 1.7e-06 on the confidence channel - so
this is a preprocessing mismatch, not an export defect.

MT-004 exports with some drift (mean IoU 0.955), which is expected: VisDrone
images have mixed aspect ratios, so no single fixed shape reproduces the
PyTorch letterbox for every image. A deployed RGB pipeline must letterbox to
the exported shape itself.

The exports are verified numerically but their latency has not yet been
measured, because the installed ONNX Runtime has no CUDA provider.

### Shared-Failure Diagnosis

MT-007 was re-run at confidence 0.10 and compared against MT-005 at the same
threshold, then the persons missed by both models were characterised in detail.

Lowering the threshold recovers 44 of the 131 shared failures (33.6%), leaving
87 that neither model finds at all. The cost is poor: about 15 extra false
positives per person recovered.

Characterising the remaining 87 produced the key result of this phase. The
problem is **not** small-object detection:

| Finding                   | Failures | Detected persons |
| ------------------------- | -------- | ---------------- |
| Classified small          | 86 of 87 | 2,394 of 2,397   |
| Median box width          | 12.0 px  | 12.0 px          |
| Neighbour within 10 px    | 32.8%    | 5.7%             |
| Median nearest neighbour  | ~13 px   | 29 px            |

Size does not separate the failures from the successes at all. Two causes do:

1. **Crowding-induced localisation failure (78%).** The model produces a box on
   the person, but people standing close together yield boxes that fall just
   below IoU 0.50 - 91% of them land between 0.25 and 0.50.
2. **Daylight thermal contrast (22%).** Recognition failures are 2.9x
   over-represented in daylight, where warm backgrounds suppress the heat
   signature entirely.

This explains the MT-006 and MT-007 results directly: both addressed target
scale, which was never the limiting factor.

### Failure-Mechanism Verification

The crowding explanation was circumstantial - nearest-neighbour distance is
only a proxy for visual crowding - so the mechanism was tested directly by
examining what each near-miss prediction actually did.

| Mechanism             | Count | Share of the 68 with a prediction |
| --------------------- | ----: | --------------------------------: |
| Merged, 2+ people     |    43 |                             63.2% |
| Undersized box        |    20 |                             29.4% |
| Ordinary displacement |     3 |                              4.4% |
| Claimed by neighbour  |     2 |                              2.9% |

The merged boxes average 2.03x the area of the person they should have covered,
and 41 of 43 touch exactly two annotated people. Among the 2,397 successful
detections only 4.46% involve a multi-person box, so the failures are about
**14x more likely** to be merged boxes. The mechanism is confirmed, not merely
correlated.

The undersized group turned out to be a separate mechanism rather than
displacement: those boxes are correctly centred but roughly half the annotated
area (width ratio 0.651, height ratio 0.715), because the detector locks onto
the bright heat signature while the annotation covers the whole body.

A global box-enlargement fix was tested and **rejected**. Matched detections
show no sizing bias at all (ratios 0.994 and 0.986), and a scale sweep gains
only 9 persons at 1.05 before degrading sharply - losing 400 by 1.30. The
undersized cases are per-instance failures, not a calibration bias.

### Thermal Model Ensembling

Because the three thermal models fail on different people, their predictions
were pooled and fused, then scored with the same person-level matching.

| Configuration            | Matched | Missed | Unmatched | Recall | Precision |
| ------------------------ | ------: | -----: | --------: | ------ | --------- |
| MT-005 @ 0.25 (baseline) |   2,425 |    186 |       638 | 0.9288 | 0.7917    |
| MT-005 @ 0.10            |   2,468 |    143 |     1,300 | 0.9452 | 0.6550    |
| WBF ensemble @ 0.25      |   2,476 |    135 |       722 | 0.9483 | 0.7742    |

Weighted box fusion beats plain NMS at every vote level, as predicted by the
failure analysis: averaging independent near-miss boxes recovers a
better-centred box than any single one.

Against the baseline, the ensemble recovers 51 persons for 84 extra false
positives (1.6 per person), where lowering the threshold recovers 43 for 662
(15.4 per person) - roughly a tenth of the cost.

This is a trade, not a free gain. The ensemble's precision under the custom
metric is **lower** than the MT-005 baseline's, 0.7742 against 0.7917, because
of those 84 extra unmatched boxes. It improves recall and the missed-person
count at some cost in false detections. For a rescue payload that is the right
direction, since a missed person is the costlier error, but it should be
reported as a trade.

The ensemble runs three models per frame, which the eager pipeline could not
absorb. The CUDA-graph measurement below resolves that: three graphed models
cost 9.4 ms per frame, **4.2x faster than a single model runs today**. The
ensemble is affordable.

### Latency Benchmark

A deployment latency benchmark replaced the validator throughput figures, which
include dataloader overhead and cannot be used for hardware sizing.

The measurement found that all four models are **launch-bound** at batch 1 on
the development GPU: the CPU takes as long to enqueue kernels as the entire
frame takes, so 1280 px costs the same as 640 px. Batch 8 removes the
bottleneck and recovers the expected compute scaling (18.7 / 9.9 / 4.4 ms per
image at 1280 / 960 / 640 px).

This was then confirmed directly with CUDA graphs, which submit the whole
network in one launch while running identical kernels:

| Experiment | Eager forward | CUDA graph | Speedup |
| ---------- | ------------: | ---------: | ------: |
| MT-005     |      39.07 ms |    2.43 ms |  16.08x |
| MT-006     |      39.98 ms |    4.54 ms |   8.81x |
| MT-004     |      37.25 ms |   11.37 ms |   3.28x |

Replay was verified bit-identical to eager execution on random tensors and on
real thermal images. Resolution scaling reappears in the graphed column
(2.43 / 4.54 / 11.37 ms, close to proportional to pixel count) where the eager
column was flat at 37-40 ms, confirming the eager figures were measuring host
dispatch rather than the models.

The consequence is that the useful deployment optimisation is graph export, not
a smaller model, and that companion-computer selection must be based on
post-export measurements on the target hardware.

Details in [`realtime_pipeline.md`](realtime_pipeline.md).

---

## Current Work

The detector is frozen. Thirteen experiments plus a seed repeat; MT-011b is
deployed, with weighted box fusion at 0.60. Every explanation for the residual
near-miss failures has been closed - two by experiment, one by measurement -
and the measured seed noise floor of 0.0051 mAP@50 means further single-run
accuracy comparisons cannot resolve anything.

Work has moved to integration. The binding constraint stopped being detection
accuracy some time ago; it is that nothing has flown, and that the interfaces
the payload is specified against do not yet exist in software.

---

## Task List

Ordered by value, not by phase. Each item says why it is worth doing and what
it depends on, so the ordering can be argued with rather than followed.

### A. Correctness gaps - do these first

| # | Task | Why |
|---|------|-----|
| A1 | **MAVLink ingestion (IF-4)** | **Nothing in the repository speaks MAVLink.** The ICD specifies it; the payload reads telemetry from a JSONL file. Testable today against ArduPilot SITL or a replayed `.tlog`, with no hardware. |
| A2 | **Timestamp-based frame association** | Telemetry is keyed by **frame index**. That works only because the synthetic sequence was generated with matching indices; real clocks are independent. The ICD's own figure - 1 degree of attitude error moves the fix 1.75 m at 100 m - is what a wrong association costs. Needs A1. |
| A3 | **Optical-flow health gate** | Over water, fog or snow - all plausible SAR conditions - flow has nothing to lock onto, the ego-motion transform silently becomes garbage, and movement classification reports confident labels built on nonsense. `ego_motion_failures` is already counted; nothing acts on it. Degrade to `unknown` instead. |
| A4 | **Search mode vs loiter mode** | Ego-motion costs 6.6 ms per frame and exists only to serve movement classification, which is also the sole reason the requirement is 6.7 FPS rather than 1.34. Section 28 already argues detect-and-report is the primary mission. Making that a runtime mode removes the cost and the requirement together, and may change the hardware decision. |

### B. Validation buildable without hardware

| # | Task | Why |
|---|------|-----|
| B1 | **Geolocation sensitivity sweep** | `geolocate.py` has hand-checkable spot cases but no systematic perturbation map. Sweep roll, pitch, yaw, altitude, GPS and timestamp error across 60/100/130 m against known truth. This is the software half of the ICD acceptance test, and it catches a coordinate-convention error before flight rather than after. |
| B2 | **Failure injection** | Camera disappears, MAVLink stops, disk fills, inference throws, frame is corrupt, JSONL write fails. The ICD requires that payload failure not affect flight control and that storage failure not stop detection. All reproducible on a laptop. |
| B3 | **Frame-drop robustness** | Movement classification depends on a 15-frame history with a 15 px threshold separating roughly 10 px of jitter from 49-69 px of real motion. Dropped frames change what that window spans. Untested. |
| B4 | **Tracker stress suite** | Current validation is one synthetic sequence with a single moving target - it tests false movement flags well and little else. Missing: several simultaneous movers, targets entering and leaving frame (the `edge` state), crossing tracks, scale change with altitude. Measure ID switches, fragmentation, track lifetime. |
| B5 | **JSONL schema validation** | The output contract is specified in the ICD and nothing checks conformance. Field presence, types, ranges, monotonic frames, finite values, `position_error_m >= 0`, no NaN. Cheap, and it becomes the integration acceptance check. |
| B6 | **Replay harness with evaluation** | `payload.py` already replays video plus telemetry to JSONL. The missing half is scoring, so a runtime change can be measured against a fixed mission rather than eyeballed. Subsumes the separate mission-recorder idea. |
| B7 | **`tests/` suite** | There is no test directory at all. Worth building as the container for B1-B6 rather than as a goal of its own. |

### C. Informs the hardware purchase

| # | Task | Why |
|---|------|-----|
| C1 | **Field-of-view study** | The sensor-resolution study fixed FOV at 50 degrees, but pixels-on-target depends on resolution **and** FOV **and** altitude. A narrow FOV on a cheap sensor may beat a wide FOV on an expensive one. Live procurement decision, and the geometry is exact - no simulation of appearance needed. |
| C2 | **Buy a Raspberry Pi 5 and measure** | Every frame-rate figure here is a 3-5x extrapolation from x86. The entire compute argument rests on a factor that was estimated, not measured, and INT8 in particular may behave differently on ARM. The board costs a fraction of the camera and is independent of which camera is chosen. |

### D. Demonstrator - after the above

| # | Task | Why |
|---|------|-----|
| D1 | **Ground-station consumer** | Map, target list, confidence, movement state, coordinate and uncertainty, fed from recorded JSONL. Useful and fully offline - but it demonstrates rather than de-risks, so it follows the correctness work. |
| D2 | **Frozen release package** | Weights, training config, seed, dataset version, test predictions, threshold sweep, mission-cost curve and exact WBF parameters as one reproducible bundle. The analysis is done; the packaging is not. |
| D3 | **Frame logging for offline labelling** | Save frames near the confidence threshold and where tracks broke - the highest-value frames to label. Turns every future flight into training data, with no onboard learning and no runtime cost. |
| D4 | **Simulated end-to-end mission** | The sum of A and B rather than separate work. Worth stating as the pre-hardware milestone. |

---

## Explicitly Not Doing

**More detector experiments without a specific hypothesis.** Thirteen have been
run. The seed noise floor is 0.0051 mAP@50, larger than most differences this
project had been reading as signal, so another single run cannot resolve
anything a swept comparison has not already settled.

**Asynchronous pipeline redesign.** Premature. The camera will not outrun
inference - the payload runs at roughly 7 FPS against a 6.7 FPS requirement,
or 1.34 in report-only mode. Threading buys nothing measurable, costs a class
of race conditions, and would destroy the deterministic replay that B6 depends
on. Revisit only if a measurement shows acquisition blocking inference.

**Synthetic sensor appearance modelling** - simulated blur, noise, contrast
reduction, compression. The geometric half (C1) is exact and worth doing. The
appearance half is guesswork until a real camera exists to validate it
against, and optimising against an invented degradation model risks tuning for
artefacts the real sensor does not produce.

**Simulated daylight degradation**, for the same reason. "At what thermal
contrast does the system stop being useful?" is the right question, and it
cannot be answered honestly by inventing the contrast reduction.

**Online or in-flight learning.** No label source exists in flight, the board
has no compute headroom, and a payload whose behaviour changes mid-mission
cannot be verified. Collect now, fine-tune offline later - which is the MT-011
recipe that already worked.
