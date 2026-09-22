# Human Detection Payload - Fixed-Wing eVTOL UAV

## Project

**Development of a Fixed Wing eVTOL UAV for Emergency Logistics**

This repository holds the human-detection payload for that aircraft: the
subsystem that finds people from the air during search-and-rescue and
emergency-logistics operations, converts each detection into a ground
coordinate, and reports it to the ground station.

The payload is **not** part of the flight-control loop. It consumes aircraft
state over MAVLink and emits detections. Its failure must not affect
controllability.

---

## Status

| Area | State |
|------|-------|
| Detector | 10 experiments complete, MT-005 frozen as baseline |
| Runtime | `scripts/payload.py` - detection, tracking, movement, geolocation |
| Export | ONNX, verified bit-exact at native input shape |
| Benchmarks | Accuracy, latency, CPU, memory, sensor resolution |
| Documentation | 9 documents, including payload ICD and hardware matrix |
| Hardware | **Nothing procured.** All deployment figures are extrapolated |

---

## Headline Results

Held-out HIT-UAV test split: **579 images, 2,611 person instances**, never
used for training or model selection.

| Configuration | Recall | Precision | Missed | Unmatched | Blind | Cost |
|---------------|--------|-----------|-------:|----------:|------:|------|
| MT-005, NMS (was baseline) | 0.9288 | 0.7917 | 186 | 638 | 8 | 1x |
| **MT-005, WBF** | **0.9291** | **0.8320** | **185** | **490** | **8** | **1x** |
| MT-005 + MT-006, WBF | 0.9406 | 0.7992 | 155 | 617 | 5 | 2x |

**The deployed configuration fuses overlapping boxes instead of suppressing
them, and it is free.** NMS keeps only the highest-confidence box in a
cluster, and the highest-confidence box is not necessarily the best-localised
one; weighted box fusion averages the cluster instead. That is better on every
axis than the old baseline - **148 fewer false positives, 23% of them, and
+4.0 points of precision** - with the network, the exported graph and the
Raspberry Pi 5 latency all unchanged. Verified to reproduce exactly in
`scripts/payload.py`.

The two-model ensemble still finds 31 more people and remains the
accuracy-optimal configuration, but it costs **2x inference**, which the
target board has no headroom for. Single-model fusion sheds seven times more
false positives than the ensemble, for nothing. See
[`docs/training_log.md`](docs/training_log.md) section 36.

### MT-011: aerial pretraining, and a result that was nearly missed

Starting from VisDrone-pretrained weights instead of COCO costs **nothing** at
inference - same architecture, same graph, same latency. On the conventional
metric it looks like another failure: mAP@50 0.9301 against 0.9330, six fewer
people found at confidence 0.25.

What it actually learned was **calibration**, which a fixed threshold hides by
construction:

| Conf | MT-005 found | unmatched | MT-011 found | unmatched |
|-----:|-------------:|----------:|-------------:|----------:|
| 0.25 | 2,426 | 490 | 2,420 | **395** |
| 0.10 | 2,470 | 918 | 2,469 | **716** |
| **0.05** | 2,485 | 1,338 | **2,492** | **1,005** |

Appearance does not transfer between RGB and thermal, but *"this shape, at
this scale, from this altitude, is not a person"* does. Having seen aerial
vehicles, roads and clutter, it stops firing on their thermal analogues - so
it can be run far more sensitively, which is what a search payload wants.

**At each model's own mission-optimal threshold, MT-011 finds more people at
9.4-14.7% lower cost**, and both seeds win at every missed-person cost ratio
from 1:1 to 100:1. Adopting it means also lowering the confidence threshold;
the two are one decision. See [`docs/training_log.md`](docs/training_log.md)
sections 40 and 45.

The range is not hedging. A seed repeat (MT-011b, identical but for the seed)
measured the noise floor at **0.0051 mAP@50** - larger than most of this
project's headline deltas. The MT-011 *direction* survives, because it is
consistent across 42 paired comparisons; its *magnitude* is uncertain.

### Against the published HIT-UAV baselines

Same dataset, same splits - the only externally comparable numbers available:

| Model | Person AP@0.50 |
|-------|---------------:|
| YOLOv4 (dataset paper) | 89.88% |
| SSD-512 | 85.6% |
| Faster-RCNN | 75.5% |
| **MT-005 (this project)** | **93.3%** |

With the caveat that this project trains person-only while the paper trains
four classes. See [`docs/related_work.md`](docs/related_work.md).

### The evaluation criterion was stricter than the mission needs

| Criterion | Box error rejected | People found | Unmatched | Recall |
|-----------|-------------------:|-------------:|----------:|--------|
| IoU 0.50 (CV convention) | 0.58 m | 2,426 | 490 | 0.9291 |
| **IoU 0.25 (mission-derived)** | **1.05 m** | **2,501** | **415** | **0.9579** |

Each fix is reported with a **+/- 3.8 m** position uncertainty, which is 3.6x
larger than the box error IoU 0.25 still accepts - so a detection rejected for
landing between 0.25 and 0.50 would have sent the rescue team to the same
place. **75 more people, no model change.** mAP@50 is still reported as the
comparable metric; see [`docs/training_log.md`](docs/training_log.md)
section 32 for how to state both honestly.

Under the old NMS post-processing this gain was 86 people rather than 75. It
shrank because fusion and the loosened criterion recover *the same* near-miss
boxes, so fixing them in post-processing leaves fewer for the criterion to
forgive - which is what should happen if both explanations are right.

### It works better in the dark

| Condition | Images | Recall | Precision |
|-----------|-------:|--------|-----------|
| **Night** | 396 | **0.9432** | **0.8631** |
| Day | 183 | 0.8768 | 0.7267 |

Measured in the deployed fusion configuration. Fusion adds about 4 points of
precision in *both* conditions and leaves the conclusion unchanged.

Thermal sensing does not use visible light, so darkness is the detector's
*best* condition. Sun-heated roads and rooftops reach body temperature in
daylight and the contrast collapses. **The hard case is noon, not midnight.**

---

## Pipeline

```text
   Thermal camera ----+
                      |
   RGB camera --------+---> [ Companion Computer ]
                      |
   MAVLink state -----+
                              |
        +---------------------+---------------------+
        |                                           |
        v                                           v
  Ego-motion estimate                     Detector (1 or N models)
  LK flow + affine                        weighted box fusion
        |                                           |
        |                                           v
        |                                     ByteTrack
        |                                           |
        +--------------> World-frame <--------------+
                         stabilisation
                              |
                              v
                     Movement classification
                     moving / stationary / edge / unknown
                              |
                              v
                        Geolocation
                     lat, lon, error estimate
                              |
                              v
              JSONL detection stream --> Ground station
```

Run it:

```bash
python scripts/payload.py --source flight.mp4 --ensemble MT-005 MT-006 --telemetry state.jsonl --jsonl detections.jsonl
```

---

## Experiments

| ID | Modality | Change under test | Outcome |
|----|----------|-------------------|---------|
| MT-001 | RGB | 3-epoch pilot | Pipeline verified |
| MT-002 | RGB | 640 px baseline | mAP@50 ~0.50 |
| MT-003 | RGB | 960 px | Improved |
| MT-004 | RGB | 1280 px | mAP@50 0.654, RGB reference |
| MT-005 | Thermal | 640 px baseline | **Frozen baseline** |
| MT-006 | Thermal | 960 px | Recall down, localisation up |
| MT-007 | Thermal | 256 px crop augmentation | Rejected |
| MT-008 | Thermal | mosaic = 0 | Rejected, merged boxes worse |
| MT-009 | Thermal | hard-negative mining | Rejected, no measurable gain |
| MT-010 | Thermal | box-loss weight 15.0 | Rejected, worse on every axis |
| MT-011 | Thermal | VisDrone -> HIT-UAV transfer | **Accepted** - better calibrated |
| MT-013 | Thermal | NWD localisation loss | Indistinguishable from baseline |
| MT-011b | Thermal | MT-011 seed repeat | **Noise floor: 0.0051 mAP@50** |
| MT-012 | Thermal | P2 small-object head | Rejected on compute **and** accuracy |

**Eleven directions tested; one improved the baseline.** MT-011 is the only
one, and it was nearly missed - see below.

Two of those eleven were aimed squarely at the dominant measured failure,
near-miss localisation: a P2 detection head (MT-012) and an NWD localisation
loss (MT-013). **Both produced a near-miss gap of 76 against the baseline's
75.** MT-011, aimed at nothing of the kind, produced 64. Two independent
approaches converging on the same non-result says the near-miss population is
limited by neither grid resolution nor loss geometry - most likely by the
IoU-based anchor assigner that runs upstream of both. Model capacity was ruled out on compute,
not accuracy: YOLO26s runs at an estimated 1.8-2.9 FPS on the target against a
6.7 FPS requirement.

The first seven all attacked target *scale* and moved no failure mechanism at
all. MT-010 is the first that moved its target - merged boxes fell from 57 to
49 - and still lost, because total loss is a budget: doubling the box term
halves the relative weight of classification, and blind images more than
doubled from 8 to 18. That is why MT-013 changes the *shape* of the
localisation loss rather than its weight.

Full detail, including every negative result, in
[`docs/training_log.md`](docs/training_log.md).

---

## What the Failures Actually Are

Of MT-005's unmatched predictions, measured on the training split:

| Category | Share |
|----------|-------|
| **Near-miss box on a real person** (IoU 0.25-0.50) | **50.7%** |
| Touching a real person | 5.1% |
| Genuine background firing | 44.1% |

Over half the "false positives" found a person and localised them poorly. The
evaluation counts each twice - once as a missed person, once as a false
positive - so **the precision problem and the recall problem are the same
problem**. That is why fusing two near-misses improves both columns at once.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/project_progress.md`](docs/project_progress.md) | Stage-by-stage progress |
| [`docs/training_log.md`](docs/training_log.md) | Every experiment, error analysis and benchmark |
| [`docs/thermal_baseline.md`](docs/thermal_baseline.md) | Frozen MT-005 reference and all-conditions results |
| [`docs/realtime_pipeline.md`](docs/realtime_pipeline.md) | Runtime pipeline, movement detection, latency |
| [`docs/system_architecture.md`](docs/system_architecture.md) | Architecture and design decisions |
| [`docs/deployment_target.md`](docs/deployment_target.md) | Target hardware, frame-rate requirement, what runs |
| [`docs/hardware_selection.md`](docs/hardware_selection.md) | Hardware trade study, every requirement measured |
| [`docs/payload_icd.md`](docs/payload_icd.md) | Interface control document |
| [`docs/literature_comparison.md`](docs/literature_comparison.md) | Positioning against Rizk [3] and Lygouras [4] |
| [`docs/related_work.md`](docs/related_work.md) | HIT-UAV benchmarks, comparable systems, techniques to adopt |
| [`docs/references.md`](docs/references.md) | Every source, tagged by provenance: local papers, online papers, code, datasets, tools |

---

## Scripts

Datasets, weights and training outputs are excluded from Git (see
`.gitignore`); the scripts regenerate them.

Scripts resolve the project root from their own location. When running from a
Git worktree while datasets live in the main checkout, set `HDU_ROOT`.

### Runtime

| Script | Purpose |
|--------|---------|
| `payload.py` | **Unified runtime**: detection, fusion, tracking, movement, geolocation |
| `realtime_detect.py` | Single-model pipeline; `payload.py` reuses its components |
| `ensemble_detect.py` | Ensemble inference with per-stage timing |
| `single_model_wbf.py` | Fusion vs suppression on one model, the free precision gain |
| `geolocate.py` | Pixel detections to ground coordinates |

### Dataset preparation

| Script | Purpose |
|--------|---------|
| `convert_visdrone.py` | VisDrone DET to single-class YOLO |
| `convert_hit_uav.py` | HIT-UAV COCO to single-class YOLO |
| `create_hit_uav_crops.py` | 256 px crop-augmented training set (MT-007) |
| `mine_hard_negatives.py` | Mine false positives as training negatives (MT-009) |
| `make_test_sequence.py` | Synthesise a validation sequence with known truth |
| `visualize_dataset.py` | Verify RGB annotations |
| `visualize_hit_uav.py` | Verify thermal annotations |
| `visualize_hit_uav_crops.py` | Verify generated crops |

### Training and benchmarking

| Script | Purpose |
|--------|---------|
| `train_pilot.py` | Pilot training run |
| `train_mt008.py` | Controlled experiments (mosaic, dataset, loss weights, starting weights, NWD) |
| `nwd_loss.py` | Normalized Wasserstein Distance localisation loss, with a self-test |
| `benchmark_models.py` | Uniform accuracy benchmark across models |
| `benchmark_latency.py` | Deployment latency with launch-bound diagnostic |
| `benchmark_edge_cpu.py` | CPU-only latency, Raspberry Pi proxy |
| `architecture_budget.py` | Gate: does a candidate architecture fit the target, before training it |
| `quantize_int8.py` | INT8 static quantisation, with a QDQ-fusion diagnostic and mandatory re-evaluation |
| `run_experiment_queue.py` | Chain training, test metrics, predictions and error analysis unattended |
| `export_models.py` | ONNX export with verification against PyTorch |
| `coverage_requirements.py` | Derive required frame rate from the mission |
| `sensor_resolution_study.py` | Detection vs thermal sensor resolution |
| `operating_point.py` | Pick the confidence threshold from mission cost |
| `operational_criterion.py` | Derive the IoU criterion from the geolocation error budget |

### Error analysis

| Script | Purpose |
|--------|---------|
| `evaluate_experiment.py` | Full baseline comparison for one experiment |
| `analyze_small_objects.py` | Size-stratified recall |
| `analyze_mt005_test_errors.py` | Thermal test error analysis |
| `analyze_mt006_test_errors.py` | MT-006 equivalent |
| `analyze_mt007_test_errors.py` | MT-007 equivalent |
| `analyze_mt005_test_errors_conf010.py` | Thermal test error analysis at conf 0.10 |
| `analyze_mt005_remaining_misses.py` | Categorise remaining misses |
| `analyze_common_failures.py` | Characterise shared failures |
| `verify_crowding_hypothesis.py` | Diagnose the mechanism of each miss |
| `compare_mt005_confidence.py` | Confidence-threshold diagnostic |
| `compare_mt005_vs_mt007.py` | Per-person model comparison |
| `compare_mt005_vs_mt007_conf010.py` | Per-person comparison at any threshold |
| `compare_standard_vs_tiled.py` | Tiled-inference comparison |
| `ensemble_thermal.py` | File-level fusion study |
| `test_tiled_inference.py` | Tiled inference implementation |
| `visualize_mt005_vs_mt007_errors.py` | Contact sheets of disagreements |

---

## Key Decisions

**Detect-and-report is the primary mission.** In search and rescue the
highest-priority casualties are unconscious or trapped, and therefore
stationary - a movement classifier labels them identically to a warm rock.
Movement classification is a loiter-phase capability, not a search-phase
constraint. See [`docs/training_log.md`](docs/training_log.md) section 28.

**Thermal is the detector; RGB is for delivery confirmation.** Thermal is the
light-independent modality and the one that fits the compute budget. MT-004 at
1280 px cannot run on a CPU-only board.

**Thermal sensor resolution is the binding purchase decision.** A 160 x 120
module loses one person in three. **384 x 288 is the floor.**

**Eleven runs were compared before anyone measured the noise floor.** A seed
repeat put it at 0.0051 mAP@50, which is larger than most of the deltas this
project had been reading as signal. Single-number verdicts below that are
withdrawn; swept comparisons across thresholds and cost ratios survive, and so
do the post-processing results, which involve no training and therefore no
variance. See [`docs/training_log.md`](docs/training_log.md) section 45.

**Three results were hidden by defaults nobody chose.** The IoU 0.50 matching
criterion (86 people), NMS post-processing (148 false positives) and the
confidence 0.25 threshold (MT-011 entirely). A comparison protocol is itself a
choice, and one fixed before the space of possible results is understood will
eventually hide one.

**Architecture changes are gated on compute before they are trained.** The
literature's most-supported technique for small objects - a P2 detection head -
was measured at 4.0-6.6 FPS on the target against a 6.7 FPS requirement and
**cancelled before training**. GFLOPs understates it: 1.3-1.5x latency for
1.30x arithmetic, because a stride-4 head is bandwidth-bound and bandwidth is
where a Pi 5 is weakest. Training-time changes - losses, augmentation,
datasets - are free at inference and need no gate. See
[`docs/training_log.md`](docs/training_log.md) section 33.

**The frame-rate requirement is derived, not assumed.** 6.7 FPS at 60 m,
falling to 4.0 at 100 m - set by ground coverage, not video smoothness. And it
applies to the **payload**, not the detector: tracking, geolocation, fusion and
ego-motion add 22% on top of inference, measured end to end. See
[`docs/training_log.md`](docs/training_log.md) section 39.

---

## Limitations

- **No hardware.** Every deployment figure is extrapolated from x86.
- **No real flight video.** Tracking and movement are validated on a
  synthetic sequence with one moving target.
- **No smoke or fog data.** Obscurant penetration is inferred from LWIR
  physics, not measured.
- **RGB and thermal metrics are not comparable** - unrelated datasets.
- **Daylight is the weak case** at 0.879 recall.
