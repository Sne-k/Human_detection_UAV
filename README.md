# Human Detection System for Fixed-Wing eVTOL UAV

## Project

**Development of a Fixed Wing eVTOL UAV for Emergency Logistics**

This repository contains the development work for the human-detection subsystem
of the UAV project.

The detection subsystem supports identification of people from an aerial
platform during emergency and logistics operations. Development has proceeded
in stages, beginning with RGB aerial imagery, extending to thermal/infrared
sensing, and now covering tracking, movement detection and a real-time
pipeline.

---

## Current Pipeline

```text
Aerial Image / Video / Camera
        |
        v
Ego-motion estimation  ----+
        |                  |
        v                  |
YOLO26n Human Detector     |
        |                  |
        v                  |
Confidence Filtering       |
        |                  |
        v                  |
ByteTrack Tracking         |
        |                  |
        v                  v
World-frame stabilisation
        |
        v
Movement Classification
        |
        v
Annotated video + JSONL detection stream
        |
        v
Future: UAV payload / ground station integration
```

---

## Reference Models

| Role                 | Model  | Modality | Input | Dataset         |
| -------------------- | ------ | -------- | ----: | --------------- |
| RGB reference        | MT-004 | RGB      |  1280 | VisDrone Person |
| Thermal reference    | MT-005 | Thermal  |   640 | HIT-UAV Person  |
| Thermal localization | MT-006 | Thermal  |   960 | HIT-UAV Person  |

RGB and thermal metrics are **not** comparable to each other: the two models
are trained and evaluated on unrelated datasets. See
[`docs/training_log.md`](docs/training_log.md) section 17.

---

## Documentation

| Document                                               | Contents                                                  |
| ------------------------------------------------------ | --------------------------------------------------------- |
| [`docs/project_progress.md`](docs/project_progress.md) | Stage-by-stage progress and results summary                |
| [`docs/training_log.md`](docs/training_log.md)         | All training experiments, error analyses and the benchmark |
| [`docs/realtime_pipeline.md`](docs/realtime_pipeline.md) | Runtime pipeline, movement detection, measured latency   |
| [`docs/system_architecture.md`](docs/system_architecture.md) | Architecture and design decisions                    |
| [`docs/thermal_baseline.md`](docs/thermal_baseline.md) | Frozen MT-005 reference for all future experiments |
| [`docs/deployment_target.md`](docs/deployment_target.md) | Project context, target hardware, and what runs on it |
| [`docs/literature_comparison.md`](docs/literature_comparison.md) | Positioning against the reference SAR-UAV systems |
| [`docs/payload_icd.md`](docs/payload_icd.md) | Interface control document for the payload |
| [`docs/hardware_selection.md`](docs/hardware_selection.md) | Hardware trade study with measured justification |

---

## Scripts

Datasets, weights and training outputs are excluded from Git (see
`.gitignore`); the scripts regenerate them.

### Dataset preparation

| Script                            | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| `convert_visdrone.py`             | VisDrone DET to single-class YOLO format      |
| `convert_hit_uav.py`              | HIT-UAV COCO to single-class YOLO format      |
| `create_hit_uav_crops.py`         | 256 px crop-augmented thermal training set    |
| `visualize_dataset.py`            | Verify RGB annotations                        |
| `visualize_hit_uav.py`            | Verify thermal annotations                    |
| `visualize_hit_uav_crops.py`      | Verify generated crops                        |

### Training and evaluation

| Script                            | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| `train_pilot.py`                  | Pilot training run                            |
| `benchmark_models.py`             | Uniform accuracy benchmark across all models  |
| `benchmark_latency.py`            | Deployment latency, with launch-bound check   |
| `benchmark_edge_cpu.py`           | CPU-only latency, Raspberry Pi proxy          |
| `coverage_requirements.py`        | Derive required FPS from the mission          |
| `sensor_resolution_study.py`      | Detection vs thermal sensor resolution        |
| `mine_hard_negatives.py`          | Mine false positives as training negatives    |
| `operating_point.py`              | Pick the confidence threshold from mission cost |
| `export_models.py`                | ONNX export with verification                 |
| `train_mt008.py`                  | MT-008 mosaic ablation                        |
| `evaluate_experiment.py`          | Full baseline comparison for one experiment   |

### Error analysis

| Script                            | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| `analyze_small_objects.py`        | Size-stratified recall                        |
| `analyze_mt005_test_errors.py`    | Thermal test error analysis                   |
| `analyze_mt006_test_errors.py`    | MT-006 equivalent                             |
| `analyze_mt007_test_errors.py`    | MT-007 equivalent                             |
| `analyze_mt005_remaining_misses.py` | Categorise remaining misses                 |
| `compare_mt005_confidence.py`     | Confidence-threshold diagnostic               |
| `compare_mt005_vs_mt007.py`       | Per-person model comparison                   |
| `compare_standard_vs_tiled.py`    | Tiled-inference comparison                    |
| `test_tiled_inference.py`         | Tiled inference implementation                |
| `visualize_mt005_vs_mt007_errors.py` | Contact sheets of disagreements            |
| `compare_mt005_vs_mt007_conf010.py` | Per-person comparison at any threshold      |
| `analyze_common_failures.py`      | Characterise shared failures                  |
| `verify_crowding_hypothesis.py`   | Diagnose the mechanism of each miss           |
| `ensemble_thermal.py`             | File-level fusion study                       |
| `ensemble_detect.py`              | Three-model ensemble inference pipeline       |

### Runtime

| Script                            | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| `payload.py`                      | **Unified runtime: ensemble + tracking + movement + geolocation** |
| `realtime_detect.py`              | Single-model pipeline (payload.py reuses its components) |
| `make_test_sequence.py`           | Synthesise a validation sequence with truth   |
| `geolocate.py`                    | Pixel detections to ground coordinates        |

---

## Usage

Run the payload on a video. Single model:

```bash
python scripts/payload.py --source flight.mp4 --model MT-005 --jsonl detections.jsonl
```

The best-measured configuration, with geolocated output:

```bash
python scripts/payload.py --source flight.mp4 --ensemble MT-005 MT-006 --telemetry state.jsonl --jsonl detections.jsonl
```

Benchmark accuracy across all reference models (one model per invocation):

```bash
python scripts/benchmark_models.py --models MT-005
```

Measure deployment latency:

```bash
python scripts/benchmark_latency.py --runs 200
```

Scripts resolve the project root from their own location. When running from a
Git worktree while datasets and training runs live in the main checkout, set
`HDU_ROOT` to the main checkout path.
