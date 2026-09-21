# Results

This directory holds documentation and references for experiment results.

Large training outputs, datasets and generated analysis artefacts are excluded
from Git via `.gitignore`. They are regenerated locally by the scripts in
`scripts/`.

## Experiments

| ID     | Modality | Input | Dataset                       | Status    |
| ------ | -------- | ----: | ----------------------------- | --------- |
| MT-001 | RGB      |   640 | VisDrone Person               | Completed |
| MT-002 | RGB      |   640 | VisDrone Person               | Completed |
| MT-003 | RGB      |   960 | VisDrone Person               | Completed |
| MT-004 | RGB      |  1280 | VisDrone Person               | Completed |
| MT-005 | Thermal  |   640 | HIT-UAV Person                | Completed |
| MT-006 | Thermal  |   960 | HIT-UAV Person                | Completed |
| MT-007 | Thermal  |   640 | HIT-UAV Person + 256 px crops | Completed |

## Generated output locations

| Path                            | Produced by                  |
| ------------------------------- | ---------------------------- |
| `results/benchmark/`            | `benchmark_models.py`, `benchmark_latency.py` |
| `results/error_analysis/`       | the `analyze_*` and `compare_*` scripts |
| `results/test_sequence/`        | `make_test_sequence.py`, `realtime_detect.py` |
| `results/dataset_check/`        | `visualize_dataset.py`       |
| `results/hit_uav_dataset_check/`| `visualize_hit_uav.py`       |
| `results/mt007_crop_check/`     | `visualize_hit_uav_crops.py` |

Training runs are written by Ultralytics under `runs/detect/results/training/`.

## Documentation

Detailed experiment information is maintained in:

- `docs/training_log.md` - experiments, error analyses, unified benchmark
- `docs/realtime_pipeline.md` - runtime pipeline, movement detection, latency
- `docs/project_progress.md` - stage-by-stage progress
- `docs/system_architecture.md` - architecture and design decisions
