# Project Progress

## Project

**Development of a Fixed Wing eVTOL UAV for Emergency Logistics**

## Human Detection Subsystem Progress

| Stage | Activity                                                  | Status      |
| ----- | --------------------------------------------------------- | ----------- |
| 1     | Project workspace and Python environment setup            | Completed   |
| 2     | NVIDIA CUDA environment setup                             | Completed   |
| 3     | PyTorch GPU verification                                  | Completed   |
| 4     | Ultralytics/YOLO environment setup                        | Completed   |
| 5     | VisDrone RGB dataset preparation                          | Completed   |
| 6     | VisDrone pedestrian + people class conversion to `person` | Completed   |
| 7     | Dataset annotation visualization                          | Completed   |
| 8     | MT-001 pilot training, 3 epochs @ 640 px                  | Completed   |
| 9     | MT-002 baseline training, 50 epochs @ 640 px              | Completed   |
| 10    | MT-002 result/error analysis                              | Completed   |
| 11    | MT-003 training, 50 epochs @ 960 px                       | In progress |
| 12    | MT-003 evaluation and comparison                          | Pending     |
| 13    | Small-object detection investigation                      | Pending     |
| 14    | HIT-UAV thermal dataset integration                       | Pending     |
| 15    | RGB/thermal model benchmarking                            | Pending     |
| 16    | Movement detection/tracking                               | Pending     |
| 17    | Real-time inference pipeline                              | Pending     |
| 18    | Embedded companion-computer selection                     | Pending     |
| 19    | Embedded model deployment/benchmarking                    | Pending     |
| 20    | UAV payload/system integration                            | Pending     |

---

## Completed Work

### Development Environment

A Windows-based development environment has been established for the human-detection subsystem.

The NVIDIA RTX 4050 Laptop GPU is being used for accelerated model training through CUDA-enabled PyTorch.

### RGB Dataset

The VisDrone DET dataset was prepared for single-class human detection.

The original `pedestrian` and `people` categories were combined into:

```text
0: person
```
