# System Architecture

## Purpose

The human-detection subsystem is being developed as an AI-based perception
payload for the fixed-wing eVTOL UAV designed for emergency logistics.

The subsystem detects people from aerial imagery, maintains their identity over
time, and reports whether each detected person is moving or stationary.

---

## Current Architecture

Solid boxes are implemented and validated. Dashed boxes are planned.

```text
        RGB camera                    Thermal camera
       (VisDrone-like)                 (HIT-UAV-like)
             |                               |
             +---------------+---------------+
                             |
                             v
                  +----------------------+
                  |   Frame acquisition  |
                  |  video / camera /    |
                  |  image sequence      |
                  +----------------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
   +----------------------+      +----------------------+
   |  Ego-motion estimate |      |    YOLO26n detector  |
   |  LK optical flow +   |      |  MT-004 RGB 1280 px  |
   |  partial affine fit  |      |  MT-005 IR   640 px  |
   +----------------------+      +----------------------+
              |                             |
              |                             v
              |                  +----------------------+
              |                  |  Confidence filter   |
              |                  |   operational 0.25   |
              |                  +----------------------+
              |                             |
              |                             v
              |                  +----------------------+
              |                  |   ByteTrack tracker  |
              |                  |  persistent track ID |
              |                  +----------------------+
              |                             |
              +--------------+--------------+
                             |
                             v
                  +----------------------+
                  | World-frame mapping  |
                  | stabilised track     |
                  | coordinates          |
                  +----------------------+
                             |
                             v
                  +----------------------+
                  | Movement classifier  |
                  | moving / stationary  |
                  | / edge / unknown     |
                  +----------------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
   +----------------------+      +----------------------+
   |  Annotated video     |      |  JSONL detection     |
   |  (operator view)     |      |  stream              |
   +----------------------+      +----------------------+
                                            |
                                            v
                             +- - - - - - - - - - - - - -+
                             |  Ground station /         |
                             |  payload controller       |
                             +- - - - - - - - - - - - - -+


   Planned deployment path
   -----------------------
   +- - - - - - - - -+   +- - - - - - - - -+   +- - - - - - - - -+
   |  ONNX/TensorRT  |-->|  Companion      |-->|  UAV payload    |
   |  export         |   |  computer       |   |  integration    |
   +- - - - - - - - -+   +- - - - - - - - -+   +- - - - - - - - -+
```

---

## Components

| Component            | Status      | Implementation                                     |
| -------------------- | ----------- | -------------------------------------------------- |
| Frame acquisition    | Implemented | `scripts/realtime_detect.py`                       |
| RGB detector         | Implemented | MT-004, YOLO26n @ 1280 px, VisDrone Person         |
| Thermal detector     | Implemented | MT-005, YOLO26n @ 640 px, HIT-UAV Person           |
| Confidence filtering | Implemented | Operational threshold 0.25                         |
| Tracking             | Implemented | ByteTrack via Ultralytics, persistent IDs          |
| Ego-motion estimate  | Implemented | LK optical flow + partial affine, RANSAC           |
| Movement classifier  | Implemented | World-frame displacement over a sliding window     |
| Detection output     | Implemented | Annotated video + JSONL stream                     |
| Geo-referencing      | Implemented | Pixel detection -> ground lat/lon + error estimate  |
| Latency benchmark    | Implemented | `scripts/benchmark_latency.py`                     |
| Model export         | Planned     | ONNX, then TensorRT on the target                  |
| Companion computer   | Planned     | Selection pending post-export benchmarks           |
| Payload integration  | Planned     | Ground-station link consuming the JSONL stream     |

---

## Modality Selection

The RGB and thermal detectors are trained on different datasets and are **not**
interchangeable, nor are their metrics comparable. See
[`training_log.md`](training_log.md) section 17.

| Condition                  | Detector | Reason                                  |
| -------------------------- | -------- | --------------------------------------- |
| Daylight, good visibility  | MT-004   | Only model trained on RGB imagery       |
| Night, smoke, low light    | MT-005   | Thermal signature independent of light  |

Fusing the two modalities has not been implemented. A prerequisite - a dataset
with registered RGB and thermal frames of the same scene - is not currently
available, since VisDrone and HIT-UAV are unrelated captures.

---

## Key Design Decisions

### Detection operates per frame; tracking supplies continuity

The detector is stateless. Identity and movement come from the tracking and
movement stages, so the detector can be swapped (RGB/thermal, or a future
exported model) without touching the rest of the pipeline.

### Movement is measured in a stabilised world frame

On a moving aircraft, the apparent motion of a stationary person is typically
larger than the real motion of a walking person. Measuring displacement in raw
image coordinates reports nearly everyone as moving. Validation on a controlled
sequence showed 6 false positives without compensation and 0 with it. See
[`realtime_pipeline.md`](realtime_pipeline.md) section 3.

### The movement threshold is measured, not assumed

The 15 px threshold sits in the gap between the measured jitter floor (~10 px)
and the measured displacement of a genuinely moving person (49-69 px).

### Movement state is withheld rather than guessed at frame borders

A partially visible person has an unreliable centroid. Such persons are still
detected and reported; only the movement label is withheld. For a rescue
payload, a wrong movement label is worse than no label.

### Detections are reported as ground coordinates, not pixels

A bounding box in pixels is not actionable for a rescue team. The payload
converts each detection's ground-contact point into a latitude/longitude using
the UAV's position and attitude and the camera geometry, and reports a position
error estimate alongside it. Attitude error dominates that estimate: at 100 m,
one degree of attitude error moves the ground fix by about 1.75 m.

### Deployment optimisation targets graph export, not model size

The trained models are launch-bound at batch 1 on the development GPU: the CPU
spends as long enqueuing kernels as the whole frame takes. Shrinking the model
cannot help that; fusing the graph can. See
[`realtime_pipeline.md`](realtime_pipeline.md) section 4.
