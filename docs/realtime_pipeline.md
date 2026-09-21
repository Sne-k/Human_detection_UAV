# Real-Time Detection Pipeline

This document covers the runtime side of the human-detection subsystem: the
pipeline that runs on a live video stream, the movement-detection design, the
measured latency of the trained models, and the validation performed so far.

The training experiments are documented separately in
[`training_log.md`](training_log.md).

---

## 1. Pipeline

```text
        Video / camera / image sequence
                     |
                     v
          +----------------------+
          |  Ego-motion estimate |   Lucas-Kanade optical flow
          |  (per frame)         |   + partial affine fit
          +----------------------+
                     |
                     v
          +----------------------+
          |   YOLO26n detector   |   MT-004 (RGB) or MT-005 (thermal)
          +----------------------+
                     |
                     v
          +----------------------+
          |   ByteTrack tracker  |   persistent identity per person
          +----------------------+
                     |
                     v
          +----------------------+
          |  World-frame mapping |   track centre -> stabilised coordinates
          +----------------------+
                     |
                     v
          +----------------------+
          |  Movement classifier |   moving / stationary / edge / unknown
          +----------------------+
                     |
          +----------+-----------+
          |          |           |
          v          v           v
      Annotated    JSONL      Run summary
        video      stream     (timing stats)
```

Implemented in [`scripts/realtime_detect.py`](../scripts/realtime_detect.py).

### Inputs

| Source type       | Example                            |
| ----------------- | ---------------------------------- |
| Video file        | `--source flight.mp4`              |
| Directory of images | `--source dataset/.../images/test` |
| Camera            | `--source 0`                       |

### Outputs

| Output           | Flag           | Contents                                     |
| ---------------- | -------------- | -------------------------------------------- |
| Annotated video  | `--save-video` | Boxes coloured by movement state, HUD banner |
| Detection stream | `--jsonl`      | One JSON record per frame                    |
| Run summary      | `--summary`    | Frame count, FPS, latency percentiles        |

The JSONL stream is the interface intended for downstream consumers (ground
station link, payload controller). One record per frame:

```json
{
  "frame": 42,
  "timestamp_s": 2.8,
  "persons": 3,
  "detections": [
    {
      "track_id": 1,
      "box": [79.4, 203.1, 91.2, 221.6],
      "confidence": 0.6412,
      "state": "moving",
      "displacement_px": 22.7
    }
  ]
}
```

---

## 2. Movement Detection

### The problem

Movement cannot be read off raw pixel displacement, because the camera is on a
moving aircraft. On an aerial platform, the apparent motion of a stationary
person is usually *larger* than the real motion of a walking person. A naive
frame-to-frame displacement test reports nearly everyone as moving.

### The approach

Each frame, the pipeline estimates the global image motion induced by the
aircraft:

1. Detect corner features in the previous frame
   (`cv2.goodFeaturesToTrack`).
2. Track them into the current frame with sparse Lucas-Kanade optical flow
   (`cv2.calcOpticalFlowPyrLK`).
3. Fit a partial affine transform - translation, rotation, uniform scale -
   with RANSAC (`cv2.estimateAffinePartial2D`).
4. Accumulate the per-frame transforms into a cumulative matrix that maps the
   first frame into the current frame.

Inverting the cumulative transform converts a current-frame point into a
**stabilised world frame**. Track centres are stored in that frame, and
movement is measured as displacement in world coordinates over a sliding
window.

A full perspective homography is not estimated. It needs more reliable
correspondences than low-contrast thermal imagery consistently provides, and
the partial affine model already captures what a UAV camera does over the short
intervals that matter here.

### States

| State        | Meaning                                                      |
| ------------ | ------------------------------------------------------------ |
| `moving`     | World-frame displacement over the window exceeds the threshold |
| `stationary` | Displacement below the threshold                             |
| `edge`       | Box touches the frame border; movement state withheld        |
| `unknown`    | Track too new for a decision                                 |

The `edge` state exists because a box touching the frame border is only partly
inside the field of view. Its centroid shifts as the person enters or leaves
the frame rather than as they move. Such persons are still detected, tracked
and reported - only the movement label is withheld. For a rescue payload a
wrong movement label is worse than no label.

### Threshold calibration

The threshold was measured, not guessed. On the validation sequence described
below, with compensation enabled:

| Population                   | World-frame displacement |
| ---------------------------- | ------------------------ |
| Stationary persons (jitter)  | up to ~10 px             |
| Genuinely moving person      | 49 - 69 px               |

The noise floor comes from detection box jitter on targets that are only about
12 x 19 px in size. The default threshold of **15 px over a 15-frame window**
sits in the gap with margin on both sides.

---

## 3. Validation

### Test sequence

HIT-UAV is a set of independent still frames, so it cannot be used to test
tracking or movement detection directly. A sequence with known ground truth is
synthesised instead by
[`scripts/make_test_sequence.py`](../scripts/make_test_sequence.py):

- A 512 x 384 viewport pans diagonally across a 640 x 512 thermal test image
  containing 31 annotated persons, with a 1.5 degree peak roll. This reproduces
  the dominant effect of UAV motion: every stationary person slides across the
  frame.
- One person patch is copied from the image and composited along a trajectory
  of its own at 1.4 px/frame, giving exactly one genuinely moving person.
- No upscaling is applied, so the detector sees original sensor texture.

The correct answer is therefore known: **exactly one person is moving.**

| Parameter    | Value              |
| ------------ | ------------------ |
| Source image | `1_60_80_0_00707`  |
| Frames       | 120 @ 15 fps       |
| Viewport     | 512 x 384          |
| Pan span     | 128 x 128 px       |
| Peak roll    | 1.5 degrees        |

### Result

MT-005 at confidence 0.25, ByteTrack, 15 px threshold over 15 frames:

| Configuration              | Tracks | Flagged moving | False positives |
| -------------------------- | -----: | -------------: | --------------: |
| Ego-motion compensation on |     43 |          **1** |           **0** |
| Ego-motion compensation off|     43 |              7 |               6 |

With compensation enabled, the single flagged track is track 1, observed on all
120 frames, moving from centre (79, 211) to (245, 253). The composited person's
known trajectory runs from (76.8, 211.2) to (243.4, 252.9), so the pipeline
tracked the correct target across the whole sequence and classified it
correctly.

Without compensation, six stationary people are additionally flagged. Their
track centres all drift from x ≈ 40-133 to x ≈ 5-8, which is purely the camera
pan carrying them toward the left edge.

The ego-motion estimator reported **0 failures** across the sequence.

### Limitations of this validation

- The sequence is synthetic. It reproduces camera translation and roll, but not
  altitude change, rolling-shutter effects, atmospheric variation or real
  sensor noise over time.
- 28 of 43 tracks end in the `edge` state, because the pan sweeps people off
  the left edge. That proportion is an artefact of a 128 px pan over a 512 px
  viewport and would be much lower in real forward flight.
- 43 track IDs were created for 31 persons, so ByteTrack produced identity
  switches. Identity stability has not yet been measured against ground truth.
- Only one moving target is present, so this validates the false-positive
  behaviour well and the false-negative behaviour barely at all.

Validation on real UAV video remains outstanding.

---

## 4. Measured Latency

Implemented in
[`scripts/benchmark_latency.py`](../scripts/benchmark_latency.py).

The Ultralytics validator reports throughput that includes dataloader and
per-image Python overhead. Those figures are fine for comparing runs but must
not be used to size a companion computer. This benchmark measures what a live
pipeline pays per frame, with warm-up, CUDA synchronisation, and median/p95
rather than mean.

### Single-frame latency, FP32, RTX 4050 Laptop GPU

| Experiment | imgsz | Source frame | Forward median | Forward p95 | End-to-end median | End-to-end FPS |
| ---------- | ----: | ------------ | -------------: | ----------: | ----------------: | -------------: |
| MT-004     |  1280 | 1920x1080    |       37.83 ms |    45.48 ms |          34.45 ms |           29.0 |
| MT-005     |   640 | 640x512      |       39.54 ms |    52.63 ms |          25.13 ms |           39.8 |
| MT-006     |   960 | 640x512      |       35.24 ms |    46.20 ms |          30.11 ms |           33.2 |
| MT-007     |   640 | 640x512      |       38.74 ms |    46.10 ms |          27.04 ms |           37.0 |

### The models are launch-bound, not compute-bound

The table above has an obvious anomaly: MT-004 at 1280 px costs the same as
MT-005 at 640 px, although it processes four times as many pixels. The
benchmark therefore also times the forward pass *without* CUDA
synchronisation, which measures only how long the CPU takes to enqueue kernels.

| Experiment | Batch-8 per image | Batch-8 FPS | CPU enqueue | Wall     | Ratio | Verdict      |
| ---------- | ----------------: | ----------: | ----------: | -------: | ----: | ------------ |
| MT-004     |          18.72 ms |        53.4 |    36.08 ms | 36.08 ms |   1.0 | launch-bound |
| MT-005     |           5.58 ms |       179.4 |    41.62 ms | 41.62 ms |   1.0 | launch-bound |
| MT-006     |           9.86 ms |       101.4 |    38.43 ms | 38.43 ms |   1.0 | launch-bound |
| MT-007     |           4.36 ms |       229.3 |    42.40 ms | 42.40 ms |   1.0 | launch-bound |

A ratio of 1.0 means the GPU finishes before the CPU can submit the next
kernel. At batch 1 the entire frame time is CPU-side launch overhead for a
2.4M-parameter, 5.3 GFLOP network with 120 layers.

Batch 8 removes that bottleneck and recovers the expected compute scaling:
18.7 ms/image at 1280 px versus about 4-6 ms/image at 640 px, roughly the 4x
ratio the pixel count predicts.

### Consequences for deployment

1. **Single-frame latency on this hardware says nothing about model cost.**
   Comparing MT-005 and MT-006 on the batch-1 numbers would wrongly suggest
   960 px inference is free. It is roughly twice the GPU work.

2. **The optimisation that matters is graph export, not a smaller model.**
   Reducing resolution or parameters cannot help a workload that is bounded by
   kernel launch count. Exporting to ONNX/TensorRT fuses the graph and removes
   the Python-side dispatch, which is where the real gain is.

3. **Companion-computer selection must not be based on these figures.**
   They characterise a laptop CPU's dispatch rate, not the embedded target.
   The benchmark must be re-run on candidate hardware after export.

### Full-pipeline throughput

On the 120-frame validation sequence, end to end including tracking,
ego-motion estimation, annotation and JSONL writing:

| Configuration               | Pipeline FPS | Frame median | Detect median |
| --------------------------- | -----------: | -----------: | ------------: |
| Ego-motion compensation on  |         13.5 |     41.35 ms |      35.37 ms |
| Ego-motion compensation off |         14.2 |     35.97 ms |      35.18 ms |

Ego-motion estimation costs roughly 5 ms/frame, about 12% of the frame budget.
The detector dominates, and the detector is launch-bound, so export is again
the lever.

---

## 5. Next Steps

1. Export MT-004 and MT-005 to ONNX and re-measure, to confirm the
   launch-bound analysis and quantify the gain.
2. Validate tracking and movement detection on real UAV video, including
   identity-stability measurement against ground truth.
3. Re-run both benchmarks on candidate companion computers.
4. Add the ground-station interface that consumes the JSONL stream.
