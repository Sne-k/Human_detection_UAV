"""
The human-detection payload, as one program.

Until now the pieces lived in three separate tools, and the best-performing
configuration could not run the full pipeline:

  realtime_detect.py   tracking and movement, but one model only
  ensemble_detect.py   the winning two-model fusion, but no tracking
  geolocate.py         ground coordinates, but as a separate post-process

So the configuration measured as best - MT-005 + MT-006 with weighted box
fusion - could not produce track identities, movement state, or a geolocated
output stream. This module closes that: detection (single model or ensemble),
tracking, movement classification and geolocation in one pass, emitting the
JSONL contract defined in docs/payload_icd.md.

    frame ---> detector ---> fusion ---> ByteTrack ---> movement ---> JSONL
                 |                                         ^            ^
                 |                                         |            |
                 +------> ego-motion -----------------------+            |
                                                                         |
                 aircraft state (MAVLink) ------> geolocation -----------+

Design
------

Detection is decoupled from tracking. `Detector` presents one interface
whether it wraps a single model or an ensemble, so the tracker, movement
classifier and geolocator are identical in both cases and the configuration
becomes a deployment choice rather than a different program.

ByteTrack is driven directly rather than through `model.track()`, which is
bound to a single model. It accepts any object exposing `conf`, `xywh` and
`cls`, so fused ensemble boxes feed it the same way single-model boxes do.

Usage:

    python scripts/payload.py --source flight.mp4 --model MT-005
    python scripts/payload.py --source flight.mp4 --ensemble MT-005 MT-006
    python scripts/payload.py --source 0 --ensemble MT-005 MT-006 \
        --telemetry state.jsonl --jsonl detections.jsonl
"""

import argparse
import json
import os
import statistics
import signal
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"

# Each model's native input shape, matching what it was trained and exported
# at. See docs/training_log.md section 17.
MODEL_SHAPES = {
    "MT-004": (1280, 1280),
    "MT-005": (512, 640),
    "MT-006": (768, 960),
    "MT-007": (512, 640),
    "MT-008": (512, 640),
    "MT-009": (512, 640),
    "MT-010": (512, 640),
    "MT-011": (512, 640),
    "MT-011b": (512, 640),
    "MT-012": (512, 640),
    "MT-013": (512, 640),
}

# The thermal detector to deploy. MT-011b is MT-011's seed repeat; both beat
# MT-005 at every mission cost ratio (training_log.md sections 40 and 45).
# They are the same configuration, so this picks one rather than selecting the
# better test-set score, which would be fitting the test split.
DEFAULT_THERMAL = "MT-011b"

# The RGB reference. Not a search detector - it exists for delivery-zone
# confirmation, and it is the only model here trained on visible light.
DEFAULT_RGB = "MT-004"

CONF = 0.25

# Whether detections the tracker has not confirmed are reported (ICD: with
# track_id -1). Measured in training_log.md section 51: dropping them loses
# 5-7 % of people from the output entirely; reporting them costs about one
# unconfirmed false detection per second at conf 0.25, which the ground
# station shows apart from confirmed targets. Below conf ~0.1 they flood.
UNASSOCIATED_DEFAULT = "report"
NMS_IOU = 0.70
FUSE_IOU = 0.55

# Single-model post-processing. Measured in training_log.md section 36:
# fusing overlapping boxes instead of suppressing them finds one more person,
# sheds 148 unmatched boxes and gains 4.0 points of precision on the test
# split, at zero inference cost.
#
# To fuse them the detector first has to keep them, so NMS is opened almost
# all the way and the clustering is done by weighted box fusion instead. NMS
# discards every box in a cluster but the highest-confidence one, and the
# highest-confidence box is not necessarily the best-localised one.
SINGLE_MODEL_NMS_IOU = 0.99
SINGLE_MODEL_FUSE_IOU = 0.60

# Movement classification, calibrated in docs/realtime_pipeline.md section 2.
HISTORY_FRAMES = 15
MOVEMENT_THRESHOLD_PX = 15.0
MIN_HISTORY_FOR_STATE = 5
BORDER_MARGIN_PX = 8

COLOR_MOVING = (0, 0, 255)
COLOR_STATIC = (0, 200, 0)
COLOR_EDGE = (160, 160, 160)
COLOR_UNKNOWN = (0, 200, 200)
COLOR_TEXT = (255, 255, 255)


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

def letterbox_geometry(height, width, shape):
    """
    The exact resize and padding that letterbox() applies, as one source.

    The inverse transform in Detector must use the same rounded dimensions and
    integer padding as the forward one. They used to be computed separately:
    the forward pass padded from rounded sizes, the inverse un-padded from
    unrounded ones. On the three frame sizes this project has used - 640 x 512
    thermal, 640 x 480 webcam, 512 x 384 test sequence - the two agree
    exactly, but on a sensor whose size does not scale to an integer they
    disagree by up to about a pixel, and one pixel is a quarter of the 4 px
    offset that drops a 12 x 19 px person below IoU 0.50.
    """

    target_h, target_w = shape

    ratio = min(target_h / height, target_w / width)

    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))

    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2

    return new_w, new_h, left, top


def letterbox(image, shape):
    """Resize into `shape` (h, w) preserving aspect, padding with 114."""

    target_h, target_w = shape
    height, width = image.shape[:2]

    new_w, new_h, left, top = letterbox_geometry(height, width, shape)

    resized = (
        cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if (new_w, new_h) != (width, height)
        else image
    )

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


class Detector:
    """
    One or more YOLO models behind a single interface.

    Returns boxes in original-frame pixel coordinates with confidences, so a
    single model and an ensemble are interchangeable downstream.
    """

    def __init__(self, names, device, conf=CONF, fuse_iou=FUSE_IOU,
                 explicit_weights=None, shape_override=None, classes=None,
                 single_fuse_iou=SINGLE_MODEL_FUSE_IOU):
        from ultralytics import YOLO

        self.conf = conf
        self.fuse_iou = fuse_iou

        # A command line delivers a GPU index as the string "0", which
        # PyTorch rejects. main() used to convert it, so every other caller
        # had to remember to; normalising here makes the Detector safe to
        # construct from anywhere.
        if isinstance(device, str) and device.isdigit():
            device = int(device)

        self.device = device
        self.classes = classes
        self.names = list(names)
        self.models = []

        for name in self.names:
            if explicit_weights:
                weights = Path(explicit_weights)

                if not weights.exists():
                    # Ultralytics resolves bare model names by downloading.
                    weights = Path(explicit_weights)
            else:
                weights = RUNS_ROOT / name / "weights" / "best.pt"

                if not weights.exists():
                    raise SystemExit(f"Weights not found: {weights}")

            shape = shape_override or MODEL_SHAPES.get(name, (512, 640))

            model = YOLO(str(weights))
            model.model.to(device).eval()

            self.models.append({
                "name": name,
                "shape": shape,
                "model": model,
            })

            print(f"  loaded {name} @ {shape[1]}x{shape[0]}")

        self.ensemble = len(self.models) > 1
        self.single_fuse_iou = single_fuse_iou

    def __call__(self, frame):
        """
        Returns (boxes Nx4 xyxy, scores N, votes list).

        Boxes are in original-frame pixels. `votes` is the number of distinct
        models behind each fused box - always 1 for a single model.
        """

        return self.fuse(*self.raw(frame))

    def raw(self, frame):
        """
        Every box above the confidence threshold, before fusion.

        Returned as (boxes, scores, sources) in original-frame pixels. Split
        from __call__ so a fusion rule can be evaluated on identical input.
        """

        height, width = frame.shape[:2]

        pooled_boxes = []
        pooled_scores = []
        sources = []

        for entry in self.models:
            padded = letterbox(frame, entry["shape"])

            result = entry["model"].predict(
                padded,
                imgsz=list(entry["shape"]),
                conf=self.conf,
                iou=NMS_IOU if self.ensemble else SINGLE_MODEL_NMS_IOU,
                device=self.device,
                classes=self.classes,
                verbose=False,
            )[0]

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()

            # Undo the letterbox with the geometry that produced it, so the
            # forward and inverse transforms cannot disagree.
            new_w, new_h, left, top = letterbox_geometry(
                height, width, entry["shape"]
            )

            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) * (width / new_w)
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) * (height / new_h)

            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)

            pooled_boxes.append(xyxy)
            pooled_scores.append(boxes.conf.cpu().numpy())
            sources.extend([entry["name"]] * len(xyxy))

        if not pooled_boxes:
            return np.zeros((0, 4)), np.zeros(0), []

        return (
            np.concatenate(pooled_boxes),
            np.concatenate(pooled_scores),
            sources,
        )

    def fuse(self, boxes, scores, sources):
        """Cluster and fuse, with the rule each configuration was measured with."""

        if not len(boxes):
            return np.zeros((0, 4)), np.zeros(0), []

        if not self.ensemble:
            return weighted_box_fusion(
                boxes, scores, sources, self.single_fuse_iou, score="seed"
            )

        return weighted_box_fusion(
            boxes, scores, sources, self.fuse_iou, score="mean"
        )


def iou_1_to_n(box, others):
    if not len(others):
        return np.zeros(0)

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])

    union = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0)


def weighted_box_fusion(boxes, scores, sources, fuse_iou, score="seed"):
    """
    Average overlapping boxes weighted by confidence.

    Most failures are near-miss boxes between IoU 0.25 and 0.50; averaging
    near-misses moves the result across the matching threshold, which converts
    a miss into a match and removes what would have counted as a false
    positive at the same time.

    `score` sets the fused box's confidence, and the two configurations were
    measured with different rules:

      seed  the highest-confidence member's score. Single-model fusion
            (single_model_wbf.py, training_log.md section 36) was measured
            this way on purpose: the mean lowers every multi-box cluster's
            score, and multi-box clusters are exactly the near-miss cases
            fusion exists to rescue.
      mean  the mean over members. The ensemble (ensemble_thermal.py) was
            measured this way, so a box one model found is not scored like
            one every model agreed on.

    The runtime used to take the mean for both. That moved the single-model
    operating point away from the one that was measured, and because the fused
    score feeds ByteTrack's match cost and its new-track threshold, it lowered
    the chance of the rescued boxes ever becoming output.
    """

    if score not in ("seed", "mean"):
        raise ValueError(f"score must be 'seed' or 'mean', not {score!r}")

    order = np.argsort(-scores)

    boxes = boxes[order]
    scores = scores[order]
    sources = np.asarray(sources)[order]

    used = np.zeros(len(boxes), dtype=bool)

    fused_boxes = []
    fused_scores = []
    fused_votes = []

    for index in range(len(boxes)):
        if used[index]:
            continue

        candidates = np.where(~used)[0]

        overlaps = iou_1_to_n(boxes[index], boxes[candidates])

        members = candidates[overlaps >= fuse_iou]

        if index not in members:
            members = np.append(members, index)

        used[members] = True

        weights = scores[members]
        total = weights.sum()

        if total <= 0:
            continue

        fused_boxes.append(
            (boxes[members] * weights[:, None]).sum(axis=0) / total
        )

        # Scores are sorted descending and every earlier index is already
        # used, so the seed at `index` is the cluster's highest score.
        fused_scores.append(
            float(scores[index]) if score == "seed" else float(weights.mean())
        )
        fused_votes.append(len(set(sources[members])))

    if not fused_boxes:
        return np.zeros((0, 4)), np.zeros(0), []

    return (
        np.asarray(fused_boxes),
        np.asarray(fused_scores),
        fused_votes,
    )


# ---------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------

class Detections:
    """
    Minimal results-like object accepted by BYTETracker.

    The tracker reads `conf`, `xywh` and `cls`, and slices the whole object
    with a boolean mask when it splits detections into high- and
    low-confidence sets, so `__getitem__` must return the same type.
    """

    def __init__(self, boxes, scores):
        self.conf = scores
        self.cls = np.zeros(len(scores))

        if len(boxes):
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]

            self.xywh = np.stack(
                [
                    (boxes[:, 0] + boxes[:, 2]) / 2,
                    (boxes[:, 1] + boxes[:, 3]) / 2,
                    widths,
                    heights,
                ],
                axis=1,
            )
        else:
            self.xywh = np.zeros((0, 4))

        self.xyxy = boxes

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xyxy[mask], self.conf[mask])


def transform_boxes(boxes, matrix):
    """
    Map xyxy boxes through a 3x3 similarity transform, keeping them upright.

    The centre is transformed exactly and the size scaled by the transform's
    uniform scale. That is exact for translation and scale. Under rotation the
    box keeps its image-aligned shape rather than rotating, which leaves its
    shape consistent between consecutive frames - and consecutive frames are
    all that association ever compares.
    """

    if not len(boxes):
        return np.zeros((0, 4))

    boxes = np.asarray(boxes, dtype=np.float64)

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    moved = matrix @ np.stack([cx, cy, np.ones_like(cx)])
    ncx, ncy = moved[0] / moved[2], moved[1] / moved[2]

    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))

    nw, nh = w * scale, h * scale

    return np.stack(
        [ncx - nw / 2.0, ncy - nh / 2.0, ncx + nw / 2.0, ncy + nh / 2.0],
        axis=1,
    )


# Boxes are enlarged by this factor for association only, and shrunk back
# before anything is reported. ByteTrack associates by overlap, and a person
# who moves about their own width between processed frames - a walker below
# about 2 FPS - has none, so is never confirmed (training_log section 52).
#
# Measured in evaluate_tracker_stress.py (training_log section 54). At 1.4
# FPS, 2x confirms 71 of 120 walkers instead of 33 - at the cost of five
# times as many track IDs reported on two different people (15 against 3).
# At 7 FPS it confirms 3 more of 120 and costs about one swap. The Pi runs at
# 3.6-6.1 FPS, so 1.0 stays the default; --assoc-scale is for a deliberate
# low-rate search configuration, where unconfirmed reporting (section 51)
# already puts those walkers on the map without an identity.
ASSOC_SCALE = 1.0


def scale_boxes(boxes, factor):
    """Scale xyxy boxes about their centres."""

    if factor == 1.0 or not len(boxes):
        return boxes

    boxes = np.asarray(boxes, dtype=np.float64)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    hw = (boxes[:, 2] - boxes[:, 0]) * factor / 2.0
    hh = (boxes[:, 3] - boxes[:, 1]) * factor / 2.0

    return np.stack([cx - hw, cy - hh, cx + hw, cy + hh], axis=1)


def track_frame(tracker, boxes, scores, frame, cumulative=None):
    """
    One frame through the tracker, stabilised by ego-motion when available.

    ByteTrack links detections across frames by box overlap. From a moving
    aircraft that overlap does not survive: at 100 m and 20 m/s the ground
    moves about 20 px between processed frames, and a 12 x 19 px person moved
    20 px has none left. On sequences built from the test split, raw-pixel
    tracking emitted every detection at 0 px of inter-frame motion, 55% at
    4 px and 8% at 5 px; across all 579 sequences at 20 px/frame it matched
    5% of people per frame, against 93% for the detector alone - and every
    record the payload emits comes from a track. Stabilised, it matches 90%.
    `evaluate_tracking_gate.py` reproduces this; training_log.md section 49.

    The payload already estimates camera motion every frame, but only used it
    after tracking, for movement classification. Here detections are mapped
    into the stabilised world frame first, where a stationary person does not
    move, so association sees the 0 px case whatever the aircraft is doing.
    Tracker output is mapped back into current-frame pixels for the record,
    the border check and geolocation.

    `cumulative` maps the first frame into the current one, as returned by
    EgoMotionEstimator.update. Without it the tracker runs on raw pixels -
    which works for a static or slowly panning camera and fails in flight.

    Returns (rows, world_boxes): tracker rows with boxes in current-frame
    pixels, and the same boxes in the stabilised frame for movement
    classification.
    """

    if cumulative is None:
        rows = tracker.update(
            Detections(scale_boxes(boxes, ASSOC_SCALE), scores), frame
        )

        if not len(rows):
            return rows, np.zeros((0, 4))

        rows = rows.copy()
        rows[:, :4] = scale_boxes(rows[:, :4], 1.0 / ASSOC_SCALE)

        return rows, rows[:, :4].copy()

    try:
        to_world = np.linalg.inv(cumulative)
    except np.linalg.LinAlgError:
        # Degenerate flow can leave a singular transform. Fall back to raw
        # pixels for this frame, as EgoMotionEstimator.to_world does.
        return track_frame(tracker, boxes, scores, frame, None)

    rows = tracker.update(
        Detections(scale_boxes(transform_boxes(boxes, to_world), ASSOC_SCALE),
                   scores),
        frame,
    )

    if not len(rows):
        return rows, np.zeros((0, 4))

    world = scale_boxes(rows[:, :4].astype(np.float64), 1.0 / ASSOC_SCALE)

    rows = rows.copy()
    rows[:, :4] = transform_boxes(world, cumulative)

    return rows, world


# Camera-motion estimation, measured in evaluate_ego_health.py and
# training_log.md section 50. One definition, used by the payload and by
# every evaluation of it, so what is measured is what ships.
#
#   predict          repeat the last trusted step across a flow failure,
#                    instead of assuming the camera stopped - which is wrong
#                    by the full ground motion at flight speed
#   forward_backward reject points that do not track back to where they
#                    started. Crossing from land onto water, forward-only
#                    flow reports a confident wrong step and the error
#                    reached 170 px within five frames; with the check, 0 px
#   max_corners 80   RANSAC needs a handful; 300 cost time and bought nothing.
#                    With the check, 14.2 ms per 640 x 512 frame against
#                    18.4 ms for the old configuration without it
#
# None of this rescues ground that is featureless from the first frame, and
# about 3% of steps there are confidently wrong in ways no vision-only check
# tried catches. That needs an independent motion reference from telemetry:
# task A5.
#   hint_tolerance_px 20
#                    with telemetry, flow that disagrees with the motion the
#                    aircraft's state implies by more than this, at the frame
#                    corners, is overruled by it (task A5). Over synthetic
#                    water, flow reported most steps "ok" while wrong by up
#                    to 32 px, and the stabilised frame drifted 1,500-2,100
#                    px over a pass; with the check, 139-455 px. 10 px
#                    overruled good flow against the simulator's telemetry;
#                    40 px missed bad flow. Provisional: the simulated
#                    attitude error moves more between frames than a real
#                    autopilot's, so set this from flight logs.
#                    training_log.md section 55.
EGO_SETTINGS = dict(
    min_inliers=1,
    fallback="predict",
    max_predict=30,
    forward_backward=True,
    max_corners=80,
    hint_tolerance_px=20.0,
)


def make_ego_estimator():
    from realtime_detect import EgoMotionEstimator

    return EgoMotionEstimator(**EGO_SETTINGS)


class FrameLogger:
    """
    Save the frames worth labelling - task D3.

    The detector was trained on someone else's data. The way to fix that is
    offline: fly, collect, label on the ground, fine-tune, validate, deploy a
    new frozen model - the two-stage recipe MT-011 showed works. This collects.

    Saving everything would fill the card with easy frames that teach the
    model nothing. A frame is saved when it is one the model was unsure
    about:

      uncertain    a detection scored within UNCERTAIN_MARGIN of the operating
                   point - where the decision is closest to going either way
      unconfirmed  a detection the tracker did not confirm - a target seen
                   once, or a track that broke

    Each saved frame gets a YOLO-format pre-label file of the model's own
    boxes, so labelling is correcting rather than drawing. Saving is rate
    limited and capped, and a write failure turns logging off without
    touching detection.
    """

    UNCERTAIN_MARGIN = 0.15

    def __init__(self, directory, conf, min_interval_s=1.0, max_frames=5000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.conf = conf
        self.min_interval_s = min_interval_s
        self.max_frames = max_frames
        self.saved = 0
        self.last_saved_at = None
        self.failed = None
        self.reasons = {"uncertain": 0, "unconfirmed": 0}
        self.index = (self.directory / "index.jsonl").open("a", encoding="utf-8")

    def consider(self, frame, frame_index, frame_time, boxes, scores, unconfirmed):
        if self.failed or self.saved >= self.max_frames or not len(boxes):
            return

        if (self.last_saved_at is not None
                and frame_time - self.last_saved_at < self.min_interval_s):
            return

        uncertain = bool(((scores >= self.conf)
                          & (scores < self.conf + self.UNCERTAIN_MARGIN)).any())

        if not (uncertain or unconfirmed):
            return

        reason = "uncertain" if uncertain else "unconfirmed"
        stem = f"{frame_index:07d}"
        height, width = frame.shape[:2]

        try:
            if not cv2.imwrite(str(self.directory / f"{stem}.png"), frame):
                raise OSError("image write returned false")

            lines = [
                f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
                for x1, y1, x2, y2 in boxes
            ]
            (self.directory / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            self.index.write(json.dumps({
                "frame": frame_index, "time_s": round(frame_time, 3),
                "reason": reason, "detections": int(len(boxes)),
                "min_score": round(float(scores.min()), 4),
            }) + "\n")
            self.index.flush()
        except OSError as exc:
            self.failed = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: frame logging failed ({self.failed}); "
                  "detection continues without it")
            return

        self.saved += 1
        self.reasons[reason] += 1
        self.last_saved_at = frame_time

    def close(self):
        try:
            self.index.close()
        except OSError:
            pass


def unassociated_indices(rows, count):
    """
    Detections the tracker did not emit a track for this frame.

    ByteTrack reports, in column 7 of each emitted row, the index of the
    input detection that fed it. Anything else was dropped or is a new track
    still waiting for its confirming second frame.
    """

    used = set(rows[:, 7].astype(int).tolist()) if len(rows) else set()
    return [i for i in range(count) if i not in used]


# ByteTrack's low-score band. Boxes between this and the operating point can
# continue an existing track but never start one.
TRACK_LOW_THRESH = 0.1


def usable_frame(frame, height, width):
    """A three-channel frame of the stream's size, or not worth processing."""

    return (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] == height
        and frame.shape[1] == width
    )


def build_tracker(frame_rate, conf=CONF):
    """
    ByteTrack, with its thresholds derived from the operating point.

    Every record the payload emits comes from a track, so the tracker is a
    gate on the whole output. A new track can only start from a detection at
    or above track_high_thresh that also clears new_track_thresh. Both were
    fixed at 0.25, so at a more sensitive operating point - such as the 0.05
    that mission cost selects in training_log.md section 37 - a person whose
    score never reached 0.25 could never be reported. The tracker silently
    re-imposed the threshold the operating point had just lowered.

    Both now follow `conf`. At the default of 0.25 the values are exactly the
    previous ones, so the default threshold behaviour is unchanged; a
    non-default operating point now means what it says.

    The guard against spurious tracks is ByteTrack's own: after the first
    frame, a new track is not reported until it is matched again on a later
    frame, so a single-frame false positive cannot become output. What the
    lower threshold costs in spurious tracks is measured by
    scripts/evaluate_tracking_gate.py rather than assumed.

    `frame_rate` is unused. This BYTETracker takes no frame-rate argument, so
    track_buffer is a number of frames, not a duration: about 4 s at the
    target's 7 FPS, 1 s at a 30 FPS webcam.
    """

    from types import SimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker

    args = SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=conf,
        track_low_thresh=min(TRACK_LOW_THRESH, conf),
        new_track_thresh=conf,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )

    return BYTETracker(args)


# A frame interval this many times the recent median counts as missed frames.
# ByteTrack predicts one step per update, so after a gap a moving person's
# track is predicted where they were a frame ago, the box no longer overlaps,
# and the track is lost and re-created under a new identity: in a sequence
# with bursts of lost frames one walking person became eight targets
# (training_log section 52). Stationary people survive, because in the
# stabilised frame they do not move. Stepping the tracker through the missed
# frames with no detections lets its motion model keep pace.
GAP_FACTOR = 1.5
BRIDGE_MAX_STEPS = 30   # = track_buffer; a longer gap ends every track anyway


def missed_steps(interval, recent):
    """Frames missed before this one, judged against recent intervals."""

    if not recent or interval <= 0:
        return 0

    nominal = float(np.median(recent))

    if nominal <= 0 or interval < GAP_FACTOR * nominal:
        return 0

    return min(int(round(interval / nominal)) - 1, BRIDGE_MAX_STEPS)


def bridge_gap(tracker, steps, frame):
    """Advance the tracker through missed frames."""

    empty = Detections(np.zeros((0, 4)), np.zeros(0))

    for _ in range(steps):
        tracker.update(empty, frame)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Human-detection payload: detect, track, classify, locate",
    )

    parser.add_argument("--source", required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model",
        default=None,
        help=f"Single model. Defaults to {DEFAULT_THERMAL}, the deployed detector.",
    )
    group.add_argument(
        "--ensemble",
        nargs="+",
        default=None,
        help="Two or more models to fuse, e.g. MT-011b MT-006. Costs one "
             "inference per model; not deployable on the target board.",
    )
    group.add_argument(
        "--weights",
        default=None,
        help=(
            "Arbitrary weights path, e.g. yolo26n.pt. Intended for bench "
            "trials on a webcam, where the project's aerial and thermal "
            "models are out of domain and a stock COCO model gives working "
            "detections for exercising the rest of the pipeline."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Input shape override, required with --weights",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Class ids to keep. COCO person is 0.",
    )

    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument(
        "--udp",
        default=None,
        metavar="HOST:PORT",
        help="Also send each record as a UDP datagram, e.g. to "
             "ground_station.py. Interface IF-6.",
    )
    parser.add_argument(
        "--log-frames",
        default=None,
        metavar="DIR",
        help="Save frames worth labelling - uncertain or unconfirmed "
             "detections - with pre-labels, for offline fine-tuning.",
    )
    parser.add_argument(
        "--assoc-scale",
        type=float,
        default=ASSOC_SCALE,
        help="Enlarge boxes by this factor for tracker association only. "
             "2.0 roughly doubles the walkers confirmed below 2 FPS, and "
             "multiplies identity swaps by five. See training_log.md "
             "section 54.",
    )
    parser.add_argument(
        "--unassociated",
        choices=["report", "drop"],
        default=UNASSOCIATED_DEFAULT,
        help="Detections the tracker has not confirmed. The ICD specifies "
             "reporting them with track_id -1; 'drop' emits confirmed tracks "
             "only. See training_log.md section 51.",
    )
    parser.add_argument("--device", default=0)

    parser.add_argument(
        "--mavlink",
        default=None,
        help="Live aircraft state over MAVLink, e.g. udpin:0.0.0.0:14550 or "
             "COM3. Requires a live camera source: frames are matched to "
             "telemetry by capture time on the local clock.",
    )
    parser.add_argument("--mavlink-baud", type=int, default=57600)
    parser.add_argument(
        "--telemetry",
        default=None,
        help="JSONL of per-frame aircraft state; enables geolocation",
    )
    parser.add_argument("--hfov", type=float, default=50.0)
    parser.add_argument("--mount-pitch", type=float, default=90.0)

    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--show", action="store_true")

    parser.add_argument("--history", type=int, default=HISTORY_FRAMES)
    parser.add_argument(
        "--movement-threshold", type=float, default=MOVEMENT_THRESHOLD_PX
    )
    parser.add_argument("--border-margin", type=int, default=BORDER_MARGIN_PX)
    parser.add_argument("--no-ego-motion", action="store_true")

    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--progress", type=int, default=50)

    args = parser.parse_args()

    # Reuse the verified implementations rather than duplicating them.
    from realtime_detect import (
        EgoMotionEstimator,
        MovementTracker,
        open_source,
    )
    import geolocate

    if args.weights:
        names = [Path(args.weights).stem]

        if args.imgsz is None:
            args.imgsz = [480, 640]

        print("NOTE: running arbitrary weights. This exercises the pipeline;")
        print("      it is not a validation of the project's detectors.")
    else:
        # DEFAULT_THERMAL is the deployed model. The fallback used to be a
        # hardcoded "MT-005", so the constant existed but did not decide
        # what ran unless --model was passed.
        names = args.ensemble or [args.model or DEFAULT_THERMAL]

    device = args.device

    if isinstance(device, str) and device.isdigit():
        device = int(device)

    print(f"payload: {' + '.join(names)}"
          f"{'  (ensemble, WBF)' if len(names) > 1 else ''}")

    detector = Detector(
        names, device, conf=args.conf,
        explicit_weights=args.weights,
        shape_override=tuple(args.imgsz) if args.imgsz else None,
        classes=args.classes,
    )

    frames, fps, (width, height), total = open_source(args.source)

    print(f"stream:  {width}x{height} @ {fps:.1f} fps, "
          f"{total or 'unknown'} frames")

    tracker = build_tracker(fps, args.conf)

    # track_frame reads the module setting; the flag only overrides it.
    globals()["ASSOC_SCALE"] = args.assoc_scale

    if args.no_ego_motion:
        # Ego-motion used to serve movement classification alone. It now also
        # stabilises tracking, and every record the payload emits comes from a
        # track - so without it, output collapses once the camera moves more
        # than a few pixels per frame.
        print("WARNING: --no-ego-motion also disables stabilised tracking.")
        print("         Raw-pixel tracking does not associate at flight speed")
        print("         (about 20 px/frame); expect almost no output from a")
        print("         moving aircraft. Fine for a static camera or webcam.")

    ego = None if args.no_ego_motion else make_ego_estimator()

    # Movement state is withheld until the motion estimate has been measured,
    # not predicted or lost, for a whole history window: a window containing
    # a bad step cannot support a confident moving/stationary label.
    clean_steps = 0
    ego_status_counts = {"ok": 0, "telemetry": 0, "predicted": 0, "lost": 0}

    movement = MovementTracker(
        history_frames=args.history,
        threshold_px=args.movement_threshold,
    )

    # Geolocation is enabled only when aircraft state is available. State is
    # matched to each frame by capture time, not frame index - see
    # telemetry.py for why that matters and how the clocks are aligned.
    import telemetry as telemetry_mod

    camera = None
    state_buffer = None
    receiver = None
    live_clock = False

    if args.telemetry or args.mavlink:
        camera = geolocate.Camera(
            width=width,
            height=height,
            hfov_deg=args.hfov,
            mount_pitch_deg=args.mount_pitch,
        )

    if args.mavlink:
        if not str(args.source).isdigit():
            print("WARNING: --mavlink with a recorded source. Frames are timed")
            print("         on the live clock, so state will not match a file.")

        state_buffer = telemetry_mod.StateBuffer()
        receiver = telemetry_mod.MavlinkReceiver(
            args.mavlink, state_buffer, baud=args.mavlink_baud
        ).start()
        live_clock = True

        print(f"geoloc:  MAVLink {args.mavlink}, "
              f"{args.hfov:.0f} deg HFOV, mount {args.mount_pitch:.0f} deg")
    elif args.telemetry:
        state_buffer = telemetry_mod.load_jsonl(args.telemetry, fps)

        print(f"geoloc:  {len(state_buffer)} telemetry samples, matched by time, "
              f"{args.hfov:.0f} deg HFOV, mount {args.mount_pitch:.0f} deg")
    else:
        print("geoloc:  disabled (no --telemetry or --mavlink)")

    frame_logger = (
        FrameLogger(args.log_frames, args.conf) if args.log_frames else None
    )

    udp = udp_target = None
    udp_sent = udp_errors = 0

    if args.udp:
        import socket

        host, _, port = args.udp.rpartition(":")
        udp_target = (host or "127.0.0.1", int(port))
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setblocking(False)
        print(f"output:  UDP to {udp_target[0]}:{udp_target[1]}")

    geoloc_refused = 0
    bad_frames = 0
    frame_errors = 0
    stream_failed = None
    stream_start = telemetry_mod.CLOCK()

    # Geolocation delay line. Live, a frame's capture time is newer than any
    # telemetry received, so its state could only be held - up to a sample
    # period plus link delay stale. Each frame's records are therefore queued
    # and geolocated once telemetry newer than the capture time has arrived,
    # so the state is interpolated. Measured over MAVLink in a 25 degree
    # banked turn: 0.07 degree worst-case attitude error interpolated, 3.4
    # degrees held - 0.12 m against 6 m on the ground at 100 m. The cost is
    # output latency of one telemetry period, a fraction of a second.
    # Recorded telemetry is all present, so it resolves immediately.
    import collections

    pending = collections.deque()
    geoloc_wait_s = 0.5

    def resolve(frame_time, out):
        nonlocal located, geoloc_refused

        if state_buffer is not None:
            aircraft = state_buffer.at(frame_time)

            if aircraft is None:
                # Stale or missing: say so rather than omit the field.
                geoloc_refused += 1
                out["uav"] = None
            else:
                out["uav"] = {
                    "lat": aircraft["lat"],
                    "lon": aircraft["lon"],
                    "altitude_m": aircraft["altitude_m"],
                    "state_age_s": aircraft["state_age_s"],
                }

                for record in out["detections"]:
                    fix = geolocate.locate_detection(
                        camera, record["box"], aircraft
                    )
                    record["ground"] = fix

                    if fix is not None:
                        located += 1

        nonlocal stream, stream_failed, udp_sent, udp_errors

        if udp is not None:
            # Fire and forget. A dead or saturated link must never slow or
            # stop detection; the on-board JSONL remains the record.
            try:
                udp.sendto(json.dumps(out, separators=(",", ":")).encode("utf-8"),
                           udp_target)
                udp_sent += 1
            except OSError:
                udp_errors += 1

        if stream is not None:
            try:
                stream.write(json.dumps(out) + "\n")
                stream.flush()
            except OSError as exc:
                # A full or failed disk must not stop detection (ICD IF-5).
                # Stop writing, say so once, keep going.
                stream_failed = f"{type(exc).__name__}: {exc}"
                print(f"WARNING: detection stream write failed ({stream_failed});"
                      " detection continues without it")

                try:
                    stream.close()
                except OSError:
                    pass

                stream = None

    def flush(final=False):
        while pending:
            frame_time, out = pending[0]

            if live_clock and not final and state_buffer is not None:
                waited = telemetry_mod.CLOCK() - frame_time

                if not state_buffer.ready(frame_time) and waited < geoloc_wait_s:
                    break

            pending.popleft()
            resolve(frame_time, out)

    writer = None

    if args.save_video:
        path = Path(args.save_video)
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    stream = None

    if args.jsonl:
        path = Path(args.jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("w", encoding="utf-8")

    frame_times = []
    detect_times = []

    seen_tracks = set()
    moving_tracks = set()
    located = 0

    frame_index = 0
    started = time.perf_counter()

    # A live camera never ends, and the `q` shortcut only fires when the
    # OpenCV window has keyboard focus. Without this, an unfocused or
    # unrendered window leaves no way out except killing the process, which
    # discards the run summary the trial exists to produce.
    stop = {"requested": False}

    def _on_interrupt(_signum, _frame):
        if stop["requested"]:
            raise KeyboardInterrupt

        stop["requested"] = True

        print()
        print("interrupt received - finishing this frame, then writing the "
              "summary")
        print("press Ctrl+C again to abort without one")

    # SIGBREAK as well as SIGINT: on Windows, Ctrl+Break and a signal sent to
    # a detached process group arrive as SIGBREAK, and Ctrl+C is ignored
    # outright by groups created with CREATE_NEW_PROCESS_GROUP.
    previous_handlers = []

    for name in ("SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)

        if sig is None:
            continue

        try:
            previous_handlers.append((sig, signal.signal(sig, _on_interrupt)))
        except (ValueError, OSError, AttributeError):
            # Not on the main thread, or unsupported on this platform.
            pass

    # A live camera is timed on the clock even without telemetry. Stream
    # time - frame index over the camera's nominal rate - is wrong whenever
    # processing is slower than the camera, which on the Pi it always is.
    if str(args.source).isdigit() and not args.telemetry:
        live_clock = True

    last_frame_time = None
    previous_state = None
    recent_intervals = deque(maxlen=30)
    steps_bridged = 0

    for frame in frames:
        # A source may supply its own capture time with each frame.
        frame, source_time = frame if isinstance(frame, tuple) else (frame, None)

        # A single bad frame, or a single exception anywhere in the
        # per-frame work, must cost that frame - not the rest of the
        # mission. Each is counted and the run continues.
        if not usable_frame(frame, height, width):
            bad_frames += 1
            frame_index += 1

            if (args.max_frames and frame_index >= args.max_frames) or stop["requested"]:
                break

            continue

        records = []

        try:
            frame_start = time.perf_counter()

            # Capture time: the shared local clock for a live stream with live
            # telemetry, stream time otherwise.
            if source_time is not None:
                frame_time = source_time
            elif live_clock:
                frame_time = telemetry_mod.CLOCK()
            else:
                frame_time = frame_index / fps

            if last_frame_time is not None:
                interval = frame_time - last_frame_time
                missed = missed_steps(interval, recent_intervals)
                recent_intervals.append(interval)

                if missed:
                    bridge_gap(tracker, missed, frame)
                    steps_bridged += missed

            last_frame_time = frame_time

            # The image motion the aircraft's own state implies, when it is
            # known for this frame and the last: a fallback for flow that
            # fails, and a check on flow that is confidently wrong (A5).
            hint = None

            if ego is not None and state_buffer is not None and camera is not None:
                state = state_buffer.at(frame_time)

                if state is not None and previous_state is not None:
                    hint = geolocate.predict_step(camera, previous_state, state)

                previous_state = state

            cumulative = (
                ego.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), hint)
                if ego is not None
                else None
            )

            if ego is not None and ego.status in ego_status_counts:
                ego_status_counts[ego.status] += 1
                clean_steps = clean_steps + 1 if ego.status == "ok" else 0

            detect_start = time.perf_counter()

            boxes, scores, votes = detector(frame)

            detect_times.append(time.perf_counter() - detect_start)

            # Tracks are formed in the stabilised frame; see track_frame().
            tracks, world_boxes = track_frame(
                tracker, boxes, scores, frame, cumulative
            )

            records = []

            for row, world_box in zip(tracks, world_boxes):
                x1, y1, x2, y2 = row[:4]
                track_id = int(row[4])
                score = float(row[5])

                world = (
                    (world_box[0] + world_box[2]) / 2.0,
                    (world_box[1] + world_box[3]) / 2.0,
                )

                at_border = (
                    x1 <= args.border_margin
                    or y1 <= args.border_margin
                    or x2 >= width - args.border_margin
                    or y2 >= height - args.border_margin
                )

                movement.update(track_id, world, frame_index, at_border, frame_time)
                state, displacement = movement.state(track_id)

                if ego is not None and clean_steps < args.history:
                    state = "unknown"

                seen_tracks.add(track_id)

                if state == "moving":
                    moving_tracks.add(track_id)

                record = {
                    "track_id": track_id,
                    "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                    "confidence": round(score, 4),
                    "state": state,
                    "displacement_px": round(float(displacement), 2),
                }

                records.append(record)

            if frame_logger is not None:
                frame_logger.consider(
                    frame, frame_index, frame_time, boxes, scores,
                    bool(unassociated_indices(tracks, len(boxes))),
                )

            if args.unassociated == "report":
                for index in unassociated_indices(tracks, len(boxes)):
                    x1, y1, x2, y2 = (float(v) for v in boxes[index])
                    records.append({
                        "track_id": -1,
                        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "confidence": round(float(scores[index]), 4),
                        "state": "unknown",
                    })

            movement.prune(frame_index)

            frame_times.append(time.perf_counter() - frame_start)

            instant_fps = (
                1.0 / frame_times[-1] if frame_times[-1] > 0 else 0.0
            )

            pending.append((frame_time, {
                "frame": frame_index,
                "timestamp_s": round(
                    frame_time - stream_start if live_clock else frame_time, 3
                ),
                "persons": len(records),
                "detections": records,
                **({"ego_motion": ego.status} if ego is not None else {}),
            }))

            flush()

            if writer is not None or args.show:
                annotated = annotate(
                    frame.copy(), records,
                    {
                        "frame": frame_index,
                        "persons": len(records),
                        "moving": sum(
                            1 for r in records if r["state"] == "moving"
                        ),
                        "fps": instant_fps,
                        "conf": args.conf,
                    },
                )

                if writer is not None:
                    writer.write(annotated)

                if args.show:
                    cv2.imshow("payload", annotated)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except Exception as exc:  # noqa: BLE001 - the payload must outlive it
            frame_errors += 1

            if frame_errors <= 5:
                print(f"frame {frame_index}: {type(exc).__name__}: {exc} "
                      f"- skipped, run continues")

        if args.progress and frame_index % args.progress == 0:
            print(f"frame {frame_index:6d}  persons {len(records):3d}  "
                  f"{instant_fps:5.1f} FPS")

        frame_index += 1

        if args.max_frames and frame_index >= args.max_frames:
            break

        if stop["requested"]:
            break

    # Resolve whatever is still waiting for telemetry.
    flush(final=True)

    elapsed = time.perf_counter() - started

    for sig, handler in previous_handlers:
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass

    if writer is not None:
        writer.release()

    if stream is not None:
        try:
            stream.close()
        except OSError:
            pass

    if receiver is not None:
        receiver.stop()

    if frame_logger is not None:
        frame_logger.close()

    if args.show:
        cv2.destroyAllWindows()

    if not frame_times:
        print("No frames processed.")
        return

    summary = {
        "models": names,
        "ensemble": len(names) > 1,
        "frames": frame_index,
        "elapsed_s": round(elapsed, 2),
        "pipeline_fps": round(frame_index / elapsed, 2),
        "frame_ms_median": round(statistics.median(frame_times) * 1000, 2),
        "frame_ms_p95": round(
            sorted(frame_times)[int(0.95 * (len(frame_times) - 1))] * 1000, 2
        ),
        "detect_ms_median": round(statistics.median(detect_times) * 1000, 2),
        "unique_tracks": len(seen_tracks),
        "tracks_classified_moving": len(moving_tracks),
        "detections_geolocated": located,
        "frames_without_valid_state": geoloc_refused,
        "bad_frames_skipped": bad_frames,
        "frame_errors": frame_errors,
        "stream_failed": stream_failed,
        "tracker_steps_bridged": steps_bridged,
        "udp_sent": udp_sent if udp is not None else None,
        "udp_errors": udp_errors if udp is not None else None,
        "frames_logged": (
            {"saved": frame_logger.saved, **frame_logger.reasons,
             "failed": frame_logger.failed}
            if frame_logger is not None else None
        ),
        "ego_motion": ego is not None,
        "ego_motion_failures": ego.failures if ego is not None else None,
        "ego_motion_overruled": ego.hint_rejections if ego is not None else None,
        "ego_motion_status": ego_status_counts if ego is not None else None,
    }

    print()
    print("=== run summary ===")

    for key, value in summary.items():
        print(f"{key:26s} {value}")

    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        print(f"\nWritten: {path}")


def annotate(frame, records, stats):
    for record in records:
        x1, y1, x2, y2 = (int(v) for v in record["box"])

        state = record["state"]

        color = (
            COLOR_MOVING if state == "moving"
            else COLOR_STATIC if state == "stationary"
            else COLOR_EDGE if state == "edge"
            else COLOR_UNKNOWN
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label = f"#{record['track_id']} {state[:4]}"

        # The detector score. It is NOT a calibrated probability that the box
        # contains a person - a network trained with cross-entropy is free to
        # be confidently wrong. What it is reliably good for is *ranking*, and
        # the measured precision at each operating point is in
        # training_log.md section 37.
        confidence = record.get("confidence")

        if confidence is not None:
            label += f" {confidence * 100:.0f}%"

        if record.get("ground"):
            label += f" +/-{record['ground']['position_error_m']:.0f}m"

        origin = (x1, max(12, y1 - 4))

        # A dark plate behind the text. Without it the label vanishes against
        # a bright background, which is exactly what a sunlit scene gives -
        # and the grey used for edge-state tracks is the worst case.
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )

        cv2.rectangle(
            frame,
            (origin[0], origin[1] - text_h - 3),
            (origin[0] + text_w + 3, origin[1] + 3),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame, label, origin,
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    banner = (
        f"frame {stats['frame']}  persons {stats['persons']}  "
        f"moving {stats['moving']}  {stats['fps']:.1f} FPS"
    )

    if stats.get("conf") is not None:
        banner += f"  conf>={stats['conf']:.2f}"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 20), (0, 0, 0), -1)

    cv2.putText(
        frame, banner, (6, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA,
    )

    return frame


if __name__ == "__main__":
    main()
