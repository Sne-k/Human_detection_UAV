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
}

CONF = 0.25
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

def letterbox(image, shape):
    """Resize into `shape` (h, w) preserving aspect, padding with 114."""

    target_h, target_w = shape
    height, width = image.shape[:2]

    ratio = min(target_h / height, target_w / width)

    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))

    resized = (
        cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if (new_w, new_h) != (width, height)
        else image
    )

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

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
        """Returns (boxes Nx4 xyxy, scores N, sources list)."""

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

            # Undo the letterbox back into original frame coordinates.
            target_h, target_w = entry["shape"]
            ratio = min(target_h / height, target_w / width)

            pad_x = (target_w - width * ratio) / 2
            pad_y = (target_h - height * ratio) / 2

            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / ratio
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / ratio

            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, width)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, height)

            pooled_boxes.append(xyxy)
            pooled_scores.append(boxes.conf.cpu().numpy())
            sources.extend([entry["name"]] * len(xyxy))

        if not pooled_boxes:
            return np.zeros((0, 4)), np.zeros(0), []

        boxes = np.concatenate(pooled_boxes)
        scores = np.concatenate(pooled_scores)

        if not self.ensemble:
            return weighted_box_fusion(
                boxes, scores, sources, self.single_fuse_iou
            )

        return weighted_box_fusion(boxes, scores, sources, self.fuse_iou)


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


def weighted_box_fusion(boxes, scores, sources, fuse_iou):
    """
    Average overlapping boxes weighted by confidence.

    This is the step that makes the ensemble worth its cost. Most shared
    failures are near-miss boxes between IoU 0.25 and 0.50; averaging two
    independent near-misses moves the result across the matching threshold,
    which simultaneously converts a miss into a match and removes what would
    have counted as a false positive.
    """

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

        fused_scores.append(float(weights.mean()))
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


def build_tracker(frame_rate):
    from types import SimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker

    args = SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )

    return BYTETracker(args)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Human-detection payload: detect, track, classify, locate",
    )

    parser.add_argument("--source", required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", default=None, help="Single model, e.g. MT-005")
    group.add_argument(
        "--ensemble",
        nargs="+",
        default=None,
        help="Two or more models to fuse, e.g. MT-005 MT-006",
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
    parser.add_argument("--device", default=0)

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
        names = args.ensemble or [args.model or "MT-005"]

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

    tracker = build_tracker(fps)

    ego = None if args.no_ego_motion else EgoMotionEstimator()

    movement = MovementTracker(
        history_frames=args.history,
        threshold_px=args.movement_threshold,
    )

    # Geolocation is enabled only when aircraft state is available.
    camera = None
    telemetry = {}

    if args.telemetry:
        camera = geolocate.Camera(
            width=width,
            height=height,
            hfov_deg=args.hfov,
            mount_pitch_deg=args.mount_pitch,
        )

        with open(args.telemetry, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    telemetry[record["frame"]] = record

        print(f"geoloc:  {len(telemetry)} telemetry records, "
              f"{args.hfov:.0f} deg HFOV, mount {args.mount_pitch:.0f} deg")
    else:
        print("geoloc:  disabled (no --telemetry)")

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

    for frame in frames:
        frame_start = time.perf_counter()

        if ego is not None:
            ego.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        detect_start = time.perf_counter()

        boxes, scores, votes = detector(frame)

        detect_times.append(time.perf_counter() - detect_start)

        tracks = tracker.update(Detections(boxes, scores), frame)

        records = []

        for row in tracks:
            x1, y1, x2, y2 = row[:4]
            track_id = int(row[4])
            score = float(row[5])

            centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            world = ego.to_world(centre) if ego is not None else centre

            at_border = (
                x1 <= args.border_margin
                or y1 <= args.border_margin
                or x2 >= width - args.border_margin
                or y2 >= height - args.border_margin
            )

            movement.update(track_id, world, frame_index, at_border)
            state, displacement = movement.state(track_id)

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

            if camera is not None:
                aircraft = telemetry.get(frame_index)

                if aircraft is not None:
                    fix = geolocate.locate_detection(
                        camera, record["box"], aircraft
                    )

                    record["ground"] = fix

                    if fix is not None:
                        located += 1

            records.append(record)

        movement.prune(frame_index)

        frame_times.append(time.perf_counter() - frame_start)

        instant_fps = (
            1.0 / frame_times[-1] if frame_times[-1] > 0 else 0.0
        )

        if stream is not None:
            payload = {
                "frame": frame_index,
                "timestamp_s": round(frame_index / fps, 3),
                "persons": len(records),
                "detections": records,
            }

            aircraft = telemetry.get(frame_index)

            if aircraft is not None:
                payload["uav"] = {
                    "lat": aircraft["lat"],
                    "lon": aircraft["lon"],
                    "altitude_m": aircraft["altitude_m"],
                }

            stream.write(json.dumps(payload) + "\n")

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
                },
            )

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow("payload", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if args.progress and frame_index % args.progress == 0:
            print(f"frame {frame_index:6d}  persons {len(records):3d}  "
                  f"{instant_fps:5.1f} FPS")

        frame_index += 1

        if args.max_frames and frame_index >= args.max_frames:
            break

    elapsed = time.perf_counter() - started

    if writer is not None:
        writer.release()

    if stream is not None:
        stream.close()

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
        "ego_motion": ego is not None,
        "ego_motion_failures": ego.failures if ego is not None else None,
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

        if record.get("ground"):
            label += f" +/-{record['ground']['position_error_m']:.0f}m"

        cv2.putText(
            frame, label, (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    banner = (
        f"frame {stats['frame']}  persons {stats['persons']}  "
        f"moving {stats['moving']}  {stats['fps']:.1f} FPS"
    )

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 20), (0, 0, 0), -1)

    cv2.putText(
        frame, banner, (6, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA,
    )

    return frame


if __name__ == "__main__":
    main()
