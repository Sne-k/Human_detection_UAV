"""
Real-time human-detection pipeline for the UAV payload.

This is the runtime counterpart to the training experiments: it takes a video
file, image sequence or camera stream, runs the selected detector on every
frame, keeps a stable identity per person across frames, and decides whether
each tracked person is moving or stationary.

    frame -> detector -> tracker -> ego-motion compensation -> movement state

The movement stage is the part that needs care on a UAV. The camera itself is
moving, so raw pixel displacement of a tracked box says nothing about whether
the person moved. Every frame the pipeline estimates the global image motion
caused by the aircraft, accumulates it, and converts each track position into a
stabilised world frame. Movement is then measured in that frame, so a
stationary person on a panning camera is correctly reported as stationary.

Outputs:

  - an annotated video (optional)
  - a JSONL detection stream, one record per frame, for downstream consumers
  - a run summary with timing statistics

Usage:

    python scripts/realtime_detect.py --source flight.mp4 --model MT-005
    python scripts/realtime_detect.py --source 0 --model MT-005 --show
    python scripts/realtime_detect.py --source frames/ --model MT-004 \
        --save-video out.mp4 --jsonl detections.jsonl
"""

import argparse
import json
import os
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"

# Trained reference models and the input size each was trained at.
MODELS = {
    "MT-004": {"imgsz": 1280, "modality": "RGB"},
    "MT-005": {"imgsz": 640, "modality": "Thermal"},
    "MT-006": {"imgsz": 960, "modality": "Thermal"},
    "MT-007": {"imgsz": 640, "modality": "Thermal"},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Movement classification
HISTORY_FRAMES = 15          # positions kept per track
MOVEMENT_THRESHOLD_PX = 6.0  # stabilised displacement to call a track moving
MIN_HISTORY_FOR_STATE = 5    # frames needed before a state is reported

# Colours (BGR)
COLOR_MOVING = (0, 0, 255)
COLOR_STATIC = (0, 200, 0)
COLOR_UNKNOWN = (0, 200, 200)
COLOR_TEXT = (255, 255, 255)


# ---------------------------------------------------------------------
# Ego-motion
# ---------------------------------------------------------------------

class EgoMotionEstimator:
    """
    Estimates frame-to-frame camera motion and accumulates it.

    Uses sparse Lucas-Kanade optical flow on corner features plus a partial
    affine fit (translation, rotation, uniform scale), which matches what a
    UAV camera does over short intervals. Full perspective is not estimated:
    it needs more correspondences than thermal imagery reliably provides, and
    it is not required for the displacement magnitudes used here.

    `cumulative` maps the first frame into the current frame. Inverting it
    turns a current-frame point into a stabilised world coordinate.
    """

    FEATURE_PARAMS = {
        "maxCorners": 300,
        "qualityLevel": 0.01,
        "minDistance": 8,
        "blockSize": 7,
    }

    LK_PARAMS = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }

    def __init__(self):
        self.previous_gray = None
        self.cumulative = np.eye(3, dtype=np.float64)
        self.failures = 0

    def update(self, gray):
        """Feed the next grayscale frame. Returns the 3x3 cumulative matrix."""

        if self.previous_gray is None:
            self.previous_gray = gray
            return self.cumulative

        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            mask=None,
            **self.FEATURE_PARAMS,
        )

        step = np.eye(3, dtype=np.float64)

        if previous_points is not None and len(previous_points) >= 6:

            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                previous_points,
                None,
                **self.LK_PARAMS,
            )

            if current_points is not None:
                status = status.reshape(-1).astype(bool)

                source = previous_points.reshape(-1, 2)[status]
                target = current_points.reshape(-1, 2)[status]

                if len(source) >= 6:
                    affine, inliers = cv2.estimateAffinePartial2D(
                        source,
                        target,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                    )

                    if affine is not None:
                        step[:2, :] = affine
                    else:
                        self.failures += 1
                else:
                    self.failures += 1
            else:
                self.failures += 1
        else:
            self.failures += 1

        self.cumulative = step @ self.cumulative
        self.previous_gray = gray

        return self.cumulative

    def to_world(self, point):
        """Map a current-frame point into the stabilised world frame."""

        try:
            inverse = np.linalg.inv(self.cumulative)
        except np.linalg.LinAlgError:
            return point

        homogeneous = np.array([point[0], point[1], 1.0])
        world = inverse @ homogeneous

        if abs(world[2]) < 1e-9:
            return point

        return (world[0] / world[2], world[1] / world[2])


# ---------------------------------------------------------------------
# Movement state
# ---------------------------------------------------------------------

class MovementTracker:
    """Per-track history in stabilised world coordinates."""

    def __init__(
        self,
        history_frames=HISTORY_FRAMES,
        threshold_px=MOVEMENT_THRESHOLD_PX,
        min_history=MIN_HISTORY_FOR_STATE,
    ):
        self.history = {}
        self.last_seen = {}
        self.history_frames = history_frames
        self.threshold_px = threshold_px
        self.min_history = min_history

    def update(self, track_id, world_point, frame_index):
        positions = self.history.setdefault(
            track_id,
            deque(maxlen=self.history_frames),
        )

        positions.append(world_point)
        self.last_seen[track_id] = frame_index

    def state(self, track_id):
        """Return (state, displacement_px) for a track."""

        positions = self.history.get(track_id)

        if positions is None or len(positions) < self.min_history:
            return "unknown", 0.0

        first = np.array(positions[0])
        last = np.array(positions[-1])

        displacement = float(np.linalg.norm(last - first))

        if displacement >= self.threshold_px:
            return "moving", displacement

        return "stationary", displacement

    def prune(self, frame_index, max_age=60):
        """Drop tracks that have not been seen recently."""

        stale = [
            track_id
            for track_id, seen in self.last_seen.items()
            if frame_index - seen > max_age
        ]

        for track_id in stale:
            self.history.pop(track_id, None)
            self.last_seen.pop(track_id, None)


# ---------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------

def open_source(source):
    """
    Return (iterator over BGR frames, fps, (width, height), total_frames).

    Accepts a video file, a directory of images, or a camera index.
    """

    path = Path(source)

    if path.is_dir():
        images = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise SystemExit(f"No images found in {path}")

        first = cv2.imread(str(images[0]))

        if first is None:
            raise SystemExit(f"Could not read {images[0]}")

        height, width = first.shape[:2]

        def frames():
            for image_path in images:
                frame = cv2.imread(str(image_path))

                if frame is not None:
                    yield frame

        return frames(), 10.0, (width, height), len(images)

    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def frames():
        try:
            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                yield frame
        finally:
            capture.release()

    return frames(), fps, (width, height), total


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------

def annotate(frame, records, stats):
    for record in records:
        x1, y1, x2, y2 = (int(v) for v in record["box"])

        if record["state"] == "moving":
            color = COLOR_MOVING
        elif record["state"] == "stationary":
            color = COLOR_STATIC
        else:
            color = COLOR_UNKNOWN

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label = f"#{record['track_id']} {record['state'][:4]}"

        cv2.putText(
            frame,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    banner = (
        f"frame {stats['frame']}  "
        f"persons {stats['persons']}  "
        f"moving {stats['moving']}  "
        f"{stats['fps']:.1f} FPS"
    )

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 20), (0, 0, 0), -1)

    cv2.putText(
        frame,
        banner,
        (6, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    return frame


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def resolve_weights(args):
    if args.weights:
        return Path(args.weights), args.imgsz

    entry = MODELS[args.model]
    weights = RUNS_ROOT / args.model / "weights" / "best.pt"

    return weights, args.imgsz or entry["imgsz"]


def run(args):
    weights, imgsz = resolve_weights(args)

    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    print(f"model:   {weights}")
    print(f"imgsz:   {imgsz}")
    print(f"source:  {args.source}")

    model = YOLO(str(weights))

    frames, fps, (width, height), total = open_source(args.source)

    print(f"stream:  {width}x{height} @ {fps:.1f} fps, {total or 'unknown'} frames")

    writer = None

    if args.save_video:
        output_path = Path(args.save_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    jsonl = None

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl = jsonl_path.open("w", encoding="utf-8")

    ego = EgoMotionEstimator() if not args.no_ego_motion else None
    movement = MovementTracker(
        history_frames=args.history,
        threshold_px=args.movement_threshold,
    )

    frame_times = []
    detect_times = []

    seen_tracks = set()
    moving_tracks = set()

    frame_index = 0
    started = time.perf_counter()

    for frame in frames:
        frame_start = time.perf_counter()

        if ego is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ego.update(gray)

        detect_start = time.perf_counter()

        results = model.track(
            frame,
            imgsz=imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            tracker=args.tracker,
            persist=True,
            verbose=False,
        )

        detect_times.append(time.perf_counter() - detect_start)

        records = []

        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()

            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                # The tracker drops IDs on frames with no association.
                ids = np.full(len(xyxy), -1, dtype=int)

            for box, confidence, track_id in zip(xyxy, confidences, ids):
                x1, y1, x2, y2 = box

                centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                if ego is not None:
                    world = ego.to_world(centre)
                else:
                    world = centre

                if track_id >= 0:
                    movement.update(track_id, world, frame_index)
                    state, displacement = movement.state(track_id)

                    seen_tracks.add(int(track_id))

                    if state == "moving":
                        moving_tracks.add(int(track_id))
                else:
                    state, displacement = "unknown", 0.0

                records.append({
                    "track_id": int(track_id),
                    "box": [
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2), 1),
                        round(float(y2), 1),
                    ],
                    "confidence": round(float(confidence), 4),
                    "state": state,
                    "displacement_px": round(float(displacement), 2),
                })

        movement.prune(frame_index)

        frame_times.append(time.perf_counter() - frame_start)

        instant_fps = 1.0 / frame_times[-1] if frame_times[-1] > 0 else 0.0

        stats = {
            "frame": frame_index,
            "persons": len(records),
            "moving": sum(1 for r in records if r["state"] == "moving"),
            "fps": instant_fps,
        }

        if jsonl is not None:
            jsonl.write(json.dumps({
                "frame": frame_index,
                "timestamp_s": round(frame_index / fps, 3),
                "persons": len(records),
                "detections": records,
            }) + "\n")

        if writer is not None or args.show:
            annotated = annotate(frame.copy(), records, stats)

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow("UAV human detection", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if args.progress and frame_index % args.progress == 0:
            print(
                f"frame {frame_index:6d}  "
                f"persons {len(records):3d}  "
                f"{instant_fps:5.1f} FPS"
            )

        frame_index += 1

        if args.max_frames and frame_index >= args.max_frames:
            break

    elapsed = time.perf_counter() - started

    if writer is not None:
        writer.release()

    if jsonl is not None:
        jsonl.close()

    if args.show:
        cv2.destroyAllWindows()

    if not frame_times:
        print("No frames processed.")
        return

    summary = {
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
        "ego_motion": ego is not None,
        "ego_motion_failures": ego.failures if ego is not None else None,
    }

    print()
    print("=== run summary ===")

    for key, value in summary.items():
        print(f"{key:26s} {value}")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWritten: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time UAV human detection with movement classification",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Video file, directory of images, or camera index",
    )
    parser.add_argument(
        "--model",
        default="MT-005",
        choices=sorted(MODELS),
        help="Trained reference model to use",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Explicit weights path, overrides --model",
    )
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default=0)
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config (bytetrack.yaml or botsort.yaml)",
    )

    parser.add_argument("--save-video", default=None)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--show", action="store_true")

    parser.add_argument(
        "--history",
        type=int,
        default=HISTORY_FRAMES,
        help="Frames of track history used for the movement decision",
    )
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=MOVEMENT_THRESHOLD_PX,
        help="Stabilised displacement in pixels to classify a track as moving",
    )
    parser.add_argument(
        "--no-ego-motion",
        action="store_true",
        help="Disable UAV ego-motion compensation (for comparison)",
    )

    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--progress",
        type=int,
        default=50,
        help="Print a progress line every N frames (0 disables)",
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
