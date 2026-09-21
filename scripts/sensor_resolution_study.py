"""
How much thermal sensor resolution does the detector actually need?

The thermal camera has not been bought yet, and this is the specification that
matters most. HIT-UAV was captured at 640 x 512, so every accuracy number in
this project assumes a 640 x 512 thermal sensor. Budget LWIR modules are
usually much coarser:

    FLIR Lepton 3.5      160 x 120
    MLX90640              32 x 24
    typical USB modules  256 x 192
    HIT-UAV / training   640 x 512

That matters more here than in most detection problems, because the failure
analysis established that the remaining misses are already concentrated in
targets around 12 x 19 px. A sensor at 256 x 192 sees 40% of the linear scale,
turning that person into roughly 5 x 8 px - below the size at which the
detector currently succeeds at all.

This script measures the effect rather than predicting it. Each test image is
downscaled to a candidate sensor resolution, then resampled back to the model's
input size, which reproduces what the detector would see if the camera had
captured at that resolution in the first place. Ground truth is unchanged, so
recall is directly comparable across rows.

The result is a purchasing specification: the coarsest thermal sensor that
still supports the mission.

Usage:

    python scripts/sensor_resolution_study.py
    python scripts/sensor_resolution_study.py --limit 200 --device cpu
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

GT_DIR = PROJECT / "dataset" / "hit_uav_person" / "labels" / "test"
IMAGE_DIR = PROJECT / "dataset" / "hit_uav_person" / "images" / "test"
WEIGHTS = (
    PROJECT / "runs" / "detect" / "results" / "training"
    / "MT-005" / "weights" / "best.pt"
)

OUTPUT_DIR = PROJECT / "results" / "sensor_study"

IOU_THRESHOLD = 0.50
CONF = 0.25

# Candidate sensors, widest first. 640 x 512 is the native training resolution
# and acts as the control.
SENSORS = [
    (640, 512, "HIT-UAV native / training resolution"),
    (384, 288, "mid-range LWIR module"),
    (320, 256, "half linear resolution"),
    (256, 192, "typical budget USB thermal module"),
    (160, 120, "FLIR Lepton 3.5 class"),
]


def load_gt(path, width, height):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        xc = float(parts[1]) * width
        yc = float(parts[2]) * height
        w = float(parts[3]) * width
        h = float(parts[4]) * height

        boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])

    return boxes


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def greedy_match(gt, preds):
    pairs = []

    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            score = iou(g, p)

            if score >= IOU_THRESHOLD:
                pairs.append((score, gi, pi))

    pairs.sort(reverse=True)

    used_gt = set()
    used_pred = set()

    for _, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

    return len(used_gt), len(preds) - len(used_pred)


def simulate(image, sensor_w, sensor_h):
    """
    Reproduce what the detector would see from a coarser sensor.

    Downscale with INTER_AREA, which averages like a real sensor integrating
    over a larger pixel, then resample back up so the model receives its
    expected input size. The information lost in the first step is gone.
    """

    height, width = image.shape[:2]

    if (sensor_w, sensor_h) == (width, height):
        return image

    small = cv2.resize(
        image, (sensor_w, sensor_h), interpolation=cv2.INTER_AREA
    )

    return cv2.resize(
        small, (width, height), interpolation=cv2.INTER_LINEAR
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)

    args = parser.parse_args()

    from ultralytics import YOLO

    if not WEIGHTS.exists():
        raise SystemExit(f"Weights not found: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if args.limit:
        images = images[:args.limit]

    print(f"MT-005 on {len(images)} HIT-UAV test images, conf {CONF}, "
          f"IoU >= {IOU_THRESHOLD}")
    print("Simulating coarser thermal sensors by downscale-then-upscale.")
    print()

    rows = []

    for sensor_w, sensor_h, label in SENSORS:
        total_gt = 0
        total_matched = 0
        total_pred = 0
        total_unmatched = 0
        blind = 0

        for path in images:
            frame = cv2.imread(str(path))

            if frame is None:
                continue

            height, width = frame.shape[:2]

            degraded = simulate(frame, sensor_w, sensor_h)

            result = model.predict(
                degraded,
                imgsz=args.imgsz,
                conf=CONF,
                iou=0.70,
                device=args.device,
                verbose=False,
            )[0]

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                preds = []
            else:
                preds = boxes.xyxy.cpu().numpy().tolist()

            gt = load_gt(GT_DIR / f"{path.stem}.txt", width, height)

            matched, unmatched = greedy_match(gt, preds)

            total_gt += len(gt)
            total_matched += matched
            total_pred += len(preds)
            total_unmatched += unmatched

            if gt and not preds:
                blind += 1

        recall = total_matched / total_gt if total_gt else 0.0

        # Scale a representative person: the median HIT-UAV person is ~12 x 19.
        scale = sensor_w / 640.0

        row = {
            "sensor": f"{sensor_w}x{sensor_h}",
            "label": label,
            "linear_scale": round(scale, 3),
            "person_px": f"{12 * scale:.1f}x{19 * scale:.1f}",
            "gt": total_gt,
            "predicted": total_pred,
            "matched": total_matched,
            "missed": total_gt - total_matched,
            "unmatched": total_unmatched,
            "recall": round(recall, 4),
            "blind_images": blind,
        }

        rows.append(row)

        print(
            f"{row['sensor']:>9s}  person ~{row['person_px']:>9s} px  "
            f"matched {row['matched']:5d}/{row['gt']:<5d}  "
            f"recall {row['recall']:.4f}  "
            f"missed {row['missed']:4d}  blind {row['blind_images']:3d}   "
            f"{label}"
        )

    baseline = rows[0]["recall"]

    print()
    print("=" * 74)
    print("Recall relative to the native 640 x 512 sensor")
    print("=" * 74)

    for row in rows:
        retained = row["recall"] / baseline if baseline else 0.0
        lost = row["missed"] - rows[0]["missed"]

        print(
            f"  {row['sensor']:>9s}  {row['recall']:.4f}  "
            f"({retained:6.1%} of native)  "
            f"{lost:+5d} additional people missed"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "sensor_resolution.json").write_text(
        json.dumps(rows, indent=2)
    )

    print()
    print(f"Written: {OUTPUT_DIR / 'sensor_resolution.json'}")


if __name__ == "__main__":
    main()
