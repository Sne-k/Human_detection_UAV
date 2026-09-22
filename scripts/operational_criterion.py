
"""
Is IoU 0.50 the right criterion for a search-and-rescue payload?

Every recall figure in this project uses IoU >= 0.50, which is the computer
vision convention inherited from PASCAL VOC and COCO. It is not a mission
requirement, and nothing in this project has checked whether it matches what
the payload actually has to deliver.

It probably does not. The payload does not hand a rescue team a bounding box.
It hands them a latitude and longitude with a stated uncertainty. A detection
is operationally useful if it puts the team close enough to find the person,
and "close enough" is set by the geolocation error budget, not by box overlap.

This script derives the criterion from the delivery mechanism instead of
assuming it:

1. Ground sampling distance at the operating altitudes.
2. The ground displacement each IoU threshold corresponds to, for a
   typically-sized target.
3. That displacement compared against the geolocation error budget.
4. Recall measured at each threshold.

The argument is only honest if both numbers are reported. mAP@50 remains the
comparable academic metric and must still be quoted; the operational figure is
reported alongside it with this derivation attached, never in place of it.

Usage:

    python scripts/operational_criterion.py
    python scripts/operational_criterion.py --altitude 60 --hfov 50
"""

import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image

PROJECT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

GT_DIR = PROJECT / "dataset" / "hit_uav_person" / "labels" / "test"
IMAGE_DIR = PROJECT / "dataset" / "hit_uav_person" / "images" / "test"

PRED_ROOTS = [
    PROJECT / "runs" / "detect" / "results" / "error_analysis",
    PROJECT / "results" / "error_analysis",
]

OUTPUT_DIR = PROJECT / "results" / "operational_criterion"

THRESHOLDS = [0.50, 0.40, 0.30, 0.25, 0.20, 0.10]

# Median HIT-UAV person, from the failure analysis.
PERSON_W = 12
PERSON_H = 19

# Geolocation 1-sigma position error at 100 m nadir, from geolocate.py.
GEOLOCATION_ERROR_M = 3.8


def load_boxes(path, width, height, conf_min=0.0):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        conf = float(parts[5]) if len(parts) >= 6 else 1.0

        if conf < conf_min:
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


def match(gt, preds, threshold):
    pairs = []

    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            score = iou(g, p)

            if score >= threshold:
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


def gsd(altitude_m, hfov_deg, sensor_px):
    """Ground sampling distance: metres per pixel."""

    swath = 2.0 * altitude_m * math.tan(math.radians(hfov_deg) / 2.0)

    return swath / sensor_px, swath


def displacement_for_iou(target, w, h):
    """
    Pixel offset between two equal boxes of size w x h that yields `target` IoU.

    Offsetting along one axis: intersection = (w - d) * h, and
    union = 2wh - intersection, so target = i / (2wh - i) gives
    i = 2wh * target / (1 + target).
    """

    intersection = 2.0 * w * h * target / (1.0 + target)

    return w - intersection / h


def find_labels(name):
    for root in PRED_ROOTS:
        path = root / name / "labels"

        if path.exists():
            return path

    raise SystemExit(f"Predictions not found: {name}/labels")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--labels", default="MT-005-test-analysis")
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--hfov", type=float, default=50.0)
    parser.add_argument("--sensor-px", type=int, default=640)

    args = parser.parse_args()

    label_dir = find_labels(args.labels)

    metres_per_px, swath = gsd(args.altitude, args.hfov, args.sensor_px)

    print("=" * 74)
    print("Is IoU 0.50 the right criterion for this payload?")
    print("=" * 74)
    print(f"Predictions : {label_dir.parent.name}")
    print(f"Geometry    : {args.altitude:.0f} m altitude, {args.hfov:.0f} deg "
          f"HFOV, {args.sensor_px} px sensor")
    print(f"              swath {swath:.1f} m, GSD {metres_per_px:.4f} m/px")
    print(f"              a {PERSON_W}x{PERSON_H} px person is "
          f"{PERSON_W * metres_per_px:.2f} x {PERSON_H * metres_per_px:.2f} m")
    print()

    # ------------------------------------------------------------------
    # What each threshold means on the ground
    # ------------------------------------------------------------------

    data = []

    for gt_path in sorted(GT_DIR.glob("*.txt")):
        image_path = IMAGE_DIR / f"{gt_path.stem}.jpg"

        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        data.append((
            load_boxes(gt_path, width, height),
            load_boxes(label_dir / f"{gt_path.stem}.txt", width, height),
        ))

    gt_total = sum(len(g) for g, _ in data)

    rows = []

    header = (
        f"{'IoU':>6s} {'px offset':>10s} {'ground err':>11s} "
        f"{'matched':>9s} {'missed':>8s} {'recall':>9s} {'unmatched':>11s}"
    )
    print(header)
    print("-" * len(header))

    for threshold in THRESHOLDS:
        matched = unmatched = 0

        for gt, preds in data:
            a, b = match(gt, preds, threshold)
            matched += a
            unmatched += b

        offset_px = displacement_for_iou(threshold, PERSON_W, PERSON_H)
        offset_m = offset_px * metres_per_px

        rows.append({
            "iou": threshold,
            "offset_px": round(offset_px, 2),
            "offset_m": round(offset_m, 3),
            "matched": matched,
            "missed": gt_total - matched,
            "recall": round(matched / gt_total, 4),
            "unmatched": unmatched,
        })

        print(
            f"{threshold:6.2f} {offset_px:10.1f} {offset_m:10.2f}m "
            f"{matched:9d} {gt_total - matched:8d} {matched / gt_total:9.4f} "
            f"{unmatched:11d}"
        )

    # ------------------------------------------------------------------
    # The argument
    # ------------------------------------------------------------------

    strict = rows[0]
    operational = next(r for r in rows if abs(r["iou"] - 0.25) < 1e-6)

    print()
    print("=" * 74)
    print("Against the geolocation error budget")
    print("=" * 74)
    print(f"  Position error reported per fix      : +/- "
          f"{GEOLOCATION_ERROR_M:.1f} m")
    print(f"  Box error rejected by IoU 0.50       : "
          f"{strict['offset_m']:.2f} m")
    print(f"  Box error rejected by IoU 0.25       : "
          f"{operational['offset_m']:.2f} m")
    print()

    ratio = GEOLOCATION_ERROR_M / operational["offset_m"]

    print(f"  The coordinate handed to the rescue team is uncertain by")
    print(f"  {GEOLOCATION_ERROR_M:.1f} m, which is {ratio:.1f}x larger than the "
          f"{operational['offset_m']:.2f} m box error that")
    print(f"  IoU 0.25 would still accept. A detection rejected for falling")
    print(f"  between 0.25 and 0.50 would have been reported to the same")
    print(f"  place on the ground as one that passed.")
    print()

    gained = operational["matched"] - strict["matched"]

    print(f"  People found under IoU 0.50 : {strict['matched']}  "
          f"(recall {strict['recall']:.4f})")
    print(f"  People found under IoU 0.25 : {operational['matched']}  "
          f"(recall {operational['recall']:.4f})")
    print(f"  Difference                  : {gained} people")
    print()
    print("  Recall saturates below 0.25, so this is not an arbitrary slide")
    print("  toward an easier number - it is where 'the box is on the person'")
    print("  stops excluding anyone.")

    print()
    print("=" * 74)
    print("How to report this")
    print("=" * 74)
    print("  Quote mAP@50 as the comparable academic metric. It is what other")
    print("  work reports and it must remain in the results.")
    print()
    print("  Report the operational recall alongside it, with this derivation")
    print("  attached. Never in place of it, and never without saying which")
    print("  criterion produced which number.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "operational_criterion.json").write_text(
        json.dumps(
            {
                "labels": str(label_dir),
                "altitude_m": args.altitude,
                "hfov_deg": args.hfov,
                "gsd_m_per_px": round(metres_per_px, 5),
                "geolocation_error_m": GEOLOCATION_ERROR_M,
                "rows": rows,
            },
            indent=2,
        )
    )

    print()
    print(f"Written: {OUTPUT_DIR / 'operational_criterion.json'}")


if __name__ == "__main__":
    main()
