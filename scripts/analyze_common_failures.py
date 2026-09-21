"""
Characterise the persons that BOTH thermal models fail to detect.

The per-person comparison established that MT-005 and MT-007 miss a shared core
of test-set persons. Those are the ones that matter: a person missed by only
one model can be recovered by ensembling, but a person missed by both is a
genuine capability gap.

Aggregate metrics cannot say what these failures have in common, and the
earlier size analysis showed that box dimensions alone do not separate them
from the persons that are detected successfully. This script therefore
characterises them along every axis the dataset actually provides:

  geometry   width, height, area, aspect ratio
  position   normalised centre, distance to the nearest image border
  crowding   persons in the same image, distance to the nearest other person
  scene      day/night, flight altitude, camera angle, parsed from the
             HIT-UAV filename convention <light>_<altitude>_<angle>_0_<id>
  failure    whether any prediction overlapped the person at all

The last axis is the important one. A miss with an overlapping prediction that
fell short of IoU 0.50 is a localisation failure; a miss with no overlapping
prediction anywhere is a recognition failure. Those two call for completely
different fixes, and the remedy for one does nothing for the other.

Every statistic is reported against the detected persons as a baseline, because
a property of the failures is only interesting if it differs from the
population as a whole.

Usage:

    python scripts/analyze_common_failures.py
    python scripts/analyze_common_failures.py --conf 0.25
"""

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter
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

IOU_THRESHOLD = 0.50

PRED_DIRS = {
    ("MT-005", "0.25"): "MT-005-test-analysis",
    ("MT-005", "0.10"): "MT-005-test-conf010",
    ("MT-007", "0.25"): "MT-007-test-analysis",
    ("MT-007", "0.10"): "MT-007-test-conf010",
}

# Project-wide small-object convention.
SMALL_PX = 32


def conf_key(value):
    return f"{value:.2f}"


def find_label_dir(model, confidence):
    name = PRED_DIRS[(model, conf_key(confidence))]

    for root in PRED_ROOTS:
        path = root / name / "labels"

        if path.exists():
            return path

    raise SystemExit(f"Missing predictions: {name}/labels")


def load_boxes(path, image_w, image_h):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        xc = float(parts[1]) * image_w
        yc = float(parts[2]) * image_h
        w = float(parts[3]) * image_w
        h = float(parts[4]) * image_h

        boxes.append({
            "x1": xc - w / 2,
            "y1": yc - h / 2,
            "x2": xc + w / 2,
            "y2": yc + h / 2,
            "cx": xc,
            "cy": yc,
            "w": w,
            "h": h,
            "conf": float(parts[5]) if len(parts) >= 6 else None,
        })

    return boxes


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    intersection = iw * ih

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def greedy_match(gt_boxes, pred_boxes):
    candidates = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            score = iou(gt, pred)

            if score >= IOU_THRESHOLD:
                candidates.append((score, gi, pi))

    candidates.sort(reverse=True)

    used_gt = set()
    used_pred = set()

    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

    return used_gt


def parse_scene(stem):
    """HIT-UAV filenames encode <light>_<altitude>_<angle>_0_<id>."""

    parts = stem.split("_")

    if len(parts) != 5:
        return {"light": None, "altitude": None, "angle": None}

    try:
        return {
            "light": "night" if parts[0] == "1" else "day",
            "altitude": int(parts[1]),
            "angle": int(parts[2]),
        }
    except ValueError:
        return {"light": None, "altitude": None, "angle": None}


def best_overlap(gt, pred_boxes):
    """Highest IoU with any prediction, regardless of threshold."""

    best = 0.0

    for pred in pred_boxes:
        score = iou(gt, pred)

        if score > best:
            best = score

    return best


def describe(values):
    if not values:
        return None

    ordered = sorted(values)

    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 2),
    }


def share_table(records, key):
    counts = Counter(str(r[key]) for r in records)
    total = sum(counts.values()) or 1

    return {
        value: {"count": count, "share": round(count / total, 4)}
        for value, count in sorted(
            counts.items(), key=lambda kv: -kv[1]
        )
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    mt005_dir = find_label_dir("MT-005", args.conf)
    mt007_dir = find_label_dir("MT-007", args.conf)

    output_dir = Path(args.output) if args.output else (
        PROJECT / "results" / "error_analysis"
        / f"common_failures_conf{conf_key(args.conf).replace('.', '')}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MT-005 @ {conf_key(args.conf)}: {mt005_dir}")
    print(f"MT-007 @ {conf_key(args.conf)}: {mt007_dir}")

    records = []

    for gt_path in sorted(GT_DIR.glob("*.txt")):
        image_path = IMAGE_DIR / f"{gt_path.stem}.jpg"

        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            image_w, image_h = image.size

        gt_boxes = load_boxes(gt_path, image_w, image_h)

        if not gt_boxes:
            continue

        preds = {
            "MT-005": load_boxes(
                mt005_dir / f"{gt_path.stem}.txt", image_w, image_h
            ),
            "MT-007": load_boxes(
                mt007_dir / f"{gt_path.stem}.txt", image_w, image_h
            ),
        }

        matched = {
            model: greedy_match(gt_boxes, boxes)
            for model, boxes in preds.items()
        }

        scene = parse_scene(gt_path.stem)

        for gi, gt in enumerate(gt_boxes):
            in5 = gi in matched["MT-005"]
            in7 = gi in matched["MT-007"]

            if in5 and in7:
                category = "both_matched"
            elif in5:
                category = "mt005_only"
            elif in7:
                category = "mt007_only"
            else:
                category = "both_missed"

            # Distance to the nearest other annotated person.
            neighbour = None

            for gj, other in enumerate(gt_boxes):
                if gj == gi:
                    continue

                distance = math.dist(
                    (gt["cx"], gt["cy"]), (other["cx"], other["cy"])
                )

                if neighbour is None or distance < neighbour:
                    neighbour = distance

            overlap5 = best_overlap(gt, preds["MT-005"])
            overlap7 = best_overlap(gt, preds["MT-007"])
            overlap = max(overlap5, overlap7)

            if category == "both_missed":
                failure = (
                    "localisation" if overlap > 0.0 else "recognition"
                )
            else:
                failure = ""

            records.append({
                "image": gt_path.stem,
                "gt_index": gi,
                "category": category,
                "failure_mode": failure,
                "width_px": round(gt["w"], 1),
                "height_px": round(gt["h"], 1),
                "area_px2": round(gt["w"] * gt["h"], 1),
                "aspect_hw": round(gt["h"] / gt["w"], 3) if gt["w"] else None,
                "centre_x_norm": round(gt["cx"] / image_w, 4),
                "centre_y_norm": round(gt["cy"] / image_h, 4),
                "border_distance_px": round(
                    min(
                        gt["x1"], gt["y1"],
                        image_w - gt["x2"], image_h - gt["y2"],
                    ), 1
                ),
                "persons_in_image": len(gt_boxes),
                "nearest_person_px": (
                    round(neighbour, 1) if neighbour is not None else ""
                ),
                "best_overlap_iou": round(overlap, 4),
                "light": scene["light"],
                "altitude_m": scene["altitude"],
                "camera_angle_deg": scene["angle"],
                "size_class": (
                    "small"
                    if gt["w"] < SMALL_PX or gt["h"] < SMALL_PX
                    else "medium_or_larger"
                ),
            })

    csv_path = output_dir / "person_records.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    missed = [r for r in records if r["category"] == "both_missed"]
    detected = [r for r in records if r["category"] == "both_matched"]

    def column(rows, key):
        return [r[key] for r in rows if isinstance(r[key], (int, float))]

    numeric_keys = [
        "width_px",
        "height_px",
        "area_px2",
        "aspect_hw",
        "border_distance_px",
        "persons_in_image",
        "nearest_person_px",
    ]

    summary = {
        "confidence": conf_key(args.conf),
        "iou_threshold": IOU_THRESHOLD,
        "total_persons": len(records),
        "both_missed": len(missed),
        "both_matched": len(detected),
        "failure_modes": dict(
            Counter(r["failure_mode"] for r in missed)
        ),
        "geometry": {
            key: {
                "missed": describe(column(missed, key)),
                "detected": describe(column(detected, key)),
            }
            for key in numeric_keys
        },
        "scene": {
            key: {
                "missed": share_table(missed, key),
                "detected": share_table(detected, key),
            }
            for key in ("light", "altitude_m", "camera_angle_deg", "size_class")
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"Persons missed by BOTH models at confidence "
        f"{conf_key(args.conf)}: {len(missed)} of {len(records)}"
    )
    print("=" * 70)

    print()
    print("Failure mode")
    print("-" * 70)

    for mode, count in Counter(r["failure_mode"] for r in missed).most_common():
        share = count / len(missed) * 100 if missed else 0
        print(f"  {mode:16s} {count:5d}  ({share:5.1f}%)")

    print()
    print("Geometry: missed vs detected (mean / median / max)")
    print("-" * 70)
    print(f"{'Measurement':22s} {'missed':>22s} {'detected':>22s}")

    for key in numeric_keys:
        m = describe(column(missed, key))
        d = describe(column(detected, key))

        if not m or not d:
            continue

        print(
            f"{key:22s} "
            f"{m['mean']:7.1f}/{m['median']:7.1f}/{m['max']:6.1f} "
            f"{d['mean']:7.1f}/{d['median']:7.1f}/{d['max']:6.1f}"
        )

    print()
    print("Scene distribution: missed vs detected (share)")
    print("-" * 70)

    for key in ("light", "camera_angle_deg", "altitude_m"):
        print(f"\n{key}")

        missed_share = share_table(missed, key)
        detected_share = share_table(detected, key)

        for value in sorted(
            set(missed_share) | set(detected_share),
            key=lambda v: -missed_share.get(v, {}).get("count", 0),
        ):
            m = missed_share.get(value, {"count": 0, "share": 0.0})
            d = detected_share.get(value, {"count": 0, "share": 0.0})

            ratio = (
                m["share"] / d["share"] if d["share"] else float("inf")
            )

            print(
                f"  {value:>8s}  missed {m['count']:4d} ({m['share']:6.2%})  "
                f"detected {d['count']:5d} ({d['share']:6.2%})  "
                f"over-representation x{ratio:.2f}"
            )

    print()
    print(f"Written: {csv_path}")
    print(f"Written: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
