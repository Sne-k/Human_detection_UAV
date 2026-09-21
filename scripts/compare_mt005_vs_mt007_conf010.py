"""
Person-level comparison of MT-005 and MT-007 at a chosen confidence threshold.

This extends `compare_mt005_vs_mt007.py`, which compared both models at
confidence 0.25, to any prediction directory. The motivating question is
whether the persons that both thermal models miss are missed because their
detections are scored below the operating threshold, or because the models do
not produce a usable box for them at all.

Comparing MT-005 at 0.25 against MT-007 at 0.10 would confound the two
variables, since MT-005 itself changes substantially when its threshold is
lowered. The default therefore compares both models at 0.10, holding
everything else fixed:

    model        MT-005      MT-007
    confidence   0.10        0.10
    input size   640         640
    test set     same        same
    ground truth same        same
    IoU          0.50        0.50

Matching is the same one-to-one greedy procedure used by the 0.25 comparison -
all candidate pairs with IoU >= 0.50, highest IoU first - so the outputs of the
two scripts are directly comparable.

Usage:

    python scripts/compare_mt005_vs_mt007_conf010.py
    python scripts/compare_mt005_vs_mt007_conf010.py --conf 0.25
    python scripts/compare_mt005_vs_mt007_conf010.py --mt005-conf 0.25 --mt007-conf 0.10
"""

import argparse
import csv
import json
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

IOU_THRESHOLD = 0.50

# Prediction directory name per (model, confidence).
PRED_DIRS = {
    ("MT-005", "0.25"): "MT-005-test-analysis",
    ("MT-005", "0.10"): "MT-005-test-conf010",
    ("MT-007", "0.25"): "MT-007-test-analysis",
    ("MT-007", "0.10"): "MT-007-test-conf010",
}


def conf_key(value):
    return f"{value:.2f}"


def find_label_dir(model, confidence):
    key = (model, conf_key(confidence))

    if key not in PRED_DIRS:
        raise SystemExit(
            f"No prediction directory registered for {model} at "
            f"confidence {conf_key(confidence)}. Known: "
            + ", ".join(f"{m}@{c}" for m, c in PRED_DIRS)
        )

    name = PRED_DIRS[key]

    for root in PRED_ROOTS:
        path = root / name / "labels"

        if path.exists():
            return path

    raise SystemExit(
        f"Could not find prediction labels for {model} at "
        f"confidence {conf_key(confidence)}.\nExpected directory "
        f"'{name}/labels' under one of:\n"
        + "\n".join(str(r) for r in PRED_ROOTS)
    )


def load_yolo_labels(path, image_w, image_h):
    """YOLO format: class x_center y_center width height [confidence]"""

    boxes = []

    if not path.exists():
        return boxes

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            xc = float(parts[1]) * image_w
            yc = float(parts[2]) * image_h
            w = float(parts[3]) * image_w
            h = float(parts[4]) * image_h

            conf = float(parts[5]) if len(parts) >= 6 else None

            boxes.append({
                "cls": cls,
                "x1": xc - w / 2,
                "y1": yc - h / 2,
                "x2": xc + w / 2,
                "y2": yc + h / 2,
                "w": w,
                "h": h,
                "conf": conf,
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

    if union <= 0:
        return 0.0

    return intersection / union


def greedy_match(gt_boxes, pred_boxes):
    """One-to-one greedy matching at IoU >= 0.50, highest IoU first."""

    candidates = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            score = iou(gt, pred)

            if score >= IOU_THRESHOLD:
                candidates.append((score, gi, pi))

    candidates.sort(reverse=True)

    used_gt = set()
    used_pred = set()

    matches = {}

    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

        matches[gi] = {"pred_index": pi, "iou": score}

    return matches, used_pred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Confidence threshold for both models",
    )
    parser.add_argument("--mt005-conf", type=float, default=None)
    parser.add_argument("--mt007-conf", type=float, default=None)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    mt005_conf = args.mt005_conf if args.mt005_conf is not None else args.conf
    mt007_conf = args.mt007_conf if args.mt007_conf is not None else args.conf

    mt005_dir = find_label_dir("MT-005", mt005_conf)
    mt007_dir = find_label_dir("MT-007", mt007_conf)

    tag = f"MT005-{conf_key(mt005_conf)}_vs_MT007-{conf_key(mt007_conf)}"

    output_dir = Path(args.output) if args.output else (
        PROJECT / "results" / "error_analysis" / tag
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MT-005 @ {conf_key(mt005_conf)}: {mt005_dir}")
    print(f"MT-007 @ {conf_key(mt007_conf)}: {mt007_dir}")

    gt_files = sorted(GT_DIR.glob("*.txt"))

    print(f"GT label files: {len(gt_files)}")

    rows = []

    totals = {
        "both_matched": 0,
        "mt005_only": 0,
        "mt007_only": 0,
        "both_missed": 0,
    }

    # Per-model aggregates
    stats = {
        model: {
            "gt": 0,
            "pred": 0,
            "matched": 0,
            "unmatched_pred": 0,
            "images_with_pred": 0,
            "zero_pred_images": 0,
            "zero_pred_images_with_gt": 0,
            "zero_pred_gt_persons": 0,
        }
        for model in ("MT-005", "MT-007")
    }

    images_total = 0
    images_with_gt = 0

    for gt_path in gt_files:
        image_path = IMAGE_DIR / f"{gt_path.stem}.jpg"

        if not image_path.exists():
            continue

        images_total += 1

        with Image.open(image_path) as image:
            image_w, image_h = image.size

        gt_boxes = load_yolo_labels(gt_path, image_w, image_h)

        if gt_boxes:
            images_with_gt += 1

        predictions = {
            "MT-005": load_yolo_labels(
                mt005_dir / f"{gt_path.stem}.txt", image_w, image_h
            ),
            "MT-007": load_yolo_labels(
                mt007_dir / f"{gt_path.stem}.txt", image_w, image_h
            ),
        }

        matched = {}

        for model, pred_boxes in predictions.items():
            entry = stats[model]

            entry["gt"] += len(gt_boxes)
            entry["pred"] += len(pred_boxes)

            if pred_boxes:
                entry["images_with_pred"] += 1
            else:
                entry["zero_pred_images"] += 1

                if gt_boxes:
                    entry["zero_pred_images_with_gt"] += 1
                    entry["zero_pred_gt_persons"] += len(gt_boxes)

            model_matches, used_pred = greedy_match(gt_boxes, pred_boxes)

            matched[model] = model_matches

            entry["matched"] += len(model_matches)
            entry["unmatched_pred"] += len(pred_boxes) - len(used_pred)

        for gi, gt in enumerate(gt_boxes):
            m5 = matched["MT-005"].get(gi)
            m7 = matched["MT-007"].get(gi)

            if m5 and m7:
                category = "both_matched"
            elif m5:
                category = "mt005_only"
            elif m7:
                category = "mt007_only"
            else:
                category = "both_missed"

            totals[category] += 1

            rows.append({
                "image": gt_path.stem,
                "gt_index": gi,
                "image_width": image_w,
                "image_height": image_h,
                "gt_width_px": round(gt["w"], 1),
                "gt_height_px": round(gt["h"], 1),
                "gt_area_px2": round(gt["w"] * gt["h"], 3),
                "category": category,
                "mt005_matched": 1 if m5 else 0,
                "mt005_iou": round(m5["iou"], 4) if m5 else "",
                "mt005_conf": (
                    predictions["MT-005"][m5["pred_index"]]["conf"] if m5 else ""
                ),
                "mt007_matched": 1 if m7 else 0,
                "mt007_iou": round(m7["iou"], 4) if m7 else "",
                "mt007_conf": (
                    predictions["MT-007"][m7["pred_index"]]["conf"] if m7 else ""
                ),
            })

    csv_path = output_dir / "person_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_gt = len(rows)

    for model, entry in stats.items():
        entry["missed"] = entry["gt"] - entry["matched"]
        entry["matched_ratio"] = (
            round(entry["matched"] / entry["gt"], 4) if entry["gt"] else None
        )

    summary = {
        "mt005_confidence": conf_key(mt005_conf),
        "mt007_confidence": conf_key(mt007_conf),
        "iou_threshold": IOU_THRESHOLD,
        "images": images_total,
        "images_with_gt": images_with_gt,
        "ground_truth_persons": total_gt,
        "per_model": stats,
        "categories": totals,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------

    print()
    print("=" * 66)
    print(f"Per-model results  (IoU >= {IOU_THRESHOLD})")
    print("=" * 66)

    header = f"{'Measurement':36s} {'MT-005':>12s} {'MT-007':>12s}"
    print(header)
    print("-" * 66)

    fields = [
        ("Ground-truth persons", "gt"),
        ("Predicted boxes", "pred"),
        ("Matched persons", "matched"),
        ("Missed persons", "missed"),
        ("Unmatched prediction boxes", "unmatched_pred"),
        ("Images producing predictions", "images_with_pred"),
        ("Zero-prediction images", "zero_pred_images"),
        ("Zero-prediction images with people", "zero_pred_images_with_gt"),
        ("Persons in those images", "zero_pred_gt_persons"),
    ]

    for label, key in fields:
        print(
            f"{label:36s} {stats['MT-005'][key]:12d} {stats['MT-007'][key]:12d}"
        )

    print(
        f"{'Matched ratio':36s} "
        f"{stats['MT-005']['matched_ratio']:12.4f} "
        f"{stats['MT-007']['matched_ratio']:12.4f}"
    )

    print()
    print("=" * 66)
    print("Per-person agreement")
    print("=" * 66)

    for category in ("both_matched", "mt005_only", "mt007_only", "both_missed"):
        count = totals[category]
        share = count / total_gt * 100 if total_gt else 0.0
        print(f"  {category:16s} {count:6d}  ({share:5.1f}%)")

    print()
    print(f"Written: {csv_path}")
    print(f"Written: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
