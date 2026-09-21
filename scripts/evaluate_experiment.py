"""
Full evaluation of one thermal experiment against the frozen MT-005 baseline.

Produces the comparison table defined in `docs/thermal_baseline.md`, so that a
new model is judged on the mechanism it was designed to move rather than on
mAP alone. MT-006 is the cautionary case: it improved mAP@50-95 while losing
recall and tripling the number of images where a person is present and nothing
is reported.

The headline metric here is the **single-model merged-box count**.

That needs care. The figure of 43 merged failures quoted elsewhere in this
project describes persons missed by MT-005 *and* MT-007 together at confidence
0.10 - an intersection of two models. It is not comparable to a single model's
count, and using it as a baseline for MT-008 would be an apples-to-oranges
comparison. This script therefore computes a self-contained per-model metric:

    Of the persons THIS model misses at THIS confidence, how many have a
    best-overlapping prediction that covers 2+ annotated people and is at
    least 1.6x the area of the person it should have covered?

Run it on MT-005 to obtain the reference value, then on the new experiment.
Both numbers are then produced the same way from the same test split.

Mechanisms, as in `verify_crowding_hypothesis.py`:

    merged      one prediction spans 2+ people (crowding)
    undersized  centred but well under the annotated area (partial signature)
    offset      right size, displaced (ordinary regression error)
    stolen      the overlapping prediction was matched to a neighbour
    no_overlap  nothing overlapped the person at all (recognition failure)

Usage:

    python scripts/evaluate_experiment.py --labels MT-005-test-analysis
    python scripts/evaluate_experiment.py --labels MT-008-test-analysis --conf 0.25
"""

import argparse
import csv
import json
import os
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

OUTPUT_DIR = PROJECT / "results" / "error_analysis" / "experiment_evaluation"

IOU_THRESHOLD = 0.50
TOUCH_IOU = 0.10
MERGE_AREA_RATIO = 1.60
UNDERSIZE_AREA_RATIO = 0.70
SMALL_PX = 32

# Frozen MT-005 reference, from docs/thermal_baseline.md.
BASELINE = {
    "matched": 2425,
    "missed": 186,
    "unmatched": 638,
    "custom_precision": 0.7917,
    "recall": 0.9288,
    "blind_images": 8,
    "small_recall": 0.9294,
}


def find_label_dir(name):
    for root in PRED_ROOTS:
        path = root / name / "labels"

        if path.exists():
            return path

    raise SystemExit(
        f"Prediction labels not found: {name}/labels\nChecked:\n"
        + "\n".join(str(r) for r in PRED_ROOTS)
    )


def load_boxes(path, width, height):
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

        boxes.append({
            "x1": xc - w / 2,
            "y1": yc - h / 2,
            "x2": xc + w / 2,
            "y2": yc + h / 2,
            "w": w,
            "h": h,
            "area": w * h,
            "conf": float(parts[5]) if len(parts) >= 6 else 1.0,
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
    mapping = {}

    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)
        mapping[gi] = pi

    return mapping, used_pred


def classify(gt_index, gt_boxes, pred_boxes, matched):
    target = gt_boxes[gt_index]

    best_pi = None
    best_iou = 0.0

    for pi, pred in enumerate(pred_boxes):
        score = iou(target, pred)

        if score > best_iou:
            best_iou = score
            best_pi = pi

    if best_pi is None or best_iou <= 0.0:
        return "no_overlap", 0.0, 0, 0.0

    prediction = pred_boxes[best_pi]

    touched = sum(
        1 for gt in gt_boxes if iou(gt, prediction) >= TOUCH_IOU
    )

    area_ratio = prediction["area"] / target["area"] if target["area"] else 0.0

    claimed = any(
        pi == best_pi and gi != gt_index for gi, pi in matched.items()
    )

    if touched >= 2 and area_ratio >= MERGE_AREA_RATIO:
        return "merged", best_iou, touched, area_ratio

    if claimed:
        return "stolen", best_iou, touched, area_ratio

    if touched >= 2:
        return "merged", best_iou, touched, area_ratio

    if area_ratio < UNDERSIZE_AREA_RATIO:
        return "undersized", best_iou, touched, area_ratio

    return "offset", best_iou, touched, area_ratio


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        required=True,
        help="Prediction directory name, e.g. MT-008-test-analysis",
    )
    parser.add_argument("--label", default=None, help="Display name")
    parser.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()

    name = args.label or args.labels.split("-test")[0]

    label_dir = find_label_dir(args.labels)

    print(f"{name}: {label_dir}")

    gt_total = 0
    pred_total = 0
    matched_total = 0
    unmatched_total = 0

    images_with_pred = 0
    zero_pred_images = 0
    blind_images = 0
    blind_persons = 0

    small_gt = 0
    small_matched = 0

    mechanisms = Counter()
    rows = []

    for gt_path in sorted(GT_DIR.glob("*.txt")):
        image_path = IMAGE_DIR / f"{gt_path.stem}.jpg"

        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        gt_boxes = load_boxes(gt_path, width, height)
        pred_boxes = load_boxes(label_dir / f"{gt_path.stem}.txt", width, height)

        if pred_boxes:
            images_with_pred += 1
        else:
            zero_pred_images += 1

            if gt_boxes:
                blind_images += 1
                blind_persons += len(gt_boxes)

        matched, used_pred = greedy_match(gt_boxes, pred_boxes)

        gt_total += len(gt_boxes)
        pred_total += len(pred_boxes)
        matched_total += len(matched)
        unmatched_total += len(pred_boxes) - len(used_pred)

        for gi, gt in enumerate(gt_boxes):
            is_small = gt["w"] < SMALL_PX or gt["h"] < SMALL_PX

            if is_small:
                small_gt += 1

                if gi in matched:
                    small_matched += 1

            if gi in matched:
                continue

            mechanism, best_iou, touched, area_ratio = classify(
                gi, gt_boxes, pred_boxes, matched
            )

            mechanisms[mechanism] += 1

            rows.append({
                "image": gt_path.stem,
                "gt_index": gi,
                "mechanism": mechanism,
                "best_iou": round(best_iou, 4),
                "touched_gt": touched,
                "area_ratio": round(area_ratio, 3),
                "width_px": round(gt["w"], 1),
                "height_px": round(gt["h"], 1),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slug = f"{name}_conf{args.conf:.2f}".replace(".", "")

    csv_path = OUTPUT_DIR / f"{slug}_misses.csv"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "experiment": name,
        "labels": str(label_dir),
        "confidence": args.conf,
        "iou_threshold": IOU_THRESHOLD,
        "ground_truth_persons": gt_total,
        "predicted_boxes": pred_total,
        "matched": matched_total,
        "missed": gt_total - matched_total,
        "unmatched": unmatched_total,
        "recall": round(matched_total / gt_total, 4) if gt_total else None,
        "custom_precision": (
            round(matched_total / pred_total, 4) if pred_total else None
        ),
        "images_with_predictions": images_with_pred,
        "zero_prediction_images": zero_pred_images,
        "blind_images": blind_images,
        "blind_persons": blind_persons,
        "small_gt": small_gt,
        "small_matched": small_matched,
        "small_recall": (
            round(small_matched / small_gt, 4) if small_gt else None
        ),
        "mechanisms": dict(mechanisms),
    }

    (OUTPUT_DIR / f"{slug}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def delta(value, base, higher_is_better=True):
        if base is None:
            return ""

        diff = value - base
        good = (diff >= 0) if higher_is_better else (diff <= 0)

        sign = "+" if diff > 0 else ""

        return f"  {sign}{diff:g} {'OK' if good else '<-- worse'}"

    print()
    print("=" * 72)
    print(f"{name} vs frozen MT-005 baseline  "
          f"(confidence {args.conf}, IoU >= {IOU_THRESHOLD})")
    print("=" * 72)
    print(f"{'Metric':28s} {'baseline':>10s} {name:>12s}   delta")
    print("-" * 72)

    comparisons = [
        ("Matched persons", "matched", summary["matched"], True),
        ("Missed persons", "missed", summary["missed"], False),
        ("Unmatched boxes", "unmatched", summary["unmatched"], False),
        ("Blind images", "blind_images", summary["blind_images"], False),
    ]

    for title, key, value, higher in comparisons:
        base = BASELINE.get(key)
        print(
            f"{title:28s} {base:10d} {value:12d}"
            f"{delta(value, base, higher)}"
        )

    for title, key, value in [
        ("Recall", "recall", summary["recall"]),
        ("Custom precision", "custom_precision", summary["custom_precision"]),
        ("Small-person recall", "small_recall", summary["small_recall"]),
    ]:
        base = BASELINE.get(key)
        print(
            f"{title:28s} {base:10.4f} {value:12.4f}"
            f"{delta(value, base, True)}"
        )

    print()
    print("=" * 72)
    print("Failure mechanisms among this model's own misses")
    print("=" * 72)

    total_missed = summary["missed"]

    for mechanism in (
        "merged", "undersized", "offset", "stolen", "no_overlap"
    ):
        count = mechanisms.get(mechanism, 0)
        share = count / total_missed * 100 if total_missed else 0

        marker = "  <-- primary MT-008 target" if mechanism == "merged" else ""

        print(f"  {mechanism:12s} {count:5d}  ({share:5.1f}%){marker}")

    print()
    print(f"Written: {OUTPUT_DIR / f'{slug}_summary.json'}")

    if rows:
        print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
