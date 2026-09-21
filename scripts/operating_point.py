
"""
Choose the deployment confidence threshold from the mission, not convention.

Every number in this project so far uses confidence 0.25, which is the
Ultralytics default. Nothing about search and rescue makes 0.25 correct.

The right threshold depends on what the two error types actually cost:

  A missed person  - the aircraft flies over a casualty and reports nothing.
                     In a search mission this can be fatal, and there may be
                     no second pass over that ground.

  A false positive - a rescue team is directed to a spot with nobody there.
                     Costs time, fuel and attention, all of which are scarce
                     during a response, but nobody dies of it.

These are not equal, and treating them as equal - which is what optimising
F1 does - is the wrong objective for this payload. This script scores
thresholds under an explicit cost ratio instead:

    cost = (missed persons x MISS_COST) + (false positives x FP_COST)

and reports the threshold that minimises it, along with how sensitive that
choice is to the ratio. The ratio is a mission decision, not a technical one,
so the script shows a range rather than asserting a single answer.

It reads the prediction label files already generated at each threshold, so it
needs no GPU and no re-inference.

Usage:

    python scripts/operating_point.py
    python scripts/operating_point.py --miss-cost 50
"""

import argparse
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

OUTPUT_DIR = PROJECT / "results" / "operating_point"

IOU_THRESHOLD = 0.50

# The conf-0.10 prediction run stores every detection down to 0.10, so any
# threshold at or above that can be evaluated by filtering it. No re-inference
# is needed.
SOURCE_DIR = "MT-005-test-conf010"

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]

# How many false positives one missed person is worth. The default of 20 is a
# deliberately conservative reading of "a missed casualty is much worse than a
# wasted search", not a derived constant.
DEFAULT_MISS_COST = 20.0


def find_labels(name):
    for root in PRED_ROOTS:
        path = root / name / "labels"

        if path.exists():
            return path

    raise SystemExit(f"Predictions not found: {name}/labels")


def load_boxes(path, width, height, min_conf=0.0):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        conf = float(parts[5]) if len(parts) >= 6 else 1.0

        if conf < min_conf:
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

    inter = iw * ih

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])

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


def evaluate(label_dir, threshold):
    gt_total = 0
    matched_total = 0
    unmatched_total = 0
    blind = 0

    for gt_path in sorted(GT_DIR.glob("*.txt")):
        image_path = IMAGE_DIR / f"{gt_path.stem}.jpg"

        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        gt = load_boxes(gt_path, width, height)
        preds = load_boxes(
            label_dir / f"{gt_path.stem}.txt", width, height, threshold
        )

        matched, unmatched = greedy_match(gt, preds)

        gt_total += len(gt)
        matched_total += matched
        unmatched_total += unmatched

        if gt and not preds:
            blind += 1

    return {
        "threshold": threshold,
        "gt": gt_total,
        "matched": matched_total,
        "missed": gt_total - matched_total,
        "unmatched": unmatched_total,
        "recall": round(matched_total / gt_total, 4) if gt_total else 0.0,
        "blind_images": blind,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--miss-cost",
        type=float,
        default=DEFAULT_MISS_COST,
        help="False positives one missed person is worth",
    )
    parser.add_argument("--labels", default=SOURCE_DIR)

    args = parser.parse_args()

    label_dir = find_labels(args.labels)

    print(f"Predictions: {label_dir}")
    print(f"Thresholds evaluated by filtering a single conf-0.10 run.")
    print()

    rows = [evaluate(label_dir, t) for t in THRESHOLDS]

    header = (
        f"{'conf':>6s} {'matched':>8s} {'missed':>7s} {'unmatched':>10s} "
        f"{'recall':>8s} {'blind':>6s}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['threshold']:6.2f} {row['matched']:8d} {row['missed']:7d} "
            f"{row['unmatched']:10d} {row['recall']:8.4f} "
            f"{row['blind_images']:6d}"
        )

    # ------------------------------------------------------------------
    # Cost under several mission weightings
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Optimal threshold vs how much a missed person costs")
    print("=" * 70)
    print(f"{'miss cost':>10s} {'best conf':>10s} {'matched':>9s} "
          f"{'missed':>8s} {'unmatched':>11s}")
    print("-" * 70)

    sweep = {}

    for miss_cost in (1, 5, 10, 20, 50, 100):
        best = min(
            rows,
            key=lambda r: r["missed"] * miss_cost + r["unmatched"],
        )

        sweep[miss_cost] = best["threshold"]

        print(
            f"{miss_cost:10.0f} {best['threshold']:10.2f} "
            f"{best['matched']:9d} {best['missed']:8d} "
            f"{best['unmatched']:11d}"
        )

    chosen = min(
        rows,
        key=lambda r: r["missed"] * args.miss_cost + r["unmatched"],
    )

    baseline = next(r for r in rows if abs(r["threshold"] - 0.25) < 1e-6)

    print()
    print("=" * 70)
    print(f"At the stated cost ratio of {args.miss_cost:.0f}:1")
    print("=" * 70)
    print(f"  recommended threshold : {chosen['threshold']:.2f}")
    print(f"  current default       : 0.25")
    print()
    print(f"{'':22s} {'conf 0.25':>12s} {'recommended':>13s} {'change':>9s}")
    print("-" * 60)

    for label, key, better_is_low in [
        ("matched persons", "matched", False),
        ("missed persons", "missed", True),
        ("unmatched boxes", "unmatched", True),
        ("blind images", "blind_images", True),
    ]:
        delta = chosen[key] - baseline[key]
        print(
            f"  {label:20s} {baseline[key]:12d} {chosen[key]:13d} "
            f"{delta:+9d}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "operating_point.json").write_text(
        json.dumps(
            {
                "iou_threshold": IOU_THRESHOLD,
                "rows": rows,
                "optimum_by_miss_cost": sweep,
                "stated_miss_cost": args.miss_cost,
                "recommended": chosen["threshold"],
            },
            indent=2,
        )
    )

    print()
    print("The cost ratio is a mission decision, not a technical one. The")
    print("sweep above shows how much the answer actually moves with it.")
    print()
    print(f"Written: {OUTPUT_DIR / 'operating_point.json'}")


if __name__ == "__main__":
    main()
