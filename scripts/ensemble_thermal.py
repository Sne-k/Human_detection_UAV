"""
Fuse the thermal models' predictions and measure the result.

The per-person comparison showed that MT-005, MT-006 and MT-007 fail on
different people: the union of what they find covers 96.09% of the test-set
persons, against 92.88% for the best single model. That union is an oracle
bound, not a system - it assumes the fused boxes can be produced without also
inheriting every model's false positives.

This script measures the real thing. It concatenates the three models'
predictions, fuses overlapping boxes, and scores the result with the same
person-level matching used everywhere else in this project, so the numbers sit
directly beside the single-model rows.

Two fusion strategies are implemented:

  nms  Keep the highest-scoring box in each cluster of overlapping boxes.
       Simple, and preserves one model's box geometry exactly.

  wbf  Weighted box fusion: average the cluster's coordinates weighted by
       confidence. Costs nothing extra and tends to help precisely where these
       models fail, because averaging several near-miss boxes recovers a
       better-centred box than any single one of them.

The second is worth testing specifically because 78% of the shared failures are
localisation failures with best-overlap IoU between 0.25 and 0.50 - boxes that
are close but not close enough. Averaging independent near-misses is exactly
the situation where fusion can cross the IoU 0.50 line.

A vote threshold is also supported: requiring a box to be seen by at least two
models trades recall for precision, which is the knob that decides whether an
ensemble is better than simply lowering a single model's confidence.

Usage:

    python scripts/ensemble_thermal.py
    python scripts/ensemble_thermal.py --method wbf --min-votes 2
    python scripts/ensemble_thermal.py --models MT-005 MT-006
"""

import argparse
import csv
import itertools
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

PRED_DIRS = {
    ("MT-005", "0.25"): "MT-005-test-analysis",
    ("MT-005", "0.10"): "MT-005-test-conf010",
    ("MT-006", "0.25"): "MT-006-test-analysis",
    ("MT-007", "0.25"): "MT-007-test-analysis",
    ("MT-007", "0.10"): "MT-007-test-conf010",
}

IOU_THRESHOLD = 0.50     # person-level matching
FUSE_IOU = 0.55          # clustering threshold for fusion

OUTPUT_DIR = PROJECT / "results" / "error_analysis" / "ensemble"


def conf_key(value):
    return f"{value:.2f}"


def find_label_dir(model, confidence):
    key = (model, conf_key(confidence))

    if key not in PRED_DIRS:
        raise SystemExit(
            f"No predictions registered for {model} @ {conf_key(confidence)}"
        )

    for root in PRED_ROOTS:
        path = root / PRED_DIRS[key] / "labels"

        if path.exists():
            return path

    raise SystemExit(f"Missing: {PRED_DIRS[key]}/labels")


def load_boxes(path, image_w, image_h, source=None):
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
            "w": w,
            "h": h,
            "conf": float(parts[5]) if len(parts) >= 6 else 1.0,
            "source": source,
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


def cluster(boxes, threshold):
    """Greedy clustering: seed on the highest-confidence box."""

    remaining = sorted(boxes, key=lambda b: -b["conf"])

    clusters = []

    while remaining:
        seed = remaining.pop(0)

        group = [seed]
        leftover = []

        for box in remaining:
            if iou(seed, box) >= threshold:
                group.append(box)
            else:
                leftover.append(box)

        clusters.append(group)
        remaining = leftover

    return clusters


def fuse(boxes, method, fuse_iou, min_votes):
    fused = []

    for group in cluster(boxes, fuse_iou):
        votes = len({b["source"] for b in group})

        if votes < min_votes:
            continue

        if method == "nms":
            best = max(group, key=lambda b: b["conf"])

            result = dict(best)

        else:  # wbf
            weight = sum(b["conf"] for b in group) or 1.0

            result = {
                key: sum(b[key] * b["conf"] for b in group) / weight
                for key in ("x1", "y1", "x2", "y2")
            }

            # Confidence is the mean over the models that voted, so a box
            # found by one model of three is not scored like a unanimous one.
            result["conf"] = sum(b["conf"] for b in group) / len(group)

        result["votes"] = votes
        result["cluster_size"] = len(group)

        fused.append(result)

    return fused


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

    return used_gt, used_pred


def evaluate(models, confidences, method, fuse_iou, min_votes):
    label_dirs = {
        model: find_label_dir(model, confidences[model])
        for model in models
    }

    stats = {
        "gt": 0,
        "pred": 0,
        "matched": 0,
        "unmatched_pred": 0,
        "images_with_pred": 0,
        "zero_pred_images": 0,
        "zero_pred_images_with_gt": 0,
        "zero_pred_gt_persons": 0,
    }

    for gt_path in sorted(GT_DIR.glob("*.txt")):
        image_path = IMAGE_DIR / f"{gt_path.stem}.jpg"

        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            image_w, image_h = image.size

        gt_boxes = load_boxes(gt_path, image_w, image_h)

        pooled = []

        for model, directory in label_dirs.items():
            pooled.extend(
                load_boxes(
                    directory / f"{gt_path.stem}.txt",
                    image_w, image_h,
                    source=model,
                )
            )

        fused = fuse(pooled, method, fuse_iou, min_votes)

        stats["gt"] += len(gt_boxes)
        stats["pred"] += len(fused)

        if fused:
            stats["images_with_pred"] += 1
        else:
            stats["zero_pred_images"] += 1

            if gt_boxes:
                stats["zero_pred_images_with_gt"] += 1
                stats["zero_pred_gt_persons"] += len(gt_boxes)

        matched, used_pred = greedy_match(gt_boxes, fused)

        stats["matched"] += len(matched)
        stats["unmatched_pred"] += len(fused) - len(used_pred)

    stats["missed"] = stats["gt"] - stats["matched"]
    stats["matched_ratio"] = (
        round(stats["matched"] / stats["gt"], 4) if stats["gt"] else 0.0
    )
    stats["precision"] = (
        round(stats["matched"] / stats["pred"], 4) if stats["pred"] else 0.0
    )

    return stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models", nargs="*", default=["MT-005", "MT-006", "MT-007"]
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--method", default=None, choices=["nms", "wbf"])
    parser.add_argument("--min-votes", type=int, default=None)
    parser.add_argument("--fuse-iou", type=float, default=FUSE_IOU)

    args = parser.parse_args()

    confidences = {model: args.conf for model in args.models}

    methods = [args.method] if args.method else ["nms", "wbf"]
    vote_levels = (
        [args.min_votes] if args.min_votes else list(range(1, len(args.models) + 1))
    )

    print(f"models: {', '.join(args.models)} @ conf {conf_key(args.conf)}")
    print(f"fusion IoU: {args.fuse_iou}   matching IoU: {IOU_THRESHOLD}")

    rows = []

    for method, votes in itertools.product(methods, vote_levels):
        stats = evaluate(
            args.models, confidences, method, args.fuse_iou, votes
        )

        stats["method"] = method
        stats["min_votes"] = votes

        rows.append(stats)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "ensemble_results.json").write_text(
        json.dumps(rows, indent=2)
    )

    with (OUTPUT_DIR / "ensemble_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    header = (
        f"{'method':6s} {'votes':>5s} {'pred':>7s} {'matched':>8s} "
        f"{'missed':>7s} {'unmatched':>10s} {'recall':>8s} {'prec':>7s} "
        f"{'blind imgs':>11s}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['method']:6s} {row['min_votes']:5d} {row['pred']:7d} "
            f"{row['matched']:8d} {row['missed']:7d} "
            f"{row['unmatched_pred']:10d} {row['matched_ratio']:8.4f} "
            f"{row['precision']:7.4f} {row['zero_pred_images_with_gt']:11d}"
        )

    print()
    print(f"Written: {OUTPUT_DIR / 'ensemble_results.csv'}")


if __name__ == "__main__":
    main()
