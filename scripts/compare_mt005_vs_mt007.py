from pathlib import Path
from PIL import Image
import csv
import re

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

GT_DIR = PROJECT / "dataset" / "hit_uav_person" / "labels" / "test"
IMAGE_DIR = PROJECT / "dataset" / "hit_uav_person" / "images" / "test"

# The outputs may be under runs/detect/... depending on where
# Ultralytics created them.
PRED_ROOTS = [
    PROJECT / "runs" / "detect" / "results" / "error_analysis",
    PROJECT / "results" / "error_analysis",
]

IOU_THRESHOLD = 0.50

OUTPUT_DIR = PROJECT / "results" / "error_analysis" / "MT005_vs_MT007"
OUTPUT_CSV = OUTPUT_DIR / "person_comparison.csv"


# ============================================================
# HELPERS
# ============================================================

def find_label_dir(model_name):
    candidates = [
        root / f"{model_name}-test-analysis" / "labels"
        for root in PRED_ROOTS
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find prediction labels for {model_name}.\n"
        f"Checked:\n" +
        "\n".join(str(p) for p in candidates)
    )


def load_yolo_labels(path, image_w, image_h):
    """
    YOLO format:
    class x_center y_center width height [confidence]
    """

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

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            boxes.append({
                "cls": cls,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
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

    area_a = max(0.0, a["x2"] - a["x1"]) * max(
        0.0, a["y2"] - a["y1"]
    )

    area_b = max(0.0, b["x2"] - b["x1"]) * max(
        0.0, b["y2"] - b["y1"]
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def greedy_match(gt_boxes, pred_boxes):
    """
    One-to-one greedy matching.

    Each GT person can match at most one prediction.
    Each prediction can match at most one GT.

    Matching threshold: IoU >= 0.50
    """

    matches = []

    candidates = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            score = iou(gt, pred)

            if score >= IOU_THRESHOLD:
                candidates.append((score, gi, pi))

    # Highest IoU first
    candidates.sort(reverse=True)

    used_gt = set()
    used_pred = set()

    for score, gi, pi in candidates:

        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

        matches.append({
            "gt_index": gi,
            "pred_index": pi,
            "iou": score,
        })

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mt005_dir = find_label_dir("MT-005")
    mt007_dir = find_label_dir("MT-007")

    print("MT-005 labels:")
    print(mt005_dir)

    print("\nMT-007 labels:")
    print(mt007_dir)

    gt_files = sorted(GT_DIR.glob("*.txt"))

    print(f"\nGT label files: {len(gt_files)}")

    rows = []

    # Aggregate counters
    totals = {
        "both_matched": 0,
        "mt005_only": 0,
        "mt007_only": 0,
        "both_missed": 0,
    }

    for gt_path in gt_files:

        image_path = IMAGE_DIR / f"{gt_path.stem}.jpg"

        if not image_path.exists():
            image_path = IMAGE_DIR / f"{gt_path.stem}.png"

        if not image_path.exists():
            print(f"WARNING: image missing: {gt_path.name}")
            continue

        with Image.open(image_path) as img:
            image_w, image_h = img.size

        gt_boxes = load_yolo_labels(
            gt_path,
            image_w,
            image_h
        )

        mt005_path = mt005_dir / gt_path.name
        mt007_path = mt007_dir / gt_path.name

        mt005_boxes = load_yolo_labels(
            mt005_path,
            image_w,
            image_h
        )

        mt007_boxes = load_yolo_labels(
            mt007_path,
            image_w,
            image_h
        )

        mt005_matches = greedy_match(
            gt_boxes,
            mt005_boxes
        )

        mt007_matches = greedy_match(
            gt_boxes,
            mt007_boxes
        )

        mt005_by_gt = {
            m["gt_index"]: m
            for m in mt005_matches
        }

        mt007_by_gt = {
            m["gt_index"]: m
            for m in mt007_matches
        }

        # Analyze every GT person individually
        for gi, gt in enumerate(gt_boxes):

            m5 = mt005_by_gt.get(gi)
            m7 = mt007_by_gt.get(gi)

            if m5 is not None and m7 is not None:
                category = "both_matched"
                totals["both_matched"] += 1

            elif m5 is not None and m7 is None:
                category = "mt005_only"
                totals["mt005_only"] += 1

            elif m5 is None and m7 is not None:
                category = "mt007_only"
                totals["mt007_only"] += 1

            else:
                category = "both_missed"
                totals["both_missed"] += 1

            row = {
                "image": gt_path.stem,
                "gt_index": gi,

                "image_width": image_w,
                "image_height": image_h,

                "gt_width_px": round(gt["w"], 3),
                "gt_height_px": round(gt["h"], 3),
                "gt_area_px2": round(gt["w"] * gt["h"], 3),

                "category": category,

                "mt005_matched": int(m5 is not None),
                "mt005_iou": round(m5["iou"], 4) if m5 else "",
                "mt005_conf": (
                    round(mt005_boxes[m5["pred_index"]]["conf"], 4)
                    if m5 and mt005_boxes[m5["pred_index"]]["conf"] is not None
                    else ""
                ),

                "mt007_matched": int(m7 is not None),
                "mt007_iou": round(m7["iou"], 4) if m7 else "",
                "mt007_conf": (
                    round(mt007_boxes[m7["pred_index"]]["conf"], 4)
                    if m7 and mt007_boxes[m7["pred_index"]]["conf"] is not None
                    else ""
                ),
            }

            rows.append(row)

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "image",
        "gt_index",
        "image_width",
        "image_height",
        "gt_width_px",
        "gt_height_px",
        "gt_area_px2",
        "category",

        "mt005_matched",
        "mt005_iou",
        "mt005_conf",

        "mt007_matched",
        "mt007_iou",
        "mt007_conf",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    total_gt = len(rows)

    print("\n" + "=" * 60)
    print("MT-005 vs MT-007 PERSON-LEVEL COMPARISON")
    print("=" * 60)

    print(f"GT persons analyzed: {total_gt}")

    print(
        f"\nBoth models matched : "
        f"{totals['both_matched']}"
    )

    print(
        f"MT-005 only         : "
        f"{totals['mt005_only']}"
    )

    print(
        f"MT-007 only         : "
        f"{totals['mt007_only']}"
    )

    print(
        f"Both missed         : "
        f"{totals['both_missed']}"
    )

    if total_gt > 0:

        print("\nPercentages:")

        for key in [
            "both_matched",
            "mt005_only",
            "mt007_only",
            "both_missed",
        ]:

            value = totals[key]
            pct = value / total_gt * 100

            print(
                f"{key:20s}: "
                f"{value:4d} "
                f"({pct:6.2f}%)"
            )

    # ========================================================
    # SIZE ANALYSIS
    # ========================================================

    categories = [
        "both_matched",
        "mt005_only",
        "mt007_only",
        "both_missed",
    ]

    print("\nAverage GT size by category:")

    for category in categories:

        subset = [
            r for r in rows
            if r["category"] == category
        ]

        if not subset:
            print(f"{category:20s}: no samples")
            continue

        avg_w = sum(
            r["gt_width_px"]
            for r in subset
        ) / len(subset)

        avg_h = sum(
            r["gt_height_px"]
            for r in subset
        ) / len(subset)

        avg_area = sum(
            r["gt_area_px2"]
            for r in subset
        ) / len(subset)

        print(
            f"{category:20s}: "
            f"N={len(subset):4d}, "
            f"W={avg_w:6.2f}px, "
            f"H={avg_h:6.2f}px, "
            f"Area={avg_area:8.2f}px²"
        )

    print("\nCSV saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()