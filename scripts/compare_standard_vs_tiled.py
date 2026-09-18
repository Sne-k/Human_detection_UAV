from pathlib import Path
from PIL import Image

# =========================================================
# Configuration
# =========================================================

VAL_IMAGES = Path("dataset/visdrone_person/images/val")
GT_LABELS = Path("dataset/visdrone_person/labels/val")

STANDARD_LABELS = Path(
    "runs/detect/runs/detect/results/error_analysis/MT-004-final/labels"
)

TILED_LABELS = Path(
    "runs/detect/results/error_analysis/MT-004-tiled-corrected/labels"
)

IOU_THRESHOLD = 0.50


# =========================================================
# Utilities
# =========================================================

def load_yolo_labels(path):
    """
    Read YOLO labels.

    GT format:
        class xc yc w h

    Prediction format:
        class xc yc w h confidence
    """

    boxes = []

    if not path.exists():
        return boxes

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            confidence = (
                float(parts[5])
                if len(parts) >= 6
                else None
            )

            boxes.append({
                "cls": cls,
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
                "conf": confidence
            })

    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    xc = box["xc"] * img_w
    yc = box["yc"] * img_h
    w = box["w"] * img_w
    h = box["h"] * img_h

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return [x1, y1, x2, y2]


def iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def greedy_match(gt_boxes, pred_boxes):

    matches = []

    used_predictions = set()

    # Sort possible matches by IoU descending
    candidates = []

    for gi, gt in enumerate(gt_boxes):

        for pi, pred in enumerate(pred_boxes):

            if pred["cls"] != gt["cls"]:
                continue

            value = iou(gt["xyxy"], pred["xyxy"])

            if value >= IOU_THRESHOLD:
                candidates.append((value, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()

    for value, gi, pi in candidates:

        if gi in matched_gt:
            continue

        if pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)

        matches.append((gi, pi, value))

    return matches


def size_category(box):

    """
    Consistent size definition based on BOTH
    width and height in the ORIGINAL image.

    Small:
        width < 32 OR height < 32

    Medium:
        width >= 32 AND height >= 32
        and either dimension < 96

    Large:
        width >= 96 AND height >= 96
    """

    w = box["w_px"]
    h = box["h_px"]

    if w < 32 or h < 32:
        return "small"

    if w < 96 or h < 96:
        return "medium"

    return "large"


# =========================================================
# Main evaluation
# =========================================================

image_files = sorted(
    p for p in VAL_IMAGES.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

print("=" * 70)
print("MT-004 STANDARD vs TILED EVALUATION")
print("=" * 70)
print(f"Validation images: {len(image_files)}")
print(f"IoU threshold: {IOU_THRESHOLD}")
print()

results = {
    "standard": {
        "gt": 0,
        "matched": 0,
        "predictions": 0,
        "images_with_gt": 0,
        "zero_detection_images": 0,
        "small_gt": 0,
        "small_matched": 0,
        "medium_gt": 0,
        "medium_matched": 0,
        "large_gt": 0,
        "large_matched": 0,
    },

    "tiled": {
        "gt": 0,
        "matched": 0,
        "predictions": 0,
        "images_with_gt": 0,
        "zero_detection_images": 0,
        "small_gt": 0,
        "small_matched": 0,
        "medium_gt": 0,
        "medium_matched": 0,
        "large_gt": 0,
        "large_matched": 0,
    }
}


# ---------------------------------------------------------
# Evaluate both methods
# ---------------------------------------------------------

for index, image_path in enumerate(image_files, start=1):

    stem = image_path.stem

    gt_path = GT_LABELS / f"{stem}.txt"
    standard_path = STANDARD_LABELS / f"{stem}.txt"
    tiled_path = TILED_LABELS / f"{stem}.txt"

    with Image.open(image_path) as im:
        img_w, img_h = im.size

    gt_raw = load_yolo_labels(gt_path)
    standard_raw = load_yolo_labels(standard_path)
    tiled_raw = load_yolo_labels(tiled_path)

    gt_boxes = []

    for gt in gt_raw:

        xyxy = yolo_to_xyxy(gt, img_w, img_h)

        gt_boxes.append({
            **gt,
            "xyxy": xyxy,
            "w_px": gt["w"] * img_w,
            "h_px": gt["h"] * img_h,
        })

    for method, pred_raw in [
        ("standard", standard_raw),
        ("tiled", tiled_raw)
    ]:

        pred_boxes = []

        for pred in pred_raw:

            xyxy = yolo_to_xyxy(pred, img_w, img_h)

            pred_boxes.append({
                **pred,
                "xyxy": xyxy
            })

        r = results[method]

        r["gt"] += len(gt_boxes)
        r["predictions"] += len(pred_boxes)

        if len(gt_boxes) > 0:
            r["images_with_gt"] += 1

        if len(pred_boxes) == 0:
            r["zero_detection_images"] += 1

        matches = greedy_match(gt_boxes, pred_boxes)

        r["matched"] += len(matches)

        matched_gt_indices = {
            gi for gi, pi, value in matches
        }

        for gi, gt in enumerate(gt_boxes):

            category = size_category(gt)

            r[f"{category}_gt"] += 1

            if gi in matched_gt_indices:
                r[f"{category}_matched"] += 1

    if index % 50 == 0:
        print(f"Processed {index}/{len(image_files)} images")


# =========================================================
# Results
# =========================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

for method in ["standard", "tiled"]:

    r = results[method]

    overall = (
        r["matched"] / r["gt"]
        if r["gt"] > 0
        else 0
    )

    small = (
        r["small_matched"] / r["small_gt"]
        if r["small_gt"] > 0
        else 0
    )

    medium = (
        r["medium_matched"] / r["medium_gt"]
        if r["medium_gt"] > 0
        else 0
    )

    large = (
        r["large_matched"] / r["large_gt"]
        if r["large_gt"] > 0
        else 0
    )

    print()
    print(f"--- {method.upper()} ---")

    print(f"Ground-truth persons:       {r['gt']}")
    print(f"Predicted boxes:            {r['predictions']}")
    print(f"Matched persons:            {r['matched']}")
    print(f"Overall matched ratio:      {overall:.4f}")

    print(
        f"Images with GT:             "
        f"{r['images_with_gt']}"
    )

    print(
        f"Zero-detection images:      "
        f"{r['zero_detection_images']}"
    )

    print()
    print(
        f"Small:   GT={r['small_gt']:5d} "
        f"Matched={r['small_matched']:5d} "
        f"Recall={small:.4f}"
    )

    print(
        f"Medium:  GT={r['medium_gt']:5d} "
        f"Matched={r['medium_matched']:5d} "
        f"Recall={medium:.4f}"
    )

    print(
        f"Large:   GT={r['large_gt']:5d} "
        f"Matched={r['large_matched']:5d} "
        f"Recall={large:.4f}"
    )


# =========================================================
# Direct comparison
# =========================================================

s = results["standard"]
t = results["tiled"]

print()
print("=" * 70)
print("TILED - STANDARD")
print("=" * 70)

print(
    f"Matched persons: "
    f"{t['matched'] - s['matched']:+d}"
)

print(
    f"Predicted boxes: "
    f"{t['predictions'] - s['predictions']:+d}"
)

print(
    f"Zero-detection images: "
    f"{t['zero_detection_images'] - s['zero_detection_images']:+d}"
)

print(
    f"Small matched: "
    f"{t['small_matched'] - s['small_matched']:+d}"
)

print(
    f"Medium matched: "
    f"{t['medium_matched'] - s['medium_matched']:+d}"
)

print(
    f"Large matched: "
    f"{t['large_matched'] - s['large_matched']:+d}"
)

print()
print("Evaluation complete.")