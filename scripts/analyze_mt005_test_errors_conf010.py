"""
MT-005 Thermal Test-Set Error Analysis
Compares YOLO ground-truth labels against prediction labels.

Expected:
GT:    class x_center y_center width height
Pred:  class x_center y_center width height confidence

Default paths match the current Human_detection_UAV project.
"""

from pathlib import Path
import csv
import cv2

ROOT = Path(r"D:\7th sem\major project\chode\Human_detection_UAV")

IMAGE_DIR = ROOT / "dataset" / "hit_uav_person" / "images" / "test"
GT_DIR = ROOT / "dataset" / "hit_uav_person" / "labels" / "test"
PRED_DIR = ROOT / "runs" / "detect" / "results" / "error_analysis" / "MT-005-test-conf010" / "labels"

OUT_DIR = ROOT / "results" / "error_analysis" / "MT-005-test-conf010"
OUT_CSV = OUT_DIR / "comparison.csv"

IOU_THRESHOLD = 0.50


def read_labels(path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            try:
                cls = int(float(p[0]))
                xc, yc, w, h = map(float, p[1:5])
                conf = float(p[5]) if len(p) >= 6 else None
                rows.append((cls, xc, yc, w, h, conf))
            except ValueError:
                continue
    return rows


def yolo_to_xyxy(box, img_w, img_h):
    _, xc, yc, w, h, _ = box
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    return (
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2,
    )


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def size_bin(w_px, h_px):
    # Consistent with the later RGB standard-vs-tiled analysis:
    # small: width < 32 OR height < 32
    # medium: both >= 32 and at least one < 96
    # large: both >= 96
    if w_px < 32 or h_px < 32:
        return "small"
    if w_px < 96 or h_px < 96:
        return "medium"
    return "large"


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found:\n{IMAGE_DIR}")
    if not GT_DIR.exists():
        raise FileNotFoundError(f"Ground-truth directory not found:\n{GT_DIR}")
    if not PRED_DIR.exists():
        raise FileNotFoundError(f"Prediction directory not found:\n{PRED_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    total_gt = 0
    total_pred = 0
    total_matched = 0

    gt_images = 0
    pred_images = 0
    zero_pred_with_gt = []
    zero_pred_without_gt = []

    matched_by_size = {"small": 0, "medium": 0, "large": 0}
    gt_by_size = {"small": 0, "medium": 0, "large": 0}

    false_positive_boxes = 0
    false_negative_boxes = 0

    rows = []

    for img_path in image_paths:
        gt_path = GT_DIR / f"{img_path.stem}.txt"
        pred_path = PRED_DIR / f"{img_path.stem}.txt"

        gt = [x for x in read_labels(gt_path) if x[0] == 0]
        pred = [x for x in read_labels(pred_path) if x[0] == 0]

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"WARNING: could not read image: {img_path.name}")
            continue

        img_h, img_w = image.shape[:2]

        gt_boxes = [yolo_to_xyxy(x, img_w, img_h) for x in gt]
        pred_boxes = [yolo_to_xyxy(x, img_w, img_h) for x in pred]

        total_gt += len(gt)
        total_pred += len(pred)

        if gt:
            gt_images += 1
        if pred:
            pred_images += 1

        for g in gt:
            _, _, _, nw, nh, _ = g
            w_px = nw * img_w
            h_px = nh * img_h
            gt_by_size[size_bin(w_px, h_px)] += 1

        # Greedy one-to-one matching using highest IoU first.
        candidates = []
        for gi, gb in enumerate(gt_boxes):
            for pi, pb in enumerate(pred_boxes):
                score = iou(gb, pb)
                if score >= IOU_THRESHOLD:
                    candidates.append((score, gi, pi))

        candidates.sort(reverse=True)

        used_gt = set()
        used_pred = set()
        matches = []

        for score, gi, pi in candidates:
            if gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append((gi, pi, score))

        matched = len(matches)
        total_matched += matched

        unmatched_gt = len(gt) - matched
        unmatched_pred = len(pred) - matched
        false_negative_boxes += unmatched_gt
        false_positive_boxes += unmatched_pred

        if gt and not pred:
            zero_pred_with_gt.append((img_path.name, len(gt)))
        elif not gt and not pred:
            zero_pred_without_gt.append(img_path.name)

        for gi, pi, score in matches:
            g = gt[gi]
            _, _, _, nw, nh, _ = g
            w_px = nw * img_w
            h_px = nh * img_h
            matched_by_size[size_bin(w_px, h_px)] += 1

        rows.append({
            "image": img_path.name,
            "width": img_w,
            "height": img_h,
            "gt_persons": len(gt),
            "pred_persons": len(pred),
            "matched_iou50": matched,
            "missed_gt": unmatched_gt,
            "false_positive_pred": unmatched_pred,
        })

    print("\n=== MT-005 THERMAL TEST-SET ERROR ANALYSIS ===")
    print(f"Test images:                 {len(image_paths)}")
    print(f"Images with GT persons:      {gt_images}")
    print(f"Images with predictions:     {pred_images}")
    print(f"Images with zero predictions:{len(image_paths) - pred_images}")
    print(f"GT persons:                  {total_gt}")
    print(f"Predicted boxes:             {total_pred}")
    print(f"Matched persons (IoU>=0.50): {total_matched}")
    print(f"Missed GT persons:           {false_negative_boxes}")
    print(f"Unmatched prediction boxes:  {false_positive_boxes}")
    print(f"Overall matched ratio:       {total_matched / total_gt:.4f}" if total_gt else "Overall matched ratio:       N/A")

    print("\n--- ZERO-DETECTION IMAGES ---")
    print(f"Zero detections with GT persons: {len(zero_pred_with_gt)}")
    print(f"Zero detections with no persons: {len(zero_pred_without_gt)}")

    if zero_pred_with_gt:
        print("\nActual complete-failure images:")
        for name, n in zero_pred_with_gt:
            print(f"  {name}  -> {n} GT person(s)")

    print("\n--- SIZE ANALYSIS ---")
    for s in ("small", "medium", "large"):
        gt_n = gt_by_size[s]
        m_n = matched_by_size[s]
        recall = m_n / gt_n if gt_n else 0.0
        print(f"{s.capitalize():8s}: GT={gt_n:4d}  Matched={m_n:4d}  Recall={recall:.4f}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
            "image", "width", "height", "gt_persons", "pred_persons",
            "matched_iou50", "missed_gt", "false_positive_pred"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-image comparison saved to:\n{OUT_CSV}")


if __name__ == "__main__":
    main()
