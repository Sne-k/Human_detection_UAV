from pathlib import Path
import cv2

ROOT = Path(r"D:\7th sem\major project\chode\Human_detection_UAV")

IMAGE_DIR = ROOT / "dataset" / "hit_uav_person" / "images" / "test"
GT_DIR = ROOT / "dataset" / "hit_uav_person" / "labels" / "test"

PRED25 = ROOT / "runs" / "detect" / "results" / "error_analysis" / "MT-005-test-analysis" / "labels"
PRED10 = ROOT / "runs" / "detect" / "results" / "error_analysis" / "MT-005-test-conf010" / "labels"

IOU_THRESHOLD = 0.50


def read_labels(path):
    result = []

    if not path.exists():
        return result

    for line in path.read_text().splitlines():
        p = line.split()

        if len(p) >= 5:
            cls = int(float(p[0]))
            xc, yc, w, h = map(float, p[1:5])
            conf = float(p[5]) if len(p) >= 6 else None

            if cls == 0:
                result.append((cls, xc, yc, w, h, conf))

    return result


def box_xyxy(box, W, H):
    _, xc, yc, w, h, _ = box

    xc *= W
    yc *= H
    w *= W
    h *= H

    return (
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2
    )


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    intersection = iw * ih

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


remaining = []

for image in sorted(IMAGE_DIR.iterdir()):

    if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    gt = read_labels(GT_DIR / f"{image.stem}.txt")
    pred25 = read_labels(PRED25 / f"{image.stem}.txt")
    pred10 = read_labels(PRED10 / f"{image.stem}.txt")

    if not gt:
        continue

    img = cv2.imread(str(image))

    if img is None:
        continue

    H, W = img.shape[:2]

    gt_boxes = [box_xyxy(x, W, H) for x in gt]
    pred25_boxes = [box_xyxy(x, W, H) for x in pred25]
    pred10_boxes = [box_xyxy(x, W, H) for x in pred10]

    for gi, gt_box in enumerate(gt_boxes):

        best25 = max(
            [iou(gt_box, p) for p in pred25_boxes],
            default=0.0
        )

        best10 = max(
            [iou(gt_box, p) for p in pred10_boxes],
            default=0.0
        )

        # Already matched at 0.25
        if best25 >= IOU_THRESHOLD:
            continue

        # Recovered at 0.10
        if best10 >= IOU_THRESHOLD:
            continue

        _, _, _, nw, nh, _ = gt[gi]

        gt_w_px = nw * W
        gt_h_px = nh * H

        if best10 == 0:
            category = "NO_OVERLAPPING_PREDICTION"
        else:
            category = "PREDICTION_BUT_IOU_LT_0.50"

        remaining.append({
            "image": image.name,
            "gt_width": gt_w_px,
            "gt_height": gt_h_px,
            "best_iou_conf010": best10,
            "best_iou_conf025": best25,
            "category": category
        })


print("\n=== MT-005 REMAINING MISS ANALYSIS ===")

print(f"Remaining missed persons: {len(remaining)}")

no_overlap = [
    x for x in remaining
    if x["category"] == "NO_OVERLAPPING_PREDICTION"
]

poor_iou = [
    x for x in remaining
    if x["category"] == "PREDICTION_BUT_IOU_LT_0.50"
]

print(f"\nNo overlapping prediction: {len(no_overlap)}")
print(f"Prediction exists, IoU < 0.50: {len(poor_iou)}")

print("\n--- REMAINING MISS SIZE ---")

if remaining:
    widths = [x["gt_width"] for x in remaining]
    heights = [x["gt_height"] for x in remaining]

    print(f"Average width:  {sum(widths)/len(widths):.2f} px")
    print(f"Average height: {sum(heights)/len(heights):.2f} px")
    print(f"Minimum width:  {min(widths):.2f} px")
    print(f"Minimum height: {min(heights):.2f} px")
    print(f"Maximum width:  {max(widths):.2f} px")
    print(f"Maximum height: {max(heights):.2f} px")

print("\n--- NO-OVERLAP EXAMPLES ---")

for x in no_overlap[:30]:
    print(
        f"{x['image']} | "
        f"GT={x['gt_width']:.1f}x{x['gt_height']:.1f}px | "
        f"best IoU @0.10={x['best_iou_conf010']:.3f}"
    )

print("\n--- LOW-IOU EXAMPLES ---")

for x in poor_iou[:30]:
    print(
        f"{x['image']} | "
        f"GT={x['gt_width']:.1f}x{x['gt_height']:.1f}px | "
        f"best IoU @0.10={x['best_iou_conf010']:.3f}"
    )