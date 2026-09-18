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
            result.append((cls, xc, yc, w, h, conf))

    return [x for x in result if x[0] == 0]


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

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)

    inter = iw * ih

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union if union else 0


recovered = []

for image in sorted(IMAGE_DIR.iterdir()):

    if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    gt = read_labels(GT_DIR / f"{image.stem}.txt")
    pred25 = read_labels(PRED25 / f"{image.stem}.txt")
    pred10 = read_labels(PRED10 / f"{image.stem}.txt")

    if not gt or not pred10:
        continue

    img = cv2.imread(str(image))
    if img is None:
        continue

    H, W = img.shape[:2]

    gt_boxes = [box_xyxy(x, W, H) for x in gt]
    p25_boxes = [box_xyxy(x, W, H) for x in pred25]
    p10_boxes = [box_xyxy(x, W, H) for x in pred10]

    for gi, gt_box in enumerate(gt_boxes):

        best25 = max(
            [iou(gt_box, p) for p in p25_boxes],
            default=0
        )

        if best25 >= IOU_THRESHOLD:
            continue

        candidates = []

        for pi, pred_box in enumerate(p10_boxes):
            score = iou(gt_box, pred_box)

            if score >= IOU_THRESHOLD:
                candidates.append((score, pi))

        if not candidates:
            continue

        score, pi = max(candidates)

        pred = pred10[pi]

        _, _, _, nw, nh, conf = gt[gi]

        recovered.append({
            "image": image.name,
            "confidence": pred[5],
            "iou": score,
            "gt_width_px": nw * W,
            "gt_height_px": nh * H,
        })


print("\n=== MT-005 CONFIDENCE RECOVERY ANALYSIS ===")
print(f"Recovered GT persons: {len(recovered)}")

if recovered:
    confidences = [
        r["confidence"] for r in recovered
        if r["confidence"] is not None
    ]

    print("\nRecovered detection confidence range:")
    print(f"Minimum: {min(confidences):.4f}")
    print(f"Maximum: {max(confidences):.4f}")
    print(f"Average: {sum(confidences)/len(confidences):.4f}")

    print("\nRecovered detections:")
    for r in recovered:
        print(
            f"{r['image']} | "
            f"conf={r['confidence']:.4f} | "
            f"IoU={r['iou']:.3f} | "
            f"GT={r['gt_width_px']:.1f}x{r['gt_height_px']:.1f}px"
        )
else:
    print("No recovered detections found.")
