from pathlib import Path
import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# Change ONLY this path when analyzing a different experiment.
PREDICTION_DIR = Path("runs/detect/results/error_analysis/MT-003-analysis/labels")

GT_LABEL_DIR = Path(
    "dataset/visdrone_person/labels/val"
)

IMAGE_DIR = Path(
    "dataset/visdrone_person/images/val"
)

IOU_THRESHOLD = 0.50

# Size bins are based on BOUNDING-BOX AREA.
# These match the original MT-003 analysis methodology.
SMALL_AREA = 32 * 32
LARGE_AREA = 96 * 96


# ============================================================
# FUNCTIONS
# ============================================================

def load_labels(label_path):
    """
    Load YOLO-format labels.

    Returns:
        list of [class_id, x_center, y_center, width, height, confidence]
    """

    if not label_path.exists():
        return []

    labels = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            values = list(map(float, parts))

            class_id = int(values[0])
            x = values[1]
            y = values[2]
            w = values[3]
            h = values[4]

            confidence = values[5] if len(values) >= 6 else 1.0

            labels.append([
                class_id,
                x,
                y,
                w,
                h,
                confidence
            ])

    return labels


def yolo_to_xyxy(label, image_width, image_height):
    """
    Convert normalized YOLO coordinates to pixel coordinates.
    """

    _, x, y, w, h, _ = label

    x_center = x * image_width
    y_center = y * image_height
    box_width = w * image_width
    box_height = h * image_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    """
    Calculate IoU between two XYXY boxes.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def get_size_category(box):
    """
    Categorize object using bounding-box AREA.

    Small:
        area < 32^2

    Medium:
        32^2 <= area < 96^2

    Large:
        area >= 96^2
    """

    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])

    area = width * height

    if area < SMALL_AREA:
        return "small"

    elif area < LARGE_AREA:
        return "medium"

    else:
        return "large"


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 55)
print("SMALL-OBJECT ERROR ANALYSIS")
print("=" * 55)
print()

print("Prediction directory:")
print(f"  {PREDICTION_DIR}")
print()

if not PREDICTION_DIR.exists():
    print("ERROR: Prediction directory does not exist.")
    print()
    print("Check the path above.")
    raise SystemExit(1)

if not GT_LABEL_DIR.exists():
    print("ERROR: Ground-truth directory does not exist.")
    print(f"  {GT_LABEL_DIR}")
    raise SystemExit(1)

if not IMAGE_DIR.exists():
    print("ERROR: Image directory does not exist.")
    print(f"  {IMAGE_DIR}")
    raise SystemExit(1)


# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

total_images_with_gt = 0
images_with_no_predictions = 0

total_gt = 0
total_matched = 0

size_stats = {
    "small": {
        "gt": 0,
        "matched": 0
    },
    "medium": {
        "gt": 0,
        "matched": 0
    },
    "large": {
        "gt": 0,
        "matched": 0
    }
}


# ------------------------------------------------------------
# Process validation images
# ------------------------------------------------------------

gt_files = sorted(GT_LABEL_DIR.glob("*.txt"))

for gt_file in gt_files:

    gt_labels = load_labels(gt_file)

    # Only person class
    gt_labels = [
        label for label in gt_labels
        if label[0] == 0
    ]

    if len(gt_labels) == 0:
        continue

    total_images_with_gt += 1

    image_name = gt_file.stem

    # Find corresponding image
    image_path = None

    for extension in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = IMAGE_DIR / f"{image_name}{extension}"

        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"WARNING: Image not found for {image_name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read {image_path}")
        continue

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Ground-truth boxes
    # --------------------------------------------------------

    gt_boxes = []

    for label in gt_labels:

        box = yolo_to_xyxy(
            label,
            image_width,
            image_height
        )

        gt_boxes.append(box)

        category = get_size_category(box)

        size_stats[category]["gt"] += 1
        total_gt += 1

    # --------------------------------------------------------
    # Prediction file
    # --------------------------------------------------------

    pred_file = PREDICTION_DIR / gt_file.name

    pred_labels = load_labels(pred_file)

    # Only person class
    pred_labels = [
        label for label in pred_labels
        if label[0] == 0
    ]

    if len(pred_labels) == 0:
        images_with_no_predictions += 1

    pred_boxes = []

    for label in pred_labels:

        box = yolo_to_xyxy(
            label,
            image_width,
            image_height
        )

        confidence = label[5]

        pred_boxes.append({
            "box": box,
            "confidence": confidence
        })

    # --------------------------------------------------------
    # Greedy IoU matching
    # --------------------------------------------------------

    matched_gt = set()

    # Highest-confidence predictions first
    pred_boxes.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    for prediction in pred_boxes:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt_box in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            iou = calculate_iou(
                prediction["box"],
                gt_box
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= IOU_THRESHOLD
        ):

            matched_gt.add(best_gt_index)

            category = get_size_category(
                gt_boxes[best_gt_index]
            )

            size_stats[category]["matched"] += 1
            total_matched += 1


# ============================================================
# RESULTS
# ============================================================

print(f"Images with ground-truth people: {total_images_with_gt}")
print(f"Images with no predictions:      {images_with_no_predictions}")
print(f"Total ground-truth persons:      {total_gt}")
print(f"Matched detections (IoU >= {IOU_THRESHOLD:.2f}): {total_matched}")
print()

print("Size-wise detection recall:")
print("-" * 55)

labels = {
    "small": "Small (< 32x32)",
    "medium": "Medium (32x32 - 96x96)",
    "large": "Large (>= 96x96)"
}

for category in ["small", "medium", "large"]:

    gt = size_stats[category]["gt"]
    matched = size_stats[category]["matched"]

    recall = matched / gt if gt > 0 else 0.0

    print(
        f"{labels[category]:30s}"
        f"GT={gt:6d} "
        f"Matched={matched:6d} "
        f"Recall={recall:.3f}"
    )

print()

overall_ratio = (
    total_matched / total_gt
    if total_gt > 0
    else 0.0
)

print(f"Overall matched ratio: {overall_ratio:.3f}")

print()
print("=" * 55)
print("END")
print("=" * 55)