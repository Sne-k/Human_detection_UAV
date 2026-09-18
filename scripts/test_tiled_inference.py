from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL = r"runs/detect/results/training/MT-004/weights/best.pt"
SOURCE_DIR = r"dataset/visdrone_person/images/val"
OUTPUT_DIR = r"runs/detect/results/error_analysis/MT-004-tiled-corrected"

TILE_SIZE = 640
OVERLAP = 0.20

CONF = 0.25
IOU = 0.50
DEVICE = 0

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
model = YOLO(MODEL)

source_dir = Path(SOURCE_DIR)
output_dir = Path(OUTPUT_DIR)
label_dir = output_dir / "labels"

label_dir.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = sorted(
    p for p in source_dir.iterdir()
    if p.suffix.lower() in image_extensions
)

print(f"Images found: {len(images)}")
print(f"Tile size: {TILE_SIZE}")
print(f"Overlap: {OVERLAP}")
print(f"Confidence: {CONF}")
print()

# ---------------------------------------------------------
# Tile coordinate generator
# ---------------------------------------------------------
def get_positions(length, tile_size, overlap):
    stride = int(tile_size * (1.0 - overlap))

    positions = list(range(0, max(length - tile_size, 0) + 1, stride))

    last_position = max(length - tile_size, 0)

    if not positions or positions[-1] != last_position:
        positions.append(last_position)

    return sorted(set(positions))


# ---------------------------------------------------------
# Process images
# ---------------------------------------------------------
total_detections_before_nms = 0
total_detections_after_nms = 0

for image_index, image_path in enumerate(images, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[WARNING] Could not read: {image_path.name}")
        continue

    image_h, image_w = image.shape[:2]

    xs = get_positions(image_w, TILE_SIZE, OVERLAP)
    ys = get_positions(image_h, TILE_SIZE, OVERLAP)

    all_boxes = []
    all_scores = []
    all_classes = []

    # -----------------------------------------------------
    # Run inference on each tile
    # -----------------------------------------------------
    for y in ys:
        for x in xs:

            x2 = min(x + TILE_SIZE, image_w)
            y2 = min(y + TILE_SIZE, image_h)

            tile = image[y:y2, x:x2]

            results = model.predict(
                source=tile,
                imgsz=TILE_SIZE,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                verbose=False
            )

            result = results[0]

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls in zip(boxes, scores, classes):

                bx1, by1, bx2, by2 = box

                # Map tile coordinates to original image
                gx1 = bx1 + x
                gy1 = by1 + y
                gx2 = bx2 + x
                gy2 = by2 + y

                # Clamp to image boundaries
                gx1 = max(0, min(gx1, image_w))
                gy1 = max(0, min(gy1, image_h))
                gx2 = max(0, min(gx2, image_w))
                gy2 = max(0, min(gy2, image_h))

                if gx2 <= gx1 or gy2 <= gy1:
                    continue

                all_boxes.append([gx1, gy1, gx2, gy2])
                all_scores.append(float(score))
                all_classes.append(int(cls))

    total_detections_before_nms += len(all_boxes)

    # -----------------------------------------------------
    # Correct NMS format:
    # OpenCV expects [x, y, width, height]
    # -----------------------------------------------------
    final_indices = []

    if all_boxes:

        nms_boxes = []

        for box in all_boxes:
            x1, y1, x2, y2 = box

            nms_boxes.append([
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            all_scores,
            score_threshold=CONF,
            nms_threshold=IOU
        )

        if len(indices) > 0:
            final_indices = np.array(indices).reshape(-1).tolist()

    total_detections_after_nms += len(final_indices)

    # -----------------------------------------------------
    # Save YOLO-format labels
    # -----------------------------------------------------
    label_path = label_dir / f"{image_path.stem}.txt"

    with open(label_path, "w") as f:

        for idx in final_indices:

            x1, y1, x2, y2 = all_boxes[idx]
            score = all_scores[idx]
            cls = all_classes[idx]

            xc = ((x1 + x2) / 2.0) / image_w
            yc = ((y1 + y2) / 2.0) / image_h
            bw = (x2 - x1) / image_w
            bh = (y2 - y1) / image_h

            f.write(
                f"{cls} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f} "
                f"{score:.6f}\n"
            )

    print(
        f"[{image_index:04d}/{len(images):04d}] "
        f"{image_path.name} | "
        f"tiles={len(xs) * len(ys)} | "
        f"raw={len(all_boxes)} | "
        f"final={len(final_indices)}"
    )

print()
print("===================================================")
print("TILED INFERENCE COMPLETE")
print("===================================================")
print(f"Images processed: {len(images)}")
print(f"Raw tile detections: {total_detections_before_nms}")
print(f"Final detections after NMS: {total_detections_after_nms}")
print(f"Labels saved to: {label_dir}")
print("===================================================")