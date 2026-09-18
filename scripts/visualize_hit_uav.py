from pathlib import Path
import random

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "dataset" / "hit_uav_person" / "images" / "train"
LABEL_DIR = PROJECT_ROOT / "dataset" / "hit_uav_person" / "labels" / "train"

OUTPUT_DIR = PROJECT_ROOT / "results" / "hit_uav_dataset_check"

NUM_IMAGES = 12


def draw_labels(image, label_path):
    h, w = image.shape[:2]

    if not label_path.exists():
        return image

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()

        if len(parts) < 5:
            continue

        class_id, xc, yc, bw, bh = map(float, parts[:5])

        # YOLO normalized -> pixel coordinates
        xc *= w
        yc *= h
        bw *= w
        bh *= h

        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1
        )

        cv2.putText(
            image,
            "person",
            (x1, max(y1 - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_files = [
        p for p in LABEL_DIR.glob("*.txt")
        if p.stat().st_size > 0
    ]

    random.seed(42)
    selected = random.sample(
        label_files,
        min(NUM_IMAGES, len(label_files))
    )

    print(f"Found {len(label_files)} non-empty training labels.")
    print(f"Creating {len(selected)} verification images.")
    print(f"Output: {OUTPUT_DIR}")

    for label_path in selected:

        image_path = IMAGE_DIR / f"{label_path.stem}.jpg"

        if not image_path.exists():
            print(f"WARNING: Missing image: {image_path}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: Could not read: {image_path}")
            continue

        image = draw_labels(image, label_path)

        output_path = OUTPUT_DIR / image_path.name

        cv2.imwrite(str(output_path), image)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()