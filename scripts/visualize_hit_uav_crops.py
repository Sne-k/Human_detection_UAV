from pathlib import Path
import cv2
import random

ROOT = Path("dataset/hit_uav_person_crops")
IMAGE_DIR = ROOT / "images/train"
LABEL_DIR = ROOT / "labels/train"
OUT_DIR = Path("results/mt007_crop_check")

NUM_IMAGES = 20
random.seed(42)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    random.shuffle(images)

    selected = []

    # Prefer person-containing crops for verification.
    for image_path in images:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if label_path.exists() and label_path.read_text().strip():
            selected.append(image_path)

        if len(selected) >= NUM_IMAGES:
            break

    print(f"Found {len(selected)} person-containing crops.")
    print(f"Output: {OUT_DIR}")

    for i, image_path in enumerate(selected, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        h, w = image.shape[:2]

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        for line in label_path.read_text().splitlines():
            parts = line.split()

            if len(parts) < 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts[:5])

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                "person",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        output = OUT_DIR / f"check_{i:02d}_{image_path.name}"
        cv2.imwrite(str(output), image)

    print("Visualization complete.")


if __name__ == "__main__":
    main()
