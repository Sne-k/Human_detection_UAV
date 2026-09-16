from pathlib import Path
import random
import cv2

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "visdrone_person"

IMAGE_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"

OUTPUT_DIR = PROJECT_DIR / "results" / "dataset_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

NUMBER_OF_IMAGES = 10


# ---------------------------------------------------------
# GET IMAGES
# ---------------------------------------------------------

image_files = list(IMAGE_DIR.glob("*.jpg"))

if not image_files:
    print("ERROR: No images found.")
    print(IMAGE_DIR)
    exit()

selected_images = random.sample(
    image_files,
    min(NUMBER_OF_IMAGES, len(image_files))
)


# ---------------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------------

for image_path in selected_images:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        continue

    image_height, image_width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"No label found: {image_path.name}")
        continue

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            center_x = float(values[1])
            center_y = float(values[2])
            width = float(values[3])
            height = float(values[4])

            # Convert normalized YOLO coordinates
            # back to pixel coordinates

            center_x *= image_width
            center_y *= image_height

            width *= image_width
            height *= image_height

            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)

            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_width - 1, x2)
            y2 = min(image_height - 1, y2)

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                image,
                "person",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    # Save result
    output_path = OUTPUT_DIR / image_path.name

    cv2.imwrite(
        str(output_path),
        image
    )

    print(f"Saved: {output_path.name}")


print("\nDataset visualization complete.")
print(f"Results saved to:")
print(OUTPUT_DIR)