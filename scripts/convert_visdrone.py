from pathlib import Path
import shutil
import cv2

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "raw_data" / "VisDrone2019-DET-train"
VAL_DIR = PROJECT_DIR / "raw_data" / "VisDrone2019-DET-val"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "visdrone_person"


# ---------------------------------------------------------
# VISDRONE CLASSES
# ---------------------------------------------------------
# VisDrone:
# 1 = pedestrian
# 2 = people
#
# We combine both into:
# 0 = person
# ---------------------------------------------------------

PERSON_CLASSES = {1, 2}


def convert_split(source_dir, split_name):
    """
    Convert one VisDrone split into YOLO format.
    """

    image_dir = source_dir / "images"
    annotation_dir = source_dir / "annotations"

    output_image_dir = OUTPUT_DIR / "images" / split_name
    output_label_dir = OUTPUT_DIR / "labels" / split_name

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*"))

    total_images = 0
    images_with_person = 0
    total_persons = 0

    print(f"\nProcessing {split_name}...")
    print(f"Images found: {len(image_files)}")

    for image_path in image_files:

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        total_images += 1

        # Read image to obtain width and height
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: Could not read {image_path.name}")
            continue

        image_height, image_width = image.shape[:2]

        annotation_path = annotation_dir / f"{image_path.stem}.txt"

        yolo_annotations = []

        if annotation_path.exists():

            with open(annotation_path, "r") as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    values = line.split(",")

                    if len(values) < 8:
                        continue

                    # VisDrone format:
                    # bbox_left
                    # bbox_top
                    # bbox_width
                    # bbox_height
                    # score
                    # class_id
                    # truncation
                    # occlusion

                    x = float(values[0])
                    y = float(values[1])
                    w = float(values[2])
                    h = float(values[3])

                    score = int(values[4])
                    class_id = int(values[5])

                    # Keep only pedestrian and people
                    if class_id not in PERSON_CLASSES:
                        continue

                    # Ignore invalid/ignored annotations
                    if score == 0:
                        continue

                    # Convert VisDrone bbox to YOLO format
                    center_x = x + w / 2
                    center_y = y + h / 2

                    center_x /= image_width
                    center_y /= image_height
                    w /= image_width
                    h /= image_height

                    # YOLO class 0 = person
                    yolo_annotations.append(
                        f"0 {center_x:.6f} {center_y:.6f} "
                        f"{w:.6f} {h:.6f}"
                    )

        # Copy image
        destination_image = output_image_dir / image_path.name
        shutil.copy2(image_path, destination_image)

        # Create label file
        destination_label = output_label_dir / f"{image_path.stem}.txt"

        with open(destination_label, "w") as file:
            file.write("\n".join(yolo_annotations))

        if yolo_annotations:
            images_with_person += 1
            total_persons += len(yolo_annotations)

    print(f"Finished {split_name}")
    print(f"Total images       : {total_images}")
    print(f"Images with person : {images_with_person}")
    print(f"Person annotations : {total_persons}")

    return total_images, images_with_person, total_persons


def create_yaml():
    """
    Create YOLO dataset configuration file.
    """

    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: person
"""

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w") as file:
        file.write(yaml_content)

    print(f"\nCreated dataset configuration:")
    print(yaml_path)


def main():

    print("=" * 60)
    print("VisDrone → YOLO PERSON DATASET CONVERTER")
    print("=" * 60)

    # Check source folders
    if not TRAIN_DIR.exists():
        print("\nERROR: Training dataset not found:")
        print(TRAIN_DIR)
        return

    if not VAL_DIR.exists():
        print("\nERROR: Validation dataset not found:")
        print(VAL_DIR)
        return

    # Convert training and validation sets
    train_stats = convert_split(TRAIN_DIR, "train")
    val_stats = convert_split(VAL_DIR, "val")

    # Create data.yaml
    create_yaml()

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print("\nTraining:")
    print(f"  Images: {train_stats[0]}")
    print(f"  Images containing person: {train_stats[1]}")
    print(f"  Person annotations: {train_stats[2]}")

    print("\nValidation:")
    print(f"  Images: {val_stats[0]}")
    print(f"  Images containing person: {val_stats[1]}")
    print(f"  Person annotations: {val_stats[2]}")

    print("\nOutput:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()