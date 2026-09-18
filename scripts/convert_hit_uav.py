import json
from pathlib import Path
import shutil

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "raw_data"
    / "suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c"
    / "normal_json"
)

OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "hit_uav_person"

# COCO category ID for Person in HIT-UAV
PERSON_CATEGORY_ID = 0

SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def convert_split(split):
    json_path = SOURCE_ROOT / "annotations" / f"{split}.json"

    image_source_dir = SOURCE_ROOT / split

    image_output_dir = OUTPUT_ROOT / "images" / split
    label_output_dir = OUTPUT_ROOT / "labels" / split

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split}...")
    print(f"JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -----------------------------------------------------
    # Build image ID -> image information lookup
    # -----------------------------------------------------

    images = {
        image["id"]: image
        for image in data["images"]
    }

    # -----------------------------------------------------
    # Group Person annotations by image
    # -----------------------------------------------------

    annotations_by_image = {}

    person_annotations = 0

    for ann in data["annotation"]:

        if ann["category_id"] != PERSON_CATEGORY_ID:
            continue

        image_id = ann["image_id"]

        annotations_by_image.setdefault(image_id, []).append(ann)

        person_annotations += 1

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    total_images = len(images)
    images_with_person = 0
    images_without_person = 0
    copied_images = 0
    missing_images = 0

    # -----------------------------------------------------
    # Convert each image
    # -----------------------------------------------------

    for image_id, image_info in images.items():

        filename = image_info["filename"]

        width = image_info["width"]
        height = image_info["height"]

        source_image = image_source_dir / filename
        output_image = image_output_dir / filename
        output_label = label_output_dir / (
            Path(filename).stem + ".txt"
        )

        # -------------------------------------------------
        # Copy image
        # -------------------------------------------------

        if not source_image.exists():
            print(f"WARNING: Missing image: {source_image}")
            missing_images += 1
            continue

        shutil.copy2(source_image, output_image)
        copied_images += 1

        # -------------------------------------------------
        # Get Person annotations
        # -------------------------------------------------

        anns = annotations_by_image.get(image_id, [])

        if anns:
            images_with_person += 1
        else:
            images_without_person += 1

        # -------------------------------------------------
        # Write YOLO labels
        #
        # YOLO:
        # class x_center y_center width height
        #
        # All values normalized to [0, 1]
        # -------------------------------------------------

        lines = []

        for ann in anns:

            x, y, box_w, box_h = ann["bbox"]

            # Convert COCO xywh -> YOLO xywh
            x_center = x + box_w / 2
            y_center = y + box_h / 2

            x_center /= width
            y_center /= height
            box_w /= width
            box_h /= height

            # Clamp values against numerical edge cases
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            box_w = min(max(box_w, 0.0), 1.0)
            box_h = min(max(box_h, 0.0), 1.0)

            lines.append(
                f"0 {x_center:.6f} {y_center:.6f} "
                f"{box_w:.6f} {box_h:.6f}"
            )

        # -------------------------------------------------
        # Write label file
        # -------------------------------------------------

        with open(output_label, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print(f"\n{split.upper()} SUMMARY")
    print("-" * 50)
    print(f"Images in JSON:          {total_images}")
    print(f"Images copied:           {copied_images}")
    print(f"Images with Person:      {images_with_person}")
    print(f"Images without Person:   {images_without_person}")
    print(f"Person annotations:      {person_annotations}")
    print(f"Missing images:           {missing_images}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("HIT-UAV -> YOLO PERSON-ONLY CONVERTER")
    print("=" * 60)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        convert_split(split)

    # -----------------------------------------------------
    # Create data.yaml
    # -----------------------------------------------------

    yaml_path = OUTPUT_ROOT / "data.yaml"

    yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: person
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT_ROOT}")
    print(f"YAML:   {yaml_path}")


if __name__ == "__main__":
    main()