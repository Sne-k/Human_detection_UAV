from pathlib import Path
import cv2
import yaml
import shutil

# ============================================================
# HIT-UAV Small-Object Crop Dataset Generator
# MT-007
# ============================================================

SRC_ROOT = Path("dataset/hit_uav_person")
OUT_ROOT = Path("dataset/hit_uav_person_crops")

CROP_SIZE = 256
OVERLAP = 0.25

# Keep original training images.
KEEP_ORIGINALS = True

# Maximum empty crops retained per source image.
MAX_EMPTY_PER_IMAGE = 2


def load_labels(label_path):
    labels = []

    if not label_path.exists():
        return labels

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        cls, xc, yc, w, h = map(float, parts[:5])

        labels.append({
            "cls": int(cls),
            "xc": xc,
            "yc": yc,
            "w": w,
            "h": h,
        })

    return labels


def yolo_to_xyxy(label, img_w, img_h):
    xc = label["xc"] * img_w
    yc = label["yc"] * img_h
    w = label["w"] * img_w
    h = label["h"] * img_h

    return (
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2,
    )


def make_starts(length, crop_size, stride):
    if length <= crop_size:
        return [0]

    starts = list(range(0, length - crop_size + 1, stride))

    last = length - crop_size

    if starts[-1] != last:
        starts.append(last)

    return sorted(set(starts))


def intersection(box, crop):
    bx0, by0, bx1, by1 = box
    cx0, cy0, cx1, cy1 = crop

    ix0 = max(bx0, cx0)
    iy0 = max(by0, cy0)
    ix1 = min(bx1, cx1)
    iy1 = min(by1, cy1)

    if ix1 <= ix0 or iy1 <= iy0:
        return None

    return ix0, iy0, ix1, iy1


def convert_box(box, crop_x0, crop_y0):
    x0, y0, x1, y1 = box

    x0 -= crop_x0
    x1 -= crop_x0
    y0 -= crop_y0
    y1 -= crop_y0

    w = x1 - x0
    h = y1 - y0

    if w < 2 or h < 2:
        return None

    xc = (x0 + x1) / 2 / CROP_SIZE
    yc = (y0 + y1) / 2 / CROP_SIZE
    nw = w / CROP_SIZE
    nh = h / CROP_SIZE

    return (
        min(max(xc, 0.0), 1.0),
        min(max(yc, 0.0), 1.0),
        min(max(nw, 0.0), 1.0),
        min(max(nh, 0.0), 1.0),
    )


def process_train():
    image_dir = SRC_ROOT / "images" / "train"
    label_dir = SRC_ROOT / "labels" / "train"

    out_image_dir = OUT_ROOT / "images" / "train"
    out_label_dir = OUT_ROOT / "labels" / "train"

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    stride = int(CROP_SIZE * (1 - OVERLAP))

    total_output = 0
    person_crops = 0
    empty_crops = 0
    total_annotations = 0

    for index, image_path in enumerate(images, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: failed to read {image_path}")
            continue

        img_h, img_w = image.shape[:2]

        labels = load_labels(
            label_dir / f"{image_path.stem}.txt"
        )

        gt_boxes = [
            (
                label,
                yolo_to_xyxy(label, img_w, img_h)
            )
            for label in labels
        ]

        # ----------------------------------------------------
        # Preserve original training image.
        # ----------------------------------------------------
        if KEEP_ORIGINALS:

            output_name = (
                f"orig_{image_path.stem}{image_path.suffix}"
            )

            cv2.imwrite(
                str(out_image_dir / output_name),
                image
            )

            original_label = out_label_dir / (
                f"orig_{image_path.stem}.txt"
            )

            original_label.write_text(
                "\n".join(
                    f"{label['cls']} "
                    f"{label['xc']:.6f} "
                    f"{label['yc']:.6f} "
                    f"{label['w']:.6f} "
                    f"{label['h']:.6f}"
                    for label in labels
                )
            )

            total_output += 1
            total_annotations += len(labels)

        xs = make_starts(img_w, CROP_SIZE, stride)
        ys = make_starts(img_h, CROP_SIZE, stride)

        empty_count = 0

        for row, y0 in enumerate(ys):
            for col, x0 in enumerate(xs):

                crop_x1 = min(x0 + CROP_SIZE, img_w)
                crop_y1 = min(y0 + CROP_SIZE, img_h)

                # Only use full-size crops.
                if crop_x1 - x0 != CROP_SIZE:
                    continue

                if crop_y1 - y0 != CROP_SIZE:
                    continue

                crop = image[
                    y0:crop_y1,
                    x0:crop_x1
                ]

                crop_box = (
                    x0,
                    y0,
                    crop_x1,
                    crop_y1
                )

                crop_labels = []

                for label, box in gt_boxes:

                    clipped = intersection(
                        box,
                        crop_box
                    )

                    if clipped is None:
                        continue

                    bx0, by0, bx1, by1 = box

                    original_area = max(
                        1.0,
                        (bx1 - bx0) * (by1 - by0)
                    )

                    clipped_area = (
                        (clipped[2] - clipped[0])
                        * (clipped[3] - clipped[1])
                    )

                    visible_fraction = (
                        clipped_area / original_area
                    )

                    # Do not train on heavily truncated people.
                    if visible_fraction < 0.50:
                        continue

                    converted = convert_box(
                        clipped,
                        x0,
                        y0
                    )

                    if converted is None:
                        continue

                    crop_labels.append(
                        (
                            label["cls"],
                            *converted
                        )
                    )

                # Limit background-only crops.
                if not crop_labels:

                    if empty_count >= MAX_EMPTY_PER_IMAGE:
                        continue

                    empty_count += 1
                    empty_crops += 1

                else:
                    person_crops += 1

                crop_name = (
                    f"{image_path.stem}"
                    f"_r{row:02d}"
                    f"_c{col:02d}.jpg"
                )

                cv2.imwrite(
                    str(out_image_dir / crop_name),
                    crop
                )

                label_path = (
                    out_label_dir /
                    crop_name.replace(".jpg", ".txt")
                )

                label_path.write_text(
                    "\n".join(
                        f"{cls} "
                        f"{xc:.6f} "
                        f"{yc:.6f} "
                        f"{w:.6f} "
                        f"{h:.6f}"
                        for cls, xc, yc, w, h
                        in crop_labels
                    )
                )

                total_output += 1
                total_annotations += len(crop_labels)

        if index % 200 == 0:
            print(
                f"train: processed "
                f"{index}/{len(images)}"
            )

    print()
    print("=== MT-007 TRAIN DATASET ===")
    print(f"Original images: {len(images)}")
    print(f"Output images:   {total_output}")
    print(f"Person crops:    {person_crops}")
    print(f"Empty crops:     {empty_crops}")
    print(f"Annotations:     {total_annotations}")


def copy_original_split(split):

    src_images = SRC_ROOT / "images" / split
    src_labels = SRC_ROOT / "labels" / split

    dst_images = OUT_ROOT / "images" / split
    dst_labels = OUT_ROOT / "labels" / split

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        src_images,
        dst_images,
        dirs_exist_ok=True
    )

    shutil.copytree(
        src_labels,
        dst_labels,
        dirs_exist_ok=True
    )

    print(
        f"Copied original {split} split unchanged."
    )


def main():

    if OUT_ROOT.exists():
        print(
            f"Removing previous dataset: {OUT_ROOT}"
        )
        shutil.rmtree(OUT_ROOT)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Creating 256x256 MT-007 crops...")
    print()

    process_train()

    # IMPORTANT:
    # Validation and test are copied exactly.
    copy_original_split("val")
    copy_original_split("test")

    yaml_data = {
        "path": str(
            OUT_ROOT.resolve()
        ).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "person"
        }
    }

    yaml_path = OUT_ROOT / "data.yaml"

    with yaml_path.open("w") as f:
        yaml.safe_dump(
            yaml_data,
            f,
            sort_keys=False
        )

    print()
    print("======================================")
    print("MT-007 dataset creation complete")
    print("======================================")
    print(f"Dataset: {OUT_ROOT}")
    print(f"YAML:    {yaml_path}")
    print()
    print("Crop size: 256x256")
    print("Overlap:   25%")
    print("Validation: ORIGINAL")
    print("Test:       ORIGINAL")


if __name__ == "__main__":
    main()
