"""
Mine hard negatives from the detector's own false positives.

MT-005 emits 638 unmatched prediction boxes against 2,425 matched persons on
the test split - a custom precision of 0.79. Nothing in this project has yet
tried to reduce that directly.

Lygouras et al. (2019), reference [4] in the literature survey, solved the
equivalent problem by collecting a second round of images containing
specifically the objects their detector was falsely firing on - boats, in
their case - at a 1:1 ratio with positives, and report it raised both recall
and mAP. This applies the same idea.

Where the negatives come from matters
-------------------------------------

They are mined from the **training** split, never the test split. Running the
detector on test data, harvesting its mistakes, and training on them would
leak the test set and invalidate every number this project reports.

Mining from training data is legitimate and is the classical bootstrapping
approach: the model has already seen those images with correct labels, but the
regions where it still fires incorrectly are, by definition, the ones it finds
hard. Cropping those regions and adding them as explicit background images
upweights exactly the patches the model has not learned to reject.

What gets written
-----------------

Each mined crop is saved as a training image with an **empty** label file,
which is how YOLO represents a background image. The originals are copied
unchanged, so MT-009 differs from MT-005 only by the added negatives.

A crop is rejected if it overlaps any annotated person, since a crop
containing a real person taught as background would actively harm the model.
That check uses a deliberately generous margin.

Usage:

    python scripts/mine_hard_negatives.py --analyse
    python scripts/mine_hard_negatives.py --build
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml
from PIL import Image

PROJECT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

SRC_ROOT = PROJECT / "dataset" / "hit_uav_person"
OUT_ROOT = PROJECT / "dataset" / "hit_uav_person_negmine"

PRED_DIR = (
    PROJECT / "runs" / "detect" / "results" / "error_analysis"
    / "MT-005-train-mining" / "labels"
)

OUTPUT_DIR = PROJECT / "results" / "hard_negatives"

IOU_THRESHOLD = 0.50

# A candidate crop is discarded if it overlaps an annotated person by more
# than this, in units of the person's own area. Deliberately strict: teaching
# a real person as background is far worse than discarding a good negative.
PERSON_CONTAMINATION = 0.05

# Crop size around each false positive.
#
# 256 px was the first choice, for context, but it yielded only 156 usable
# crops from 1,729 false positives: 1,572 were rejected because a real person
# fell inside the window. 95.9% of these false positives occur on images that
# already contain people, so a large window almost always catches one.
#
# 128 px still carries roughly 7x the area of a typical 13 x 18 px target,
# which is ample context for a background patch, while being far less likely
# to enclose an annotated person.
CROP_SIZE = 128

# Cap per source image so a single pathological frame cannot dominate.
MAX_PER_IMAGE = 4


def load_boxes(path, width, height):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        xc = float(parts[1]) * width
        yc = float(parts[2]) * height
        w = float(parts[3]) * width
        h = float(parts[4]) * height

        boxes.append({
            "x1": xc - w / 2,
            "y1": yc - h / 2,
            "x2": xc + w / 2,
            "y2": yc + h / 2,
            "cx": xc,
            "cy": yc,
            "w": w,
            "h": h,
            "conf": float(parts[5]) if len(parts) >= 6 else 1.0,
        })

    return boxes


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def overlap_fraction(region, box):
    """Fraction of `box` that lies inside `region`."""

    x1 = max(region[0], box["x1"])
    y1 = max(region[1], box["y1"])
    x2 = min(region[2], box["x2"])
    y2 = min(region[3], box["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area = max(1e-6, box["w"] * box["h"])

    return inter / area


def find_false_positives(stem, width, height):
    """Predictions on a training image that match no annotated person."""

    gt = load_boxes(SRC_ROOT / "labels" / "train" / f"{stem}.txt", width, height)
    preds = load_boxes(PRED_DIR / f"{stem}.txt", width, height)

    if not preds:
        return [], gt

    matched_pred = set()

    pairs = []

    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            score = iou(g, p)

            if score >= IOU_THRESHOLD:
                pairs.append((score, gi, pi))

    pairs.sort(reverse=True)

    used_gt = set()

    for _, gi, pi in pairs:
        if gi in used_gt or pi in matched_pred:
            continue

        used_gt.add(gi)
        matched_pred.add(pi)

    false_positives = [
        p for pi, p in enumerate(preds) if pi not in matched_pred
    ]

    return false_positives, gt


def scan():
    """Collect every false positive across the training split."""

    image_dir = SRC_ROOT / "images" / "train"

    records = []

    for image_path in sorted(image_dir.glob("*.jpg")):
        with Image.open(image_path) as image:
            width, height = image.size

        false_positives, gt = find_false_positives(
            image_path.stem, width, height
        )

        for fp in false_positives:
            records.append({
                "image": image_path.stem,
                "width": width,
                "height": height,
                "conf": round(fp["conf"], 4),
                "box_w": round(fp["w"], 1),
                "box_h": round(fp["h"], 1),
                "cx": round(fp["cx"], 1),
                "cy": round(fp["cy"], 1),
                "persons_in_image": len(gt),
                "fp": fp,
                "gt": gt,
            })

    return records


def analyse(records):
    if not records:
        print("No false positives found. Has the mining prediction run?")
        print(f"Expected labels at: {PRED_DIR}")
        return

    print(f"False positives on the training split: {len(records)}")
    print()

    confs = sorted(r["conf"] for r in records)

    print("Confidence distribution")
    print("-" * 52)

    for lo, hi in [(0.25, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.01)]:
        n = sum(1 for c in confs if lo <= c < hi)
        print(f"  {lo:.2f} - {hi:.2f}   {n:5d}  ({n / len(confs):5.1%})")

    print()
    print("Scene context")
    print("-" * 52)

    empty = sum(1 for r in records if r["persons_in_image"] == 0)

    print(f"  on images with NO annotated person : {empty:5d} "
          f"({empty / len(records):5.1%})")
    print(f"  on images that do contain persons  : "
          f"{len(records) - empty:5d} "
          f"({(len(records) - empty) / len(records):5.1%})")

    light = Counter(
        "night" if r["image"].split("_")[0] == "1" else "day"
        for r in records
    )

    print()
    print(f"  day   {light['day']:5d}   night {light['night']:5d}")

    sizes = sorted(r["box_w"] * r["box_h"] for r in records)

    print()
    print(f"  median FP area {sizes[len(sizes) // 2]:.0f} px2, "
          f"min {sizes[0]:.0f}, max {sizes[-1]:.0f}")

    worst = Counter(r["image"] for r in records)

    print()
    print("Images producing the most false positives")
    print("-" * 52)

    for stem, count in worst.most_common(8):
        print(f"  {stem}  {count}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "false_positives.json").write_text(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ("fp", "gt")}
                for r in records
            ],
            indent=2,
        )
    )

    print()
    print(f"Written: {OUTPUT_DIR / 'false_positives.json'}")


def build(records):
    """Create the augmented dataset: originals plus mined negative crops."""

    if OUT_ROOT.exists():
        print(f"Removing previous dataset: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy every split unchanged first. Only the training split gains images.
    for split in ("train", "val", "test"):
        shutil.copytree(
            SRC_ROOT / "images" / split,
            OUT_ROOT / "images" / split,
            dirs_exist_ok=True,
        )
        shutil.copytree(
            SRC_ROOT / "labels" / split,
            OUT_ROOT / "labels" / split,
            dirs_exist_ok=True,
        )

        print(f"Copied {split} unchanged.")

    by_image = {}

    for record in records:
        by_image.setdefault(record["image"], []).append(record)

    written = 0
    rejected = 0

    for stem, items in sorted(by_image.items()):
        image_path = SRC_ROOT / "images" / "train" / f"{stem}.jpg"

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        # Highest-confidence false positives first: those are the ones the
        # model is most wrong about.
        items.sort(key=lambda r: -r["conf"])

        kept = 0

        for record in items:
            if kept >= MAX_PER_IMAGE:
                break

            cx, cy = record["cx"], record["cy"]

            x0 = int(round(cx - CROP_SIZE / 2))
            y0 = int(round(cy - CROP_SIZE / 2))

            x0 = max(0, min(x0, width - CROP_SIZE))
            y0 = max(0, min(y0, height - CROP_SIZE))

            if width < CROP_SIZE or height < CROP_SIZE:
                continue

            region = (x0, y0, x0 + CROP_SIZE, y0 + CROP_SIZE)

            # Reject if any annotated person intrudes into the crop.
            contaminated = any(
                overlap_fraction(region, person) > PERSON_CONTAMINATION
                for person in record["gt"]
            )

            if contaminated:
                rejected += 1
                continue

            crop = image[region[1]:region[3], region[0]:region[2]]

            if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
                continue

            name = f"neg_{stem}_{kept:02d}"

            cv2.imwrite(
                str(OUT_ROOT / "images" / "train" / f"{name}.jpg"), crop
            )

            # Empty label file: YOLO's representation of a background image.
            (OUT_ROOT / "labels" / "train" / f"{name}.txt").write_text("")

            kept += 1
            written += 1

    config = {
        "path": str(OUT_ROOT.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "person"},
    }

    with (OUT_ROOT / "data.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    originals = len(list((SRC_ROOT / "images" / "train").glob("*.jpg")))

    print()
    print("=" * 56)
    print("MT-009 hard-negative training set")
    print("=" * 56)
    print(f"  original training images : {originals}")
    print(f"  mined negative crops     : {written}")
    print(f"  rejected (person inside) : {rejected}")
    print(f"  total training images    : {originals + written}")
    print()
    print(f"  validation and test splits copied unchanged")
    print(f"Written: {OUT_ROOT / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--build", action="store_true")

    args = parser.parse_args()

    if not PRED_DIR.exists():
        raise SystemExit(
            f"Mining predictions not found at {PRED_DIR}\n"
            "Run MT-005 over the TRAINING split first, saving labels to "
            "MT-005-train-mining."
        )

    records = scan()

    if args.build:
        analyse(records)
        build(records)
    else:
        analyse(records)


if __name__ == "__main__":
    main()
