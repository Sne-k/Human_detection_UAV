"""
Test whether the thermal localisation failures are really a crowding effect.

The failure characterisation found that persons missed by every thermal model
are 5.8x more likely than detected persons to have a neighbour within 10 px,
and that their boxes are squarer than a standing person seen from above. That
is consistent with boxes spanning two adjacent people - but "consistent with"
is not evidence, and nearest-neighbour distance is only a proxy for visual
crowding. It could equally be a correlation with some third factor.

This script tests the mechanism directly, by asking what the near-miss
prediction actually did. Three distinguishable things can go wrong:

  merged    One prediction covers two or more annotated people. Its box is
            larger than a single person and overlaps several of them. The
            detector saw a group and emitted one object.

  stolen    The prediction that best overlaps the missed person was matched to
            a different, neighbouring person. The detector found only one of
            two adjacent people, and greedy matching gave the box to the other.

  undersized The prediction covers this person alone but is markedly smaller
            than the annotation. In thermal imagery the detector locks onto the
            bright heat signature while the annotation covers the whole body,
            so the box is correctly centred but too small to reach IoU 0.50.

  offset    The prediction covers this person alone, is about the right size,
            and is simply displaced. Ordinary box-regression error, nothing to
            do with crowding.

Only the first two are crowding. If the latter two dominate, the crowding
explanation is wrong and the real problem is box scale or regression
precision, which would point at completely different fixes.

The test is run against detected persons as a control, because near-miss
geometry is only meaningful relative to what success looks like.

Usage:

    python scripts/verify_crowding_hypothesis.py
    python scripts/verify_crowding_hypothesis.py --conf 0.25 --render 24
"""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

PROJECT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

GT_DIR = PROJECT / "dataset" / "hit_uav_person" / "labels" / "test"
IMAGE_DIR = PROJECT / "dataset" / "hit_uav_person" / "images" / "test"

PRED_ROOTS = [
    PROJECT / "runs" / "detect" / "results" / "error_analysis",
    PROJECT / "results" / "error_analysis",
]

PRED_DIRS = {
    ("MT-005", "0.25"): "MT-005-test-analysis",
    ("MT-005", "0.10"): "MT-005-test-conf010",
    ("MT-006", "0.25"): "MT-006-test-analysis",
    ("MT-007", "0.25"): "MT-007-test-analysis",
    ("MT-007", "0.10"): "MT-007-test-conf010",
}

IOU_THRESHOLD = 0.50

# A prediction is considered to "touch" a ground-truth person above this IoU.
TOUCH_IOU = 0.10

# A prediction counts as oversized when it is this much larger than the person
# it should have covered.
MERGE_AREA_RATIO = 1.60

# Below this share of the annotated area, a single-person prediction is
# undersized rather than merely displaced.
UNDERSIZE_AREA_RATIO = 0.70

OUTPUT_DIR = PROJECT / "results" / "error_analysis" / "crowding_check"

# Render colours (BGR)
COLOR_GT = (0, 220, 0)
COLOR_TARGET = (0, 128, 255)
COLOR_PRED = (0, 0, 255)


def conf_key(value):
    return f"{value:.2f}"


def find_label_dir(model, confidence):
    name = PRED_DIRS[(model, conf_key(confidence))]

    for root in PRED_ROOTS:
        path = root / name / "labels"

        if path.exists():
            return path

    raise SystemExit(f"Missing predictions: {name}/labels")


def load_boxes(path, image_w, image_h):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        xc = float(parts[1]) * image_w
        yc = float(parts[2]) * image_h
        w = float(parts[3]) * image_w
        h = float(parts[4]) * image_h

        boxes.append({
            "x1": xc - w / 2,
            "y1": yc - h / 2,
            "x2": xc + w / 2,
            "y2": yc + h / 2,
            "cx": xc,
            "cy": yc,
            "w": w,
            "h": h,
            "area": w * h,
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

    intersection = iw * ih

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def greedy_match(gt_boxes, pred_boxes):
    """Returns {gt_index: pred_index} at IoU >= 0.50."""

    candidates = []

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            score = iou(gt, pred)

            if score >= IOU_THRESHOLD:
                candidates.append((score, gi, pi))

    candidates.sort(reverse=True)

    used_gt = set()
    used_pred = set()
    mapping = {}

    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)
        mapping[gi] = pi

    return mapping


def classify(gt_index, gt_boxes, pred_boxes, matched):
    """
    Diagnose the best-overlapping prediction for one missed person.

    Returns (mechanism, detail dict) or (None, ...) when no prediction
    overlaps the person at all.
    """

    target = gt_boxes[gt_index]

    best_pi = None
    best_iou = 0.0

    for pi, pred in enumerate(pred_boxes):
        score = iou(target, pred)

        if score > best_iou:
            best_iou = score
            best_pi = pi

    if best_pi is None or best_iou <= 0.0:
        return "no_overlap", {}

    prediction = pred_boxes[best_pi]

    # How many annotated people does this one prediction touch?
    touched = [
        gi for gi, gt in enumerate(gt_boxes)
        if iou(gt, prediction) >= TOUCH_IOU
    ]

    area_ratio = prediction["area"] / target["area"] if target["area"] else 0.0

    # Was this prediction awarded to a neighbouring person?
    owner = None

    for gi, pi in matched.items():
        if pi == best_pi:
            owner = gi
            break

    detail = {
        "best_iou": round(best_iou, 4),
        "touched_gt": len(touched),
        "area_ratio": round(area_ratio, 3),
        "claimed_by_neighbour": owner is not None and owner != gt_index,
    }

    if len(touched) >= 2 and area_ratio >= MERGE_AREA_RATIO:
        return "merged", detail

    if detail["claimed_by_neighbour"]:
        return "stolen", detail

    if len(touched) >= 2:
        return "merged", detail

    if area_ratio < UNDERSIZE_AREA_RATIO:
        return "undersized", detail

    return "offset", detail


def render(records, output_dir, limit, pad=28):
    """Crop each case so the mechanism can be inspected by eye."""

    render_dir = output_dir / "cases"
    render_dir.mkdir(parents=True, exist_ok=True)

    written = 0

    for record in records[:limit]:
        image_path = IMAGE_DIR / f"{record['image']}.jpg"

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        target = record["_target"]

        x1 = int(max(0, target["x1"] - pad))
        y1 = int(max(0, target["y1"] - pad))
        x2 = int(min(width, target["x2"] + pad))
        y2 = int(min(height, target["y2"] + pad))

        crop = image[y1:y2, x1:x2].copy()

        if crop.size == 0:
            continue

        scale = 6

        crop = cv2.resize(
            crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

        def draw(box, color, thickness=1):
            cv2.rectangle(
                crop,
                (int((box["x1"] - x1) * scale), int((box["y1"] - y1) * scale)),
                (int((box["x2"] - x1) * scale), int((box["y2"] - y1) * scale)),
                color,
                thickness,
            )

        for gt in record["_neighbours"]:
            draw(gt, COLOR_GT)

        draw(target, COLOR_TARGET, 2)

        if record["_prediction"] is not None:
            draw(record["_prediction"], COLOR_PRED, 2)

        label = f"{record['mechanism']} IoU={record['best_iou']}"

        cv2.putText(
            crop, label, (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        name = (
            f"{record['mechanism']}_{record['image']}"
            f"_{record['gt_index']}.jpg"
        )

        cv2.imwrite(str(render_dir / name), crop)

        written += 1

    return written, render_dir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument(
        "--models", nargs="*", default=["MT-005", "MT-007"],
    )
    parser.add_argument(
        "--render", type=int, default=30,
        help="Number of failure crops to render for inspection (0 disables)",
    )

    args = parser.parse_args()

    label_dirs = {
        model: find_label_dir(model, args.conf) for model in args.models
    }

    for model, path in label_dirs.items():
        print(f"{model} @ {conf_key(args.conf)}: {path}")

    failures = []
    control = []

    for gt_path in sorted(GT_DIR.glob("*.txt")):
        image_path = IMAGE_DIR / f"{gt_path.stem}.jpg"

        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            image_w, image_h = image.size

        gt_boxes = load_boxes(gt_path, image_w, image_h)

        if not gt_boxes:
            continue

        preds = {
            model: load_boxes(
                directory / f"{gt_path.stem}.txt", image_w, image_h
            )
            for model, directory in label_dirs.items()
        }

        matched = {
            model: greedy_match(gt_boxes, boxes)
            for model, boxes in preds.items()
        }

        for gi, gt in enumerate(gt_boxes):
            found_by = [m for m in args.models if gi in matched[m]]

            neighbours = [
                other for gj, other in enumerate(gt_boxes) if gj != gi
            ]

            if found_by:
                # Control: successful detections, diagnosed the same way.
                if len(found_by) == len(args.models):
                    model = args.models[0]

                    prediction = preds[model][matched[model][gi]]

                    touched = sum(
                        1 for other in gt_boxes
                        if iou(other, prediction) >= TOUCH_IOU
                    )

                    control.append({
                        "touched_gt": touched,
                        "area_ratio": (
                            prediction["area"] / gt["area"] if gt["area"] else 0
                        ),
                    })

                continue

            # Missed by every model: diagnose against each model, and keep the
            # most informative verdict.
            verdicts = []

            for model in args.models:
                mechanism, detail = classify(
                    gi, gt_boxes, preds[model], matched[model]
                )

                verdicts.append((mechanism, detail, model))

            priority = {
                "merged": 0,
                "stolen": 1,
                "undersized": 2,
                "offset": 3,
                "no_overlap": 4,
            }

            mechanism, detail, model = min(
                verdicts, key=lambda v: priority[v[0]]
            )

            if mechanism == "no_overlap":
                prediction = None
            else:
                best_pi = None
                best_iou = 0.0

                for pi, pred in enumerate(preds[model]):
                    score = iou(gt, pred)

                    if score > best_iou:
                        best_iou = score
                        best_pi = pi

                prediction = preds[model][best_pi] if best_pi is not None else None

            failures.append({
                "image": gt_path.stem,
                "gt_index": gi,
                "mechanism": mechanism,
                "diagnosed_on": model,
                "best_iou": detail.get("best_iou", 0.0),
                "touched_gt": detail.get("touched_gt", 0),
                "area_ratio": detail.get("area_ratio", 0.0),
                "claimed_by_neighbour": detail.get(
                    "claimed_by_neighbour", False
                ),
                "persons_in_image": len(gt_boxes),
                "_target": gt,
                "_neighbours": neighbours,
                "_prediction": prediction,
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "image", "gt_index", "mechanism", "diagnosed_on", "best_iou",
        "touched_gt", "area_ratio", "claimed_by_neighbour", "persons_in_image",
    ]

    csv_path = OUTPUT_DIR / "mechanisms.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for record in failures:
            writer.writerow({key: record[key] for key in fields})

    counts = Counter(r["mechanism"] for r in failures)

    total = len(failures)
    overlapping = total - counts.get("no_overlap", 0)

    crowding = counts.get("merged", 0) + counts.get("stolen", 0)

    control_multi = (
        sum(1 for c in control if c["touched_gt"] >= 2) / len(control) * 100
        if control else 0.0
    )

    summary = {
        "confidence": conf_key(args.conf),
        "models": args.models,
        "missed_by_all": total,
        "with_overlapping_prediction": overlapping,
        "mechanisms": dict(counts),
        "crowding_share_of_overlapping": (
            round(crowding / overlapping, 4) if overlapping else None
        ),
        "control_detected_multi_person_boxes_pct": round(control_multi, 2),
        "control_size": len(control),
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 68)
    print(f"Persons missed by {' + '.join(args.models)} at "
          f"confidence {conf_key(args.conf)}: {total}")
    print("=" * 68)
    print()
    print(f"{'Mechanism':14s} {'Count':>6s} {'Share of overlapping':>22s}")
    print("-" * 68)

    for mechanism in (
        "merged", "stolen", "undersized", "offset", "no_overlap"
    ):
        count = counts.get(mechanism, 0)

        if mechanism == "no_overlap":
            print(f"{mechanism:14s} {count:6d} {'(no prediction)':>22s}")
        else:
            share = count / overlapping * 100 if overlapping else 0
            print(f"{mechanism:14s} {count:6d} {share:21.1f}%")

    print("-" * 68)
    print(
        f"{'crowding':14s} {crowding:6d} "
        f"{crowding / overlapping * 100 if overlapping else 0:21.1f}%"
    )

    print()
    print("Control: successfully detected persons")
    print("-" * 68)
    print(
        f"  predictions covering 2+ annotated people: "
        f"{control_multi:.2f}%  (n={len(control)})"
    )

    if args.render:
        written, render_dir = render(failures, OUTPUT_DIR, args.render)
        print()
        print(f"Rendered {written} annotated crops to {render_dir}")
        print("  orange = missed person, green = other annotated people, "
              "red = best-overlapping prediction")

    print()
    print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
