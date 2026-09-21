from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv
import math

PROJECT = Path(__file__).resolve().parents[1]

GT_DIR = PROJECT / "dataset" / "hit_uav_person" / "labels" / "test"
IMAGE_DIR = PROJECT / "dataset" / "hit_uav_person" / "images" / "test"

PRED_ROOT = (
    PROJECT
    / "runs"
    / "detect"
    / "results"
    / "error_analysis"
)

MT005_DIR = PRED_ROOT / "MT-005-test-analysis" / "labels"
MT007_DIR = PRED_ROOT / "MT-007-test-analysis" / "labels"

COMPARISON_CSV = (
    PROJECT
    / "results"
    / "error_analysis"
    / "MT005_vs_MT007"
    / "person_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "error_analysis"
    / "MT005_vs_MT007"
    / "contact_sheets"
)

IOU_THRESHOLD = 0.50

# Number of examples per group
MAX_BOTH_MISSED = 24
MAX_MT005_ONLY = 24
MAX_MT007_ONLY = 24

# Contact-sheet layout
COLS = 3
CELL_W = 500
CELL_H = 350
HEADER_H = 55


def find_image(stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = IMAGE_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def load_labels(path, image_w, image_h):
    boxes = []

    if not path.exists():
        return boxes

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            xc = float(parts[1]) * image_w
            yc = float(parts[2]) * image_h
            w = float(parts[3]) * image_w
            h = float(parts[4]) * image_h

            conf = (
                float(parts[5])
                if len(parts) >= 6
                else None
            )

            boxes.append({
                "x1": xc - w / 2,
                "y1": yc - h / 2,
                "x2": xc + w / 2,
                "y2": yc + h / 2,
                "w": w,
                "h": h,
                "conf": conf,
            })

    return boxes


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)

    inter = iw * ih

    area_a = (
        max(0, a["x2"] - a["x1"])
        * max(0, a["y2"] - a["y1"])
    )

    area_b = (
        max(0, b["x2"] - b["x1"])
        * max(0, b["y2"] - b["y1"])
    )

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def best_match(gt, predictions):
    best = None
    best_iou = 0.0

    for i, pred in enumerate(predictions):
        score = iou(gt, pred)

        if score > best_iou:
            best_iou = score
            best = i

    if best is not None and best_iou >= IOU_THRESHOLD:
        return best, best_iou

    return None, best_iou


def draw_box(draw, box, scale_x, scale_y, label, outline, width=3):

    x1 = box["x1"] * scale_x
    y1 = box["y1"] * scale_y
    x2 = box["x2"] * scale_x
    y2 = box["y2"] * scale_y

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=outline,
        width=width,
    )

    draw.text(
        (x1 + 3, max(0, y1 - 16)),
        label,
        fill=outline,
    )


def make_panel(row):

    stem = row["image"]

    image_path = find_image(stem)

    if image_path is None:
        return None

    image = Image.open(image_path).convert("RGB")

    original_w, original_h = image.size

    # Scale image to fit panel
    available_w = CELL_W
    available_h = CELL_H - HEADER_H

    scale = min(
        available_w / original_w,
        available_h / original_h,
    )

    display_w = max(1, int(original_w * scale))
    display_h = max(1, int(original_h * scale))

    image = image.resize(
        (display_w, display_h),
        Image.Resampling.LANCZOS,
    )

    panel = Image.new(
        "RGB",
        (CELL_W, CELL_H),
        "white",
    )

    draw = ImageDraw.Draw(panel)

    panel.paste(
        image,
        (
            (CELL_W - display_w) // 2,
            HEADER_H,
        ),
    )

    offset_x = (CELL_W - display_w) // 2
    offset_y = HEADER_H

    scale_x = display_w / original_w
    scale_y = display_h / original_h

    gt_path = GT_DIR / f"{stem}.txt"
    mt005_path = MT005_DIR / f"{stem}.txt"
    mt007_path = MT007_DIR / f"{stem}.txt"

    gt_boxes = load_labels(
        gt_path,
        original_w,
        original_h,
    )

    mt005_boxes = load_labels(
        mt005_path,
        original_w,
        original_h,
    )

    mt007_boxes = load_labels(
        mt007_path,
        original_w,
        original_h,
    )

    # Draw GT first
    for gt in gt_boxes:

        shifted = gt.copy()

        shifted["x1"] += offset_x / scale_x
        shifted["x2"] += offset_x / scale_x
        shifted["y1"] -= HEADER_H / scale_y
        shifted["y2"] -= HEADER_H / scale_y

        draw_box(
            draw,
            shifted,
            scale_x,
            scale_y,
            "GT",
            "lime",
            3,
        )

    # Predictions
    if row["mt005_matched"] == "1":

        matches = []

        for pred in mt005_boxes:
            for gt in gt_boxes:
                score = iou(gt, pred)

                if score >= IOU_THRESHOLD:
                    matches.append((score, pred))

        if matches:
            score, pred = max(matches, key=lambda x: x[0])

            draw_box(
                draw,
                pred,
                scale_x,
                scale_y,
                f"MT5 {score:.2f}",
                "yellow",
                3,
            )

    if row["mt007_matched"] == "1":

        matches = []

        for pred in mt007_boxes:
            for gt in gt_boxes:
                score = iou(gt, pred)

                if score >= IOU_THRESHOLD:
                    matches.append((score, pred))

        if matches:
            score, pred = max(matches, key=lambda x: x[0])

            draw_box(
                draw,
                pred,
                scale_x,
                scale_y,
                f"MT7 {score:.2f}",
                "cyan",
                3,
            )

    # Header
    category = row["category"]

    title = (
        f"{category} | "
        f"{stem}"
    )

    draw.text(
        (8, 8),
        title,
        fill="black",
    )

    size_text = (
        f"GT: "
        f"{float(row['gt_width_px']):.1f}"
        f"x"
        f"{float(row['gt_height_px']):.1f}px"
    )

    draw.text(
        (8, 28),
        size_text,
        fill="black",
    )

    return panel


def select_rows(rows, category, maximum):

    candidates = [
        r for r in rows
        if r["category"] == category
    ]

    # Sort by GT area so the sheet samples
    # a range of target sizes rather than only
    # taking arbitrary first entries.
    candidates.sort(
        key=lambda r: float(r["gt_area_px2"])
    )

    if len(candidates) <= maximum:
        return candidates

    # Evenly sample the sorted list.
    selected = []

    for i in range(maximum):
        idx = round(
            i * (len(candidates) - 1)
            / (maximum - 1)
        )
        selected.append(candidates[idx])

    return selected


def create_sheet(rows, category, filename):

    selected = select_rows(
        rows,
        category,
        {
            "both_missed": MAX_BOTH_MISSED,
            "mt005_only": MAX_MT005_ONLY,
            "mt007_only": MAX_MT007_ONLY,
        }[category],
    )

    panels = []

    for row in selected:

        panel = make_panel(row)

        if panel is not None:
            panels.append(panel)

    if not panels:
        print(f"No images available for {category}")
        return

    rows_count = math.ceil(
        len(panels) / COLS
    )

    sheet = Image.new(
        "RGB",
        (
            COLS * CELL_W,
            rows_count * CELL_H,
        ),
        "white",
    )

    for i, panel in enumerate(panels):

        x = (i % COLS) * CELL_W
        y = (i // COLS) * CELL_H

        sheet.paste(
            panel,
            (x, y),
        )

    output_path = OUTPUT_DIR / filename

    sheet.save(
        output_path,
        quality=95,
    )

    print(
        f"{category}: "
        f"{len(panels)} examples -> "
        f"{output_path}"
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"Comparison CSV not found:\n"
            f"{COMPARISON_CSV}"
        )

    with open(
        COMPARISON_CSV,
        "r",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    print(
        f"Loaded {len(rows)} person-level records."
    )

    create_sheet(
        rows,
        "both_missed",
        "01_both_missed.jpg",
    )

    create_sheet(
        rows,
        "mt005_only",
        "02_mt005_only.jpg",
    )

    create_sheet(
        rows,
        "mt007_only",
        "03_mt007_only.jpg",
    )

    print("\nVisualization complete.")
    print(f"Output directory:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()