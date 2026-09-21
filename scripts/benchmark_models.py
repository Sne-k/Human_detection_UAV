"""
Controlled benchmark of the trained human-detection models.

Every model is evaluated with the Ultralytics validator on a fixed
evaluation split so that the reported numbers are directly comparable
within a modality.

RGB (VisDrone) and thermal (HIT-UAV) numbers are reported separately.
They must not be compared across modalities, because the two datasets
have different imagery, annotation policies and object statistics.

Usage:

    python scripts/benchmark_models.py --models MT-005
    python scripts/benchmark_models.py --models MT-006

Each evaluated model is cached as results/benchmark/models/<ID>.json and the
summary table is rebuilt from every cached model, so the benchmark can be
filled in one model per invocation. That is the recommended way to run it:
evaluating several models inside a single process keeps the previous models
resident in VRAM on a 6 GB GPU and inflates the reported inference time.

The project root is taken from the script location and can be overridden
with the HDU_ROOT environment variable, which is useful when the script
is executed from a Git worktree while the datasets and training runs live
in the main checkout.
"""

import argparse
import csv
import json
import os
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"
OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark"
MODEL_DIR = OUTPUT_DIR / "models"

DEVICE = 0
CONF = 0.001          # Ultralytics AP default; keeps mAP threshold-independent
IOU = 0.70            # NMS IoU, Ultralytics default

# ---------------------------------------------------------------------
# Benchmark definitions
#
# imgsz is the input size the model was trained at; evaluating at the
# training resolution is what makes each row a fair report of that model.
# ---------------------------------------------------------------------

BENCHMARKS = [
    {
        "id": "MT-004",
        "modality": "RGB",
        "trained_on": "VisDrone Person",
        "train_imgsz": 1280,
        "weights": RUNS_ROOT / "MT-004" / "weights" / "best.pt",
        "data": PROJECT_ROOT / "dataset" / "visdrone_person" / "data.yaml",
        "split": "val",
        "imgsz": 1280,
        "batch": 1,
    },
    {
        "id": "MT-005",
        "modality": "Thermal",
        "trained_on": "HIT-UAV Person",
        "train_imgsz": 640,
        "weights": RUNS_ROOT / "MT-005" / "weights" / "best.pt",
        "data": PROJECT_ROOT / "dataset" / "hit_uav_person" / "data.yaml",
        "split": "test",
        "imgsz": 640,
        "batch": 1,
    },
    {
        "id": "MT-006",
        "modality": "Thermal",
        "trained_on": "HIT-UAV Person",
        "train_imgsz": 960,
        "weights": RUNS_ROOT / "MT-006" / "weights" / "best.pt",
        "data": PROJECT_ROOT / "dataset" / "hit_uav_person" / "data.yaml",
        "split": "test",
        "imgsz": 960,
        "batch": 1,
    },
    {
        "id": "MT-007",
        "modality": "Thermal",
        "trained_on": "HIT-UAV Person + 256 px crops",
        "train_imgsz": 640,
        "weights": RUNS_ROOT / "MT-007" / "weights" / "best.pt",
        # Evaluated on the untouched HIT-UAV test split, not on crops,
        # so MT-005 / MT-006 / MT-007 share one evaluation set.
        "data": PROJECT_ROOT / "dataset" / "hit_uav_person" / "data.yaml",
        "split": "test",
        "imgsz": 640,
        "batch": 1,
    },
]


def count_images(entry):
    """Number of images in the evaluation split, read from the dataset."""

    data_root = Path(entry["data"]).parent
    image_dir = data_root / "images" / entry["split"]

    if not image_dir.exists():
        return None

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    return sum(
        1 for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    )


def evaluate(entry):
    weights = entry["weights"]

    if not weights.exists():
        print(f"SKIP {entry['id']}: weights not found at {weights}")
        return None

    if not Path(entry["data"]).exists():
        print(f"SKIP {entry['id']}: data.yaml not found at {entry['data']}")
        return None

    print()
    print("=" * 62)
    print(f"{entry['id']} - {entry['modality']} - {entry['split']} split")
    print(f"weights: {weights}")
    print(f"imgsz:   {entry['imgsz']}")
    print("=" * 62)

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(entry["data"]),
        split=entry["split"],
        imgsz=entry["imgsz"],
        batch=entry["batch"],
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        workers=0,
        plots=False,
        save_json=False,
        project=str(OUTPUT_DIR / "runs"),
        name=entry["id"],
        exist_ok=True,
        verbose=False,
    )

    speed = metrics.speed

    return {
        "experiment": entry["id"],
        "modality": entry["modality"],
        "trained_on": entry["trained_on"],
        "train_imgsz": entry["train_imgsz"],
        "eval_split": entry["split"],
        "eval_imgsz": entry["imgsz"],
        "images": count_images(entry),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "preprocess_ms": round(float(speed.get("preprocess", 0.0)), 2),
        "inference_ms": round(float(speed.get("inference", 0.0)), 2),
        "postprocess_ms": round(float(speed.get("postprocess", 0.0)), 2),
        "total_ms": round(
            float(speed.get("preprocess", 0.0))
            + float(speed.get("inference", 0.0))
            + float(speed.get("postprocess", 0.0)),
            2,
        ),
    }


def cache_result(row):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    path = MODEL_DIR / f"{row['experiment']}.json"
    path.write_text(json.dumps(row, indent=2))


def load_cached_results():
    """Every previously evaluated model, in BENCHMARKS order."""

    rows = []

    for entry in BENCHMARKS:
        path = MODEL_DIR / f"{entry['id']}.json"

        if path.exists():
            rows.append(json.loads(path.read_text()))

    return rows


def write_outputs(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "benchmark.csv"
    json_path = OUTPUT_DIR / "benchmark.json"
    md_path = OUTPUT_DIR / "benchmark.md"

    fields = list(rows[0].keys())

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2))

    lines = [
        "| Experiment | Modality | Split | imgsz | Images | P | R | mAP@50 | mAP@50-95 | Inference (ms) |",
        "| ---------- | -------- | ----- | ----: | -----: | -: | -: | -----: | --------: | -------------: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['modality']} | {row['eval_split']} "
            f"| {row['eval_imgsz']} | {row['images']} "
            f"| {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['mAP50']:.3f} | {row['mAP50_95']:.3f} "
            f"| {row['inference_ms']:.1f} |"
        )

    md_path.write_text("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print()
    print(f"Written: {csv_path}")
    print(f"Written: {json_path}")
    print(f"Written: {md_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of experiment IDs to evaluate, e.g. MT-005 MT-006",
    )

    args = parser.parse_args()

    selected = [
        entry for entry in BENCHMARKS
        if args.models is None or entry["id"] in args.models
    ]

    evaluated = 0

    for entry in selected:
        result = evaluate(entry)

        if result is not None:
            cache_result(result)
            evaluated += 1

    rows = load_cached_results()

    if not rows:
        print("No models were evaluated.")
        return

    print()
    print(f"Evaluated this run: {evaluated}. Models in summary: {len(rows)}.")

    write_outputs(rows)


if __name__ == "__main__":
    main()
