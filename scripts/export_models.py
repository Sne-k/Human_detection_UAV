"""
Export trained models for embedded inference, and verify the exports.

The latency benchmark found that the PyTorch models are launch-bound at batch 1
on the development GPU: the CPU spends as long enqueuing kernels as the whole
frame takes, which is why 1280 px costs the same as 640 px. A workload bounded
by kernel launch count cannot be improved by shrinking the model, so the useful
optimisation is a fused, exported graph.

This script performs the export and, importantly, checks that the exported
model still produces the same detections. An export that is fast and wrong is
worse than no export, so every format is verified against the PyTorch model on
real dataset images rather than on random tensors.

Verification compares decoded detections - box count, box IoU and confidence
agreement - against the PyTorch model on real dataset images rather than on
random tensors.

Export shape matters more than it looks. An ONNX graph has a fixed input
shape, while the PyTorch predict path letterboxes rectangularly to a stride
multiple. For a 640 x 512 thermal sensor, PyTorch feeds the network 512 x 640
but a square 640 x 640 export pads to a different shape, which silently changes
which marginal detections survive:

    exported at 640 x 640   792 -> 778 boxes, mean IoU 0.972, 27 lost
    exported at 512 x 640   792 -> 792 boxes, mean IoU 1.000,  0 lost

The graph itself is faithful either way - raw pre-NMS outputs agree to 1.7e-6
on the confidence channel - so this is a preprocessing mismatch, not an export
defect. Each model therefore declares the shape it is exported at, matching the
shape its own predict path uses.

Usage:

    python scripts/export_models.py --models MT-005
    python scripts/export_models.py --models MT-004 MT-005 --format onnx
    python scripts/export_models.py --models MT-005 --no-verify
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"
OUTPUT_DIR = PROJECT_ROOT / "results" / "export"

# Images used for verification, per modality.
VERIFY_SOURCES = {
    "RGB": PROJECT_ROOT / "dataset" / "visdrone_person" / "images" / "val",
    "Thermal": PROJECT_ROOT / "dataset" / "hit_uav_person" / "images" / "test",
}

VERIFY_IMAGES = 20

# Agreement thresholds. Export is numerically lossy, so exact equality is not
# required; these bound how much drift is acceptable. When the export shape
# matches the predict shape the thermal models agree exactly, so a failure here
# means something genuinely changed.
MAX_BOX_COUNT_DRIFT = 0.02   # fraction of detections gained or lost
MIN_MEAN_IOU = 0.99          # mean IoU between matched boxes
MAX_CONF_DELTA = 0.02        # mean absolute confidence difference

# export_shape is the fixed input shape baked into the exported graph, as
# [height, width]. It must match the shape the PyTorch predict path letterboxes
# to, or marginal detections diverge. HIT-UAV imagery is 640 x 512, so the
# thermal models export at 512 x 640 rather than square.
#
# VisDrone images have mixed aspect ratios, so no single fixed shape reproduces
# the PyTorch path for every image. MT-004 therefore exports square and is
# expected to show some drift; a deployed RGB pipeline should letterbox to the
# exported shape itself rather than relying on the Ultralytics predict path.
MODELS = {
    "MT-004": {"imgsz": 1280, "export_shape": 1280, "modality": "RGB"},
    "MT-005": {"imgsz": 640, "export_shape": [512, 640], "modality": "Thermal"},
    "MT-006": {"imgsz": 960, "export_shape": [768, 960], "modality": "Thermal"},
    "MT-007": {"imgsz": 640, "export_shape": [512, 640], "modality": "Thermal"},
}


def verification_images(modality, count=VERIFY_IMAGES):
    source = VERIFY_SOURCES.get(modality)

    if source is None or not source.exists():
        return []

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    images = sorted(
        p for p in source.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        return []

    # Spread the sample across the split rather than taking the first N,
    # which would all come from one flight or one scene.
    stride = max(1, len(images) // count)

    return [str(p) for p in images[::stride][:count]]


def iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])

    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    union = area_a + area_b - intersection

    return np.where(union > 0, intersection / union, 0.0)


def detections(model, images, imgsz, device):
    """Run prediction and return per-image (boxes, confidences)."""

    output = []

    for image in images:
        result = model.predict(
            image,
            imgsz=imgsz,
            conf=0.25,
            iou=0.70,
            device=device,
            verbose=False,
        )[0]

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            output.append((np.zeros((0, 4)), np.zeros((0,))))
            continue

        output.append((
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
        ))

    return output


def compare(reference, candidate):
    """Compare two detection sets image by image."""

    total_reference = 0
    total_candidate = 0
    total_matched = 0

    ious = []
    conf_deltas = []
    unmatched_reference_conf = []
    unmatched_candidate_conf = []

    for (ref_boxes, ref_conf), (cand_boxes, cand_conf) in zip(
        reference, candidate
    ):
        total_reference += len(ref_boxes)
        total_candidate += len(cand_boxes)

        if len(ref_boxes) == 0 or len(cand_boxes) == 0:
            unmatched_reference_conf.extend(float(c) for c in ref_conf)
            unmatched_candidate_conf.extend(float(c) for c in cand_conf)
            continue

        matrix = iou_matrix(ref_boxes, cand_boxes)

        # Greedy one-to-one matching, highest IoU first.
        used = set()
        matched_reference = set()

        for ref_index in np.argsort(-ref_conf):
            row = matrix[ref_index]

            order = np.argsort(-row)

            for cand_index in order:
                if cand_index in used:
                    continue

                if row[cand_index] <= 0.0:
                    break

                used.add(int(cand_index))
                matched_reference.add(int(ref_index))
                total_matched += 1

                ious.append(float(row[cand_index]))

                conf_deltas.append(
                    abs(float(ref_conf[ref_index]) - float(cand_conf[cand_index]))
                )

                break

        unmatched_reference_conf.extend(
            float(ref_conf[i])
            for i in range(len(ref_boxes))
            if i not in matched_reference
        )

        unmatched_candidate_conf.extend(
            float(cand_conf[i])
            for i in range(len(cand_boxes))
            if i not in used
        )

    count_drift = (
        abs(total_candidate - total_reference) / total_reference
        if total_reference
        else 0.0
    )

    report = {
        "reference_boxes": total_reference,
        "exported_boxes": total_candidate,
        "matched_boxes": total_matched,
        "box_count_drift": round(count_drift, 4),
        "mean_iou": round(float(np.mean(ious)), 5) if ious else None,
        "min_iou": round(float(np.min(ious)), 5) if ious else None,
        "mean_conf_delta": (
            round(float(np.mean(conf_deltas)), 5) if conf_deltas else None
        ),
        "max_conf_delta": (
            round(float(np.max(conf_deltas)), 5) if conf_deltas else None
        ),
        "unmatched_reference": len(unmatched_reference_conf),
        "unmatched_exported": len(unmatched_candidate_conf),
        "unmatched_reference_max_conf": (
            round(float(np.max(unmatched_reference_conf)), 4)
            if unmatched_reference_conf else None
        ),
        "unmatched_exported_max_conf": (
            round(float(np.max(unmatched_candidate_conf)), 4)
            if unmatched_candidate_conf else None
        ),
    }

    checks = {
        "box_count_drift_ok": count_drift <= MAX_BOX_COUNT_DRIFT,
        "mean_iou_ok": bool(ious) and float(np.mean(ious)) >= MIN_MEAN_IOU,
        "conf_delta_ok": (
            bool(conf_deltas)
            and float(np.mean(conf_deltas)) <= MAX_CONF_DELTA
        ),
    }

    report["checks"] = checks
    report["passed"] = all(checks.values())

    return report


def export_one(name, entry, args):
    weights = RUNS_ROOT / name / "weights" / "best.pt"

    if not weights.exists():
        print(f"SKIP {name}: weights not found at {weights}")
        return None

    imgsz = entry["imgsz"]
    export_shape = entry.get("export_shape", imgsz)

    print()
    print("=" * 62)
    print(f"{name} - {entry['modality']} - export shape {export_shape} "
          f"-> {args.format}")
    print("=" * 62)

    model = YOLO(str(weights))

    exported_path = model.export(
        format=args.format,
        imgsz=export_shape,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
        device=args.device,
    )

    exported_path = Path(exported_path)

    row = {
        "experiment": name,
        "modality": entry["modality"],
        "imgsz": imgsz,
        "export_shape": export_shape,
        "format": args.format,
        "source_weights": str(weights),
        "exported": str(exported_path),
        "size_mb": round(exported_path.stat().st_size / (1024 * 1024), 2),
    }

    print(f"exported: {exported_path} ({row['size_mb']} MB)")

    if args.no_verify:
        return row

    images = verification_images(entry["modality"], args.verify_images)

    if not images:
        print(
            f"WARNING: no verification images for {entry['modality']}; "
            "export not verified"
        )
        row["verification"] = None
        return row

    print(f"verifying on {len(images)} images...")

    reference = detections(model, images, imgsz, args.device)

    exported_model = YOLO(str(exported_path))

    # The exported graph is fixed-shape, so it must be driven at the shape it
    # was exported at, not at the nominal training imgsz.
    candidate = detections(exported_model, images, export_shape, args.device)

    report = compare(reference, candidate)
    report["images"] = len(images)

    row["verification"] = report

    print(
        f"  boxes {report['reference_boxes']} -> {report['exported_boxes']}  "
        f"(drift {report['box_count_drift']:.4f})"
    )
    print(
        f"  mean IoU {report['mean_iou']}  min IoU {report['min_iou']}  "
        f"mean conf delta {report['mean_conf_delta']}"
    )
    print(f"  VERDICT: {'PASS' if report['passed'] else 'FAIL'}")

    if not report["passed"]:
        failed = [k for k, v in report["checks"].items() if not v]
        print(f"  failed checks: {', '.join(failed)}")

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", nargs="*", default=["MT-004", "MT-005"])
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--verify-images", type=int, default=VERIFY_IMAGES)

    args = parser.parse_args()

    rows = []

    for name in args.models:
        if name not in MODELS:
            print(f"SKIP {name}: unknown model")
            continue

        result = export_one(name, MODELS[name], args)

        if result is not None:
            rows.append(result)

    if not rows:
        print("Nothing exported.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUT_DIR / f"export_{args.format}.json"
    report_path.write_text(json.dumps(rows, indent=2))

    print()
    print(f"Written: {report_path}")

    verified = [
        r for r in rows
        if r.get("verification") is not None
    ]

    if verified:
        passed = sum(1 for r in verified if r["verification"]["passed"])
        print(f"Verified {passed}/{len(verified)} exports.")


if __name__ == "__main__":
    main()
