"""
Three-model thermal ensemble as a real inference pipeline.

`ensemble_thermal.py` fused prediction *files* to answer an analysis question:
is an ensemble worth building? It is - the fusion recovers 51 persons over the
MT-005 baseline at 1.6 false positives each. This script is the answer to the
follow-up question: what does it cost to actually run.

    frame -> preprocess -+-> MT-005 -+
                         +-> MT-006 -+-> WBF -> final detections
                         +-> MT-007 -+

The distinction matters. Forward-pass timings alone are not the engineering
number; the payload pays for preprocessing, three inferences, fusion and
post-processing on every frame. This runs all of it on real images and times
each stage separately, so the budget can be read rather than guessed.

Two execution modes:

  eager       ordinary PyTorch inference
  cudagraph   each model captured into a CUDA graph and replayed

The second exists because the models are launch-bound at batch 1: on the
development GPU the CPU spends as long enqueuing kernels as the whole frame
takes, and graph replay removes that while running bit-identical kernels.
Graph capture requires a fixed input shape, which a fixed-resolution sensor
provides.

Fusion is weighted box fusion, which beat plain NMS at every vote level in the
file-level study. That is expected rather than lucky: most of the shared
localisation failures are near misses between IoU 0.25 and 0.50, and averaging
independent near-miss boxes moves the result across the threshold.

Usage:

    python scripts/ensemble_detect.py --limit 100
    python scripts/ensemble_detect.py --mode cudagraph --limit 200
    python scripts/ensemble_detect.py --mode cudagraph --save-labels
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes

try:
    # Ultralytics 8.4 moved NMS out of utils.ops into its own module.
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # pragma: no cover - older Ultralytics
    from ultralytics.utils.ops import non_max_suppression

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"
IMAGE_DIR = PROJECT_ROOT / "dataset" / "hit_uav_person" / "images" / "test"
GT_DIR = PROJECT_ROOT / "dataset" / "hit_uav_person" / "labels" / "test"

OUTPUT_DIR = PROJECT_ROOT / "results" / "ensemble_pipeline"

# Each model runs at its own training resolution, which is what it was
# evaluated at. The network input shape is fixed per model so CUDA graphs can
# be captured.
ENSEMBLE = [
    {"id": "MT-005", "shape": (512, 640)},
    {"id": "MT-006", "shape": (768, 960)},
    {"id": "MT-007", "shape": (512, 640)},
]

CONF = 0.25
NMS_IOU = 0.70
FUSE_IOU = 0.55
IOU_THRESHOLD = 0.50   # person-level matching


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def letterbox(image, shape):
    """Resize into `shape` (h, w) preserving aspect, padding with 114."""

    target_h, target_w = shape
    height, width = image.shape[:2]

    ratio = min(target_h / height, target_w / width)

    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))

    if (new_w, new_h) != (width, height):
        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        resized = image

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas, ratio, (left, top)


def to_tensor(image, device):
    array = image[:, :, ::-1].transpose(2, 0, 1)

    tensor = torch.from_numpy(np.ascontiguousarray(array))

    return tensor.to(device).float().div_(255.0).unsqueeze(0)


# ---------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------

class EagerRunner:
    def __init__(self, module):
        self.module = module

    def __call__(self, tensor):
        with torch.inference_mode():
            output = self.module(tensor)

        return output[0] if isinstance(output, (list, tuple)) else output


class GraphRunner:
    """CUDA-graph replay. Requires a fixed input shape."""

    def __init__(self, module, shape, device):
        self.static_input = torch.zeros(
            1, 3, shape[0], shape[1], device=device
        )

        with torch.inference_mode():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(side):
                for _ in range(5):
                    module(self.static_input)

            torch.cuda.current_stream().wait_stream(side)

            self.graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(self.graph):
                output = module(self.static_input)

        self.static_output = (
            output[0] if isinstance(output, (list, tuple)) else output
        )

    def __call__(self, tensor):
        self.static_input.copy_(tensor)
        self.graph.replay()

        return self.static_output


# ---------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------

def iou_1_to_n(box, others):
    if not len(others):
        return np.zeros(0)

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])

    union = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0)


def weighted_box_fusion(boxes, scores, sources, fuse_iou, min_votes):
    """Cluster overlapping boxes and average each cluster by confidence."""

    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)

    order = np.argsort(-scores)

    boxes = boxes[order]
    scores = scores[order]
    sources = np.asarray(sources)[order]

    used = np.zeros(len(boxes), dtype=bool)

    fused_boxes = []
    fused_scores = []

    for index in range(len(boxes)):
        if used[index]:
            continue

        candidates = np.where(~used)[0]

        overlaps = iou_1_to_n(boxes[index], boxes[candidates])

        member_idx = candidates[overlaps >= fuse_iou]

        if index not in member_idx:
            member_idx = np.append(member_idx, index)

        used[member_idx] = True

        votes = len(set(sources[member_idx]))

        if votes < min_votes:
            continue

        weights = scores[member_idx]
        total = weights.sum()

        if total <= 0:
            continue

        fused_boxes.append(
            (boxes[member_idx] * weights[:, None]).sum(axis=0) / total
        )

        fused_scores.append(float(weights.mean()))

    if not fused_boxes:
        return np.zeros((0, 4)), np.zeros(0)

    return np.asarray(fused_boxes), np.asarray(fused_scores)


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def load_gt(path, width, height):
    boxes = []

    if not path.exists():
        return np.zeros((0, 4))

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        xc = float(parts[1]) * width
        yc = float(parts[2]) * height
        w = float(parts[3]) * width
        h = float(parts[4]) * height

        boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])

    return np.asarray(boxes) if boxes else np.zeros((0, 4))


def match(gt, preds):
    """One-to-one greedy matching at IoU >= 0.50."""

    if len(gt) == 0 or len(preds) == 0:
        return 0, len(preds)

    pairs = []

    for gi in range(len(gt)):
        overlaps = iou_1_to_n(gt[gi], preds)

        for pi in np.where(overlaps >= IOU_THRESHOLD)[0]:
            pairs.append((overlaps[pi], gi, int(pi)))

    pairs.sort(reverse=True)

    used_gt = set()
    used_pred = set()

    for _, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

    return len(used_gt), len(preds) - len(used_pred)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def build_runners(mode, device):
    runners = []

    for entry in ENSEMBLE:
        weights = RUNS_ROOT / entry["id"] / "weights" / "best.pt"

        if not weights.exists():
            raise SystemExit(f"Missing weights: {weights}")

        module = YOLO(str(weights)).model.to(device).eval().fuse()

        if mode == "cudagraph":
            if device.type != "cuda":
                raise SystemExit("cudagraph mode requires CUDA")

            runner = GraphRunner(module, entry["shape"], device)
        else:
            runner = EagerRunner(module)

        runners.append({
            "id": entry["id"],
            "shape": entry["shape"],
            "run": runner,
        })

        print(f"  loaded {entry['id']} @ {entry['shape']} ({mode})")

    return runners


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", default="eager", choices=["eager", "cudagraph"]
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--save-labels", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}  mode: {args.mode}")

    runners = build_runners(args.mode, device)

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if args.limit:
        images = images[:args.limit]

    print(f"images: {len(images)}")

    label_dir = OUTPUT_DIR / f"labels_{args.mode}"

    if args.save_labels:
        label_dir.mkdir(parents=True, exist_ok=True)

    timings = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
        "fusion": [],
        "total": [],
    }

    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_unmatched = 0
    blind_images = 0

    for index, image_path in enumerate(images):
        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        height, width = frame.shape[:2]

        warming = index < args.warmup

        start_total = time.perf_counter()

        # --- preprocess -------------------------------------------------
        start = time.perf_counter()

        prepared = []

        for runner in runners:
            padded, ratio, offset = letterbox(frame, runner["shape"])
            prepared.append((to_tensor(padded, device), ratio, offset))

        if device.type == "cuda":
            torch.cuda.synchronize()

        preprocess_s = time.perf_counter() - start

        # --- inference --------------------------------------------------
        start = time.perf_counter()

        raw_outputs = [
            runner["run"](tensor)
            for runner, (tensor, _, _) in zip(runners, prepared)
        ]

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_s = time.perf_counter() - start

        # --- postprocess (NMS per model) --------------------------------
        start = time.perf_counter()

        pooled_boxes = []
        pooled_scores = []
        pooled_sources = []

        for runner, output, (tensor, ratio, offset) in zip(
            runners, raw_outputs, prepared
        ):
            detections = non_max_suppression(
                output.clone(),
                conf_thres=args.conf,
                iou_thres=NMS_IOU,
                max_det=300,
            )[0]

            if detections is None or not len(detections):
                continue

            boxes = detections[:, :4].clone()

            boxes = scale_boxes(
                tensor.shape[2:], boxes, (height, width)
            )

            pooled_boxes.append(boxes.cpu().numpy())
            pooled_scores.append(detections[:, 4].cpu().numpy())
            pooled_sources.extend([runner["id"]] * len(detections))

        postprocess_s = time.perf_counter() - start

        # --- fusion -----------------------------------------------------
        start = time.perf_counter()

        if pooled_boxes:
            boxes = np.concatenate(pooled_boxes)
            scores = np.concatenate(pooled_scores)

            fused_boxes, fused_scores = weighted_box_fusion(
                boxes, scores, pooled_sources, FUSE_IOU, args.min_votes
            )
        else:
            fused_boxes = np.zeros((0, 4))
            fused_scores = np.zeros(0)

        fusion_s = time.perf_counter() - start

        total_s = time.perf_counter() - start_total

        if not warming:
            timings["preprocess"].append(preprocess_s)
            timings["inference"].append(inference_s)
            timings["postprocess"].append(postprocess_s)
            timings["fusion"].append(fusion_s)
            timings["total"].append(total_s)

        # --- scoring ----------------------------------------------------
        gt = load_gt(GT_DIR / f"{image_path.stem}.txt", width, height)

        matched, unmatched = match(gt, fused_boxes)

        total_gt += len(gt)
        total_pred += len(fused_boxes)
        total_matched += matched
        total_unmatched += unmatched

        if len(fused_boxes) == 0 and len(gt) > 0:
            blind_images += 1

        if args.save_labels:
            lines = []

            for box, score in zip(fused_boxes, fused_scores):
                xc = (box[0] + box[2]) / 2 / width
                yc = (box[1] + box[3]) / 2 / height
                bw = (box[2] - box[0]) / width
                bh = (box[3] - box[1]) / height

                lines.append(
                    f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {score:.6f}"
                )

            (label_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(lines)
            )

    def stats(values):
        if not values:
            return None

        ordered = sorted(v * 1000 for v in values)

        return {
            "median_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
            "mean_ms": round(statistics.fmean(ordered), 2),
        }

    report = {
        "mode": args.mode,
        "models": [e["id"] for e in ENSEMBLE],
        "confidence": args.conf,
        "min_votes": args.min_votes,
        "images": len(timings["total"]),
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else "cpu"
        ),
        "stages": {key: stats(values) for key, values in timings.items()},
        "accuracy": {
            "ground_truth_persons": total_gt,
            "predicted_boxes": total_pred,
            "matched": total_matched,
            "missed": total_gt - total_matched,
            "unmatched": total_unmatched,
            "recall": round(total_matched / total_gt, 4) if total_gt else None,
            "precision": (
                round(total_matched / total_pred, 4) if total_pred else None
            ),
            "blind_images": blind_images,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / f"pipeline_{args.mode}.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print("=" * 60)
    print(f"Full-pipeline budget per frame ({args.mode})")
    print("=" * 60)

    for stage in ("preprocess", "inference", "postprocess", "fusion", "total"):
        entry = report["stages"][stage]

        if entry:
            print(
                f"  {stage:12s} median {entry['median_ms']:7.2f} ms   "
                f"p95 {entry['p95_ms']:7.2f} ms"
            )

    total_ms = report["stages"]["total"]["median_ms"]

    print(f"\n  pipeline FPS (median): {1000 / total_ms:.1f}")

    accuracy = report["accuracy"]

    print()
    print("=" * 60)
    print("Accuracy (IoU >= 0.50, confidence "
          f"{args.conf}, min votes {args.min_votes})")
    print("=" * 60)

    for key, value in accuracy.items():
        print(f"  {key:22s} {value}")

    print()
    print(f"Written: {OUTPUT_DIR / f'pipeline_{args.mode}.json'}")


if __name__ == "__main__":
    main()
