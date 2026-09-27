"""
What the tracker costs the payload, measured on sequences.

The payload's output is tracks, not detections: every record it emits comes
from ByteTrack. That makes the tracker a gate between the detector and the
rescue team, and it had never been measured - because every accuracy figure in
this project comes from the HIT-UAV test split, 579 independent still images,
across which tracking is meaningless. `verify_deployed_config.py` validates
the detection stage and has to stop there.

This script builds sequences with exact per-frame ground truth from those same
test images, runs the payload's own Detector, ego-motion estimator and
`track_frame` over them, and scores what the payload would actually emit.

How the sequences are built
---------------------------

Each sequence is one test image drifting across a fixed canvas of the same
size, a set number of pixels per frame - a camera moving over a static scene.
Shifts are whole pixels, so every person stays at native scale and the
detector sees what it saw on the still split. Ground truth is the source
labels moved by the known offset, keeping anyone at least half inside the
frame. Shifts that are not multiples of the network stride change grid
alignment, which varies each person's score between frames much as aliasing
does in real footage.

Two speeds matter. A slow pan of a few pixels per frame is what every earlier
test in this project used. **Flight is about 20 px per frame**: at 100 m the
ground sampling distance is 0.146 m/px, so 20 m/s cruise moves the ground
137 px/s, or 20 px between frames at the target's 7 FPS (34 px at 60 m).

What this is not: flight video. Appearance barely changes between frames and
the motion is pure translation, so ego-motion is estimated under easy
conditions. The stabilised result is therefore a best case for optical flow;
over water, fog or featureless ground, flow degrades and so would tracking.

What is compared
----------------

Every configuration runs on identical detector output. The detector is run
once at the lowest operating point and filtered for each higher one, which
gives exactly the boxes a run at that threshold would: suppression here only
ever removes a box in favour of a higher-scoring one, and any such box also
clears the higher threshold.

  detector      boxes after fusion, before tracking - an upper bound, and
                the stage verify_deployed_config.py measures
  before-fix    the runtime as merged: fused scores averaged, tracker
                thresholds fixed at 0.25 / 0.1 / 0.25, raw pixels
  unstabilised  seed scores, thresholds from the operating point, raw pixels
  current       seed scores, thresholds from the operating point, tracking in
                the stabilised frame - the runtime after this fix

Usage:

    python scripts/evaluate_tracking_gate.py
    python scripts/evaluate_tracking_gate.py --drifts 20 --confs 0.05
    python scripts/evaluate_tracking_gate.py --images 100
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

DATASET = PROJECT_ROOT / "dataset" / "hit_uav_person"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tracking_gate"

MATCH_IOU = 0.50
FRAMES = 24          # about 3.4 s at the target's 7 FPS - near the 100 m dwell
DRIFTS = (5, 20)     # a slow pan, and flight at 100 m / 20 m/s / 7 FPS
MIN_VISIBLE = 0.5    # fraction of a person inside the frame to count as truth

# The fixed values the runtime used before thresholds followed --conf are
# exactly what build_tracker derives at 0.25, so building with 0.25 recreates
# the old tracker with every other argument unchanged.
LEGACY_CONF = 0.25

CONFIGS = ("detector", "before-fix", "unstabilised", "current")


def load_truth(stem, width, height):
    path = DATASET / "labels" / "test" / f"{stem}.txt"

    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        xc, yc, w, h = (float(v) for v in parts[1:5])

        boxes.append([
            (xc - w / 2) * width, (yc - h / 2) * height,
            (xc + w / 2) * width, (yc + h / 2) * height,
        ])

    return boxes


def offsets(frames, drift):
    """Horizontal drift, centred so the image passes through the frame."""

    centre = (frames - 1) / 2

    return [(int(round(drift * (t - centre))), 0) for t in range(frames)]


def shift_frame(image, ox, oy):
    """Move the image by (ox, oy) on a same-size canvas padded with 114."""

    height, width = image.shape[:2]
    canvas = np.full_like(image, 114)

    src_x0, src_y0 = max(0, -ox), max(0, -oy)
    dst_x0, dst_y0 = max(0, ox), max(0, oy)

    w = width - abs(ox)
    h = height - abs(oy)

    if w > 0 and h > 0:
        canvas[dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = \
            image[src_y0:src_y0 + h, src_x0:src_x0 + w]

    return canvas


def shift_truth(boxes, ox, oy, width, height):
    """Truth for one frame: (person index, box) for anyone at least half in view."""

    visible = []

    for index, (x1, y1, x2, y2) in enumerate(boxes):
        x1, x2 = x1 + ox, x2 + ox
        y1, y2 = y1 + oy, y2 + oy

        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(width), x2), min(float(height), y2)

        inside = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)

        if area > 0 and inside / area >= MIN_VISIBLE:
            visible.append((index, [cx1, cy1, cx2, cy2]))

    return visible


def iou_matrix(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def greedy_pairs(truth_boxes, predicted):
    """One-to-one matching, highest IoU first - the project's convention."""

    if not len(truth_boxes) or not len(predicted):
        return []

    matrix = iou_matrix(truth_boxes, predicted)

    candidates = np.argwhere(matrix >= MATCH_IOU)
    order = np.argsort(-matrix[candidates[:, 0], candidates[:, 1]])

    used_t, used_p, pairs = set(), set(), []

    for ti, pi in candidates[order]:
        if ti in used_t or pi in used_p:
            continue

        used_t.add(ti)
        used_p.add(pi)
        pairs.append((int(ti), int(pi)))

    return pairs


def empty_tally():
    return {
        "truth_boxes": 0,
        "emitted": 0,
        "matched": 0,
        "persons_present": 0,
        "persons_reported": 0,
        "tracks": 0,
        "spurious_tracks": 0,
        "spurious_tracks_background": 0,
    }


def run_sequence(image, source_truth, drift, frames, detector, confs, payload,
                 realtime_detect, cv2):
    """Score one drifting sequence for every configuration and threshold."""

    height, width = image.shape[:2]
    background = not source_truth

    # Detector and ego-motion once per frame, shared by every configuration.
    ego = realtime_detect.EgoMotionEstimator()
    prepared = []

    for ox, oy in offsets(frames, drift):
        frame = shift_frame(image, ox, oy)
        cumulative = ego.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).copy()
        prepared.append((
            frame,
            detector.raw(frame),
            cumulative,
            shift_truth(source_truth, ox, oy, width, height),
        ))

    results = {}

    for conf in confs:
        trackers = {
            "before-fix": payload.build_tracker(7.0, LEGACY_CONF),
            "unstabilised": payload.build_tracker(7.0, conf),
            "current": payload.build_tracker(7.0, conf),
        }

        tally = {name: empty_tally() for name in CONFIGS}
        present = set()
        reported = {name: set() for name in CONFIGS}
        track_hit = {name: {} for name in CONFIGS if name != "detector"}

        for frame, (boxes, scores, sources), cumulative, truth in prepared:
            keep = scores >= conf
            kept = (boxes[keep], scores[keep],
                    [s for s, k in zip(sources, keep) if k])

            seed = detector.fuse(*kept)
            mean = (
                payload.weighted_box_fusion(
                    *kept, payload.SINGLE_MODEL_FUSE_IOU, score="mean"
                )
                if len(kept[0]) else (np.zeros((0, 4)), np.zeros(0), [])
            )

            truth_ids = [pid for pid, _ in truth]
            truth_boxes = [box for _, box in truth]
            present.update(truth_ids)

            emitted = {"detector": (seed[0], None)}

            for name, fused, stab in (("before-fix", mean, None),
                                      ("unstabilised", seed, None),
                                      ("current", seed, cumulative)):
                rows, _ = payload.track_frame(
                    trackers[name], fused[0], fused[1], frame, stab
                )

                if len(rows):
                    emitted[name] = (rows[:, :4], rows[:, 4].astype(int))
                else:
                    emitted[name] = (np.zeros((0, 4)), np.zeros(0, dtype=int))

            for name, (out_boxes, track_ids) in emitted.items():
                pairs = greedy_pairs(truth_boxes, out_boxes)

                tally[name]["truth_boxes"] += len(truth_boxes)
                tally[name]["emitted"] += len(out_boxes)
                tally[name]["matched"] += len(pairs)

                for ti, _ in pairs:
                    reported[name].add(truth_ids[ti])

                if track_ids is not None:
                    hits = {pi for _, pi in pairs}

                    for pi, tid in enumerate(track_ids):
                        track_hit[name][tid] = (
                            track_hit[name].get(tid, False) or pi in hits
                        )

        for name in CONFIGS:
            tally[name]["persons_present"] = len(present)
            tally[name]["persons_reported"] = len(reported[name])

            if name != "detector":
                spurious = sum(1 for hit in track_hit[name].values() if not hit)
                tally[name]["tracks"] = len(track_hit[name])
                tally[name]["spurious_tracks"] = spurious
                tally[name]["spurious_tracks_background"] = (
                    spurious if background else 0
                )

        results[conf] = tally

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None)
    parser.add_argument("--confs", type=float, nargs="+", default=[0.25, 0.05])
    parser.add_argument("--drifts", type=int, nargs="+", default=list(DRIFTS))
    parser.add_argument("--images", type=int, default=0, help="0 = all")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--device", default="0")

    args = parser.parse_args()

    import cv2
    import payload
    import realtime_detect

    model = args.model or payload.DEFAULT_THERMAL
    confs = sorted(set(args.confs), reverse=True)

    images = sorted((DATASET / "images" / "test").glob("*.jpg"))

    if args.images:
        images = images[: args.images]

    print("=" * 78)
    print("Tracker gate - what the payload emits, not what the detector sees")
    print("=" * 78)
    print(f"model       {model}")
    print(f"sequences   {len(images)} test images x {args.frames} frames")
    print(f"drift       {', '.join(f'{d} px/frame' for d in args.drifts)}")
    print(f"thresholds  {', '.join(f'{c:.2f}' for c in confs)}")
    print()

    detector = payload.Detector([model], device=args.device, conf=min(confs))

    totals = {
        d: {c: {name: empty_tally() for name in CONFIGS} for c in confs}
        for d in args.drifts
    }

    started = time.perf_counter()

    for n, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]
        truth = load_truth(image_path.stem, width, height)

        for drift in args.drifts:
            seq = run_sequence(image, truth, drift, args.frames, detector,
                               confs, payload, realtime_detect, cv2)

            for conf, tally in seq.items():
                for name, counts in tally.items():
                    for key, value in counts.items():
                        totals[drift][conf][name][key] += value

        if n % 50 == 0 or n == len(images):
            print(f"  {n}/{len(images)} images, "
                  f"{time.perf_counter() - started:.0f}s")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    for drift in args.drifts:
        for conf in confs:
            print()
            print("=" * 78)
            print(f"Drift {drift} px/frame, operating point conf {conf:.2f}")
            print("=" * 78)

            header = (f"{'config':<14s} {'frame recall':>13s} {'precision':>10s} "
                      f"{'persons':>13s} {'tracks':>7s} {'spurious':>9s} "
                      f"{'on bkgd':>8s}")
            print(header)
            print("-" * len(header))

            for name in CONFIGS:
                t = totals[drift][conf][name]
                recall = t["matched"] / t["truth_boxes"] if t["truth_boxes"] else 0.0
                precision = t["matched"] / t["emitted"] if t["emitted"] else 0.0
                persons = f"{t['persons_reported']}/{t['persons_present']}"

                dash = name == "detector"

                print(f"{name:<14s} {recall:13.4f} {precision:10.4f} "
                      f"{persons:>13s} "
                      f"{'-' if dash else t['tracks']:>7} "
                      f"{'-' if dash else t['spurious_tracks']:>9} "
                      f"{'-' if dash else t['spurious_tracks_background']:>8}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "tracking_gate.json").write_text(
        json.dumps(
            {
                "model": model,
                "sequences": len(images),
                "frames": args.frames,
                "match_iou": MATCH_IOU,
                "min_visible": MIN_VISIBLE,
                "results": {
                    f"drift_{d}": {f"{c:.2f}": totals[d][c] for c in confs}
                    for d in args.drifts
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Written: {OUTPUT_DIR / 'tracking_gate.json'}")


if __name__ == "__main__":
    main()
