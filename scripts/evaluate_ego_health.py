"""
When does ego-motion fail, and what should the payload do when it does?

Tracking now runs in the stabilised frame (training_log.md section 49), so
every record the payload emits depends on the frame-to-frame motion estimate.
That estimate comes from optical flow on image corners, and some of the
conditions a search aircraft flies over offer almost none: open water, snow,
fog, featureless ground. When flow fails, the estimator used to substitute an
identity step - "the camera did not move" - which at flight speed is wrong by
the full ground motion, about 20 px per frame.

This script measures the estimator against known motion. Each test image
drifts across the frame at a set rate, so the true step is exact and the
estimator's error is directly measurable. Three conditions:

  normal       the test image as captured
  featureless  everything except the people replaced by flat background
               with sensor noise - water, snow, uniform ground
  fog          contrast cut to 10%, with sensor noise

It reports, per condition, how the error depends on the RANSAC inlier count -
which is what a health gate can see at run time - and then compares fallback
policies by what the tracker emits.

Usage:

    python scripts/evaluate_ego_health.py
    python scripts/evaluate_ego_health.py --images 60
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "ego_health"

DRIFT = 20     # px per frame, flight at 100 m
FRAMES = 24
NOISE = 2.0    # sensor noise, grey levels
INLIER_BINS = [0, 1, 10, 20, 30, 50, 100, 1000]


def degrade(image, boxes, condition, rng):
    if condition == "normal":
        return image

    noise = rng.normal(0, NOISE, image.shape)

    if condition == "fog":
        mean = image.mean()
        out = 0.1 * image.astype(float) + 0.9 * mean + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    # featureless: keep the people, flatten everything else
    out = np.full(image.shape, float(np.median(image))) + noise
    h, w = image.shape[:2]

    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1) - 3), max(0, int(y1) - 3)
        x2, y2 = min(w, int(x2) + 3), min(h, int(y2) + 3)
        out[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    return np.clip(out, 0, 255).astype(np.uint8)


def step_error(step, width, height):
    """Distance, at the frame centre, between estimated and true motion."""

    centre = np.array([width / 2.0, height / 2.0, 1.0])
    moved = step @ centre
    return float(np.hypot(moved[0] - (centre[0] + DRIFT), moved[1] - centre[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=120)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tracking-conditions", nargs="+",
                        default=["normal", "featureless", "fog"])
    args = parser.parse_args()

    import cv2
    import evaluate_tracking_gate as gate
    import payload
    import realtime_detect

    rng = np.random.default_rng(0)

    paths = [
        p for p in sorted((gate.DATASET / "images" / "test").glob("*.jpg"))
        if gate.load_truth(p.stem, 640, 512)
    ][: args.images]

    print("=" * 74)
    print(f"Ego-motion against known {DRIFT} px/frame drift, "
          f"{len(paths)} sequences x {FRAMES} frames")
    print("=" * 74)

    conditions = ("normal", "featureless", "fog")
    samples = {c: [] for c in conditions}
    sequences = {c: [] for c in conditions}

    for path in paths:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        truth = gate.load_truth(path.stem, width, height)

        for condition in conditions:
            degraded = degrade(image, truth, condition, rng)
            frames = [gate.shift_frame(degraded, ox, oy)
                      for ox, oy in gate.offsets(FRAMES, DRIFT)]
            sequences[condition].append((path.stem, degraded, truth))

            ego = realtime_detect.EgoMotionEstimator()

            for frame in frames:
                ego.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                if ego.status != "init":
                    samples[condition].append(
                        (ego.last_inliers, step_error(ego.last_step, width, height))
                    )

    # ---------------------------------------------------------------- part 1
    print()
    print("Estimator error by condition (identity fallback, as shipped)")
    print(f"{'condition':<12s} {'steps':>6s} {'median inl':>11s} "
          f"{'median err px':>14s} {'p95 err px':>11s} {'err > 3 px':>11s}")

    report = {"by_condition": {}, "by_inliers": {}}

    for condition in conditions:
        inl = np.array([s[0] for s in samples[condition]])
        err = np.array([s[1] for s in samples[condition]])
        row = {
            "steps": int(len(err)),
            "median_inliers": float(np.median(inl)),
            "median_error_px": round(float(np.median(err)), 2),
            "p95_error_px": round(float(np.percentile(err, 95)), 2),
            "share_error_over_3px": round(float((err > 3).mean()), 4),
        }
        report["by_condition"][condition] = row
        print(f"{condition:<12s} {row['steps']:6d} {row['median_inliers']:11.0f} "
              f"{row['median_error_px']:14.2f} {row['p95_error_px']:11.2f} "
              f"{row['share_error_over_3px']:11.1%}")

    print()
    print("Error by inlier count, all conditions pooled")
    print(f"{'inliers':<12s} {'steps':>6s} {'median err px':>14s} "
          f"{'p95 err px':>11s} {'err > 3 px':>11s}")

    pooled = [s for c in conditions for s in samples[c]]
    inl = np.array([s[0] for s in pooled])
    err = np.array([s[1] for s in pooled])

    for lo, hi in zip(INLIER_BINS[:-1], INLIER_BINS[1:]):
        mask = (inl >= lo) & (inl < hi)

        if not mask.any():
            continue

        e = err[mask]
        label = f"{lo}-{hi - 1}"
        report["by_inliers"][label] = {
            "steps": int(mask.sum()),
            "median_error_px": round(float(np.median(e)), 2),
            "p95_error_px": round(float(np.percentile(e, 95)), 2),
            "share_error_over_3px": round(float((e > 3).mean()), 4),
        }
        print(f"{label:<12s} {int(mask.sum()):6d} {np.median(e):14.2f} "
              f"{np.percentile(e, 95):11.2f} {(e > 3).mean():11.1%}")

    # ---------------------------------------------------------------- part 2
    print()
    print("What the tracker emits, by fallback policy (conf 0.25)")

    policies = {
        "identity (as shipped)": dict(min_inliers=0, fallback="identity"),
        "predict, min 10, 5 fr": dict(min_inliers=10, fallback="predict", max_predict=5),
        "predict, min 1, 5 fr": dict(min_inliers=1, fallback="predict", max_predict=5),
        "predict, min 1, 30 fr": dict(min_inliers=1, fallback="predict", max_predict=30),
    }

    detector = payload.Detector([payload.DEFAULT_THERMAL], device=args.device,
                                conf=0.25)

    print(f"{'condition':<12s} {'policy':<24s} {'frame recall':>13s} "
          f"{'status lost':>12s} {'predicted':>10s}")

    report["tracking"] = {}

    for condition in args.tracking_conditions:
        prepared = []

        for stem, degraded, truth in sequences[condition]:
            frames = []

            for ox, oy in gate.offsets(FRAMES, DRIFT):
                frame = gate.shift_frame(degraded, ox, oy)
                frames.append((frame, detector(frame),
                               gate.shift_truth(truth, ox, oy, 640, 512)))

            prepared.append(frames)

        for name, kwargs in policies.items():
            matched = total = lost = predicted = steps = 0

            for frames in prepared:
                ego = realtime_detect.EgoMotionEstimator(**kwargs)
                tracker = payload.build_tracker(7.0, 0.25)

                for frame, (boxes, scores, _), truth_frame in frames:
                    cumulative = ego.update(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).copy()
                    rows, _ = payload.track_frame(tracker, boxes, scores,
                                                  frame, cumulative)

                    truth_boxes = [b for _, b in truth_frame]
                    out = rows[:, :4] if len(rows) else np.zeros((0, 4))
                    matched += len(gate.greedy_pairs(truth_boxes, out))
                    total += len(truth_boxes)

                    if ego.status != "init":
                        steps += 1
                        lost += ego.status == "lost"
                        predicted += ego.status == "predicted"

            recall = matched / total if total else 0.0
            report["tracking"][f"{condition} / {name}"] = {
                "frame_recall": round(recall, 4),
                "lost_share": round(lost / steps, 4),
                "predicted_share": round(predicted / steps, 4),
            }
            print(f"{condition:<12s} {name:<24s} {recall:13.4f} "
                  f"{lost / steps:12.1%} {predicted / steps:10.1%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ego_health_probe.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Written: {OUTPUT_DIR / 'ego_health_probe.json'}")


if __name__ == "__main__":
    main()
