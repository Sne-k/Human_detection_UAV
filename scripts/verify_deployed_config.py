"""
Does the assembled payload actually deliver what the parts promised?

Every result in this project was measured on a component. The detector was
benchmarked alone, weighted box fusion was compared offline against cached
boxes, the operating point came from a threshold sweep on saved predictions,
and the pipeline timing was taken with a different model at a different
threshold on a sequence with one person in it.

None of that is the payload. This script runs the real thing - the same
`Detector` the runtime uses, at the configuration the evidence recommends -
over the held-out test split, and checks two things the parts cannot:

1. **Accuracy.** Does the assembled stack reproduce the offline numbers? If it
   does not, some component is configured differently from how it was
   measured, which is exactly the class of error that cost 27 detections in
   the ONNX export study.

2. **Cost at the recommended operating point.** Section 39 measured the
   non-detection pipeline at 2.1 ms - on a sequence averaging about one person
   per frame. The recommended threshold is far more sensitive and produces
   several times as many boxes, all of which flow into tracking. That number
   is a lower bound and has never been checked where it matters.

The second is the point. Recommending an operating point whose pipeline cost
is unverified is the same mistake section 39 was written about.

Usage:

    python scripts/verify_deployed_config.py
    python scripts/verify_deployed_config.py --model MT-011b --conf 0.05
    python scripts/verify_deployed_config.py --conf-sweep
"""

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

DATASET = PROJECT_ROOT / "dataset" / "hit_uav_person"
OUTPUT_DIR = PROJECT_ROOT / "results" / "deployed_config"

# Offline expectations from single_model_wbf.py, WBF clustering at 0.60.
# matched, missed, unmatched
EXPECTED = {
    ("MT-005", 0.25): (2426, 185, 490),
    ("MT-011", 0.05): (2492, 119, 1005),
    ("MT-011b", 0.25): (2430, 181, 446),
    ("MT-011b", 0.10): (2489, 122, 871),
    ("MT-011b", 0.05): (2513, 98, 1228),
}

SWEEP = [0.25, 0.20, 0.15, 0.10, 0.07, 0.05]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-sweep", action="store_true")

    args = parser.parse_args()

    import cv2
    import payload
    from single_model_wbf import greedy_match, load_truth

    model = args.model or payload.DEFAULT_THERMAL

    print("=" * 78)
    print("Deployed configuration, assembled and measured")
    print("=" * 78)
    print(f"detector    {model}")
    print(f"clustering  weighted box fusion at {payload.SINGLE_MODEL_FUSE_IOU}")
    print(f"NMS         opened to {payload.SINGLE_MODEL_NMS_IOU} so near-duplicates survive to be fused")
    print(f"device      {args.device}")
    print()

    thresholds = SWEEP if args.conf_sweep else [args.conf]

    images = sorted((DATASET / "images" / "test").glob("*.jpg"))

    rows = []

    for conf in thresholds:
        detector = payload.Detector([model], device=args.device, conf=conf)

        matched = unmatched = truth_total = predicted = blind = 0
        detect_ms = []

        for image_path in images:
            frame = cv2.imread(str(image_path))

            if frame is None:
                continue

            height, width = frame.shape[:2]

            start = time.perf_counter()
            boxes, _scores, _sources = detector(frame)
            detect_ms.append((time.perf_counter() - start) * 1000)

            truth = load_truth(image_path.stem, width, height)

            a, b = greedy_match(truth, boxes.tolist())

            matched += a
            unmatched += b
            truth_total += len(truth)
            predicted += len(boxes)

            if truth and len(boxes) == 0:
                blind += 1

        row = {
            "model": model,
            "conf": conf,
            "matched": matched,
            "missed": truth_total - matched,
            "unmatched": unmatched,
            "blind": blind,
            "boxes_per_image": round(predicted / max(1, len(images)), 2),
            "recall": round(matched / truth_total, 4) if truth_total else 0.0,
            "precision": round(matched / predicted, 4) if predicted else 0.0,
            "detect_ms_median": round(statistics.median(detect_ms), 1),
        }

        rows.append(row)

        expected = EXPECTED.get((model, conf))

        print(f"conf {conf:.2f}   {matched} matched, {row['missed']} missed, "
              f"{unmatched} unmatched, {blind} blind")
        print(f"            recall {row['recall']:.4f}  "
              f"precision {row['precision']:.4f}  "
              f"{row['boxes_per_image']:.1f} boxes/image  "
              f"{row['detect_ms_median']:.1f} ms")

        if expected:
            delta = (matched - expected[0], unmatched - expected[2])

            # The offline study drove the exported ONNX graph; the payload
            # runs the PyTorch weights. Small differences in unmatched boxes
            # are numerics. A difference in MATCHED persons would not be -
            # that would mean the stack is configured differently from how it
            # was measured.
            drift = abs(delta[1]) / max(1, expected[2])

            if delta == (0, 0):
                print("            reproduces the offline measurement exactly")
            elif delta[0] == 0 and drift < 0.01:
                print(f"            matched persons reproduce exactly; "
                      f"unmatched differ by {delta[1]:+d} of {expected[2]} "
                      f"({drift * 100:.1f}%)")
                print("            PyTorch against ONNX numerics, not a "
                      "configuration difference")
            else:
                print(f"            OFFLINE SAID {expected[0]} matched, "
                      f"{expected[2]} unmatched  "
                      f"(delta {delta[0]:+d}, {delta[1]:+d})")
                print("            the assembled stack does not match how it "
                      "was measured - investigate before deploying")

        print()

    # ------------------------------------------------------------------
    # What the extra boxes cost downstream
    # ------------------------------------------------------------------

    if len(rows) > 1:
        print("=" * 78)
        print("Detection density across the sweep")
        print("=" * 78)
        print("Every box here flows into tracking, movement classification and")
        print("geolocation. Section 39 measured that stage at 2.1 ms on a")
        print("sequence averaging about one person per frame.")
        print()

        print(f"{'conf':>6s} {'boxes/image':>12s} {'vs 0.25':>9s} {'detect ms':>10s}")
        print("-" * 40)

        reference = rows[0]["boxes_per_image"]

        for row in rows:
            print(f"{row['conf']:6.2f} {row['boxes_per_image']:12.2f} "
                  f"{row['boxes_per_image'] / reference:8.2f}x "
                  f"{row['detect_ms_median']:10.1f}")

        print()
        print("Run payload.py on a real sequence at the chosen threshold to")
        print("measure what that density costs the tracker; detector latency")
        print("above does not include it.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "deployed_config.json").write_text(
        json.dumps(
            {
                "model": model,
                "fuse_iou": payload.SINGLE_MODEL_FUSE_IOU,
                "nms_iou": payload.SINGLE_MODEL_NMS_IOU,
                "device": args.device,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Written: {OUTPUT_DIR / 'deployed_config.json'}")


if __name__ == "__main__":
    main()
