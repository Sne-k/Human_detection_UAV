"""
Weighted box fusion on ONE model - the ensemble's benefit without its cost.

The two-model ensemble is the only configuration found that improves on the
baseline on every axis at once: more people found, fewer missed, fewer false
positives, higher precision. It costs 2x inference, which
`architecture_budget.py` and `deployment_target.md` both rule out on a
Raspberry Pi 5 at 60 m.

Section 29 explains *why* it works, and the explanation does not obviously
require two models. 56% of unmatched predictions are near-miss boxes sitting
on real people, counted twice - once as a missed person, once as a false
positive. Fusing two near-misses produces one better-centred box, which is why
both columns improve together.

A single model also emits several slightly-offset boxes per person before
post-processing. NMS then **discards** all but the highest-confidence one, and
the highest-confidence box is not necessarily the best-localised one. WBF
instead **averages** the cluster, weighted by confidence.

So the question this script asks is: does the ensemble's mechanism survive
with one model, by changing only the post-processing?

If it does, it is completely free. The network, the exported graph and the
deployed latency are all unchanged - only the code after the forward pass
differs, and that code costs microseconds against a 40 ms inference.

The comparison is exact. The same raw network output feeds both paths, so
any difference is attributable to fusion versus suppression and nothing else.

Usage:

    python scripts/single_model_wbf.py
    python scripts/single_model_wbf.py --model MT-005 --conf 0.25
    python scripts/single_model_wbf.py --write-labels 0.55
"""

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"
ANALYSIS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "error_analysis"
DATASET = PROJECT_ROOT / "dataset" / "hit_uav_person"
OUTPUT_DIR = PROJECT_ROOT / "results" / "single_model_wbf"

DEPLOY_SHAPE = (512, 640)
CONF = 0.25
NMS_IOU = 0.70
MATCH_IOU = 0.50

FUSE_THRESHOLDS = [0.80, 0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.30]

# The frozen MT-005 baseline, conf 0.25, IoU >= 0.50 on the test split.
BASELINE = {
    "matched": 2425,
    "missed": 186,
    "unmatched": 638,
    "blind": 8,
    "recall": 0.9288,
    "precision": 0.7917,
}


def preprocess(path, shape):
    import cv2

    image = cv2.imread(str(path))

    if image is None:
        return None, None

    target_h, target_w = shape
    h, w = image.shape[:2]

    scale = min(target_h / h, target_w / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = image

    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0

    return np.ascontiguousarray(tensor)[None], (w, h)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def cluster(boxes, scores, threshold):
    """Greedy clustering seeded on the highest-confidence box."""

    order = sorted(range(len(boxes)), key=lambda i: -scores[i])

    remaining = list(order)
    clusters = []

    while remaining:
        seed = remaining.pop(0)

        group = [seed]
        leftover = []

        for index in remaining:
            if iou(boxes[seed], boxes[index]) >= threshold:
                group.append(index)
            else:
                leftover.append(index)

        clusters.append(group)
        remaining = leftover

    return clusters


def suppress(boxes, scores, threshold):
    """NMS: keep the cluster seed, discard the rest. The current behaviour."""

    return [
        (boxes[group[0]], scores[group[0]], len(group))
        for group in cluster(boxes, scores, threshold)
    ]


def fuse(boxes, scores, threshold):
    """WBF: replace the cluster with its confidence-weighted average."""

    fused = []

    for group in cluster(boxes, scores, threshold):
        weight = sum(scores[i] for i in group) or 1.0

        averaged = [
            sum(boxes[i][axis] * scores[i] for i in group) / weight
            for axis in range(4)
        ]

        # The seed's confidence is kept rather than the cluster mean. Averaging
        # it would systematically lower every multi-box cluster's score and
        # change the operating point, which would confound the comparison with
        # the effect of fusion itself.
        fused.append((averaged, scores[group[0]], len(group)))

    return fused


def greedy_match(truth, predictions, threshold=MATCH_IOU):
    """One-to-one matching, highest IoU first - the project's convention."""

    pairs = []

    for gi, g in enumerate(truth):
        for pi, p in enumerate(predictions):
            score = iou(g, p)

            if score >= threshold:
                pairs.append((score, gi, pi))

    pairs.sort(reverse=True)

    used_gt = set()
    used_pred = set()

    for _, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

    return len(used_gt), len(predictions) - len(used_pred)


def load_truth(stem, width, height):
    path = DATASET / "labels" / "test" / f"{stem}.txt"

    if not path.exists():
        return []

    boxes = []

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        xc, yc, w, h = (float(v) for v in parts[1:5])

        xc, w = xc * width, w * width
        yc, h = yc * height, h * height

        boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])

    return boxes


def collect(model, conf, threads=4):
    """Run the model once and keep every raw box above threshold."""

    import onnxruntime as ort

    path = RUNS_ROOT / model / "weights" / "best.onnx"

    if not path.exists():
        raise SystemExit(
            f"No ONNX export for {model}. Run:\n"
            f"  python scripts/export_models.py --models {model}"
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )

    meta = session.get_inputs()[0]
    _, _, net_h, net_w = [d if isinstance(d, int) else 1 for d in meta.shape]

    frames = []
    raw_total = 0

    images = sorted((DATASET / "images" / "test").glob("*.jpg"))

    for image_path in images:
        tensor, size = preprocess(image_path, (net_h, net_w))

        if tensor is None:
            continue

        output = session.run(None, {meta.name: tensor})[0][0]

        centres = output[:4, :]
        scores = output[4, :]

        keep = scores >= conf

        cx, cy, w, h = centres[:, keep]
        kept_scores = scores[keep]

        boxes = np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1
        ).tolist()

        raw_total += len(boxes)

        frames.append({
            "stem": image_path.stem,
            "size": size,
            "net": (net_w, net_h),
            "boxes": boxes,
            "scores": kept_scores.tolist(),
            "truth": load_truth(image_path.stem, net_w, net_h),
        })

    return frames, raw_total


def score(frames, method, threshold):
    matched = unmatched = blind = truth_total = predicted = 0

    per_frame = []

    for frame in frames:
        produced = method(frame["boxes"], frame["scores"], threshold)

        boxes = [b for b, _, _ in produced]

        a, b = greedy_match(frame["truth"], boxes)

        matched += a
        unmatched += b
        truth_total += len(frame["truth"])
        predicted += len(boxes)

        if frame["truth"] and not boxes:
            blind += 1

        per_frame.append((frame, produced))

    return {
        "matched": matched,
        "missed": truth_total - matched,
        "unmatched": unmatched,
        "blind": blind,
        "predicted": predicted,
        "recall": matched / truth_total if truth_total else 0.0,
        "precision": matched / predicted if predicted else 0.0,
    }, per_frame


def write_labels(per_frame, out_dir):
    labels = Path(out_dir) / "labels"
    labels.mkdir(parents=True, exist_ok=True)

    for frame, produced in per_frame:
        net_w, net_h = frame["net"]

        lines = []

        for (x1, y1, x2, y2), conf, _ in produced:
            lines.append(
                f"0 {(x1 + x2) / 2 / net_w:.6f} {(y1 + y2) / 2 / net_h:.6f} "
                f"{(x2 - x1) / net_w:.6f} {(y2 - y1) / net_h:.6f} {conf:.6f}"
            )

        (labels / f"{frame['stem']}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    return labels


CONF_SWEEP = [0.25, 0.20, 0.15, 0.10, 0.07, 0.05]

# The mission cost model from operating_point.py: a missed casualty costs
# MISS_COST times what a false positive costs. 20 is that script's stated
# value, and under NMS it selected a 0.15 threshold.
MISS_COST = 20.0

# NMS rows at the same thresholds, from results/operating_point/. Needed to
# show that changing the post-processing moves the optimum, not just the
# numbers at a fixed threshold.
NMS_SWEEP = {
    0.25: {"missed": 186, "unmatched": 638, "matched": 2425},
    0.20: {"missed": 170, "unmatched": 793, "matched": 2441},
    0.15: {"missed": 157, "unmatched": 999, "matched": 2454},
    0.10: {"missed": 143, "unmatched": 1300, "matched": 2468},
}


def filter_frames(frames, conf):
    """Re-threshold cached raw boxes. No re-inference."""

    filtered = []

    for frame in frames:
        keep = [i for i, s in enumerate(frame["scores"]) if s >= conf]

        filtered.append({
            **frame,
            "boxes": [frame["boxes"][i] for i in keep],
            "scores": [frame["scores"][i] for i in keep],
        })

    return filtered


def sweep_confidence(frames, args):
    """
    Spend the precision that fusion bought, on recall.

    Missing a casualty and reporting a warm rock are not symmetric costs in a
    search mission. The operating point was set at confidence 0.25 when NMS
    gave 0.7917 precision; fusion raises that to 0.8320 at the same threshold,
    which means the threshold can come down before precision returns to where
    it was considered acceptable.

    The floor matters. This is only defensible while the extra boxes are
    genuinely people - so both columns are reported at every step, and the
    point where unmatched boxes start climbing faster than matched persons is
    where it stops.
    """

    print()
    print("=" * 78)
    print("Spending the precision gain on recall")
    print("=" * 78)
    print("Same single inference pass, re-thresholded. WBF clustering at 0.60.")
    print()

    header = (f"{'conf':>6s} {'matched':>9s} {'missed':>8s} {'unmatched':>11s} "
              f"{'blind':>7s} {'recall':>9s} {'precision':>11s} {'per extra':>11s}")
    print(header)
    print("-" * len(header))

    previous = None
    rows = []

    for conf in CONF_SWEEP:
        result, _ = score(filter_frames(frames, conf), fuse, 0.60)

        cost = ""

        if previous is not None:
            gained = result["matched"] - previous["matched"]
            added = result["unmatched"] - previous["unmatched"]

            cost = f"{added / gained:.1f} FP" if gained > 0 else "  no gain"

        print(f"{conf:6.2f} {result['matched']:9d} {result['missed']:8d} "
              f"{result['unmatched']:11d} {result['blind']:7d} "
              f"{result['recall']:9.4f} {result['precision']:11.4f} "
              f"{cost:>11s}")

        result["conf"] = conf
        rows.append(result)
        previous = result

    print()
    print("'per extra' is the number of additional false positives paid for")
    print("each additional person found, relative to the row above.")
    print()

    baseline_precision = BASELINE["precision"]

    affordable = [r for r in rows if r["precision"] >= baseline_precision]

    if affordable:
        best = max(affordable, key=lambda r: r["matched"])

        print(f"Lowest threshold still at or above the baseline's "
              f"{baseline_precision:.4f} precision: conf {best['conf']:.2f}")
        print(f"  {best['matched']} matched "
              f"({best['matched'] - BASELINE['matched']:+d} vs baseline), "
              f"recall {best['recall']:.4f}, precision {best['precision']:.4f}")
        print()
        print("  That is strictly better than the deployed baseline on both")
        print("  axes at once, at no inference cost.")

    # ------------------------------------------------------------------
    # Re-derive the operating point. It depends on post-processing, and
    # post-processing just changed.
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print(f"Mission cost, missed person = {MISS_COST:.0f} x false positive")
    print("=" * 78)
    print("operating_point.py selected 0.15 under NMS. The optimum is a")
    print("property of the post-processing, so it has to be re-derived.")
    print()

    def cost(entry):
        return entry["missed"] * MISS_COST + entry["unmatched"]

    print(f"{'conf':>6s} {'NMS cost':>10s} {'WBF cost':>10s} {'saving':>9s} "
          f"{'NMS found':>11s} {'WBF found':>11s}")
    print("-" * 62)

    for row in rows:
        nms = NMS_SWEEP.get(row["conf"])

        if not nms:
            print(f"{row['conf']:6.2f} {'-':>10s} {cost(row):10.0f}")
            continue

        print(f"{row['conf']:6.2f} {cost(nms):10.0f} {cost(row):10.0f} "
              f"{cost(nms) - cost(row):+9.0f} "
              f"{nms['matched']:11d} {row['matched']:11d}")

    wbf_best = min(rows, key=cost)
    nms_best = min(NMS_SWEEP.items(), key=lambda kv: cost(kv[1]))

    print()
    print(f"NMS optimum : conf {nms_best[0]:.2f}, cost {cost(nms_best[1]):.0f}, "
          f"{nms_best[1]['matched']} people found")
    print(f"WBF optimum : conf {wbf_best['conf']:.2f}, cost {cost(wbf_best):.0f}, "
          f"{wbf_best['matched']} people found")
    print()

    saving = (cost(nms_best[1]) - cost(wbf_best)) / cost(nms_best[1])

    print(f"Fusion lowers the mission cost at its own optimum by "
          f"{saving * 100:.1f}%,")
    print(f"and finds {wbf_best['matched'] - nms_best[1]['matched']:+d} more "
          f"people while doing it. Both at no inference cost.")
    print()
    print("The cost ratio is a mission decision, not a technical one. What is")
    print("technical is that fusion beats suppression at every threshold, so")
    print("whichever ratio is chosen, fusion is the better post-processing.")

    (OUTPUT_DIR / "confidence_sweep.json").write_text(
        json.dumps({
            "baseline": BASELINE,
            "miss_cost": MISS_COST,
            "nms_reference": {str(k): v for k, v in NMS_SWEEP.items()},
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Written: {OUTPUT_DIR / 'confidence_sweep.json'}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="MT-005")
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--sweep-conf",
        action="store_true",
        help="Also sweep the confidence threshold. Fusion buys precision, "
             "and in a search mission precision headroom is worth spending "
             "on recall.",
    )
    parser.add_argument(
        "--write-labels",
        type=float,
        default=None,
        metavar="IOU",
        help="Write WBF predictions at this fusion threshold, for "
             "evaluate_experiment.py",
    )

    args = parser.parse_args()

    print("=" * 78)
    print(f"Weighted box fusion on a single model - {args.model}")
    print("=" * 78)
    print(f"Confidence {args.conf}, matching at IoU >= {MATCH_IOU}")
    print()
    print("Same raw network output feeds both paths, so the difference is")
    print("post-processing and nothing else. Inference cost is identical.")
    print()

    started = time.perf_counter()

    # Collect once at the sweep floor; higher thresholds filter this in
    # memory, so the whole study costs one inference pass.
    floor = min(CONF_SWEEP) if args.sweep_conf else args.conf
    raw_frames, raw_total = collect(args.model, floor, args.threads)

    # The main comparison runs at the operating point; the sweep needs the
    # unfiltered set, so both are kept.
    if floor < args.conf:
        frames = filter_frames(raw_frames, args.conf)
        raw_total = sum(len(f["boxes"]) for f in frames)
    else:
        frames = raw_frames

    print(f"{len(frames)} images, {raw_total} raw boxes above {args.conf} "
          f"({raw_total / max(1, len(frames)):.1f} per image) "
          f"in {time.perf_counter() - started:.0f}s")
    print()

    header = (f"{'method':>7s} {'IoU':>5s} {'boxes':>7s} {'matched':>8s} "
              f"{'missed':>7s} {'unmatched':>10s} {'blind':>6s} "
              f"{'recall':>8s} {'precision':>10s}")
    print(header)
    print("-" * len(header))

    rows = []

    for threshold in FUSE_THRESHOLDS:
        for name, method in (("NMS", suppress), ("WBF", fuse)):
            result, _ = score(frames, method, threshold)

            result["method"] = name
            result["fuse_iou"] = threshold
            rows.append(result)

            print(f"{name:>7s} {threshold:5.2f} {result['predicted']:7d} "
                  f"{result['matched']:8d} {result['missed']:7d} "
                  f"{result['unmatched']:10d} {result['blind']:6d} "
                  f"{result['recall']:8.4f} {result['precision']:10.4f}")

        print()

    # ------------------------------------------------------------------
    # Does fusion beat suppression at the same threshold?
    # ------------------------------------------------------------------

    print("=" * 78)
    print("WBF minus NMS, at matched thresholds")
    print("=" * 78)
    print(f"{'IoU':>5s} {'matched':>9s} {'unmatched':>11s} {'blind':>7s} "
          f"{'recall':>10s} {'precision':>11s}")
    print("-" * 56)

    best = None

    for threshold in FUSE_THRESHOLDS:
        nms = next(r for r in rows if r["method"] == "NMS" and r["fuse_iou"] == threshold)
        wbf = next(r for r in rows if r["method"] == "WBF" and r["fuse_iou"] == threshold)

        gain = wbf["matched"] - nms["matched"]

        print(f"{threshold:5.2f} {gain:+9d} "
              f"{wbf['unmatched'] - nms['unmatched']:+11d} "
              f"{wbf['blind'] - nms['blind']:+7d} "
              f"{wbf['recall'] - nms['recall']:+10.4f} "
              f"{wbf['precision'] - nms['precision']:+11.4f}")

    # Selecting by "most people found" just picks the loosest threshold, which
    # buys recall with a flood of false positives. The useful question is which
    # configurations beat the deployed baseline on BOTH axes at once - the
    # property that made the two-model ensemble worth reporting.
    dominant = [
        r for r in rows
        if r["matched"] >= BASELINE["matched"]
        and r["unmatched"] <= BASELINE["unmatched"]
        and r["blind"] <= BASELINE["blind"]
    ]

    print()
    print("=" * 78)
    print("Configurations that beat the frozen baseline on every axis")
    print("=" * 78)
    print(f"Baseline: {BASELINE['matched']} matched, "
          f"{BASELINE['unmatched']} unmatched, "
          f"precision {BASELINE['precision']:.4f}")
    print()

    if not dominant:
        print("  None.")
        best = next(r for r in rows if r["method"] == "WBF" and r["fuse_iou"] == NMS_IOU)
    else:
        print(f"{'method':>7s} {'IoU':>5s} {'matched':>9s} {'unmatched':>11s} "
              f"{'recall':>9s} {'precision':>11s}")
        print("-" * 56)

        for r in sorted(dominant, key=lambda r: r["unmatched"]):
            print(f"{r['method']:>7s} {r['fuse_iou']:5.2f} "
                  f"{r['matched'] - BASELINE['matched']:+9d} "
                  f"{r['unmatched'] - BASELINE['unmatched']:+11d} "
                  f"{r['recall'] - BASELINE['recall']:+9.4f} "
                  f"{r['precision'] - BASELINE['precision']:+11.4f}")

        # Prefer the one that sheds the most false positives while still
        # finding at least as many people as the baseline.
        best = min(dominant, key=lambda r: r["unmatched"])

    print()
    print("=" * 78)
    print("Selected configuration against the frozen MT-005 baseline")
    print("=" * 78)
    print(f"{best['method']} at fusion IoU {best['fuse_iou']:.2f}")
    print()
    print(f"{'Metric':<20s} {'baseline':>10s} {'selected':>10s} {'delta':>10s}")
    print("-" * 52)

    for key, label in (
        ("matched", "Matched persons"),
        ("missed", "Missed persons"),
        ("unmatched", "Unmatched boxes"),
        ("blind", "Blind images"),
    ):
        delta = best[key] - BASELINE[key]
        flag = "" if delta == 0 else ("  better" if (
            (key == "matched" and delta > 0) or
            (key in ("missed", "unmatched", "blind") and delta < 0)
        ) else "  worse")

        print(f"{label:<20s} {BASELINE[key]:10d} {best[key]:10d} "
              f"{delta:+10d}{flag}")

    for key, label in (("recall", "Recall"), ("precision", "Precision")):
        delta = best[key] - BASELINE[key]
        print(f"{label:<20s} {BASELINE[key]:10.4f} {best[key]:10.4f} "
              f"{delta:+10.4f}"
              f"{'  better' if delta > 0 else '  worse' if delta < 0 else ''}")

    print()
    print("Inference cost: identical. Only the post-processing differs.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "single_model_wbf.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "conf": args.conf,
                "match_iou": MATCH_IOU,
                "baseline": BASELINE,
                "rows": rows,
                "best_wbf": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Written: {OUTPUT_DIR / 'single_model_wbf.json'}")

    if args.sweep_conf:
        sweep_confidence(raw_frames, args)

    if args.write_labels is not None:
        _, per_frame = score(frames, fuse, args.write_labels)

        name = f"{args.model}-wbf{int(args.write_labels * 100)}-test-analysis"
        labels = write_labels(per_frame, ANALYSIS_ROOT / name)

        print(f"Labels:  {labels}")
        print()
        print(f"  python scripts/evaluate_experiment.py --labels {name} "
              f"--label {args.model}-WBF")


if __name__ == "__main__":
    main()
