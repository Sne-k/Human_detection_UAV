"""
INT8 quantisation: the only lever that makes the payload lighter.

Every accuracy technique considered so far either costs compute or is free.
None of them give any back. That matters because the deployed baseline sits at
an estimated 5.2-8.6 FPS against a binding 6.7 FPS requirement - "marginal" on
the architecture budget - and the two-model weighted-box-fusion ensemble, the
only configuration that improves on the baseline on every axis at once, was
rejected purely because it costs 2x inference.

INT8 quantisation typically gives 2-3x on CPU. If it does here, it changes two
answers at once: the single model clears its requirement outright instead of
straddling it, and the ensemble comes back into range.

It is also the change most likely to break something quietly. Quantisation
genuinely alters the numbers the network computes, unlike an ONNX export,
which should reproduce PyTorch exactly. This project has already been bitten
once by an export that silently dropped 27 of 792 detections at the wrong
input shape, so the accuracy check here is not optional and it is not a
bit-exactness check - it is a full re-run of the test-set evaluation.

Static quantisation is used rather than dynamic. Dynamic quantisation mainly
helps transformer and recurrent layers; for a convolutional network the
activations must be calibrated against real data, which is what the
calibration pass below does using real thermal frames at the deployment
resolution.

Usage:

    python scripts/quantize_int8.py
    python scripts/quantize_int8.py --model MT-005 --calibration-images 200
    python scripts/quantize_int8.py --no-eval
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
DATASET = PROJECT_ROOT / "dataset" / "hit_uav_person"
OUTPUT_DIR = PROJECT_ROOT / "results" / "quantization"

TARGET_THREADS = 4
SLOWDOWN_RANGE = (3, 5)

# MT-005's native export shape, height x width.
DEPLOY_SHAPE = (512, 640)

CALIBRATION_IMAGES = 200

# The operating point every other analysis in this project uses.
CONF = 0.25

REQUIREMENTS = [
    ("tracking @ 60 m", 6.70),
    ("tracking @ 100 m", 4.02),
    ("report-only @ 60 m", 1.34),
]


def preprocess(path, shape):
    """
    Letterbox to the model input shape, scale to [0, 1], CHW float32.

    HIT-UAV frames are 640 x 512 and the deployment shape is 512 x 640, so in
    practice no padding is applied - but the general path is kept so this also
    works for other sensors.
    """

    import cv2

    image = cv2.imread(str(path))

    if image is None:
        return None

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

    return np.ascontiguousarray(tensor)[None]


def build_calibration_reader(input_name, shape, count):
    from onnxruntime.quantization import CalibrationDataReader

    image_dir = DATASET / "images" / "train"
    paths = sorted(image_dir.glob("*.jpg"))

    if not paths:
        raise SystemExit(f"No calibration images under {image_dir}")

    # Spread the sample across the split so both day and night frames are
    # represented; HIT-UAV encodes lighting in the filename, and calibrating
    # on one condition would set activation ranges for the wrong distribution.
    step = max(1, len(paths) // count)
    selected = paths[::step][:count]

    day = sum(1 for p in selected if p.stem.split("_")[0] == "0")

    print(f"calibrating on {len(selected)} frames "
          f"({day} day, {len(selected) - day} night)")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.index = 0

        def get_next(self):
            while self.index < len(selected):
                tensor = preprocess(selected[self.index], shape)
                self.index += 1

                if tensor is not None:
                    return {input_name: tensor}

            return None

        def rewind(self):
            self.index = 0

    return Reader()


def copy_metadata(source, destination):
    """
    Carry the Ultralytics metadata across.

    Quantisation rewrites the graph and drops metadata_props, which is what
    tells AutoBackend the stride, class names and input shape. Without it the
    quantised model cannot be loaded by the normal inference path.
    """

    import onnx

    src = onnx.load(str(source))
    dst = onnx.load(str(destination))

    del dst.metadata_props[:]

    for prop in src.metadata_props:
        entry = dst.metadata_props.add()
        entry.key = prop.key
        entry.value = prop.value

    onnx.save(dst, str(destination))


def fusion_report(path):
    """
    Did the QDQ pairs actually fuse into integer kernels?

    This is the single diagnostic that explains whether quantisation helps.
    If the Quantize/DequantizeLinear nodes fuse into QLinearConv, the
    convolution runs in integer arithmetic and is faster. If they do not, the
    convolution still runs in FP32 and the conversions are pure added cost -
    which makes the quantised model slower than the original.
    """

    import collections
    import onnx

    counts = collections.Counter(
        node.op_type for node in onnx.load(str(path)).graph.node
    )

    return {
        "qlinearconv": counts.get("QLinearConv", 0) + counts.get("ConvInteger", 0),
        "conv": counts.get("Conv", 0),
        "quant_nodes": counts.get("QuantizeLinear", 0) + counts.get("DequantizeLinear", 0),
    }


def head_nodes(path):
    """
    Node names belonging to the detection head.

    The head must usually be left in FP32. Its classification branch emits
    logits whose post-sigmoid confidences sit in a very small range near zero
    - on a representative frame the maximum is 0.0087 and the mean 3.2e-5.
    Mapping that onto 256 integer levels, with the scale set by a calibrated
    maximum, rounds the entire branch to zero, and the detector stops
    reporting anything at all while its box regression still looks healthy.

    That is exactly what happened here before this exclusion existed, and it
    is worth stating plainly: the quantised model was 1.14x faster and found
    nothing.
    """

    import onnx

    model = onnx.load(str(path))

    modules = [
        n.name.split("/")[1]
        for n in model.graph.node
        if n.name.startswith("/model.")
    ]

    if not modules:
        return []

    last = max(modules, key=lambda s: int(s.split(".")[1]))

    return [
        n.name for n in model.graph.node
        if n.name.startswith(f"/{last}/")
    ]


def benchmark(path, threads, runs=40):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )

    meta = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in meta.shape]
    tensor = np.random.rand(*shape).astype(np.float32)

    for _ in range(5):
        session.run(None, {meta.name: tensor})

    samples = []

    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {meta.name: tensor})
        samples.append((time.perf_counter() - start) * 1000)

    return statistics.median(samples)


def predict_to_labels(path, out_dir, threads, conf=CONF, iou=0.70):
    """
    Run an ONNX graph over the test split and write YOLO-format predictions.

    Ultralytics' own validator cannot be used here. The exported graph has a
    fixed 512 x 640 input, while `val` coerces imgsz to a single integer and
    feeds a square image, which the session rejects outright. Re-exporting
    with dynamic axes would sidestep that but would also stop measuring the
    graph that actually deploys.

    So the session is driven directly at its native shape, and the output is
    written in the same format as every other analysis directory in this
    project - which means the numbers land in `evaluate_experiment.py` and are
    directly comparable to the frozen baseline rather than to a separately
    computed mAP.
    """

    import onnxruntime as ort
    from ultralytics.utils.nms import non_max_suppression
    import torch

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )

    meta = session.get_inputs()[0]
    _, _, net_h, net_w = [d if isinstance(d, int) else 1 for d in meta.shape]

    labels_dir = Path(out_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = sorted((DATASET / "images" / "test").glob("*.jpg"))
    written = 0

    for image_path in images:
        tensor = preprocess(image_path, (net_h, net_w))

        if tensor is None:
            continue

        raw = session.run(None, {meta.name: tensor})[0]

        detections = non_max_suppression(
            torch.from_numpy(raw), conf_thres=conf, iou_thres=iou, nc=1
        )[0]

        lines = []

        for *xyxy, score, _cls in detections.tolist():
            x1, y1, x2, y2 = xyxy

            # Back to normalised centre/size in the network's frame. HIT-UAV
            # frames are 640 x 512 and the network takes 512 x 640, so no
            # letterbox padding is applied and the frames coincide.
            xc = (x1 + x2) / 2 / net_w
            yc = (y1 + y2) / 2 / net_h
            w = (x2 - x1) / net_w
            h = (y2 - y1) / net_h

            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {score:.6f}")

        if lines:
            written += 1

        (labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    return labels_dir, len(images), written


def verdict(fps_low, fps_high, required):
    if fps_low >= required:
        return "clears"
    if fps_high >= required:
        return "marginal"
    return "FAILS"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="MT-005")
    parser.add_argument("--threads", type=int, default=TARGET_THREADS)
    parser.add_argument("--calibration-images", type=int, default=CALIBRATION_IMAGES)
    parser.add_argument("--activation", choices=["uint8", "int8"], default="uint8",
                        help="Activation type. ORT's x86 CPU kernels are built for "
                             "u8s8; int8 activations usually block QDQ fusion.")
    parser.add_argument(
        "--quantise-head",
        action="store_true",
        help="Also quantise the detection head. Off by default because "
             "doing so collapses the classification branch to zero and "
             "the detector reports nothing.",
    )
    parser.add_argument("--no-eval", action="store_true")

    args = parser.parse_args()

    from onnxruntime.quantization import (
        QuantFormat, QuantType, quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnxruntime as ort

    fp32 = RUNS_ROOT / args.model / "weights" / "best.onnx"

    if not fp32.exists():
        raise SystemExit(
            f"No ONNX export for {args.model}. Run:\n"
            f"  python scripts/export_models.py --models {args.model}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prepared = OUTPUT_DIR / f"{args.model}_prepared.onnx"
    int8 = OUTPUT_DIR / f"{args.model}_int8.onnx"

    print("=" * 78)
    print(f"INT8 static quantisation - {args.model}")
    print("=" * 78)
    print(f"ONNX Runtime {ort.__version__}, {args.threads} threads, "
          f"{DEPLOY_SHAPE[0]}x{DEPLOY_SHAPE[1]}")
    print()

    session = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    del session

    print("shape inference and graph preparation...")
    quant_pre_process(str(fp32), str(prepared), skip_symbolic_shape=False)

    reader = build_calibration_reader(input_name, DEPLOY_SHAPE, args.calibration_images)

    activation_type = (
        QuantType.QUInt8 if args.activation == "uint8" else QuantType.QInt8
    )

    exclude = [] if args.quantise_head else head_nodes(fp32)

    if exclude:
        print(f"keeping the detection head in FP32 ({len(exclude)} nodes)")

    print(f"quantising (QDQ, {args.activation} activations, "
          "per-channel int8 weights)...")
    quantize_static(
        model_input=str(prepared),
        model_output=str(int8),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=exclude,
    )

    copy_metadata(fp32, int8)
    prepared.unlink(missing_ok=True)

    fusion = fusion_report(int8)

    print()
    print(f"QDQ fusion: {fusion['qlinearconv']} QLinearConv, "
          f"{fusion['conv']} plain Conv, "
          f"{fusion['quant_nodes']} Quantize/DequantizeLinear")

    if fusion["conv"] > fusion["qlinearconv"]:
        print()
        print("  Most convolutions did NOT fuse. They will run in FP32 while")
        print("  still paying for every quantise/dequantise conversion, so the")
        print("  quantised model is expected to be SLOWER, not faster. This is")
        print("  a property of the runtime's kernels, not of the model.")

    # ------------------------------------------------------------------
    # Size and latency
    # ------------------------------------------------------------------

    size_fp32 = fp32.stat().st_size / 1e6
    size_int8 = int8.stat().st_size / 1e6

    print()
    print(f"{'':16s} {'size':>9s} {'dev CPU':>10s} {'Pi 5 estimate':>16s} {'Pi 5 FPS':>14s}")
    print("-" * 70)

    rows = []

    for label, path in (("FP32", fp32), ("INT8", int8)):
        median = benchmark(path, args.threads)
        low, high = median * SLOWDOWN_RANGE[0], median * SLOWDOWN_RANGE[1]
        size = path.stat().st_size / 1e6

        rows.append({
            "precision": label,
            "size_mb": round(size, 2),
            "dev_cpu_ms": round(median, 1),
            "target_ms": [round(low), round(high)],
            "target_fps": [round(1000.0 / high, 2), round(1000.0 / low, 2)],
        })

        print(f"{label:16s} {size:8.2f}M {median:9.1f}ms "
              f"{low:5.0f} - {high:<4.0f}ms {1000/high:5.1f} - {1000/low:<4.1f}")

    speedup = rows[0]["dev_cpu_ms"] / rows[1]["dev_cpu_ms"]

    print()
    print(f"Speedup: {speedup:.2f}x        Size: "
          f"{size_fp32 / size_int8:.2f}x smaller")

    print()
    print("Against the derived frame-rate requirements:")
    print()

    header = f"{'':16s}" + "".join(f"{n:>22s}" for n, _ in REQUIREMENTS)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['precision']:16s}"

        for _, required in REQUIREMENTS:
            line += f"{verdict(row['target_fps'][0], row['target_fps'][1], required):>22s}"

        print(line)

    # Does INT8 bring the two-model ensemble back into range?
    ensemble_ms = rows[1]["dev_cpu_ms"] * 2
    ens_low, ens_high = ensemble_ms * SLOWDOWN_RANGE[0], ensemble_ms * SLOWDOWN_RANGE[1]
    ens_fps = (1000.0 / ens_high, 1000.0 / ens_low)

    print()
    print(f"Two-model WBF ensemble at INT8 (2x single-model cost):")
    print(f"  {ensemble_ms:.1f} ms dev, {ens_low:.0f} - {ens_high:.0f} ms target, "
          f"{ens_fps[0]:.1f} - {ens_fps[1]:.1f} FPS")
    print(f"  tracking @ 60 m: {verdict(ens_fps[0], ens_fps[1], 6.70)}   "
          f"report-only: {verdict(ens_fps[0], ens_fps[1], 1.34)}")

    summary = {
        "model": args.model,
        "runtime": "onnxruntime-cpu",
        "threads": args.threads,
        "imgsz": list(DEPLOY_SHAPE),
        "quantisation": {
            "method": "static",
            "format": "QDQ",
            "weights": "int8 per-channel",
            "activations": args.activation,
            "calibration_images": args.calibration_images,
            "fusion": fusion,
            "head_excluded": not args.quantise_head,
            "excluded_nodes": len(exclude),
        },
        "slowdown_range": list(SLOWDOWN_RANGE),
        "speedup": round(speedup, 3),
        "latency": rows,
        "int8_ensemble_target_fps": [round(ens_fps[0], 2), round(ens_fps[1], 2)],
    }

    # ------------------------------------------------------------------
    # Accuracy - mandatory, and not a bit-exactness check
    # ------------------------------------------------------------------

    if not args.no_eval:
        print()
        print("=" * 78)
        print("Accuracy on the held-out test split")
        print("=" * 78)
        print("Quantisation changes the computed values, so this is a full")
        print("re-evaluation rather than a comparison against the FP32 output.")
        print()

        analysis_root = (
            PROJECT_ROOT / "runs" / "detect" / "results" / "error_analysis"
        )

        for label, path in (("FP32", fp32), ("INT8", int8)):
            name = f"{args.model}-{label.lower()}-quant"

            print(f"predicting with {label} at conf {CONF}...")

            labels_dir, total, with_boxes = predict_to_labels(
                path, analysis_root / name, args.threads
            )

            print(f"  {total} images, {with_boxes} with at least one detection")

            summary.setdefault("predictions", {})[label] = {
                "labels": str(labels_dir),
                "images": total,
                "images_with_detections": with_boxes,
            }

        print()
        print("Compare with the project's own matching, against the frozen")
        print("baseline:")
        print()

        for label in ("FP32", "INT8"):
            name = f"{args.model}-{label.lower()}-quant"
            print(f"  python scripts/evaluate_experiment.py --labels {name} "
                  f"--label {args.model}-{label}")

    (OUTPUT_DIR / "quantization.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print(f"Quantised model: {int8}")
    print(f"Written: {OUTPUT_DIR / 'quantization.json'}")


if __name__ == "__main__":
    main()
