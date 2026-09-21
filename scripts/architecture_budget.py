"""
Does a candidate architecture fit on the target board, before training it?

Every experiment so far has changed training - loss weights, augmentation,
datasets - and left the network untouched, so the deployed cost never moved.
The literature survey in `docs/related_work.md` recommends two changes that do
touch the network: a P2 small-object detection head, and a larger backbone.
Both were proposed on accuracy grounds alone.

That is the wrong order. The companion computer is a Raspberry Pi 5 class
board, the binding frame-rate requirement is 6.7 FPS derived in
`coverage_requirements.py`, and the deployed baseline already sits at an
estimated 5.4-9.0 FPS. There is very little headroom to spend, so an
architecture that cannot clear the requirement is not worth a 50-epoch
training run to discover.

This script measures that first. It builds each candidate from its YAML at the
deployment class count, exports to ONNX at the deployment input shape, and
times it through ONNX Runtime on CPU with threads restricted to the target's
core count - the same harness and the same assumptions as
`benchmark_edge_cpu.py`, so the numbers are directly comparable.

Weights are random. Latency does not depend on their values, only on the
graph, so an untrained network gives the same timing as a trained one. That is
the point: the cost is knowable before the accuracy is.

Usage:

    python scripts/architecture_budget.py
    python scripts/architecture_budget.py --configs yolo26n.yaml yolo26n-p2.yaml
    python scripts/architecture_budget.py --imgsz 768 960
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark"

# Raspberry Pi 5: 4x Cortex-A76 @ 2.4 GHz.
TARGET_THREADS = 4

# Per-core slowdown of the target relative to this development CPU, as an
# estimated range. Identical to benchmark_edge_cpu.py so the two agree.
SLOWDOWN_RANGE = (3, 5)

# MT-005's native export shape, which is what the payload actually runs.
DEPLOY_SHAPE = (512, 640)

# Frame-rate requirements derived in coverage_requirements.py. The first is
# the binding one.
REQUIREMENTS = [
    ("tracking @ 60 m", 6.70),
    ("tracking @ 100 m", 4.02),
    ("report-only @ 60 m", 1.34),
]

CANDIDATES = [
    ("YOLO26n", "yolo26.yaml", "n", "deployed baseline (MT-005)"),
    ("YOLO26n-p2", "yolo26-p2.yaml", "n", "P2 small-object head"),
    ("YOLO26s", "yolo26.yaml", "s", "larger backbone"),
]


def single_class_config(source_name, scale, workdir):
    """
    Write a copy of an Ultralytics model YAML with nc set to 1.

    Ultralytics infers the scale from the filename, so the copy is named for
    the scale it should be built at.
    """

    from ultralytics.utils import ROOT

    matches = list((ROOT / "cfg" / "models").rglob(source_name))

    if not matches:
        raise SystemExit(f"Model config not found: {source_name}")

    stem = Path(source_name).stem
    family, _, suffix = stem.partition("-")

    target = workdir / f"{family}{scale}{'-' + suffix if suffix else ''}.yaml"

    text = matches[0].read_text(encoding="utf-8")
    lines = []

    for line in text.splitlines():
        if line.startswith("nc:"):
            line = "nc: 1 # single class: person"
        lines.append(line)

    target.write_text("\n".join(lines), encoding="utf-8")

    return target


def measure(session, runs):
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


def verdict(fps_low, fps_high, required):
    if fps_low >= required:
        return "clears"
    if fps_high >= required:
        return "marginal"
    return "FAILS"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--imgsz", type=int, nargs=2, default=list(DEPLOY_SHAPE))
    parser.add_argument("--threads", type=int, default=TARGET_THREADS)
    parser.add_argument("--runs", type=int, default=40)

    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime is required:  pip install onnxruntime")

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    shape = tuple(args.imgsz)

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    workdir = Path(tempfile.mkdtemp(prefix="arch_budget_"))
    rows = []

    try:
        for label, source, scale, note in CANDIDATES:
            config = single_class_config(source, scale, workdir)

            model = YOLO(str(config))

            params = sum(p.numel() for p in model.model.parameters())
            gflops = get_flops(model.model, imgsz=list(shape))

            exported = model.export(
                format="onnx", imgsz=shape, device="cpu",
                opset=17, verbose=False,
            )

            onnx_path = workdir / f"{label}.onnx"
            shutil.move(str(exported), str(onnx_path))

            session = ort.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )

            median = measure(session, args.runs)

            low, high = median * SLOWDOWN_RANGE[0], median * SLOWDOWN_RANGE[1]

            rows.append({
                "architecture": label,
                "note": note,
                "params_m": round(params / 1e6, 3),
                "gflops": round(gflops, 1),
                "dev_cpu_ms": round(median, 1),
                "target_ms": [round(low), round(high)],
                "target_fps": [round(1000.0 / high, 2), round(1000.0 / low, 2)],
            })
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print("=" * 78)
    print("Architecture compute budget")
    print("=" * 78)
    print(f"ONNX Runtime {ort.__version__}, CPU, {args.threads} threads, "
          f"{shape[0]}x{shape[1]}, nc=1, random weights")
    print("x86 measurement scaled by an estimated "
          f"{SLOWDOWN_RANGE[0]}-{SLOWDOWN_RANGE[1]}x for the ARM target.")
    print()

    header = (f"{'Architecture':<14s} {'params':>8s} {'GFLOPs':>7s} "
              f"{'dev CPU':>9s} {'target est':>14s} {'target FPS':>13s}")
    print(header)
    print("-" * len(header))

    for row in rows:
        print(f"{row['architecture']:<14s} {row['params_m']:7.2f}M "
              f"{row['gflops']:7.1f} {row['dev_cpu_ms']:8.1f}ms "
              f"{row['target_ms'][0]:5d} - {row['target_ms'][1]:<4d}ms "
              f"{row['target_fps'][0]:5.1f} - {row['target_fps'][1]:<4.1f}")

    baseline = rows[0]

    print()
    print(f"Relative to the deployed baseline ({baseline['dev_cpu_ms']:.1f} ms):")

    for row in rows[1:]:
        latency_ratio = row["dev_cpu_ms"] / baseline["dev_cpu_ms"]
        flops_ratio = row["gflops"] / baseline["gflops"]

        print(f"  {row['architecture']:<14s} {latency_ratio:5.2f}x latency, "
              f"{flops_ratio:5.2f}x GFLOPs", end="")

        if latency_ratio > flops_ratio * 1.1:
            print("   <- costs more than its arithmetic predicts")
        else:
            print()

    print()
    print("Against the derived frame-rate requirements:")
    print()

    header = f"{'Architecture':<14s}" + "".join(
        f"{name:>22s}" for name, _ in REQUIREMENTS
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['architecture']:<14s}"

        for _, required in REQUIREMENTS:
            line += f"{verdict(row['target_fps'][0], row['target_fps'][1], required):>22s}"

        print(line)

    print()
    print("  clears   - satisfied across the whole estimated range")
    print("  marginal - satisfied only at the optimistic end")
    print("  FAILS    - not satisfied even at the optimistic end")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "architecture_budget.json").write_text(
        json.dumps(
            {
                "runtime": "onnxruntime-cpu",
                "threads": args.threads,
                "imgsz": list(shape),
                "nc": 1,
                "slowdown_range": list(SLOWDOWN_RANGE),
                "requirements_fps": {name: req for name, req in REQUIREMENTS},
                "note": (
                    "Random weights; latency depends on the graph, not on "
                    "weight values. x86 proxy for a Raspberry Pi 5 class "
                    "board - confirm on real hardware."
                ),
                "architectures": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Written: {OUTPUT_DIR / 'architecture_budget.json'}")


if __name__ == "__main__":
    main()
