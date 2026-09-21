"""
CPU-only inference benchmark, as a proxy for the Raspberry Pi companion computer.

The project's companion computer is a Raspberry Pi 5 class board, chosen under
budget constraints - not a Jetson. That matters more than any other single fact
about deployment, because every latency number measured so far in this project
was taken on an RTX 4050 laptop GPU:

  - The CUDA-graph result (16x speedup on MT-005) does not transfer. A Pi 5
    has no CUDA device at all.
  - The launch-bound finding does not transfer either. It described a host
    dispatching kernels faster than a GPU could be kept busy; on a CPU-only
    board there are no kernel launches to eliminate.

What does transfer is the ONNX export, which is why it was verified. This
script runs the exported graph through ONNX Runtime on CPU with the thread
count restricted to the target's core count, which is the closest honest
approximation available without the hardware in hand.

The result is an x86 measurement, not an ARM one. A Cortex-A76 at 2.4 GHz is
substantially slower per core than a Zen 4 core near 4 GHz, and the two have
different SIMD (NEON vs AVX2), so the scaling factor is an estimate rather than
a conversion. The script reports a range rather than a single number for that
reason, and the range must be confirmed on real hardware before any purchase
decision rests on it.

Usage:

    python scripts/benchmark_edge_cpu.py
    python scripts/benchmark_edge_cpu.py --threads 4 --runs 50
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"
OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark"

# Raspberry Pi 5: 4x Cortex-A76 @ 2.4 GHz.
TARGET_THREADS = 4

# Per-core slowdown of the target relative to this development CPU. A range,
# because it is an estimate: NEON and AVX2 do not scale the same way across
# operator types.
SLOWDOWN_RANGE = (3, 5)

MODELS = ["MT-005", "MT-004", "MT-006", "MT-007"]


def benchmark(session, input_meta, runs, warmup=5):
    shape = [d if isinstance(d, int) else 1 for d in input_meta.shape]

    tensor = np.random.rand(*shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_meta.name: tensor})

    samples = []

    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_meta.name: tensor})
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()

    return {
        "input_shape": shape,
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
        "min_ms": round(samples[0], 2),
        "fps": round(1000.0 / statistics.median(samples), 2),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--threads", type=int, default=TARGET_THREADS)
    parser.add_argument("--runs", type=int, default=30)

    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit(
            "onnxruntime is required:  pip install onnxruntime"
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    print(f"ONNX Runtime {ort.__version__}, CPU, {args.threads} threads")
    print("This is an x86 measurement used as a proxy for the ARM target.")

    rows = []

    for name in args.models:
        path = RUNS_ROOT / name / "weights" / "best.onnx"

        if not path.exists():
            print(f"SKIP {name}: no ONNX export at {path}")
            print(f"     run: python scripts/export_models.py --models {name}")
            continue

        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

        result = benchmark(session, session.get_inputs()[0], args.runs)
        result["experiment"] = name

        result["target_estimate_ms"] = [
            round(result["median_ms"] * SLOWDOWN_RANGE[0]),
            round(result["median_ms"] * SLOWDOWN_RANGE[1]),
        ]

        result["target_estimate_fps"] = [
            round(1000.0 / result["target_estimate_ms"][1], 2),
            round(1000.0 / result["target_estimate_ms"][0], 2),
        ]

        rows.append(result)

        print()
        print(f"{name}  input {result['input_shape']}")
        print(
            f"  dev CPU        median {result['median_ms']:7.1f} ms  "
            f"({result['fps']:5.1f} FPS)"
        )
        print(
            f"  target est.    {result['target_estimate_ms'][0]:4d}-"
            f"{result['target_estimate_ms'][1]:<4d} ms       "
            f"({result['target_estimate_fps'][0]:.1f}-"
            f"{result['target_estimate_fps'][1]:.1f} FPS)"
        )

    if not rows:
        print("\nNothing benchmarked. Export models first.")
        return

    # Three-model thermal ensemble, if all its members were measured.
    ensemble = [r for r in rows if r["experiment"] in ("MT-005", "MT-006", "MT-007")]

    summary = {
        "runtime": "onnxruntime-cpu",
        "threads": args.threads,
        "slowdown_range": list(SLOWDOWN_RANGE),
        "note": (
            "x86 measurement used as a proxy for a Raspberry Pi 5 class "
            "board. Confirm on real hardware before relying on it."
        ),
        "models": rows,
    }

    if len(ensemble) == 3:
        total = sum(r["median_ms"] for r in ensemble)

        summary["thermal_ensemble"] = {
            "dev_cpu_ms": round(total, 1),
            "dev_cpu_fps": round(1000.0 / total, 2),
            "target_estimate_ms": [
                round(total * SLOWDOWN_RANGE[0]),
                round(total * SLOWDOWN_RANGE[1]),
            ],
            "target_estimate_fps": [
                round(1000.0 / (total * SLOWDOWN_RANGE[1]), 2),
                round(1000.0 / (total * SLOWDOWN_RANGE[0]), 2),
            ],
        }

        block = summary["thermal_ensemble"]

        print()
        print("Three-model thermal ensemble (sum of members)")
        print(
            f"  dev CPU        median {block['dev_cpu_ms']:7.1f} ms  "
            f"({block['dev_cpu_fps']:5.2f} FPS)"
        )
        print(
            f"  target est.    {block['target_estimate_ms'][0]:4d}-"
            f"{block['target_estimate_ms'][1]:<4d} ms       "
            f"({block['target_estimate_fps'][0]:.2f}-"
            f"{block['target_estimate_fps'][1]:.2f} FPS)"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / "edge_cpu.json"
    path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"Written: {path}")


if __name__ == "__main__":
    main()
