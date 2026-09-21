"""
Deployment latency benchmark for the trained human-detection models.

The Ultralytics validator reports throughput that includes dataloader and
per-image Python overhead. Those numbers are fine for comparing runs but are
not what a real-time payload experiences, and they must not be used to size a
companion computer.

This script measures what the pipeline actually pays per frame:

  forward     pure model forward pass on a preloaded GPU tensor
  end_to_end  letterbox + forward + NMS, starting from a BGR frame in memory
  batched     the same forward at batch 8, to expose the throughput ceiling
  cuda_graph  the forward pass replayed from a captured CUDA graph, which
              eliminates per-kernel launch overhead entirely

Disk I/O is excluded on purpose: a live pipeline receives frames from a camera,
not from a JPEG decoder.

Every timing uses CUDA synchronisation, a warm-up phase, and reports the median
and 95th percentile rather than the mean, because a real-time system is sized
by its slow frames.

The script also records a launch-bound diagnostic. It times the forward pass
without synchronising, which measures only how long the CPU takes to enqueue
the kernels. When that figure is close to the synchronised wall time, the GPU
is finishing before the CPU can submit work, and single-frame latency is
limited by per-layer launch overhead rather than by GPU compute. That
distinction decides whether the right optimisation is a smaller model or a
compiled/exported graph.

The CUDA-graph measurement is the direct test of that conclusion. Capturing
the forward pass into a graph and replaying it submits the whole network with
one launch instead of hundreds. Replay is bit-identical to eager execution -
it runs the same kernels in the same order - so any speedup it produces is
pure launch overhead that a deployed, exported pipeline can also recover.

Usage:

    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --models MT-005 --runs 300
    python scripts/benchmark_latency.py --half
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

RUNS_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"
OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark"

WARMUP = 20
RUNS = 200

# Source frame sizes, i.e. what the sensor delivers before letterboxing.
MODELS = [
    {
        "id": "MT-004",
        "modality": "RGB",
        "imgsz": 1280,
        "frame": (1080, 1920),
        "weights": RUNS_ROOT / "MT-004" / "weights" / "best.pt",
    },
    {
        "id": "MT-005",
        "modality": "Thermal",
        "imgsz": 640,
        "frame": (512, 640),
        "graph_shape": (512, 640),
        "weights": RUNS_ROOT / "MT-005" / "weights" / "best.pt",
    },
    {
        "id": "MT-006",
        "modality": "Thermal",
        "imgsz": 960,
        "frame": (512, 640),
        "graph_shape": (768, 960),
        "weights": RUNS_ROOT / "MT-006" / "weights" / "best.pt",
    },
    {
        "id": "MT-007",
        "modality": "Thermal",
        "imgsz": 640,
        "frame": (512, 640),
        "graph_shape": (512, 640),
        "weights": RUNS_ROOT / "MT-007" / "weights" / "best.pt",
    },
]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(samples_s):
    """Convert a list of per-frame durations in seconds to a report block."""

    ms = sorted(value * 1000.0 for value in samples_s)

    return {
        "median_ms": round(statistics.median(ms), 2),
        "mean_ms": round(statistics.fmean(ms), 2),
        "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 2),
        "min_ms": round(ms[0], 2),
        "max_ms": round(ms[-1], 2),
        "fps_median": round(1000.0 / statistics.median(ms), 1),
    }


def measure_forward(model, entry, device, half, runs):
    """Pure forward pass on a resident GPU tensor."""

    module = model.model.to(device)
    module.eval()

    if half:
        module = module.half()

    dtype = torch.float16 if half else torch.float32

    tensor = torch.rand(
        1, 3, entry["imgsz"], entry["imgsz"],
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(WARMUP):
            module(tensor)

        synchronize(device)

        samples = []

        for _ in range(runs):
            start = time.perf_counter()
            module(tensor)
            synchronize(device)
            samples.append(time.perf_counter() - start)

    return summarize(samples)


def measure_batched(model, entry, device, half, runs, batch=8):
    """Forward pass at batch 8, which shows the throughput ceiling."""

    module = model.model.to(device)
    module.eval()

    if half:
        module = module.half()

    dtype = torch.float16 if half else torch.float32

    tensor = torch.rand(
        batch, 3, entry["imgsz"], entry["imgsz"],
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(max(5, WARMUP // 2)):
            module(tensor)

        synchronize(device)

        samples = []

        for _ in range(max(20, runs // 4)):
            start = time.perf_counter()
            module(tensor)
            synchronize(device)
            samples.append((time.perf_counter() - start) / batch)

    report = summarize(samples)
    report["batch"] = batch

    return report


def measure_launch_overhead(model, entry, device, half, iterations=40):
    """
    Time the forward pass with and without CUDA synchronisation.

    Without synchronisation the measurement captures only CPU-side kernel
    enqueue time. A ratio near 1.0 means the pipeline is launch-bound.
    """

    module = model.model.to(device)
    module.eval()

    if half:
        module = module.half()

    dtype = torch.float16 if half else torch.float32

    tensor = torch.rand(
        1, 3, entry["imgsz"], entry["imgsz"],
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(WARMUP):
            module(tensor)

        synchronize(device)

        start = time.perf_counter()

        for _ in range(iterations):
            module(tensor)

        cpu_ms = (time.perf_counter() - start) / iterations * 1000.0

        synchronize(device)

        wall_ms = (time.perf_counter() - start) / iterations * 1000.0

    return {
        "cpu_enqueue_ms": round(cpu_ms, 2),
        "wall_ms": round(wall_ms, 2),
        "ratio": round(cpu_ms / wall_ms, 3) if wall_ms else None,
        "launch_bound": bool(wall_ms and cpu_ms / wall_ms > 0.90),
    }


def measure_cuda_graph(model, entry, device, half, runs):
    """
    Capture the forward pass into a CUDA graph and replay it.

    Returns (report, max_abs_diff) where the difference is measured against
    eager execution on the same input. Replay should be bit-identical; a
    non-zero difference means the capture is unsound and the timing must not
    be trusted.
    """

    if device.type != "cuda":
        return None, None

    module = model.model.to(device)
    module.eval()

    if half:
        module = module.half()

    dtype = torch.float16 if half else torch.float32

    shape = entry.get("graph_shape", (entry["imgsz"], entry["imgsz"]))

    static_input = torch.rand(
        1, 3, shape[0], shape[1], device=device, dtype=dtype
    )

    def raw(output):
        return output[0] if isinstance(output, (list, tuple)) else output

    try:
        with torch.inference_mode():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(side):
                for _ in range(5):
                    module(static_input)

            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                static_output = module(static_input)
    except Exception as error:
        print(f"  CUDA graph capture failed: {type(error).__name__}: {error}")
        return None, None

    # Correctness before speed.
    worst = 0.0

    with torch.inference_mode():
        for _ in range(5):
            probe = torch.rand_like(static_input)

            eager = raw(module(probe)).clone()

            static_input.copy_(probe)
            graph.replay()

            worst = max(
                worst,
                (eager - raw(static_output)).abs().max().item(),
            )

    samples = []

    with torch.inference_mode():
        for _ in range(WARMUP):
            graph.replay()

        synchronize(device)

        for _ in range(runs):
            start = time.perf_counter()
            graph.replay()
            synchronize(device)
            samples.append(time.perf_counter() - start)

    report = summarize(samples)
    report["max_abs_diff_vs_eager"] = worst
    report["bit_identical"] = worst == 0.0
    report["shape"] = list(shape)

    return report, worst


def measure_end_to_end(model, entry, device, half, runs):
    """Letterbox + forward + NMS, from an in-memory BGR frame."""

    height, width = entry["frame"]

    rng = np.random.default_rng(0)

    frame = rng.integers(
        0, 256,
        size=(height, width, 3),
        dtype=np.uint8,
    )

    predict_kwargs = {
        "imgsz": entry["imgsz"],
        "device": device.index if device.type == "cuda" else "cpu",
        "half": half,
        "conf": 0.25,
        "iou": 0.70,
        "verbose": False,
    }

    for _ in range(WARMUP):
        model.predict(frame, **predict_kwargs)

    synchronize(device)

    samples = []

    for _ in range(runs):
        start = time.perf_counter()
        model.predict(frame, **predict_kwargs)
        synchronize(device)
        samples.append(time.perf_counter() - start)

    return summarize(samples)


def benchmark(entry, device, half, runs):
    weights = entry["weights"]

    if not weights.exists():
        print(f"SKIP {entry['id']}: weights not found at {weights}")
        return None

    print()
    print("=" * 62)
    print(f"{entry['id']} - {entry['modality']} - imgsz {entry['imgsz']}")
    print(f"source frame: {entry['frame'][1]} x {entry['frame'][0]}")
    print(f"precision:    {'FP16' if half else 'FP32'}")
    print("=" * 62)

    model = YOLO(str(weights))

    end_to_end = measure_end_to_end(model, entry, device, half, runs)
    forward = measure_forward(model, entry, device, half, runs)
    batched = measure_batched(model, entry, device, half, runs)
    launch = measure_launch_overhead(model, entry, device, half)
    graphed, _ = measure_cuda_graph(model, entry, device, half, runs)

    row = {
        "experiment": entry["id"],
        "modality": entry["modality"],
        "imgsz": entry["imgsz"],
        "source_frame": f"{entry['frame'][1]}x{entry['frame'][0]}",
        "precision": "FP16" if half else "FP32",
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "warmup": WARMUP,
        "runs": runs,
        "forward": forward,
        "end_to_end": end_to_end,
        "batched": batched,
        "launch": launch,
        "cuda_graph": graphed,
    }

    print(
        f"forward    median {forward['median_ms']:6.2f} ms  "
        f"p95 {forward['p95_ms']:6.2f} ms  "
        f"{forward['fps_median']:6.1f} FPS"
    )
    print(
        f"end-to-end median {end_to_end['median_ms']:6.2f} ms  "
        f"p95 {end_to_end['p95_ms']:6.2f} ms  "
        f"{end_to_end['fps_median']:6.1f} FPS"
    )
    print(
        f"batch {batched['batch']}    median {batched['median_ms']:6.2f} ms/img"
        f"                    {batched['fps_median']:6.1f} FPS"
    )
    print(
        f"launch     cpu enqueue {launch['cpu_enqueue_ms']:.2f} ms vs wall "
        f"{launch['wall_ms']:.2f} ms  (ratio {launch['ratio']}) -> "
        f"{'LAUNCH-BOUND' if launch['launch_bound'] else 'compute-bound'}"
    )

    if graphed:
        print(
            f"cudagraph  median {graphed['median_ms']:6.2f} ms  "
            f"p95 {graphed['p95_ms']:6.2f} ms  "
            f"{graphed['fps_median']:6.1f} FPS  "
            f"speedup {forward['median_ms'] / graphed['median_ms']:.2f}x  "
            f"[{'bit-identical' if graphed['bit_identical'] else 'DIVERGENT'}]"
        )

    return row


def write_outputs(rows, half):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "fp16" if half else "fp32"

    json_path = OUTPUT_DIR / f"latency_{suffix}.json"
    md_path = OUTPUT_DIR / f"latency_{suffix}.md"

    json_path.write_text(json.dumps(rows, indent=2))

    lines = [
        f"Precision: {'FP16' if half else 'FP32'}  |  "
        f"Device: {rows[0]['device']}  |  "
        f"Warm-up: {rows[0]['warmup']}  |  Runs: {rows[0]['runs']}",
        "",
        "| Experiment | imgsz | Source frame | Forward median | Forward p95 | "
        "End-to-end median | End-to-end p95 | End-to-end FPS |",
        "| ---------- | ----: | ------------ | -------------: | ----------: | "
        "----------------: | -------------: | -------------: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['imgsz']} | {row['source_frame']} "
            f"| {row['forward']['median_ms']:.2f} ms "
            f"| {row['forward']['p95_ms']:.2f} ms "
            f"| {row['end_to_end']['median_ms']:.2f} ms "
            f"| {row['end_to_end']['p95_ms']:.2f} ms "
            f"| {row['end_to_end']['fps_median']:.1f} |"
        )

    lines += [
        "",
        "| Experiment | Batch-8 per image | Batch-8 FPS | CPU enqueue | Wall | "
        "Ratio | Verdict |",
        "| ---------- | ----------------: | ----------: | ----------: | ---: | "
        "----: | ------- |",
    ]

    for row in rows:
        lines.append(
            f"| {row['experiment']} "
            f"| {row['batched']['median_ms']:.2f} ms "
            f"| {row['batched']['fps_median']:.1f} "
            f"| {row['launch']['cpu_enqueue_ms']:.2f} ms "
            f"| {row['launch']['wall_ms']:.2f} ms "
            f"| {row['launch']['ratio']} "
            f"| {'launch-bound' if row['launch']['launch_bound'] else 'compute-bound'} |"
        )

    md_path.write_text("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print()
    print(f"Written: {json_path}")
    print(f"Written: {md_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument(
        "--half",
        action="store_true",
        help="Benchmark FP16, which is the expected embedded deployment mode.",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.half and device.type != "cuda":
        print("FP16 requires CUDA; falling back to FP32.")
        args.half = False

    selected = [
        entry for entry in MODELS
        if args.models is None or entry["id"] in args.models
    ]

    rows = []

    for entry in selected:
        result = benchmark(entry, device, args.half, args.runs)

        if result is not None:
            rows.append(result)

    if not rows:
        print("No models were benchmarked.")
        return

    write_outputs(rows, args.half)


if __name__ == "__main__":
    main()
