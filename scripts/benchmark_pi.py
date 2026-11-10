"""
Measure the payload on the companion computer itself - task C2.

Every frame rate in this project so far is an x86 measurement scaled by an
estimated factor. This is the measurement that replaces the estimate. Run it
on the Raspberry Pi, with the cooling and power supply it will fly with:

    python scripts/benchmark_pi.py --weights release/weights/best.pt \\
        --clip release/benchmark/clip.mp4 --telemetry release/benchmark/telemetry.jsonl

It records:

  - the board: CPU, cores, memory, OS, library versions
  - each stage alone on 640 x 512 frames: the detector (ONNX Runtime, and
    PyTorch with --torch), and camera-motion estimation at full and half scale
  - the whole payload over the clip, as it would run in flight
  - CPU temperature before and after, and the firmware's throttling flags:
    a Pi that throttles part-way through a sortie is slower than it bench-tests

and compares the payload's frame rate with what the mission needs
(docs/deployment_target.md section 3b). Results go to benchmark_pi.json -
bring that file back; it is what the deployment documents should quote.
"""

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Frame rates the mission needs, from coverage_requirements.py (section 3b).
REQUIREMENTS = {
    60: {"coverage": 0.64, "confirm": 1.34, "tracking": 6.70},
    80: {"coverage": 0.48, "confirm": 1.01, "tracking": 5.03},
    100: {"coverage": 0.38, "confirm": 0.80, "tracking": 4.02},
    130: {"coverage": 0.29, "confirm": 0.62, "tracking": 3.09},
}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def temperature_c():
    raw = read("/sys/class/thermal/thermal_zone0/temp")
    return round(int(raw) / 1000.0, 1) if raw and raw.isdigit() else None


def throttled():
    """vcgencmd get_throttled, decoded. None off a Raspberry Pi."""

    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=5).stdout
        value = int(out.strip().split("=")[1], 16)
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None

    flags = {0: "under-voltage now", 1: "frequency capped now", 2: "throttled now",
             3: "soft temperature limit now", 16: "under-voltage has occurred",
             17: "frequency capping has occurred", 18: "throttling has occurred",
             19: "soft temperature limit has occurred"}
    return {"raw": hex(value), "flags": [name for bit, name in flags.items() if value >> bit & 1]}


def system():
    cpu = None
    cpuinfo = read("/proc/cpuinfo") or ""
    for line in cpuinfo.splitlines():
        if line.startswith(("Model", "model name")):
            cpu = line.split(":", 1)[1].strip()

    memory = None
    for line in (read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal"):
            memory = f"{int(line.split()[1]) / 1e6:.1f} GB"

    info = {
        "machine": platform.machine(), "system": platform.platform(),
        "cpu": cpu or platform.processor(), "cores": os.cpu_count(), "memory": memory,
        "python": platform.python_version(), "opencv": cv2.__version__,
    }

    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        info["onnxruntime"] = None

    return info


def load_clip(path, limit):
    capture = cv2.VideoCapture(str(path))
    frames = []

    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)

    if not frames:
        raise SystemExit(f"Could not read frames from {path}")

    return frames


def time_stage(fn, frames, warmup=5):
    for frame in frames[:warmup]:
        fn(frame)

    times = []
    for frame in frames[warmup:]:
        start = time.perf_counter()
        fn(frame)
        times.append(time.perf_counter() - start)

    return {"median_ms": round(1000 * statistics.median(times), 1),
            "p95_ms": round(1000 * float(np.percentile(times, 95)), 1),
            "frames": len(times)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="best.pt; best.onnx beside it is used for ONNX Runtime")
    parser.add_argument("--clip", required=True, help="640 x 512 video")
    parser.add_argument("--telemetry", default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--torch", action="store_true", help="also time the PyTorch backend")
    parser.add_argument("--threads", type=int, default=None,
                        help="Inference threads. Set to match any CPU restriction "
                             "(e.g. taskset -c 0,1 with --threads 2), or ONNX Runtime "
                             "oversubscribes the cores it has")
    parser.add_argument("--output", default="benchmark_pi.json")
    args = parser.parse_args()

    import payload

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    result = {"system": system(), "threads": args.threads,
              "temperature_start_c": temperature_c(),
              "throttled_start": throttled(), "stages": {}}
    print(json.dumps(result["system"], indent=2))

    frames = load_clip(args.clip, args.frames)
    height, width = frames[0].shape[:2]
    result["frame_size"] = [width, height]
    if (width, height) != (640, 512):
        print(f"WARNING: clip is {width} x {height}; the sensor is 640 x 512")

    backends = ["onnx"] + (["torch"] if args.torch else [])
    for backend in backends:
        with contextlib.redirect_stdout(io.StringIO()):
            detector = payload.Detector([payload.DEFAULT_THERMAL], device="cpu",
                                        explicit_weights=args.weights, backend=backend,
                                        threads=args.threads)
        result["stages"][f"detector_{backend}"] = time_stage(detector, frames)
        print(f"detector ({backend}):   {result['stages'][f'detector_{backend}']}")

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    for scale in (1.0, 0.5):
        ego = payload.make_ego_estimator(scale)
        result["stages"][f"ego_motion_scale_{scale}"] = time_stage(ego.update, gray)
        print(f"ego-motion (x{scale}):   {result['stages'][f'ego_motion_scale_{scale}']}")

    # The whole payload, as it runs in flight.
    runs = {}
    for label, extra in (("payload", []), ("payload, ego x0.5", ["--ego-scale", "0.5"])):
        summary_path = Path(args.output).with_suffix(".summary.json")
        argv = sys.argv
        sys.argv = ["payload.py", "--source", args.clip, "--weights", args.weights,
                    "--device", "cpu", "--max-frames", str(args.frames),
                    "--summary", str(summary_path), "--progress", "100000", *extra]
        if args.telemetry:
            sys.argv += ["--telemetry", args.telemetry]
        if args.threads:
            sys.argv += ["--threads", str(args.threads)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                payload.main()
        finally:
            sys.argv = argv

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs[label] = {k: summary.get(k) for k in
                       ("pipeline_fps", "frame_ms_median", "frame_ms_p95", "detect_ms_median")}
        print(f"{label + ':':<21s} {runs[label]}")

    result["payload"] = runs
    result["temperature_end_c"] = temperature_c()
    result["throttled_end"] = throttled()

    fps = runs["payload"]["pipeline_fps"]
    print()
    print(f"Payload: {fps} FPS. Against the mission (section 3b):")
    print(f"{'altitude':>9s} {'coverage':>9s} {'confirm':>8s} {'tracking':>9s}")
    verdict = {}
    for altitude, need in REQUIREMENTS.items():
        row = {k: bool(fps and fps >= v) for k, v in need.items()}
        verdict[altitude] = row
        print(f"{altitude:7d} m " + " ".join(
            f"{('yes' if row[k] else 'NO'):>{w}s}" for k, w in
            (("coverage", 9), ("confirm", 8), ("tracking", 9))))
    result["meets"] = verdict

    print(f"temperature {result['temperature_start_c']} -> {result['temperature_end_c']} C, "
          f"throttling: {result['throttled_end']}")

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
