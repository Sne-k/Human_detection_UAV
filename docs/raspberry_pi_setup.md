# Raspberry Pi 4 Model B - Setup and First Measurement

The companion computer is a **Raspberry Pi 4 Model B, 8 GB** (Broadcom
BCM2711, 4 x Cortex-A72 at 1.8 GHz). This is how to put the payload on it and
take the measurement that every frame rate in this project has been waiting
for. `deployment_target.md` section 3g gives the estimate it replaces.

## 1. Board

- **OS:** Raspberry Pi OS (64-bit), Lite is enough. The 64-bit OS is required:
  ONNX Runtime and PyTorch publish wheels for `aarch64` only.
- **Cooling:** a heatsink and fan, mounted as they will fly. The payload keeps
  all four cores busy continuously, and a Pi 4 cuts its clock once the SoC
  reaches its thermal limit. A bench test without the flight enclosure
  overstates the frame rate.
- **Power:** a 5 V, 3 A supply (or regulator from the flight battery). Under
  load with a USB camera attached, a weaker supply causes under-voltage
  throttling, which the benchmark reports.
- **Storage:** the detection stream is small, but a good-quality card or USB 3
  SSD avoids stalls when the run log is written.

## 2. Software

```bash
sudo apt update
sudo apt install -y git python3-venv
git clone https://github.com/Sne-k/Human_detection_UAV.git
cd Human_detection_UAV
python3 -m venv venv
. venv/bin/activate
pip install numpy opencv-python-headless onnxruntime pymavlink ultralytics
```

`ultralytics` brings PyTorch with it. The detector itself runs on ONNX
Runtime, but the tracker (ByteTrack) comes from Ultralytics, so PyTorch must
be installed; with 8 GB of memory that costs nothing that matters.

## 3. Weights

Copy the release folder (`results/release/thermal-v1.2` on the development
machine) to the Pi, e.g. as `~/Human_detection_UAV/release/`. Check that it
arrived intact:

```bash
python scripts/make_release.py --verify release
```

It holds `weights/best.pt` and `weights/best.onnx` (the payload uses the ONNX
file on a CPU), and `benchmark/clip.mp4` with `benchmark/telemetry.jsonl`:
30 s of the simulated mission at the sensor's 640 x 512.

## 4. Measure

Leave the board idle for a few minutes first, then:

```bash
python scripts/benchmark_pi.py --weights release/weights/best.pt --clip release/benchmark/clip.mp4 --telemetry release/benchmark/telemetry.jsonl --torch
```

It times the detector (ONNX Runtime and PyTorch), camera-motion estimation at
full and half scale, and the whole payload with and without half-scale
camera motion; records temperature and the firmware's throttling flags; and
says which mission requirements the measured frame rate meets. Bring back
`benchmark_pi.json`.

For a thermal soak, run it three times back to back and compare the last run
with the first.

## 5. Run

From a recording:

```bash
python scripts/payload.py --source release/benchmark/clip.mp4 --telemetry release/benchmark/telemetry.jsonl --weights release/weights/best.pt --device cpu --jsonl detections.jsonl
```

In flight, with the flight controller on the serial header and the ground
station on the telemetry radio's network:

```bash
python scripts/payload.py --source 0 --mavlink /dev/serial0 --mavlink-baud 921600 --weights release/weights/best.pt --device cpu --jsonl detections.jsonl --udp 192.168.1.20:14660
```

`--device cpu` selects ONNX Runtime automatically. Two options exist for a
slow board, and the benchmark says whether either is needed:

- `--ego-scale 0.5` runs camera-motion estimation on a half-size frame,
  about 2.5x cheaper, at a small cost in stabilisation accuracy.
- `--assoc-scale 2.0` keeps walking people trackable below about 2 FPS, at
  the cost of more identity swaps (training_log section 54).
