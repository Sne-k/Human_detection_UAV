"""
Freeze the deployed detector into one traceable package - task D2.

Thirteen experiments and a seed repeat have left many sets of weights on
disk. The aircraft should carry exactly one, and it should be possible, from
the file on the companion computer, to recover exactly what was evaluated:
the weights and their export, the training configuration and seed, the data
it was trained and tested on, every runtime parameter that changes what the
payload emits, and the measurements that justified shipping it.

This copies all of that into results/release/<name>/ and writes MANIFEST.json
with a SHA-256 for every file. The dataset is fingerprinted rather than
copied: a hash over every image name and label file in each split, so a
silently changed label or a reshuffled split changes the fingerprint.

Usage:

    python scripts/make_release.py
    python scripts/make_release.py --name thermal-v1 --model MT-011b
    python scripts/make_release.py --verify results/release/thermal-v1
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)
TRAINING = PROJECT_ROOT / "runs" / "detect" / "results" / "training"
DATASET = PROJECT_ROOT / "dataset" / "hit_uav_person"
RESULTS = PROJECT_ROOT / "results"

# Measurements that justify the release, copied in if present.
EVIDENCE = [
    "deployed_config/deployed_config.json",
    "tracking_gate/tracking_gate.json",
    "telemetry_sync/telemetry_sync.json",
    "geolocation_sensitivity/geolocation_sensitivity.json",
    "ego_health/ego_health_probe.json",
    "single_model_wbf/confidence_sweep.json",
    "fov_study/fov_study_100m.json",
    "frame_drops/frame_drops_long.json",
    "tracker_stress/tracker_stress.json",
    "mission/seed0_tiles24/score.json",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint():
    """Per split: image count, and a hash over names and label contents."""

    splits = {}

    for split in ("train", "val", "test"):
        images = sorted((DATASET / "images" / split).glob("*.jpg"))
        h = hashlib.sha256()

        for image in images:
            label = DATASET / "labels" / split / f"{image.stem}.txt"
            h.update(image.name.encode())
            h.update(label.read_bytes() if label.exists() else b"<no label>")

        splits[split] = {"images": len(images), "sha256": h.hexdigest()}

    return splits


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip() or None
    except OSError:
        return None


def build(name, model):
    import payload

    run = TRAINING / model
    weights = run / "weights" / "best.pt"

    if not weights.exists():
        raise SystemExit(f"No weights for {model} at {weights}")

    out = RESULTS / "release" / name

    if out.exists():
        raise SystemExit(f"{out} exists - a release is frozen; choose a new name")

    (out / "evidence").mkdir(parents=True)

    files = {"weights/best.pt": weights}

    onnx = run / "weights" / "best.onnx"
    if onnx.exists():
        files["weights/best.onnx"] = onnx

    for config in ("args.yaml", "results.csv"):
        if (run / config).exists():
            files[f"training/{config}"] = run / config

    for item in EVIDENCE:
        if (RESULTS / item).exists():
            files[f"evidence/{Path(item).name}"] = RESULTS / item

    for dest, src in files.items():
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / dest)

    runtime = {
        "model": model,
        "input_shape_hw": list(payload.MODEL_SHAPES[model]),
        "confidence_default": payload.CONF,
        "single_model_nms_iou": payload.SINGLE_MODEL_NMS_IOU,
        "single_model_fuse_iou": payload.SINGLE_MODEL_FUSE_IOU,
        "fusion_score": "seed",
        "tracker": "ByteTrack, thresholds derived from --conf, stabilised frame",
        "track_low_thresh": payload.TRACK_LOW_THRESH,
        "ego_motion": payload.EGO_SETTINGS,
        "unassociated_default": payload.UNASSOCIATED_DEFAULT,
        "assoc_scale": payload.ASSOC_SCALE,
        "gap_factor": payload.GAP_FACTOR,
        "bridge_max_steps": payload.BRIDGE_MAX_STEPS,
    }

    import realtime_detect
    runtime["movement"] = {
        "history_frames": realtime_detect.HISTORY_FRAMES,
        "threshold_px": realtime_detect.MOVEMENT_THRESHOLD_PX,
        "min_history": realtime_detect.MIN_HISTORY_FOR_STATE,
        "stationary_min_span_s": realtime_detect.STATIONARY_MIN_SPAN_S,
    }

    import geolocate
    runtime["geolocation_sigmas"] = {
        "roll_deg": geolocate.SIGMA_ROLL_DEG,
        "pitch_deg": geolocate.SIGMA_PITCH_DEG,
        "yaw_deg": geolocate.SIGMA_YAW_DEG,
        "altitude_m": geolocate.SIGMA_ALTITUDE_M,
        "gps_m": geolocate.SIGMA_GPS_M,
    }
    runtime["ground_point_person_width_to_height"] = geolocate.PERSON_WIDTH_TO_HEIGHT
    runtime["ground_station_fix"] = "inverse-variance mean; tightest single sigma"

    args_yaml = (run / "args.yaml").read_text(encoding="utf-8") if (run / "args.yaml").exists() else ""
    seed = next((line.split(":", 1)[1].strip() for line in args_yaml.splitlines()
                 if line.startswith("seed:")), None)

    import torch
    import ultralytics

    manifest = {
        "release": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "model": model,
        "training_seed": seed,
        "dataset": {"name": "HIT-UAV, single-class person", "splits": dataset_fingerprint()},
        "runtime": runtime,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "files": {dest: {"sha256": sha256(out / dest), "bytes": (out / dest).stat().st_size}
                  for dest in sorted(files)},
    }

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Release {name}: {model}, seed {seed}, commit {manifest['code_commit'][:10] if manifest['code_commit'] else '?'}")
    for split, info in manifest["dataset"]["splits"].items():
        print(f"  dataset {split:<5s} {info['images']:5d} images  {info['sha256'][:16]}")
    for dest, info in manifest["files"].items():
        print(f"  {dest:<45s} {info['bytes'] / 1e6:8.2f} MB  {info['sha256'][:16]}")
    print(f"Written: {out / 'MANIFEST.json'}")


def verify(path):
    """Check every file in a release against its manifest."""

    out = Path(path)
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [dest for dest, info in manifest["files"].items()
           if not (out / dest).exists() or sha256(out / dest) != info["sha256"]]

    if bad:
        print("MODIFIED OR MISSING: " + ", ".join(bad))
        raise SystemExit(1)

    print(f"{manifest['release']}: all {len(manifest['files'])} files match the manifest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--verify", default=None)
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
        return

    import payload

    model = args.model or payload.DEFAULT_THERMAL
    build(args.name or f"{model}-{datetime.now():%Y%m%d}", model)


if __name__ == "__main__":
    main()
