"""
MT-008 - thermal crowd-separation experiment (mosaic ablation).

The failure-mechanism analysis found that the dominant remaining thermal
failure is not small-object detection but **merged detections**: 43 of the 68
shared localisation failures are a single prediction drawn around two adjacent
people, averaging 2.03x the area of the person it should have covered. Among
successfully detected persons only 4.46% involve a multi-person box, so the
failures are roughly 14x more likely to be merged.

Mosaic augmentation composes four training images into one and downscales them,
which manufactures dense arrangements of small objects that never occur at that
density in the source imagery. The hypothesis is that this teaches the detector
to accept crowded groups as single objects.

MT-008 tests exactly that, and nothing else:

    MT-005   640 x 640, mosaic = 1.0   (baseline)
    MT-008   640 x 640, mosaic = 0.0   (this run)

Every other parameter is copied verbatim from the MT-005 run configuration -
model, dataset, split, epochs, batch, optimizer, learning rate, AMP, and every
other augmentation value - so any difference in the result is attributable to
mosaic alone.

`close_mosaic` is left at its MT-005 value of 10. With mosaic disabled it is
inert, and changing it would be a second variable.

Box-loss weighting is deliberately NOT changed here. The undersized-box failure
mode is a separate mechanism and deserves its own controlled run; altering both
at once would make the result uninterpretable.

Usage:

    python scripts/train_mt008.py
    python scripts/train_mt008.py --name MT-008b --mosaic 0.5
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(
    os.environ.get("HDU_ROOT", Path(__file__).resolve().parents[1])
)

# Absolute, so the run lands beside MT-005/006/007 regardless of the working
# directory this is launched from (the repo is also used via a Git worktree).
TRAINING_ROOT = PROJECT_ROOT / "runs" / "detect" / "results" / "training"

DATA = PROJECT_ROOT / "dataset" / "hit_uav_person" / "data.yaml"

# Copied verbatim from runs/.../MT-005/args.yaml. Only `mosaic` differs.
MT005_CONFIG = {
    "model": "yolo26n.pt",
    "epochs": 50,
    "patience": 100,
    "batch": 8,
    "imgsz": 640,
    "device": 0,
    "workers": 0,
    "pretrained": True,
    "optimizer": "auto",
    "amp": True,
    "rect": False,
    "lr0": 0.01,
    "close_mosaic": 10,
    "augment": False,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.4,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="MT-008")
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="The single variable under test. MT-005 used 1.0.",
    )
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if not DATA.exists():
        raise SystemExit(f"Dataset config not found: {DATA}")

    config = dict(MT005_CONFIG)
    config["mosaic"] = args.mosaic

    print("=" * 62)
    print(f"{args.name} - thermal crowd-separation experiment")
    print("=" * 62)
    print(f"data:    {DATA}")
    print(f"output:  {TRAINING_ROOT / args.name}")
    print(f"mosaic:  {args.mosaic}  (MT-005 baseline used 1.0)")
    print("All other parameters are identical to MT-005.")
    print("=" * 62)

    weights = config.pop("model")

    model = YOLO(weights)

    model.train(
        data=str(DATA),
        project=str(TRAINING_ROOT),
        name=args.name,
        exist_ok=False,
        resume=args.resume,
        **config,
    )


if __name__ == "__main__":
    main()
