"""
Controlled thermal experiments, expressed as a difference from MT-005.

Every parameter not named on the command line is copied verbatim from the
MT-005 run configuration, so any difference in the result is attributable to
the flags that were passed and to nothing else. That is the whole point: it is
why MT-008 through MT-013 are comparable to each other and to the baseline.

It began as the MT-008 mosaic ablation and was generalised as later
experiments needed different variables - dataset, loss weights, starting
weights, architecture, NWD blending and seed. The original motivation is kept
below because it explains the design.

---

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

    python scripts/train_experiment.py
    python scripts/train_experiment.py --name MT-008b --mosaic 0.5
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
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
    parser.add_argument(
        "--data",
        default=None,
        help="Override the dataset config, e.g. the hard-negative variant",
    )
    parser.add_argument(
        "--box",
        type=float,
        default=None,
        help="Box-loss weight. Ultralytics default and MT-005 value is 7.5.",
    )
    parser.add_argument(
        "--dfl",
        type=float,
        default=None,
        help="Distribution focal loss weight. MT-005 used 1.5.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Starting weights, or an architecture YAML. MT-005 used "
             "yolo26n.pt. MT-011 passes MT-004's VisDrone weights; MT-012 "
             "passes yolo26n-p2.yaml.",
    )
    parser.add_argument(
        "--transfer",
        default=None,
        help="Weights to transfer into an architecture given by --model. "
             "Layers whose shapes match are copied, the rest stay random. "
             "Without this, a YAML trains from scratch, which would confound "
             "an architecture change with the loss of COCO pretraining.",
    )
    parser.add_argument(
        "--nwd",
        type=float,
        default=None,
        metavar="RATIO",
        help="Blend Normalized Wasserstein Distance into the localisation "
             "loss: similarity = (1-RATIO)*CIoU + RATIO*NWD. 0.5 is the "
             "usual choice. Training-time only; the exported graph and the "
             "deployed latency are unchanged.",
    )
    parser.add_argument(
        "--nwd-constant",
        type=float,
        default=None,
        help="NWD distance scale in image pixels. Defaults to the measured "
             "mean object size of the HIT-UAV train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed. MT-005 and every experiment since used 0. A "
             "different seed with everything else identical measures how much "
             "of a result is run-to-run variation rather than the change "
             "under test.",
    )
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    data = Path(args.data) if args.data else DATA

    if not data.exists():
        raise SystemExit(f"Dataset config not found: {data}")

    config = dict(MT005_CONFIG)
    config["mosaic"] = args.mosaic

    if args.model is not None:
        config["model"] = args.model

    if args.seed is not None:
        config["seed"] = args.seed

    if args.box is not None:
        config["box"] = args.box

    if args.dfl is not None:
        config["dfl"] = args.dfl

    print("=" * 62)
    print(f"{args.name} - thermal crowd-separation experiment")
    print("=" * 62)
    print(f"data:    {data}")
    print(f"output:  {TRAINING_ROOT / args.name}")
    print(f"mosaic:  {args.mosaic}  (MT-005 baseline used 1.0)")

    if args.box is not None:
        print(f"box:     {args.box}  (MT-005 baseline used 7.5)")

    if args.dfl is not None:
        print(f"dfl:     {args.dfl}  (MT-005 baseline used 1.5)")

    if args.model is not None:
        print(f"model:   {args.model}  (MT-005 baseline used yolo26n.pt)")

    if args.transfer is not None:
        print(f"transfer:{args.transfer}  -> into the architecture above")

    if args.seed is not None:
        print(f"seed:    {args.seed}  (every run so far used 0)")

    if args.nwd is not None:
        print(f"nwd:     {args.nwd}  (MT-005 used pure CIoU)")

    print("All other parameters are identical to MT-005.")
    print("=" * 62)

    if args.nwd is not None:
        import nwd_loss

        nwd_loss.enable(
            ratio=args.nwd,
            constant=args.nwd_constant or nwd_loss.DEFAULT_CONSTANT,
        )

    weights = config.pop("model")

    model = YOLO(weights)

    if args.transfer is not None:
        # Copy every layer whose shape matches. For a P2 architecture this
        # keeps the COCO-pretrained backbone and neck and leaves only the new
        # stride-4 head random, so the run tests the head rather than the
        # absence of pretraining.
        model = model.load(args.transfer)

    model.train(
        data=str(data),
        project=str(TRAINING_ROOT),
        name=args.name,
        exist_ok=False,
        resume=args.resume,
        **config,
    )


if __name__ == "__main__":
    main()
